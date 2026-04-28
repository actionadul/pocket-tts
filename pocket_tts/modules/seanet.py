from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module

from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d


def _elu(x: TensorValue, alpha: float = 1.0) -> TensorValue:
    """Elementwise ELU: x if x >= 0 else alpha * (exp(x) - 1)."""
    zero = ops.constant(0.0, x.dtype, x.device)
    pos_mask = x >= zero
    neg_part = ops.constant(alpha, x.dtype, x.device) * (ops.exp(x) - 1.0)
    return ops.where(pos_mask, x, neg_part)


class SEANetResnetBlock(Module):
    def __init__(
        self,
        dim: int,
        kernel_sizes: list[int] | None = None,
        dilations: list[int] | None = None,
        pad_mode: str = "reflect",
        compress: int = 2,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 1]
        if dilations is None:
            dilations = [1, 1]
        assert len(kernel_sizes) == len(dilations)
        hidden = dim // compress
        self._block_layers: list[tuple[str, object]] = []
        # We mirror the legacy nn.ModuleList where Conv1d's are interleaved
        # with ELU activations. To keep weight names stable, conv layers go
        # under `block.<index>` (index matches the legacy nn.ModuleList).
        block_holder = _ModuleList()
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            # Activation slot.
            block_holder.append(("elu", None))
            # Conv slot.
            conv = StreamingConv1d(
                in_channels=in_chs,
                out_channels=out_chs,
                kernel_size=kernel_size,
                dilation=dilation,
                pad_mode=pad_mode,
                dtype=dtype,
                device=device,
            )
            block_holder.append(("conv", conv))
        self.block = block_holder

    def __call__(self, x: TensorValue, model_state) -> TensorValue:
        v = x
        for kind, layer in self.block.items():
            if kind == "elu":
                v = _elu(v, alpha=1.0)
            else:
                v = layer(v, model_state)
        return x + v


class _ModuleList(Module):
    """A minimal ModuleList equivalent that registers child modules under
    integer attribute names (`0`, `1`, `2`, ...), matching the legacy
    PyTorch `nn.ModuleList` weight-FQN convention.

    Each entry stores a `(kind, module)` tuple. `kind` is informational only
    — it says whether the slot holds an activation or a real module. Slots
    with `module is None` (e.g. activations) do not contribute to the weight
    tree. Real modules are also exposed as `self.<index>` attributes so the
    `Module._iter_named_weights` walker picks them up under `<index>.<...>`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._kinds: list[str] = []
        self._modules: list[object] = []

    def append(self, item: tuple[str, object]) -> None:
        kind, module = item
        idx = len(self._kinds)
        self._kinds.append(kind)
        self._modules.append(module)
        if module is not None:
            setattr(self, str(idx), module)

    def items(self):
        return zip(self._kinds, self._modules)

    def __len__(self) -> int:
        return len(self._kinds)

    def __call__(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("ModuleList is iteration-only; call its members directly.")


class SEANetEncoder(Module):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: list[int] | None = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "reflect",
        compress: int = 2,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        if ratios is None:
            ratios = [8, 5, 4, 2]
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2

        mult = 1
        model = _ModuleList()
        model.append(
            (
                "conv",
                StreamingConv1d(
                    in_channels=channels,
                    out_channels=mult * n_filters,
                    kernel_size=kernel_size,
                    pad_mode=pad_mode,
                    dtype=dtype,
                    device=device,
                ),
            )
        )
        for i, ratio in enumerate(self.ratios):
            for j in range(n_residual_layers):
                model.append(
                    (
                        "resnet",
                        SEANetResnetBlock(
                            dim=mult * n_filters,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base**j, 1],
                            pad_mode=pad_mode,
                            compress=compress,
                            dtype=dtype,
                            device=device,
                        ),
                    )
                )
            model.append(("elu", None))
            model.append(
                (
                    "conv",
                    StreamingConv1d(
                        in_channels=mult * n_filters,
                        out_channels=mult * n_filters * 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        pad_mode=pad_mode,
                        dtype=dtype,
                        device=device,
                    ),
                )
            )
            mult *= 2
        model.append(("elu", None))
        model.append(
            (
                "conv",
                StreamingConv1d(
                    in_channels=mult * n_filters,
                    out_channels=dimension,
                    kernel_size=last_kernel_size,
                    pad_mode=pad_mode,
                    dtype=dtype,
                    device=device,
                ),
            )
        )
        self.model = model

    def __call__(self, x: TensorValue, model_state) -> TensorValue:
        for kind, layer in self.model.items():
            if kind == "elu":
                x = _elu(x, alpha=1.0)
            elif kind == "conv":
                x = layer(x, model_state)
            else:  # resnet
                x = layer(x, model_state)
        return x


class SEANetDecoder(Module):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: list[int] | None = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "reflect",
        compress: int = 2,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        if ratios is None:
            ratios = [8, 5, 4, 2]
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2
        mult = int(2 ** len(self.ratios))
        model = _ModuleList()
        model.append(
            (
                "conv",
                StreamingConv1d(
                    in_channels=dimension,
                    out_channels=mult * n_filters,
                    kernel_size=kernel_size,
                    pad_mode=pad_mode,
                    dtype=dtype,
                    device=device,
                ),
            )
        )
        for ratio in self.ratios:
            model.append(("elu", None))
            model.append(
                (
                    "convtr",
                    StreamingConvTranspose1d(
                        in_channels=mult * n_filters,
                        out_channels=mult * n_filters // 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        groups=1,
                        bias=True,
                        dtype=dtype,
                        device=device,
                    ),
                )
            )
            for j in range(n_residual_layers):
                model.append(
                    (
                        "resnet",
                        SEANetResnetBlock(
                            dim=mult * n_filters // 2,
                            kernel_sizes=[residual_kernel_size, 1],
                            dilations=[dilation_base**j, 1],
                            pad_mode=pad_mode,
                            compress=compress,
                            dtype=dtype,
                            device=device,
                        ),
                    )
                )
            mult //= 2
        model.append(("elu", None))
        model.append(
            (
                "conv",
                StreamingConv1d(
                    in_channels=n_filters,
                    out_channels=channels,
                    kernel_size=last_kernel_size,
                    pad_mode=pad_mode,
                    dtype=dtype,
                    device=device,
                ),
            )
        )
        self.model = model

    def __call__(self, z: TensorValue, model_state) -> TensorValue:
        for kind, layer in self.model.items():
            if kind == "elu":
                z = _elu(z, alpha=1.0)
            elif kind in ("conv", "convtr"):
                z = layer(z, model_state)
            else:  # resnet
                z = layer(z, model_state)
        return z
