"""SimpleMLPAdaLN flow MLP, ported from
https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/models/diffloss.py
"""

from __future__ import annotations

import math

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Module
from typing_extensions import Self

from pocket_tts.utils.config import FlowLMConfig


def _modulate(x: TensorValue, shift: TensorValue, scale: TensorValue) -> TensorValue:
    return x * (ops.constant(1.0, scale.dtype, scale.device) + scale) + shift


class _Linear(Module):
    """Linear with bias, weight named `weight`, bias named `bias`."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.has_bias = bias
        self.dtype = dtype
        self._device = device or DeviceRef.CPU()
        self.weight = Weight("weight", dtype, [out_dim, in_dim], device=self._device)
        if bias:
            self.bias = Weight("bias", dtype, [out_dim], device=self._device)

    def __call__(self, x: TensorValue) -> TensorValue:
        out = x @ ops.transpose(self.weight, 0, 1)
        if self.has_bias:
            out = out + self.bias
        return out


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, dtype: DType = DType.float32):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Param attribute name `alpha` matches legacy `RMSNorm.alpha`.
        self.alpha = Weight("alpha", dtype, [dim], device=DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        x_dtype = x.dtype
        x_f32 = ops.cast(x, DType.float32)
        # var(dim=-1, keepdim=True), unbiased
        mean = ops.mean(x_f32, axis=-1)
        # `ops.mean` does not have an unbiased var; compute manually.
        # var = mean((x - mean)**2)  (biased)
        diff = x_f32 - mean
        var = ops.mean(diff * diff, axis=-1)
        eps = ops.constant(self.eps, DType.float32, var.device)
        rms = ops.rsqrt(var + eps)
        alpha_f32 = ops.cast(self.alpha, DType.float32)
        y = x_f32 * (alpha_f32 * rms)
        return ops.cast(y, x_dtype)


class LayerNorm(Module):
    """Reimplementation of LayerNorm matching the legacy non-jvp version."""

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        dtype: DType = DType.float32,
    ):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Weight("weight", dtype, [channels], device=DeviceRef.CPU())
            self.bias = Weight("bias", dtype, [channels], device=DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        mean = ops.mean(x, axis=-1)
        diff = x - mean
        var = ops.mean(diff * diff, axis=-1)
        eps = ops.constant(self.eps, x.dtype, x.device)
        normed = diff / ops.sqrt(var + eps)
        if self.elementwise_affine:
            normed = normed * self.weight + self.bias
        return normed


class TimestepEmbedder(Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        assert frequency_embedding_size % 2 == 0
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.dtype = dtype
        self._device = device or DeviceRef.CPU()
        # mlp = nn.Sequential(Linear, SiLU, Linear, RMSNorm)
        # Use _ModuleList-style wrapper with named attributes mlp.0, mlp.2, mlp.3
        # to match the legacy state dict naming (SiLU at index 1 has no params).
        self.mlp = _Sequential(
            [
                (
                    "linear",
                    _Linear(
                        frequency_embedding_size, hidden_size, dtype=dtype, device=self._device
                    ),
                ),
                ("silu", None),
                ("linear", _Linear(hidden_size, hidden_size, dtype=dtype, device=self._device)),
                ("rms", RMSNorm(hidden_size, dtype=dtype)),
            ]
        )
        # Pre-computed inverse frequencies, used as a constant in the graph.
        half = frequency_embedding_size // 2
        self._freqs_np = np.exp(
            -math.log(max_period) * np.arange(half, dtype=np.float32) / half
        ).astype(self.dtype.to_numpy())

    def __call__(self, t: TensorValue) -> TensorValue:
        freqs = ops.constant(self._freqs_np, self.dtype, t.device)
        # t: (..., 1) — broadcast multiply against freqs (half,)
        args = t * freqs
        embedding = ops.concat([ops.cos(args), ops.sin(args)], axis=-1)
        return self.mlp(embedding)


class _Sequential(Module):
    """Like nn.Sequential — sub-modules indexed by integer attribute names."""

    def __init__(self, items: list[tuple[str, object]]):
        super().__init__()
        self._kinds: list[str] = []
        self._mods: list[object] = []
        for idx, (kind, mod) in enumerate(items):
            self._kinds.append(kind)
            self._mods.append(mod)
            if mod is not None:
                setattr(self, str(idx), mod)

    def __call__(self, x: TensorValue) -> TensorValue:
        for kind, mod in zip(self._kinds, self._mods):
            if kind == "silu":
                x = ops.silu(x)
            elif kind == "linear":
                x = mod(x)
            elif kind == "rms":
                x = mod(x)
            else:
                raise NotImplementedError(kind)
        return x


class ResBlock(Module):
    def __init__(
        self, channels: int, dtype: DType = DType.float32, device: DeviceRef | None = None
    ):
        super().__init__()
        self.channels = channels
        self._device = device or DeviceRef.CPU()
        self.in_ln = LayerNorm(channels, eps=1e-6, dtype=dtype)
        self.mlp = _Sequential(
            [
                ("linear", _Linear(channels, channels, dtype=dtype, device=self._device)),
                ("silu", None),
                ("linear", _Linear(channels, channels, dtype=dtype, device=self._device)),
            ]
        )
        self.adaLN_modulation = _Sequential(
            [
                ("silu", None),
                ("linear", _Linear(channels, 3 * channels, dtype=dtype, device=self._device)),
            ]
        )

    def __call__(self, x: TensorValue, y: TensorValue) -> TensorValue:
        mod_out = self.adaLN_modulation(y)
        # Split into 3 along last dim.
        shift, scale, gate = ops.split(
            mod_out, [self.channels, self.channels, self.channels], axis=-1
        )
        h = _modulate(self.in_ln(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h


class FinalLayer(Module):
    def __init__(
        self,
        model_channels: int,
        out_channels: int,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.out_channels = out_channels
        self._device = device or DeviceRef.CPU()
        self.norm_final = LayerNorm(model_channels, eps=1e-6, elementwise_affine=False, dtype=dtype)
        self.linear = _Linear(model_channels, out_channels, dtype=dtype, device=self._device)
        self.adaLN_modulation = _Sequential(
            [
                ("silu", None),
                (
                    "linear",
                    _Linear(model_channels, 2 * model_channels, dtype=dtype, device=self._device),
                ),
            ]
        )

    def __call__(self, x: TensorValue, c: TensorValue) -> TensorValue:
        mod_out = self.adaLN_modulation(c)
        shift, scale = ops.split(mod_out, [self.model_channels, self.model_channels], axis=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class _ResBlockList(Module):
    """ModuleList equivalent for stacked ResBlocks."""

    def __init__(self, blocks: list[ResBlock]):
        super().__init__()
        self._blocks = blocks
        for i, b in enumerate(blocks):
            setattr(self, str(i), b)

    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return len(self._blocks)

    def __call__(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("Iterate over _ResBlockList; do not call it.")


class _TimeEmbedList(Module):
    """ModuleList equivalent for the small list of TimestepEmbedders."""

    def __init__(self, items: list[TimestepEmbedder]):
        super().__init__()
        self._items = items
        for i, item in enumerate(items):
            setattr(self, str(i), item)

    def __getitem__(self, idx: int) -> TimestepEmbedder:
        return self._items[idx]

    def __len__(self):
        return len(self._items)

    def __call__(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("Use _TimeEmbedList[i] to access an item.")


class SimpleMLPAdaLN(Module):
    """The MLP for Diffusion Loss (flow MLP). See legacy class doc."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        num_time_conds: int = 2,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        assert num_time_conds != 1
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_time_conds = num_time_conds
        self._device = device or DeviceRef.CPU()

        self.time_embed = _TimeEmbedList(
            [
                TimestepEmbedder(model_channels, dtype=dtype, device=self._device)
                for _ in range(num_time_conds)
            ]
        )
        self.cond_embed = _Linear(cond_channels, model_channels, dtype=dtype, device=self._device)
        self.input_proj = _Linear(in_channels, model_channels, dtype=dtype, device=self._device)
        self.res_blocks = _ResBlockList(
            [
                ResBlock(model_channels, dtype=dtype, device=self._device)
                for _ in range(num_res_blocks)
            ]
        )
        self.final_layer = FinalLayer(
            model_channels, out_channels, dtype=dtype, device=self._device
        )

    @classmethod
    def from_pydantic_config(
        cls,
        cfg: FlowLMConfig,
        latent_dim: int,
        cond_dim: int,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ) -> Self:
        config = cfg.flow
        flow_dim = config.dim
        flow_depth = config.depth
        return cls(
            in_channels=latent_dim,
            model_channels=flow_dim,
            out_channels=latent_dim,
            cond_channels=cond_dim,
            num_res_blocks=flow_depth,
            num_time_conds=2,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self, c: TensorValue, s: TensorValue, t: TensorValue, x: TensorValue
    ) -> TensorValue:
        ts_list = [s, t]
        x = self.input_proj(x)
        assert len(ts_list) == self.num_time_conds
        # Average over the time conditioner outputs.
        t_combined = self.time_embed[0](ts_list[0])
        for i in range(1, self.num_time_conds):
            t_combined = t_combined + self.time_embed[i](ts_list[i])
        t_combined = t_combined * ops.constant(
            1.0 / self.num_time_conds, t_combined.dtype, t_combined.device
        )
        c_emb = self.cond_embed(c)
        y = t_combined + c_emb
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)
