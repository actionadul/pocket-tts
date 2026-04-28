from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.nn import Module

from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d


class ConvDownsample1d(Module):
    """Downsampling by `stride` using a Conv1d with kernel = 2 * stride."""

    def __init__(
        self,
        stride: int,
        dimension: int,
        out_dimension: int | None = None,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        if out_dimension is None:
            out_dimension = dimension
        self.conv = StreamingConv1d(
            in_channels=dimension,
            out_channels=out_dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=1,
            bias=False,
            pad_mode="replicate",
            dtype=dtype,
            device=device,
        )

    def __call__(self, x: TensorValue, model_state) -> TensorValue:
        return self.conv(x, model_state)


class ConvTrUpsample1d(Module):
    """Upsample by `stride` using a depthwise ConvTranspose1d."""

    def __init__(
        self,
        stride: int,
        dimension: int,
        in_dimension: int | None = None,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        if in_dimension is None:
            in_dimension = dimension
        self.convtr = StreamingConvTranspose1d(
            in_channels=in_dimension,
            out_channels=dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=dimension,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def __call__(self, x: TensorValue, model_state) -> TensorValue:
        return self.convtr(x, model_state)
