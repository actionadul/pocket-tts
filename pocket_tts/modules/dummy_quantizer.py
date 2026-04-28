from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Module


class DummyQuantizer(Module):
    """Simplified quantizer that only provides output projection for TTS."""

    def __init__(
        self,
        dimension: int,
        output_dimension: int,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.dtype = dtype
        self._device = device or DeviceRef.CPU()
        # Conv1d with kernel=1, no bias. Weight shape (out_C, in_C, 1) matches
        # PyTorch's nn.Conv1d.weight layout.
        self.output_proj = _Conv1dKernel1(dimension, output_dimension, dtype, self._device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.output_proj(x)


class _Conv1dKernel1(Module):
    """Equivalent to nn.Conv1d(in, out, kernel_size=1, bias=False).

    Implemented as a 1x1 conv in time, i.e. a per-position linear map.
    Weight name and shape match torch's `output_proj.weight` layout
    `(out_channels, in_channels, 1)`.
    """

    def __init__(
        self, in_channels: int, out_channels: int, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Weight("weight", dtype, [out_channels, in_channels, 1], device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        # x: (B, in_C, T) -> matmul with W (out_C, in_C) per time step.
        # Reshape weight to (out_C, in_C), apply x as (B, in_C, T) -> (B, out_C, T).
        w = self.weight.reshape((self.out_channels, self.in_channels))
        # x: (B, C, T) -> transpose to (B, T, C); matmul with w.T -> (B, T, out_C); transpose back.
        x_bt = ops.transpose(x, 1, 2)  # (B, T, in_C)
        y = x_bt @ ops.transpose(w, 0, 1)  # (B, T, out_C)
        return ops.transpose(y, 1, 2)
