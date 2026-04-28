from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight
from max.nn import Module


class LayerScale(Module):
    def __init__(self, channels: int, init: float, dtype: DType = DType.float32):
        super().__init__()
        del init  # only used at training time
        self.scale = Weight("scale", dtype, [channels], device=DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.scale.cast(x.dtype) * x
