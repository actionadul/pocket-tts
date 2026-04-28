"""Rotary positional embedding (RoPE).

The math matches the legacy torch implementation in `apply_rope`.
"""

from __future__ import annotations

import math

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module


class RotaryEmbedding(Module):
    def __init__(self, max_period: float | int = 10_000.0):
        super().__init__()
        self.max_period = float(max_period)

    def __call__(
        self, q: TensorValue, k: TensorValue, offset: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Apply RoPE rotation to q and k.

        Args:
            q: (B, T, H, D) query tensor.
            k: (B, T, H, D) key tensor.
            offset: scalar long tensor — number of past positions before this
                slice (RoPE position offset).

        Returns:
            Rotated (q, k) with the same shapes/dtypes as inputs.
        """
        return _apply_rope(q, k, offset=offset, max_period=self.max_period)


def _apply_rope(
    q: TensorValue, k: TensorValue, offset: TensorValue, max_period: float
) -> tuple[TensorValue, TensorValue]:
    B, T, H, D = q.shape
    Bk, Tk, Hk, Dk = k.shape
    assert int(D) == int(Dk)
    assert int(D) % 2 == 0
    D_int = int(D)
    half = D_int // 2

    device = q.device

    # Frequencies depend only on D and max_period (compile-time constant).
    ds = np.arange(half, dtype=np.float32)
    freqs_np = np.exp(ds * (-math.log(max_period) * 2.0 / D_int))
    freqs = ops.constant(freqs_np, DType.float32, device)

    # Positions: offset + arange(T). We build the range dynamically from T
    # (which may be a symbolic dim).
    ts = ops.range(
        ops.constant(0, DType.int64, DeviceRef.CPU()),
        T,
        ops.constant(1, DType.int64, DeviceRef.CPU()),
        out_dim=T,
        dtype=DType.int64,
        device=device,
    )
    offset_cpu = ops.cast(offset, DType.int64).reshape(())
    if offset_cpu.device != device:
        offset_cpu = offset_cpu.to(device)
    ts = ts + offset_cpu
    ts = ops.cast(ts, DType.float32)
    ts = ops.unsqueeze(ts, -1)  # (T, 1)

    angles = ts * freqs  # (T, half)
    rotr = ops.cos(angles)
    roti = ops.sin(angles)
    # Broadcast helpers: (1, T, 1, half)
    rotr = ops.unsqueeze(ops.unsqueeze(rotr, 1), 0)
    roti = ops.unsqueeze(ops.unsqueeze(roti, 1), 0)

    def _rotate(x: TensorValue) -> TensorValue:
        # x: (B, T, H, D) → split last dim into pairs.
        x_pairs = ops.reshape(x, (x.shape[0], x.shape[1], x.shape[2], half, 2))
        xr = x_pairs[..., 0]
        xi = x_pairs[..., 1]
        cosv = ops.cast(rotr, x.dtype)
        sinv = ops.cast(roti, x.dtype)
        xor_ = xr * cosv - xi * sinv
        xoi_ = xr * sinv + xi * cosv
        out = ops.stack([xor_, xoi_], axis=-1)
        return ops.reshape(out, x.shape)

    return _rotate(q), _rotate(k)
