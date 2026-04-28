"""Streaming multi-head attention with a per-layer KV cache.

The cache is held *functionally*: K and V are inputs to the graph, and
appended-K, appended-V are outputs. The Python orchestrator threads them
between calls. RoPE position offset is also tracked explicitly.
"""

from __future__ import annotations

import math

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Module

from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule


def _build_attention_mask(
    pos_q: TensorValue, pos_k: TensorValue, context: int | None
) -> TensorValue:
    """Mirror of legacy `_build_attention_mask`.

    Args:
        pos_q: (1, T_q) int positions.
        pos_k: (1, T_k) int positions.
        context: optional sliding-window context width.

    Returns:
        Mask of shape (1, 1, T_q, T_k), True where attention is allowed.
    """
    delta = ops.unsqueeze(pos_q, 2) - ops.unsqueeze(pos_k, 1)  # (1, T_q, T_k)
    pos_k_b = ops.unsqueeze(pos_k, 1)  # (1, 1, T_k)
    nonneg_k = pos_k_b >= ops.constant(0, pos_k.dtype, pos_k.device)
    causal = delta >= ops.constant(0, delta.dtype, delta.device)
    mask = nonneg_k & causal
    if context is not None:
        mask = mask & (delta < ops.constant(context, delta.dtype, delta.device))
    return ops.unsqueeze(mask, 1)  # (1, 1, T_q, T_k)


class _StackedInProj(Module):
    """Linear with weight named `weight` matching legacy `self_attn.in_proj.weight`.

    Output dim is `3 * embed_dim` (Q | K | V interleaved).
    """

    def __init__(self, embed_dim: int, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = Weight("weight", dtype, [3 * embed_dim, embed_dim], device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return x @ ops.transpose(self.weight, 0, 1)


class _LinearNoBias(Module):
    """Linear w/o bias whose weight is named `weight`."""

    def __init__(self, in_dim: int, out_dim: int, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Weight("weight", dtype, [out_dim, in_dim], device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return x @ ops.transpose(self.weight, 0, 1)


class StreamingMultiheadAttention(StatefulModule):
    """Self-attention with rolling KV cache.

    State schema: {"k": (B, T_past, H, D), "v": (B, T_past, H, D),
    "offset": (B,) int64}.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: RotaryEmbedding,
        context: int | None = None,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // num_heads
        self.context = context
        self.rope = rope
        self._device = device or DeviceRef.CPU()
        self.dtype = dtype

        self.in_proj = _StackedInProj(embed_dim, dtype, self._device)
        self.out_proj = _LinearNoBias(embed_dim, embed_dim, dtype, self._device)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, np.ndarray]:
        np_dtype = self.dtype.to_numpy()
        return {
            "k": np.zeros((batch_size, 0, self.num_heads, self.dim_per_head), dtype=np_dtype),
            "v": np.zeros((batch_size, 0, self.num_heads, self.dim_per_head), dtype=np_dtype),
            "offset": np.zeros((batch_size,), dtype=np.int64),
        }

    def increment_step(self, state: dict[str, np.ndarray], increment: int = 1) -> None:
        state["offset"] = state["offset"] + increment

    def __call__(self, query: TensorValue, model_state) -> TensorValue:
        B, T, _ = query.shape
        H = self.num_heads
        D = self.dim_per_head

        projected = self.in_proj(query)
        # Split out Q | K | V along the last axis, then reshape to (B, T, H, D).
        q_, k_, v_ = ops.split(projected, [self.embed_dim, self.embed_dim, self.embed_dim], axis=-1)
        q = ops.reshape(q_, (B, T, H, D))
        k = ops.reshape(k_, (B, T, H, D))
        v = ops.reshape(v_, (B, T, H, D))

        if model_state is None:
            offset_scalar = ops.constant(0, DType.int64, query.device)
            q, k = self.rope(q, k, offset=offset_scalar)
            k_attn = ops.transpose(k, 1, 2)  # (B, H, T_k, D)
            v_attn = ops.transpose(v, 1, 2)
            T_k = k.shape[1]
            pos_k = ops.range(
                ops.constant(0, DType.int64, DeviceRef.CPU()),
                T_k,
                ops.constant(1, DType.int64, DeviceRef.CPU()),
                out_dim=T_k,
                dtype=DType.int64,
                device=query.device,
            )
            pos_k = ops.unsqueeze(pos_k, 0)  # (1, T_k)
            offset_per_b = ops.broadcast_to(offset_scalar, shape=(B,))
        else:
            state = self.get_state(model_state)
            offset_t = state["offset"]
            offset_scalar = ops.cast(offset_t, DType.int64).reshape(())
            q, k = self.rope(q, k, offset=offset_scalar)
            k_full = ops.concat([state["k"], k], axis=1)
            v_full = ops.concat([state["v"], v], axis=1)
            state["k"] = k_full
            state["v"] = v_full
            k_attn = ops.transpose(k_full, 1, 2)
            v_attn = ops.transpose(v_full, 1, 2)
            T_k = k_full.shape[1]
            pos_k = ops.range(
                ops.constant(0, DType.int64, DeviceRef.CPU()),
                T_k,
                ops.constant(1, DType.int64, DeviceRef.CPU()),
                out_dim=T_k,
                dtype=DType.int64,
                device=query.device,
            )
            pos_k = ops.unsqueeze(pos_k, 0)
            offset_per_b = state["offset"]

        # pos_q = offset + arange(T)
        arange_q = ops.range(
            ops.constant(0, DType.int64, DeviceRef.CPU()),
            T,
            ops.constant(1, DType.int64, DeviceRef.CPU()),
            out_dim=T,
            dtype=DType.int64,
            device=query.device,
        )
        pos_q = ops.unsqueeze(offset_per_b, 1) + ops.unsqueeze(arange_q, 0)
        attn_mask = _build_attention_mask(pos_q, pos_k, self.context)

        q_t = ops.transpose(q, 1, 2)  # (B, H, T, D)
        scale = 1.0 / math.sqrt(D)
        scores = q_t @ ops.transpose(k_attn, 2, 3)  # (B, H, T_q, T_k)
        scores = scores * ops.constant(scale, scores.dtype, scores.device)
        # Apply mask: where mask is False, set scores to a very-negative finite
        # value. We avoid -inf because the MAX compiler fuses softmax with
        # ``where(..., -inf)`` in a way that produces NaN downstream when the
        # intermediate isn't materialised.
        neg_huge = ops.constant(-1e30, scores.dtype, scores.device)
        scores = ops.where(attn_mask, scores, neg_huge)
        weights = ops.softmax(scores, axis=-1)
        attn_out = weights @ v_attn  # (B, H, T_q, D)
        attn_out = ops.transpose(attn_out, 1, 2)  # (B, T_q, H, D)
        attn_out = ops.reshape(attn_out, (B, T, self.embed_dim))
        return self.out_proj(attn_out)
