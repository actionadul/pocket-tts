"""Streaming Conv1d / ConvTranspose1d on top of MAX ops.

All modules in this file work in PyTorch channel-first ordering, i.e.
tensors of shape (B, C, T).
"""

from __future__ import annotations

import math

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.graph.type import ConvInputLayout, FilterLayout
from max.nn import Module

from pocket_tts.modules.stateful_module import StatefulModule


def get_extra_padding_for_conv1d(
    length: int, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d_numpy(x: np.ndarray, kernel_size: int, stride: int, padding_total: int = 0):
    extra_padding = get_extra_padding_for_conv1d(x.shape[-1], kernel_size, stride, padding_total)
    if extra_padding <= 0:
        return x
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, extra_padding)])


class _NativeConv1d(Module):
    """1-D convolution with PyTorch-style weight layout (out_C, in_C/groups, K)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        self._device = device or DeviceRef.CPU()
        self.weight = Weight(
            "weight", dtype, [out_channels, in_channels // groups, kernel_size], device=self._device
        )
        if bias:
            self.bias = Weight("bias", dtype, [out_channels], device=self._device)

    def __call__(self, x: TensorValue) -> TensorValue:
        # x: (B, C, T) -> NHWC (B, 1, T, C)
        x = ops.transpose(x, 1, 2)
        x = ops.unsqueeze(x, 1)
        weight = ops.unsqueeze(self.weight, 2)  # (out, in/g, 1, K) FCRS
        bias = self.bias if self.has_bias else None
        out = ops.conv2d(
            x,
            weight,
            stride=(1, self.stride),
            dilation=(1, self.dilation),
            padding=(0, 0, 0, 0),
            groups=self.groups,
            bias=bias,
            input_layout=ConvInputLayout.NHWC,
            filter_layout=FilterLayout.FCRS,
        )
        out = ops.squeeze(out, 1)
        return ops.transpose(out, 1, 2)


class _NativeConvTranspose1d(Module):
    """1-D transposed convolution with PyTorch-style weight layout
    ``(in_C, out_C, K)``.

    Currently restricted to ``kernel_size == 2 * stride``. The implementation
    avoids :func:`max.graph.ops.conv2d_transpose` because that op fails to
    lower to a CPU kernel when its filter is a :class:`~max.graph.Weight`
    (it cannot infer the ``num_groups`` template parameter). Instead we
    expand the conv-transpose into two matmuls plus a shift+add, which the
    compiler handles cleanly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        if kernel_size != 2 * stride:
            raise NotImplementedError(
                "_NativeConvTranspose1d currently assumes kernel_size == 2 * stride;"
                f" got kernel_size={kernel_size}, stride={stride}"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.has_bias = bias
        self._device = device or DeviceRef.CPU()
        self.weight = Weight(
            "weight", dtype, [in_channels, out_channels, kernel_size], device=self._device
        )
        if bias:
            self.bias = Weight("bias", dtype, [out_channels], device=self._device)

    def __call__(self, x: TensorValue) -> TensorValue:
        S = self.stride
        in_C = self.in_channels
        out_C = self.out_channels
        B = x.shape[0]
        T = x.shape[2]

        # Split the weight at the kernel midpoint. With kernel = 2 * stride,
        # the "current" half maps an input sample to its own S output slots,
        # the "previous" half maps it to the next S slots.
        w = self.weight  # (in_C, out_C, 2S)
        w_curr = w[..., :S]  # (in_C, out_C, S)
        w_prev = w[..., S:]  # (in_C, out_C, S)
        w_curr_2d = ops.reshape(w_curr, (in_C, out_C * S))
        w_prev_2d = ops.reshape(w_prev, (in_C, out_C * S))

        # x: (B, in_C, T) -> (B, T, in_C)
        x_bt = ops.transpose(x, 1, 2)
        curr = x_bt @ w_curr_2d  # (B, T, out_C * S)
        prev = x_bt @ w_prev_2d  # (B, T, out_C * S)
        curr = ops.reshape(curr, (B, T, out_C, S))
        prev = ops.reshape(prev, (B, T, out_C, S))
        # Bring out_C in front, flatten T and S.
        curr = ops.transpose(curr, 1, 2)  # (B, out_C, T, S)
        prev = ops.transpose(prev, 1, 2)
        curr = ops.reshape(curr, (B, out_C, T * S))
        prev = ops.reshape(prev, (B, out_C, T * S))

        zeros_pad = ops.broadcast_to(ops.constant(0.0, x.dtype, x.device), shape=(B, out_C, S))
        curr_padded = ops.concat([curr, zeros_pad], axis=2)
        prev_padded = ops.concat([zeros_pad, prev], axis=2)
        y = curr_padded + prev_padded
        if self.has_bias:
            y = y + ops.unsqueeze(self.bias, -1)
        return y


class _DepthwiseConvTranspose1d(Module):
    """Depthwise 1-D transposed conv (groups == channels) with kernel == 2 * stride.

    Implemented in-place because MAX's `conv2d_transpose` does not expose a
    `groups` parameter. With kernel = 2*S and stride = S, no padding,
    the output `y` of shape (B, C, T*S + S) is:

        y[..., j] = x[..., j // S] * w[..., j % S]
                  + (j >= S ? x[..., j // S - 1] * w[..., j % S + S] : 0)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        if kernel_size != 2 * stride:
            raise NotImplementedError(
                "depthwise conv-transpose currently assumes kernel_size == 2 * stride"
            )
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.has_bias = bias
        self._device = device or DeviceRef.CPU()
        self.weight = Weight("weight", dtype, [channels, 1, kernel_size], device=self._device)
        if bias:
            self.bias = Weight("bias", dtype, [channels], device=self._device)

    def __call__(self, x: TensorValue) -> TensorValue:
        S = self.stride
        C = self.channels
        B = x.shape[0]
        T = x.shape[2]
        weight2d = ops.reshape(self.weight, (C, 2 * S))
        w_curr = weight2d[:, :S]
        w_prev = weight2d[:, S:]
        w_curr_b = ops.unsqueeze(ops.unsqueeze(w_curr, 0), 2)  # (1, C, 1, S)
        w_prev_b = ops.unsqueeze(ops.unsqueeze(w_prev, 0), 2)
        x_unsq = ops.unsqueeze(x, -1)  # (B, C, T, 1)
        curr = x_unsq * w_curr_b
        prev = x_unsq * w_prev_b
        curr = ops.reshape(curr, (B, C, T * S))
        prev = ops.reshape(prev, (B, C, T * S))
        zeros_pad = ops.broadcast_to(ops.constant(0.0, x.dtype, x.device), shape=(B, C, S))
        curr_padded = ops.concat([curr, zeros_pad], axis=2)
        prev_padded = ops.concat([zeros_pad, prev], axis=2)
        y = curr_padded + prev_padded
        if self.has_bias:
            y = y + ops.unsqueeze(self.bias, -1)
        return y


class StreamingConv1d(StatefulModule):
    """Streaming Conv1d that prepends a (kernel - stride) tail from the previous
    chunk before convolving.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        assert pad_mode in ("constant", "replicate"), pad_mode
        self.pad_mode = pad_mode
        self.in_channels = in_channels
        self.dtype = dtype
        self.conv = _NativeConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            dtype=dtype,
            device=device or DeviceRef.CPU(),
        )

    @property
    def _stride(self) -> int:
        return self.conv.stride

    @property
    def _kernel_size(self) -> int:
        return self.conv.kernel_size

    @property
    def _effective_kernel_size(self) -> int:
        return (self.conv.kernel_size - 1) * self.conv.dilation + 1

    @property
    def _tail_size(self) -> int:
        return max(0, self._effective_kernel_size - self._stride)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, np.ndarray]:
        np_dtype = self.dtype.to_numpy()
        previous = np.zeros((batch_size, self.in_channels, self._tail_size), dtype=np_dtype)
        first = np.ones((batch_size,), dtype=np.bool_)
        return {"previous": previous, "first": first}

    def __call__(self, x: TensorValue, model_state) -> TensorValue:
        TP = self._tail_size
        if model_state is None:
            if TP == 0:
                return self.conv(x)
            if self.pad_mode == "replicate":
                init = x[..., :1]
                tail = ops.broadcast_to(init, shape=(x.shape[0], x.shape[1], TP))
            else:
                tail = ops.broadcast_to(
                    ops.constant(0.0, x.dtype, x.device), shape=(x.shape[0], x.shape[1], TP)
                )
            return self.conv(ops.concat([tail, x], axis=2))

        # Streaming path: pull (previous, first) from state.
        state = self.get_state(model_state)
        if TP == 0:
            return self.conv(x)

        previous = state["previous"]
        if self.pad_mode == "replicate":
            init = x[..., :1]
            init_tail = ops.broadcast_to(init, shape=(x.shape[0], x.shape[1], TP))
            first_b = state["first"]
            first_b = ops.reshape(first_b, (x.shape[0], 1, 1))
            previous_eff = ops.where(first_b, init_tail, previous)
            new_first = ops.broadcast_to(
                ops.constant(False, DType.bool, x.device), shape=state["first"].shape
            )
        else:
            previous_eff = previous
            new_first = state["first"]

        x_padded = ops.concat([previous_eff, x], axis=2)
        new_previous = x_padded[..., -TP:]
        # Update state in place — the model_state dict accumulates new values
        # that the orchestrator returns as graph outputs.
        state["previous"] = new_previous
        state["first"] = new_first
        return self.conv(x_padded)


class StreamingConvTranspose1d(StatefulModule):
    """Streaming ConvTranspose1d: the tail of the previous chunk's output is
    saved and added to the start of the next chunk's output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.has_bias = bias
        self.dtype = dtype
        if groups == 1:
            self.convtr = _NativeConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                dtype=dtype,
                device=device or DeviceRef.CPU(),
            )
        elif groups == in_channels and in_channels == out_channels:
            self.convtr = _DepthwiseConvTranspose1d(
                channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                dtype=dtype,
                device=device or DeviceRef.CPU(),
            )
        else:
            raise NotImplementedError(
                "StreamingConvTranspose1d only supports groups == 1 or depthwise"
                f" (groups == in_channels == out_channels); got"
                f" groups={groups}, in={in_channels}, out={out_channels}"
            )

    @property
    def _stride(self) -> int:
        return self.stride

    @property
    def _kernel_size(self) -> int:
        return self.kernel_size

    @property
    def _tail_size(self) -> int:
        return max(0, self._kernel_size - self._stride)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, np.ndarray]:
        np_dtype = self.dtype.to_numpy()
        partial = np.zeros((batch_size, self.out_channels, self._tail_size), dtype=np_dtype)
        return {"partial": partial}

    def __call__(self, x: TensorValue, model_state) -> TensorValue:
        y = self.convtr(x)
        PT = self._tail_size
        if PT <= 0:
            return y
        if model_state is None:
            return y[..., :-PT]

        state = self.get_state(model_state)
        partial = state["partial"]
        head = y[..., :PT] + partial
        body = y[..., PT:-PT]
        tail = y[..., -PT:]
        if self.has_bias:
            tail_for_state = tail - ops.unsqueeze(self.convtr.bias, -1)
        else:
            tail_for_state = tail
        state["partial"] = tail_for_state
        return ops.concat([head, body], axis=2)
