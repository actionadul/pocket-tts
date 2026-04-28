"""Mimi neural audio codec, MAX implementation."""

from __future__ import annotations

import logging

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.nn import Module

from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.mimi_transformer import ProjectedTransformer
from pocket_tts.modules.resample import ConvDownsample1d, ConvTrUpsample1d
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder

logger = logging.getLogger(__name__)


class MimiModel(Module):
    def __init__(
        self,
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        quantizer: DummyQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        inner_dim: int | None,
        outer_dim: int | None,
        encoder_transformer: ProjectedTransformer,
        decoder_transformer: ProjectedTransformer,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.encoder_frame_rate = encoder_frame_rate
        self.dtype = dtype
        self._device = device or DeviceRef.CPU()

        dimension = encoder.dimension
        self.dimension = dimension

        if encoder_frame_rate != frame_rate:
            assert self.encoder_frame_rate > self.frame_rate, "Cannot upsample with conv."
            downsample_stride = self.encoder_frame_rate / self.frame_rate
            assert downsample_stride == int(downsample_stride), (
                f"Only integer strides are supported, got {downsample_stride}"
            )
            self.downsample = ConvDownsample1d(
                int(downsample_stride),
                dimension=dimension,
                out_dimension=inner_dim,
                dtype=dtype,
                device=self._device,
            )
            self.upsample = ConvTrUpsample1d(
                int(downsample_stride),
                dimension=dimension,
                in_dimension=outer_dim,
                dtype=dtype,
                device=self._device,
            )

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    def _to_framerate(self, x: TensorValue) -> TensorValue:
        if self.encoder_frame_rate == self.frame_rate:
            return x
        return self.downsample(x, model_state=None)

    def _to_encoder_framerate(self, x: TensorValue, mimi_state) -> TensorValue:
        if self.encoder_frame_rate == self.frame_rate:
            return x
        return self.upsample(x, mimi_state)

    def encode_to_latent(self, x: TensorValue) -> TensorValue:
        """Encoder graph entry-point. Caller is responsible for padding `x`
        to a multiple of `frame_size` before calling this."""
        emb = self.encoder(x, model_state=None)
        (emb,) = self.encoder_transformer(emb, model_state=None)
        emb = self._to_framerate(emb)
        return emb

    def decode_from_latent(self, latent: TensorValue, mimi_state) -> TensorValue:
        emb = self._to_encoder_framerate(latent, mimi_state)
        (emb,) = self.decoder_transformer(emb, mimi_state)
        out = self.decoder(emb, mimi_state)
        return out

    def __call__(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "MimiModel is not directly callable; use encode_to_latent / decode_from_latent."
        )
