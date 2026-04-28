"""Weight loading from safetensors. Returns numpy float32 arrays.

Files in the wild may use bfloat16 storage; we convert to float32 here so
the rest of the pipeline can work with plain numpy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import SafetensorWeights


def _to_float32_numpy(weight_data) -> np.ndarray:
    data = weight_data.astype(DType.float32)
    buf = Buffer.from_dlpack(data.data) if not isinstance(data.data, Buffer) else data.data
    return buf.to_numpy().copy()


def _to_native_numpy(weight_data) -> np.ndarray:
    data = weight_data
    if data.dtype == DType.bfloat16:
        return _to_float32_numpy(data)
    buf = Buffer.from_dlpack(data.data) if not isinstance(data.data, Buffer) else data.data
    return buf.to_numpy().copy()


def _items(path: Path):
    sw = SafetensorWeights([str(path)])
    return list(sw.items())


def get_flow_lm_state_dict(path: Path) -> dict[str, np.ndarray]:
    state_dict: dict[str, np.ndarray] = {}
    for key, weight in _items(path):
        if (
            key.startswith("flow.w_s_t.")
            or key == "condition_provider.conditioners.transcript_in_segment.learnt_padding"
            or key == "condition_provider.conditioners.speaker_wavs.learnt_padding"
            or key == "num_ema_updates"
        ):
            continue
        new_name = key
        if key == "condition_provider.conditioners.transcript_in_segment.embed.weight":
            new_name = "conditioner.embed.weight"
        if key == "condition_provider.conditioners.speaker_wavs.output_proj.weight":
            new_name = "speaker_proj_weight"
        if key == "fuser.padding_value":
            new_name = "bos_before_voice"
        new_name = new_name.replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight")
        state_dict[new_name] = _to_native_numpy(weight.data())
    return state_dict


def _weight_norm(weight_v: np.ndarray, weight_g: np.ndarray, dim: int = 0) -> np.ndarray:
    """Equivalent of `torch._weight_norm(weight_v, weight_g, dim=0)`."""
    axes = tuple(i for i in range(weight_v.ndim) if i != dim)
    norm = np.sqrt((weight_v.astype(np.float32) ** 2).sum(axis=axes, keepdims=True))
    return ((weight_v / norm) * weight_g).astype(weight_v.dtype, copy=False)


def get_mimi_state_dict(path: Path) -> dict[str, np.ndarray]:
    items = dict(_items(path))
    state_dict: dict[str, np.ndarray] = {}
    for key in items.keys():
        if (
            key.startswith("model.quantizer.vq.")
            or key == "model.quantizer.logvar_proj.weight"
            or "_codebook" in key
            or key.endswith(".weight_v")
            or key == "quantizer.logvar_proj.weight"
        ):
            continue
        if key.endswith(".weight_g"):
            base = key[: -len("_g")]
            key_v = base + "_v"
            weight_v = _to_native_numpy(items[key_v].data())
            weight_g = _to_native_numpy(items[key].data())
            new_key = base.replace(".conv.conv.", ".conv.").replace(".convtr.convtr.", ".convtr.")
            state_dict[new_key] = _weight_norm(weight_v, weight_g, dim=0)
            continue
        if key in (
            "wavlm_emb_downsample.conv.conv.weight",
            "wavlm_input_resample.kernel",
            "wavlm_proj.weight",
            "quantizer.logvar_param",
        ):
            continue
        if "wavlm_emb_downsample" in key:
            continue
        new_key = (
            key.removeprefix("model.")
            .replace(".conv.conv.", ".conv.")
            .replace(".convtr.convtr.", ".convtr.")
            .replace("in_proj_weight", "in_proj.weight")
        )
        state_dict[new_key] = _to_native_numpy(items[key].data())
    return state_dict


def load_top_level_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Load a TTSModel-style flat safetensors file as numpy arrays."""
    return {key: _to_native_numpy(weight.data()) for key, weight in _items(path)}
