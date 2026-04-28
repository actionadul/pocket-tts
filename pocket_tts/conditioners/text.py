"""Text conditioner: SentencePiece tokenizer + embedding lookup table."""

from __future__ import annotations

import logging

import numpy as np
import sentencepiece
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Module

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.utils.utils import download_if_necessary

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
    """Wraps SentencePiece for natural-language conditioning.

    Output is a `TokenizedText(tokens=np.ndarray of shape (1, T))`.
    """

    def __init__(self, nbins: int, tokenizer_path: str) -> None:
        logger.info("Loading sentencepiece tokenizer from %s", tokenizer_path)
        tokenizer_path = download_if_necessary(tokenizer_path)
        self.sp = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
        assert nbins == self.sp.vocab_size(), (
            f"sentencepiece tokenizer has vocab size={self.sp.vocab_size()}"
            f" but nbins={nbins} was specified"
        )

    def __call__(self, text: str) -> TokenizedText:
        tokens = np.asarray(self.sp.encode(text, out_type=int), dtype=np.int64)
        return TokenizedText(tokens[None, :])


DEFAULT_TOKENIZER_N_BINS = 4000
DEFAULT_TOKENIZER_PATH = (
    "hf://kyutai/pocket-tts-without-voice-cloning/"
    "tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
)


def get_default_tokenizer() -> SentencePieceTokenizer:
    return SentencePieceTokenizer(DEFAULT_TOKENIZER_N_BINS, DEFAULT_TOKENIZER_PATH)


class LUTConditioner(Module):
    """Lookup-table text conditioner.

    Splits responsibilities between Python and graph:
    - `prepare(text)`: Python — tokenize + ensure shape (1, T) numpy int64.
    - `__call__(tokens)`: graph — embedding lookup (B, T) -> (B, T, dim).
    """

    def __init__(
        self,
        n_bins: int,
        tokenizer_path: str,
        dim: int,
        output_dim: int,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        if output_dim != dim:
            raise NotImplementedError("LUTConditioner currently only supports dim == output_dim")
        self.dim = dim
        self.output_dim = output_dim
        self.n_bins = n_bins
        self.tokenizer = SentencePieceTokenizer(n_bins, tokenizer_path)
        self._device = device or DeviceRef.CPU()
        # Store the embedding weight directly under `embed.weight` to mirror
        # the legacy weight tree (`conditioner.embed.weight`).
        self.embed = _Embedding(n_bins + 1, dim, dtype=dtype, device=self._device)

    def prepare(self, text: str) -> TokenizedText:
        return self.tokenizer(text)

    def __call__(self, inputs: TokenizedText | TensorValue) -> TensorValue:
        if isinstance(inputs, TokenizedText):
            raise RuntimeError(
                "LUTConditioner.__call__ expects a TensorValue inside the graph; "
                "use prepare() outside the graph and pass the resulting tokens "
                "as a graph input."
            )
        return self.embed(inputs)


class _Embedding(Module):
    """Embedding with PyTorch-compatible weight (no separate `padding_idx`)."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Weight("weight", dtype, [num_embeddings, embedding_dim], device=device)

    def __call__(self, indices: TensorValue) -> TensorValue:
        return ops.gather(self.weight, indices, axis=0)
