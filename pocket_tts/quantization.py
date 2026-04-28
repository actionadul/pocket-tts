"""Quantization stub.

The legacy implementation relied on torch's quantization runtime; it has
not been ported to MAX. The functions are kept so existing callers don't
have to special-case the import.
"""

RECOMMENDED_CONFIG: set = set()


def apply_dynamic_int8(model, config) -> None:  # pragma: no cover
    raise NotImplementedError(
        "int8 quantization is not yet implemented for the MAX-based pocket-tts; "
        "please run with quantize=False."
    )
