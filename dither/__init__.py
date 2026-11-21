"""Dither and quantization primitives for the audio-dither-quantize exercise."""

from .quantize import (
    DitherError,
    apply_dither,
    dequantize,
    dither_batch,
    measure_snr,
    noise_shape,
    quantize,
    rectangular_noise,
    tpdf_noise,
)

__all__ = [
    "DitherError",
    "apply_dither",
    "dequantize",
    "dither_batch",
    "measure_snr",
    "noise_shape",
    "quantize",
    "rectangular_noise",
    "tpdf_noise",
]
