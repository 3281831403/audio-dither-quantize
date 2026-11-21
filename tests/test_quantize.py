"""Contract tests for the audio-dither-quantize exercise.

Every test calls a stub, so the unmodified repository is expected to show 15
red tests.  Audio is a normalized ``list[float]``; the free-standing noise
generators use a 16-bit reference LSB of ``2 ** -15``.  All tests are offline,
deterministic, and contain no sleeps.
"""

import math

import pytest

from dither.quantize import (
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

LSB16 = 2.0 ** -15


def _signal(n=64):
    """A small deterministic signal with no randomness or external data."""
    return [((i * 37) % 101 - 50) / 500.0 for i in range(n)]


def test_tpdf_and_rectangular_noise_have_one_lsb_amplitude():
    """TPDF peaks at 1 LSB and RPDF spans 1 LSB, not twice that."""
    n = 4096
    triangular = tpdf_noise(20261119, n)
    rectangular = rectangular_noise(20261119, n)

    assert len(triangular) == n
    assert len(rectangular) == n

    triangular_peak = max(abs(value) for value in triangular)
    rectangular_peak = max(abs(value) for value in rectangular)
    triangular_rms = math.sqrt(sum(value * value for value in triangular) / n)
    rectangular_rms = math.sqrt(sum(value * value for value in rectangular) / n)

    assert triangular_peak <= 1.001 * LSB16
    assert triangular_peak >= 0.80 * LSB16
    assert 0.30 * LSB16 <= triangular_rms <= 0.55 * LSB16
    assert rectangular_peak <= 0.501 * LSB16
    assert rectangular_peak >= 0.40 * LSB16
    assert 0.20 * LSB16 <= rectangular_rms <= 0.40 * LSB16


def test_noise_is_deterministic_and_seeded():
    """Noise must be reproducible even after another generation call."""
    seed = 424242
    first = tpdf_noise(seed, 64)
    tpdf_noise(seed + 1, 128)
    second = tpdf_noise(seed, 64)

    assert first == second
    assert tpdf_noise(seed + 2, 64) != first

    rect_first = rectangular_noise(seed, 64)
    rectangular_noise(seed + 1, 128)
    assert rectangular_noise(seed, 64) == rect_first
    assert rectangular_noise(seed + 2, 64) != rect_first


def test_quantize_rounds_to_nearest_not_truncates():
    """Values above a half-LSB boundary must round up, not toward zero."""
    bit_depth = 8
    lsb = 2.0 ** (1 - bit_depth)
    samples = [2.6 * lsb, -2.6 * lsb, 0.49 * lsb, -0.49 * lsb, 7.7 * lsb]

    assert quantize(samples, bit_depth) == [3, -3, 0, 0, 8]


def test_quantize_saturates_instead_of_wrapping():
    """Out-of-range peaks must clip at the signed code limits."""
    bit_depth = 8
    samples = [1.5, -1.5, 1.0, -1.0, 3.0, -3.0]

    assert quantize(samples, bit_depth) == [127, -128, 127, -128, 127, -128]


def test_dequantize_uses_signed_full_scale():
    """Codes must be divided by 2 ** (bits - 1), not left unscaled."""
    codes = [-128, -64, 0, 64, 127]

    result = dequantize(codes, 8)

    assert result == pytest.approx(
        [-1.0, -0.5, 0.0, 0.5, 127.0 / 128.0], abs=1e-12
    )


def test_noise_shape_rejects_unstable_feedback():
    """A first-order shape with |c| >= 1 must fail instead of diverging."""
    shaped = noise_shape([0.25] * 80, [0.5])

    assert len(shaped) == 80
    assert all(math.isfinite(value) for value in shaped)
    assert max(abs(value) for value in shaped) <= 1.0

    with pytest.raises(DitherError):
        noise_shape([0.1] * 8, [1.25])


def test_measure_snr_uses_decibel_scale():
    """A 100:1 power ratio is 20 dB, not the raw ratio 100."""
    original = [0.5] * 64
    quantized = [0.55] * 64

    assert measure_snr(original, quantized) == pytest.approx(20.0, abs=1e-6)


def test_measure_snr_silence_returns_infinity():
    """Silence has no finite SNR and must not cause a division by zero."""
    snr = measure_snr([0.0] * 32, [0.0] * 32)

    assert math.isinf(snr)
    assert snr > 0.0


def test_dither_batch_uses_independent_noise_per_track():
    """Each track gets its own seed and an invalid track aborts atomically."""
    samples = _signal(64)
    tracks = [
        {"name": "left", "samples": samples},
        {"name": "right", "samples": samples},
    ]
    config = {"bit_depth": 8, "shape": None, "seed": 123}

    result = dither_batch(tracks, config)

    assert result["bit_depth"] == 8
    assert len(result["tracks"]) == 2
    assert result["tracks"][0]["codes"] == apply_dither(samples, 8, None, 123)
    assert result["tracks"][1]["codes"] == apply_dither(samples, 8, None, 124)

    bad_tracks = [
        {"name": "ok", "samples": [0.1] * 8},
        {"name": "bad", "samples": [0.1, "not-a-sample", 0.2]},
    ]
    snapshot = [(track["name"], list(track["samples"])) for track in bad_tracks]

    with pytest.raises(DitherError):
        dither_batch(bad_tracks, config)
    assert [
        (track["name"], list(track["samples"])) for track in bad_tracks
    ] == snapshot


def test_bit_depth_below_one_raises_dither_error():
    """Negative bit depths must be rejected by every depth-taking entry point."""
    for bad_depth in (-1, -7):
        with pytest.raises(DitherError):
            quantize([0.0], bad_depth)
        with pytest.raises(DitherError):
            dequantize([0], bad_depth)
        with pytest.raises(DitherError):
            apply_dither([0.0], bad_depth, None, 7)


def test_quantize_16_bit_error_within_half_lsb():
    """In-range 16-bit quantization error must stay within half an LSB."""
    bit_depth = 16
    lsb = 2.0 ** (1 - bit_depth)
    samples = [0.1, 0.3, 0.7, -0.1, -0.3, -0.7, 0.12345, -0.54321, 0.9]

    reconstructed = dequantize(quantize(samples, bit_depth), bit_depth)

    assert len(reconstructed) == len(samples)
    for original, result in zip(samples, reconstructed):
        assert abs(original - result) <= 0.5 * lsb + 1e-12


def test_same_seed_produces_identical_noise():
    """Two calls with the same explicit seed must return equal sequences."""
    assert tpdf_noise(777, 128) == tpdf_noise(777, 128)
    assert rectangular_noise(777, 128) == rectangular_noise(777, 128)


def test_zero_bit_depth_raises_dither_error():
    """bit_depth == 0 must raise DitherError instead of dividing by zero."""
    with pytest.raises(DitherError):
        quantize([0.1, -0.1], 0)
    with pytest.raises(DitherError):
        apply_dither([0.0], 0, None, 0)


def test_quantize_silence_stays_zero():
    """Digital silence has code zero and reconstructs to exactly 0.0."""
    codes = quantize([0.0] * 32, 12)

    assert codes == [0] * 32
    assert dequantize(codes, 12) == [0.0] * 32


def test_dither_batch_preserves_track_order_on_success():
    """Successful batches keep input order and use per-track seeds and shape."""
    tracks = [
        {"name": "intro", "samples": [0.0, 0.1, -0.1, 0.25]},
        {"name": "body", "samples": _signal(33)},
        {"name": "outro", "samples": [0.4, -0.4] * 8},
    ]
    config = {"bit_depth": 10, "shape": [0.25], "seed": 42}

    result = dither_batch(tracks, config)

    assert result["bit_depth"] == 10
    assert [track["name"] for track in result["tracks"]] == [
        "intro",
        "body",
        "outro",
    ]
    for index, (source, output) in enumerate(zip(tracks, result["tracks"])):
        assert len(output["codes"]) == len(source["samples"])
        assert output["codes"] == apply_dither(
            source["samples"], 10, [0.25], 42 + index
        )
