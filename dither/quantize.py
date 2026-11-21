"""Dither and quantization primitives for the audio-dither-quantize exercise.

Audio is represented as a normalized ``list[float]``.  For a signed ``b``-bit
format, sample ``1.0`` corresponds to ``2 ** (b - 1)`` codes, the code range is
``[-2 ** (b - 1), 2 ** (b - 1) - 1]``, and one LSB is ``2 ** (1 - b)`` in
normalized units.  The free-standing noise generators use a 16-bit reference
LSB, ``2 ** -15``.  Floating-point comparisons use a tolerance of one LSB
unless a tighter absolute tolerance is stated.
"""

from __future__ import annotations

import hashlib
import math

LSB16 = 2.0 ** -15
_HASH_DOMAIN = 1 << 256


class DitherError(ValueError):
    """Raised when dither parameters, audio data, or batch tracks are invalid."""


def _is_real_int(value: object) -> bool:
    """Return True for genuine ints (booleans are rejected as not real ints)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_seed(seed: object) -> None:
    if not _is_real_int(seed):
        raise DitherError(f"seed must be an integer, got {type(seed).__name__}")


def _validate_count(n: object) -> None:
    if not _is_real_int(n) or n < 0:
        raise DitherError(f"n must be a non-negative integer, got {n!r}")


def _validate_bit_depth(bit_depth: object) -> None:
    if not _is_real_int(bit_depth) or bit_depth < 1:
        raise DitherError(
            f"bit_depth must be an integer >= 1, got {bit_depth!r}"
        )


def _validate_samples(samples: object, *, allow_empty: bool = True) -> list:
    if not isinstance(samples, list):
        raise DitherError(
            f"samples must be a list, got {type(samples).__name__}"
        )
    if not allow_empty and len(samples) == 0:
        raise DitherError("samples must not be empty")
    for sample in samples:
        if not isinstance(sample, (int, float)) or isinstance(sample, bool):
            raise DitherError(f"sample must be numeric, got {sample!r}")
        if not math.isfinite(sample):
            raise DitherError(f"sample must be finite, got {sample!r}")
    return samples


def _uniform(seed: int, index: int, stream: int) -> float:
    """Deterministic uniform value in [0, 1) from a SHA-256 hash.

    The payload canonically binds the seed, the sample index, and a distinct
    stream number, so no global random state or salted hash() is involved and
    the output is identical on every call, process, and platform.
    """
    payload = f"audio-dither:v1:{stream}:{seed}:{index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest, "big") / _HASH_DOMAIN


def tpdf_noise(seed: int, n: int) -> list[float]:
    """Generate triangular (TPDF) dither with a peak amplitude of 1 LSB.

    The reference is a normalized 16-bit signal, so one LSB is ``2 ** -15``.
    Each output value is ``u0 - u1``, where ``u0`` and ``u1`` are independent
    uniform values in ``[0, 1)`` derived from SHA-256 hashes of a canonical
    payload containing ``seed``, the sample index, and the stream number.  The
    support is therefore ``(-1, 1)`` LSB, the peak is 1 LSB, and the RMS is
    ``1 / sqrt(6)`` LSB.  The result must be identical on every call, process,
    and platform for the same ``seed`` and ``n``; Python's global ``random``
    state and salted ``hash()`` must not be used.

    Args:
        seed: Explicit integer seed.  Negative values are allowed.
        n: Number of noise samples; must be an integer greater than or equal
            to zero.

    Returns:
        A list of ``n`` floats in the triangular dither range.  ``n == 0``
        returns an empty list.

    Raises:
        DitherError: If ``n`` is not an integer, ``n < 0``, or ``seed`` is not
            an integer.
    """
    _validate_seed(seed)
    _validate_count(n)
    return [
        (_uniform(seed, i, 0) - _uniform(seed, i, 1)) * LSB16
        for i in range(n)
    ]


def rectangular_noise(seed: int, n: int) -> list[float]:
    """Generate rectangular (RPDF) dither with a full width of 1 LSB.

    The reference is the same normalized 16-bit scale as :func:`tpdf_noise`, so
    one LSB is ``2 ** -15``.  Values are uniform in ``[-0.5, 0.5)`` LSB: the
    peak magnitude is half an LSB, the peak-to-peak width is one LSB, and the
    RMS is ``1 / sqrt(12)`` LSB.  Uniform values are derived deterministically
    from SHA-256 hashes of ``seed``, the sample index, and a distinct stream
    identifier, so repeated calls with the same arguments are identical.

    Args:
        seed: Explicit integer seed.  Negative values are allowed.
        n: Number of noise samples; must be an integer greater than or equal
            to zero.

    Returns:
        A list of ``n`` floats in the rectangular dither range.  ``n == 0``
        returns an empty list.

    Raises:
        DitherError: If ``n`` is not an integer, ``n < 0``, or ``seed`` is not
            an integer.
    """
    _validate_seed(seed)
    _validate_count(n)
    return [(_uniform(seed, i, 2) - 0.5) * LSB16 for i in range(n)]


def _round_half_away(value: float) -> int:
    """Round to the nearest integer, with ties rounded away from zero."""
    floored = math.floor(value)
    fraction = value - floored
    if fraction < 0.5:
        return floored
    if fraction > 0.5:
        return floored + 1
    # Exact tie: away from zero.  For negative values floor() already moved
    # away from zero (e.g. floor(-2.5) == -3), so the floored value is kept.
    return floored + 1 if value > 0 else floored


def quantize(samples: list[float], bit_depth: int) -> list[int]:
    """Round normalized samples to the nearest signed PCM quantization level.

    ``bit_depth`` must be an integer greater than or equal to 1.  The scaled
    value is ``sample * 2 ** (bit_depth - 1)``; ties are rounded away from
    zero (``2.5`` becomes ``3`` and ``-2.5`` becomes ``-3``).  The result is
    clipped to ``[-2 ** (bit_depth - 1), 2 ** (bit_depth - 1) - 1]``.  In
    particular, a sample of ``1.0`` saturates to the largest positive code
    rather than growing an extra code.

    Input samples are normally in ``[-1, 1]``.  Finite samples outside that
    range are saturated, never wrapped, so a large positive peak cannot become
    a negative code.  Truncation toward zero is not acceptable: it introduces a
    systematic half-LSB bias.

    Args:
        samples: Finite normalized sample values.  An empty list is allowed.
        bit_depth: Target signed PCM bit depth; must be at least 1.

    Returns:
        One integer code per input sample, in input order.

    Raises:
        DitherError: If ``bit_depth < 1``, ``bit_depth`` is not an integer, the
            input is not a sequence, or a sample is not finite.
    """
    _validate_bit_depth(bit_depth)
    _validate_samples(samples)
    scale = 1 << (bit_depth - 1)
    low, high = -scale, scale - 1
    codes = []
    for sample in samples:
        code = _round_half_away(sample * scale)
        if code < low:
            code = low
        elif code > high:
            code = high
        codes.append(code)
    return codes


def dequantize(codes: list[int], bit_depth: int) -> list[float]:
    """Convert signed PCM codes back to normalized samples.

    The conversion is ``code / 2 ** (bit_depth - 1)``; in particular, the most
    negative code maps to ``-1.0`` and the most positive code maps to
    ``1.0 - 2 ** (1 - bit_depth)``.  A code produced by :func:`quantize` for an
    in-range sample must therefore reconstruct that sample with an error no
    greater than half an LSB.

    Args:
        codes: Integer codes in the signed range
            ``[-2 ** (bit_depth - 1), 2 ** (bit_depth - 1) - 1]``.
        bit_depth: Source signed PCM bit depth; must be at least 1.

    Returns:
        One normalized float per code, in input order.

    Raises:
        DitherError: If ``bit_depth < 1``, a code is not an integer, or a code
            falls outside the signed range for ``bit_depth``.
    """
    _validate_bit_depth(bit_depth)
    if not isinstance(codes, list):
        raise DitherError(f"codes must be a list, got {type(codes).__name__}")
    scale = 1 << (bit_depth - 1)
    low, high = -scale, scale - 1
    result = []
    for code in codes:
        if not _is_real_int(code):
            raise DitherError(f"code must be an integer, got {code!r}")
        if code < low or code > high:
            raise DitherError(
                f"code {code} outside signed {bit_depth}-bit range "
                f"[{low}, {high}]"
            )
        result.append(code / scale)
    return result


def _validate_shape(shape: object) -> list[float] | None:
    """Validate a noise-shape argument and return a normalized list."""
    if shape is None or shape == []:
        return [] if shape is not None else None
    if not isinstance(shape, list) or len(shape) != 1:
        raise DitherError(
            "shape must be None, [], or a single-coefficient list [c]"
        )
    coefficient = shape[0]
    if not isinstance(coefficient, (int, float)) or isinstance(
        coefficient, bool
    ):
        raise DitherError(f"shape coefficient must be numeric, got {coefficient!r}")
    if not math.isfinite(coefficient):
        raise DitherError(
            f"shape coefficient must be finite, got {coefficient!r}"
        )
    if abs(coefficient) >= 1.0:
        raise DitherError(
            f"shape coefficient must satisfy abs(c) < 1, got {coefficient!r}"
        )
    return [float(coefficient)]


def noise_shape(samples: list[float], coefficients: list[float]) -> list[float]:
    """Apply first-order error feedback to a sequence of errors.

    ``samples`` is treated as an error sequence.  With one feedback coefficient
    ``c`` the recursion is::

        y[0] = x[0]
        y[n] = x[n] + c * y[n - 1]

    which implements the all-pole error-feedback kernel
    ``1 / (1 - c * z ** -1)``.  It is stable only when ``abs(c) < 1``; for
    ``abs(c) >= 1`` the response diverges and the function must raise
    :class:`DitherError` instead of returning growing values.  An empty
    coefficient sequence is a no-op and returns a copy of ``samples``.  A
    first-order filter accepts exactly one coefficient; longer sequences are
    invalid.

    The output is not clipped to ``[-1, 1]`` because it represents a shaped
    error signal rather than audio.  A caller applying noise shaping to audio
    must keep the coefficients in the stable range and add the shaped error
    before quantization.

    Args:
        samples: Finite error samples.
        coefficients: ``[]`` or ``[c]`` with ``abs(c) < 1``.

    Returns:
        The shaped error sequence, with the same length as ``samples``.

    Raises:
        DitherError: If a sample is not finite, a coefficient is not finite, or
            the coefficient sequence has more than one element or is unstable
            (``abs(c) >= 1``).
    """
    _validate_samples(samples)
    if not isinstance(coefficients, list):
        raise DitherError(
            "coefficients must be a list, got "
            f"{type(coefficients).__name__}"
        )
    if len(coefficients) == 0:
        return list(samples)
    if len(coefficients) > 1:
        raise DitherError(
            f"first-order shaping accepts exactly one coefficient, got "
            f"{len(coefficients)}"
        )
    coefficient = coefficients[0]
    if not isinstance(coefficient, (int, float)) or isinstance(
        coefficient, bool
    ):
        raise DitherError(
            f"coefficient must be numeric, got {coefficient!r}"
        )
    if not math.isfinite(coefficient):
        raise DitherError(f"coefficient must be finite, got {coefficient!r}")
    if abs(coefficient) >= 1.0:
        raise DitherError(
            f"feedback coefficient must satisfy abs(c) < 1, got {coefficient!r}"
        )

    shaped: list[float] = []
    previous = 0.0
    for index, sample in enumerate(samples):
        if index == 0:
            value = float(sample)
        else:
            value = sample + coefficient * previous
        shaped.append(value)
        previous = value
    return shaped


def apply_dither(
    samples: list[float],
    bit_depth: int,
    shape: list[float] | None,
    seed: int,
) -> list[int]:
    """Add TPDF dither, optionally shape the error, and quantize to PCM codes.

    ``tpdf_noise`` produces noise on the 16-bit reference scale.  For a target
    bit depth ``b`` its values are multiplied by ``2 ** (16 - b)`` so that the
    dither still has a peak amplitude of exactly one target LSB.  The dither is
    added before quantization; quantization rounds to nearest and saturates as
    described in :func:`quantize`.

    ``shape`` is ``None``/empty for plain dither, or ``[c]`` with
    ``abs(c) < 1`` for first-order error feedback.  With shaping, each sample is
    adjusted by ``c`` times the previous quantization error (where the error is
    the reconstructed value minus the adjusted value), quantized, and the new
    error is retained for the next sample.  The same ``seed`` and arguments must
    produce exactly the same codes on every call.

    Args:
        samples: Finite normalized audio samples, normally in ``[-1, 1]``.
        bit_depth: Target signed PCM bit depth; must be at least 1.
        shape: ``None``, ``[]``, or a stable single-coefficient sequence.
        seed: Explicit integer seed for the deterministic dither.

    Returns:
        One signed PCM code per input sample, in input order.

    Raises:
        DitherError: If ``bit_depth < 1``, a sample is not finite, ``seed`` is
            invalid, or ``shape`` is malformed or unstable.
    """
    _validate_bit_depth(bit_depth)
    _validate_samples(samples)
    _validate_seed(seed)
    normalized_shape = _validate_shape(shape)

    scale = 1 << (bit_depth - 1)
    low, high = -scale, scale - 1
    # Re-scale the 16-bit-reference dither to the target bit depth so its peak
    # stays at exactly one target LSB.
    dither_gain = float(1 << (16 - bit_depth)) if bit_depth <= 16 else 1.0 / (
        1 << (bit_depth - 16)
    )
    noise = tpdf_noise(seed, len(samples))
    coefficient = (
        normalized_shape[0] if normalized_shape else 0.0
    )
    use_shaping = normalized_shape is not None and len(normalized_shape) == 1

    codes: list[int] = []
    previous_error = 0.0
    for index, sample in enumerate(samples):
        dither_value = noise[index] * dither_gain
        if use_shaping:
            adjusted = sample + coefficient * previous_error + dither_value
        else:
            adjusted = sample + dither_value
        code = _round_half_away(adjusted * scale)
        if code < low:
            code = low
        elif code > high:
            code = high
        codes.append(code)
        if use_shaping:
            reconstructed = code / scale
            previous_error = reconstructed - adjusted
    return codes


def measure_snr(original: list[float], quantized: list[float]) -> float:
    """Measure signal-to-quantization-noise ratio in decibels.

    ``original`` and ``quantized`` are both normalized float signals (the
    second is a reconstructed signal, not integer codes).  Signal power is
    ``mean(original[n] ** 2)`` and noise power is
    ``mean((original[n] - quantized[n]) ** 2)``.  The result is::

        10 * log10(signal_power / noise_power)

    A fully silent original has zero signal power and no finite SNR, so the
    function returns positive infinity instead of dividing by zero.  A
    noise-free reconstruction also has zero noise power and returns positive
    infinity.

    Args:
        original: Original finite normalized signal.
        quantized: Reconstructed finite signal with the same non-zero length.

    Returns:
        SNR in dB, or ``float("inf")`` for silence or a noise-free pair.

    Raises:
        DitherError: If either sequence is empty, the lengths differ, or any
            value is not finite.
    """
    _validate_samples(original, allow_empty=False)
    _validate_samples(quantized, allow_empty=False)
    if len(original) != len(quantized):
        raise DitherError(
            f"length mismatch: original has {len(original)} samples, "
            f"quantized has {len(quantized)}"
        )

    count = len(original)
    signal_power = sum(value * value for value in original) / count
    noise_power = (
        sum(
            (a - b) * (a - b)
            for a, b in zip(original, quantized)
        )
        / count
    )

    if signal_power == 0.0 or noise_power == 0.0:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)


def dither_batch(tracks: list[dict], config: dict) -> dict:
    """Dither and quantize several tracks atomically.

    Each track is a mapping with a unique non-empty string ``name`` and a
    non-empty sequence ``samples`` of finite normalized floats.  ``config`` is a
    mapping with an integer ``bit_depth`` (at least 1), an optional ``shape``
    (``None``, ``[]``, or a stable single-coefficient sequence), and an optional
    integer ``seed`` (default ``0``).  Track ``i`` uses the independent seed
    ``config["seed"] + i``, so tracks never consume or share one noise stream.

    The return value is::

        {
            "bit_depth": int,
            "tracks": [
                {"name": str, "codes": list[int]},
                ...
            ],
        }

    The ``tracks`` list preserves input order.  Every track is validated before
    any processing starts; if any track or config value is invalid, the
    function raises :class:`DitherError` and returns no partial result.  This
    makes the batch atomic (all tracks succeed or none are reported), and the
    input track mappings are never modified.

    Args:
        tracks: List of track mappings.  An empty list is allowed and returns
            an empty result.
        config: Batch configuration mapping.

    Returns:
        A new result mapping containing the bit depth and per-track codes in
        input order.

    Raises:
        DitherError: If ``tracks``/``config`` have the wrong shape, a track name
            is missing, empty, non-string, or duplicated, samples are missing
            or non-finite, the sample list is empty, or the batch
            ``bit_depth``/``shape``/``seed`` is invalid.
    """
    if not isinstance(config, dict):
        raise DitherError(
            f"config must be a mapping, got {type(config).__name__}"
        )
    if "bit_depth" not in config:
        raise DitherError("config is missing required 'bit_depth'")
    bit_depth = config["bit_depth"]
    _validate_bit_depth(bit_depth)

    shape = config.get("shape")
    normalized_shape = _validate_shape(shape)

    seed = config.get("seed", 0)
    _validate_seed(seed)

    if not isinstance(tracks, list):
        raise DitherError(
            f"tracks must be a list, got {type(tracks).__name__}"
        )

    # Validate every track before any processing begins so the batch is
    # atomic: no partial result can be produced and inputs are never mutated.
    seen_names: set[str] = set()
    for position, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise DitherError(
                f"track {position} must be a mapping, got "
                f"{type(track).__name__}"
            )
        if "name" not in track:
            raise DitherError(f"track {position} is missing 'name'")
        name = track["name"]
        if not isinstance(name, str) or name == "":
            raise DitherError(
                f"track {position} name must be a non-empty string, got "
                f"{name!r}"
            )
        if name in seen_names:
            raise DitherError(f"duplicate track name: {name!r}")
        seen_names.add(name)
        if "samples" not in track:
            raise DitherError(f"track {name!r} is missing 'samples'")
        _validate_samples(track["samples"], allow_empty=False)

    result_tracks = []
    for index, track in enumerate(tracks):
        codes = apply_dither(
            track["samples"],
            bit_depth,
            normalized_shape,
            seed + index,
        )
        result_tracks.append({"name": track["name"], "codes": codes})

    return {"bit_depth": bit_depth, "tracks": result_tracks}
