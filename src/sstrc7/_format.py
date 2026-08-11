"""Binary layout of the SSTRC7 catalog, and the codec used to distribute it.

The catalog is a set of zone files plus one index ("accelerator") file:

    sstrc.acc        1800 x 60 pairs of little-endian uint32 (pos, length)
    s0000.cat ...    one file per declination zone, 60-byte records
    s1799.cat

Declination zones are 0.1 deg tall in south polar distance (SPD = dec + 90),
so zone ``z`` covers ``dec in [z * 0.1 - 90, (z + 1) * 0.1 - 90)``. Within a
zone, records are sorted by right ascension, and the index gives the record
offset and count of each of the 60 six-degree-wide RA sub-zones.

Every field below is verified against the shipped catalog: the summed zone
lengths in ``sstrc.acc`` reproduce each zone file's record count exactly, and
every zone file is byte-aligned to 60 and RA-sorted.
"""

from __future__ import annotations

import lzma
import struct

import numpy as np

# --- Geometry -------------------------------------------------------------

N_DEC_ZONES = 1800
N_RA_ZONES = 60
ZONE_HEIGHT_DEG = 180.0 / N_DEC_ZONES  # 0.1
ZONE_WIDTH_DEG = 360.0 / N_RA_ZONES  # 6.0

RECORD_SIZE = 60
INDEX_FILENAME = "sstrc.acc"
INDEX_SIZE = N_DEC_ZONES * N_RA_ZONES * 2 * 4  # 864000

# --- Record layout --------------------------------------------------------

#: One catalog record. ``ra``/``dec`` are milliarcseconds; ``mag`` is
#: millimagnitudes with :data:`MAG_ABSENT` marking a band with no measurement.
RECORD_DTYPE = np.dtype(
    [
        ("ra", "<i4"),  # milliarcseconds, [0, 360 deg)
        ("dec", "<i4"),  # milliarcseconds, [-90, +90 deg]
        ("pm_ra", "<i2"),  # units of PM_SCALE_MAS_PER_YEAR
        ("pm_dec", "<i2"),  # units of PM_SCALE_MAS_PER_YEAR
        ("parallax", "<i2"),  # units of PARALLAX_SCALE_MAS
        ("mag", "<i2", (18,)),  # millimagnitudes, see BAND_NAMES
        ("_reserved0", "<u2"),
        ("source_flags", "<u2"),  # bitmask, see SOURCE_FLAGS
        ("_reserved1", "<u2", (3,)),
    ]
)
assert RECORD_DTYPE.itemsize == RECORD_SIZE

MAG_SCALE = 1.0e-3  # stored millimagnitudes -> magnitudes
MAG_ABSENT = 32000  # stored sentinel for "no measurement in this band"
PM_SCALE_MAS_PER_YEAR = 0.32
PARALLAX_SCALE_MAS = 0.032

#: Photometric bands, in stored order.
BAND_NAMES: tuple[str, ...] = (
    "Gaia_G",
    "Gaia_BP",
    "Gaia_RP",
    "Johnson_B",
    "Johnson_V",
    "Johnson_R",
    "Johnson_I",
    "Sloan_g",
    "Sloan_r",
    "Sloan_i",
    "Sloan_z",
    "2MASS_J",
    "2MASS_H",
    "2MASS_Ks",
    "WISE_W1",
    "WISE_W2",
    "WISE_W3",
    "WISE_W4",
)

#: Approximate effective wavelength of each band, nanometres.
BAND_CENTERS_NM: tuple[int, ...] = (
    600,
    500,
    800,
    440,
    548,
    700,
    900,
    477,
    622,
    762,
    913,
    1235,
    1662,
    2159,
    3400,
    4600,
    12000,
    22000,
)

BAND_INDEX: dict[str, int] = {name: i for i, name in enumerate(BAND_NAMES)}

assert len(BAND_NAMES) == len(BAND_CENTERS_NM) == RECORD_DTYPE["mag"].shape[0]

#: Preference order used by :func:`visual_magnitude` for a broadband silicon
#: response, best match first.
VISUAL_BAND_PRIORITY: tuple[str, ...] = (
    "Johnson_V",
    "Johnson_R",
    "Sloan_r",
    "Gaia_G",
    "Sloan_g",
    "Johnson_B",
)

#: Meaning of each bit in the ``source_flags`` field.
SOURCE_FLAGS: dict[int, str] = {
    0x0001: "Bright Star Catalog (HR)",
    0x0002: "Henry Draper Catalog (HD)",
    0x0004: "Hipparcos Catalog",
    0x0008: "Tycho-Gaia (TGAS) Catalog",
    0x0010: "Gaia Catalog",
    0x0020: "Landolt Catalog",
    0x0040: "2MASS Catalog",
    0x0080: "AllWISE Catalog",
    0x0100: "Astrometric Standard",
    0x0200: "Extended Source",
    0x0400: "High Proper Motion Star",
    0x0800: "Multiple stars",
    0x1000: "Photometric Standard",
    0x2000: "Spectrophotometric Star",
    0x4000: "Variable star",
    0x8000: "SWIR Standard",
}

MAS_PER_DEG = 3.6e6


def zone_filename(zone_id: int) -> str:
    """Return the zone file name for a declination zone index."""
    if not 0 <= zone_id < N_DEC_ZONES:
        raise ValueError(f"zone_id must be in [0, {N_DEC_ZONES}), got {zone_id}")
    return f"s{zone_id:04d}.cat"


def zone_asset_name(zone_id: int) -> str:
    """Return the release asset name for a declination zone."""
    return f"s{zone_id:04d}{CODEC_SUFFIX}"


def zone_dec_range(zone_id: int) -> tuple[float, float]:
    """Return the (min, max) declination in degrees covered by a zone."""
    return (
        zone_id * ZONE_HEIGHT_DEG - 90.0,
        (zone_id + 1) * ZONE_HEIGHT_DEG - 90.0,
    )


def zones_for_dec_range(dec_min: float, dec_max: float) -> range:
    """Return the declination zone indices intersecting a range in degrees."""
    if dec_min > dec_max:
        dec_min, dec_max = dec_max, dec_min
    lo = int((dec_min + 90.0) / ZONE_HEIGHT_DEG)
    hi = int((dec_max + 90.0) / ZONE_HEIGHT_DEG)
    return range(max(lo, 0), min(hi, N_DEC_ZONES - 1) + 1)


def decode_source_flags(flags: int) -> list[str]:
    """Expand a ``source_flags`` bitmask into a list of descriptions."""
    return [name for bit, name in SOURCE_FLAGS.items() if flags & bit]


def magnitudes(records: np.ndarray) -> np.ndarray:
    """Return an ``(N, 18)`` float32 magnitude array, NaN where unmeasured.

    Args:
        records: array of :data:`RECORD_DTYPE`.
    """
    raw = np.atleast_2d(records["mag"])
    mag = raw.astype(np.float32) * MAG_SCALE
    mag[raw >= MAG_ABSENT] = np.nan
    return mag


def visual_magnitude(mag: np.ndarray) -> np.ndarray:
    """Collapse an ``(N, 18)`` magnitude array to one broadband magnitude.

    Takes the first band present in :data:`VISUAL_BAND_PRIORITY` for each star.
    Stars with none of those bands measured come back as NaN.
    """
    out = np.full(mag.shape[0], np.nan, dtype=np.float32)
    for name in VISUAL_BAND_PRIORITY:
        col = mag[:, BAND_INDEX[name]]
        fill = np.isnan(out) & ~np.isnan(col)
        out[fill] = col[fill]
    return out


def interpolate_magnitude(mag: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """Interpolate each star's SED across its measured bands.

    Bands are treated as points at :data:`BAND_CENTERS_NM` sorted by
    wavelength; the value is linearly interpolated and clamped to the nearest
    measured band outside that span. Stars with no measured band are NaN.

    Args:
        mag: ``(N, 18)`` magnitude array from :func:`magnitudes`.
        wavelength_nm: target wavelength in nanometres.
    """
    order = np.argsort(BAND_CENTERS_NM)
    centers = np.asarray(BAND_CENTERS_NM, dtype=np.float64)[order]
    values = mag[:, order]
    valid = ~np.isnan(values)

    n_stars = values.shape[0]
    lo_x = np.full(n_stars, np.nan)
    lo_y = np.full(n_stars, np.nan)
    hi_x = np.full(n_stars, np.nan)
    hi_y = np.full(n_stars, np.nan)

    # Nearest measured band at or below the target, scanning upward.
    for i, center in enumerate(centers):
        if center > wavelength_nm:
            break
        take = valid[:, i]
        lo_x[take] = center
        lo_y[take] = values[take, i]

    # Nearest measured band at or above the target, scanning downward.
    for i in range(len(centers) - 1, -1, -1):
        if centers[i] < wavelength_nm:
            break
        take = valid[:, i]
        hi_x[take] = centers[i]
        hi_y[take] = values[take, i]

    both = ~np.isnan(lo_x) & ~np.isnan(hi_x)
    # Substitute placeholders before dividing so the unused branch of the
    # np.where below cannot raise on NaN or a zero-width span.
    safe_lo_x = np.where(both, lo_x, 0.0)
    span = np.where(both, hi_x - safe_lo_x, 1.0)
    span[span <= 0] = 1.0
    frac = (wavelength_nm - safe_lo_x) / span

    out = np.where(both, np.where(both, lo_y, 0.0) + frac * np.where(both, hi_y - lo_y, 0.0), np.nan)
    # Outside the measured span, clamp to the nearest measured band.
    out = np.where(np.isnan(out) & ~np.isnan(lo_y), lo_y, out)
    out = np.where(np.isnan(out) & ~np.isnan(hi_y), hi_y, out)
    return out.astype(np.float32)


# --- Distribution codec ---------------------------------------------------
#
# Zone files are transposed to a column-major byte layout before compression.
# Each record is 60 bytes of mostly-similar fields, so grouping byte i of every
# record together gives the compressor long runs to work with: a representative
# zone drops from 10.40 MB to 7.24 MB with plain gzip, but to 5.15 MB this way.
# The transform is exactly invertible, so the reconstructed .cat file is
# byte-identical to the original.

CODEC_MAGIC = b"SSTRC7Z1"
CODEC_HEADER = struct.Struct("<8sII")  # magic, n_records, record_size
CODEC_SUFFIX = ".catz"
LZMA_PRESET = 6


def encode_zone(raw: bytes) -> bytes:
    """Compress raw zone-file bytes into the distributed ``.catz`` form."""
    if len(raw) % RECORD_SIZE:
        raise ValueError(f"zone data is not a multiple of {RECORD_SIZE} bytes")
    n_records = len(raw) // RECORD_SIZE
    flat = np.frombuffer(raw, dtype=np.uint8).reshape(n_records, RECORD_SIZE)
    columns = np.ascontiguousarray(flat.T)
    header = CODEC_HEADER.pack(CODEC_MAGIC, n_records, RECORD_SIZE)
    return header + lzma.compress(columns.tobytes(), format=lzma.FORMAT_XZ, preset=LZMA_PRESET)


def decode_zone(blob: bytes) -> bytes:
    """Reverse :func:`encode_zone`, returning the original zone-file bytes."""
    magic, n_records, record_size = CODEC_HEADER.unpack_from(blob)
    if magic != CODEC_MAGIC:
        raise ValueError(f"not an sstrc7 zone archive (magic {magic!r})")
    if record_size != RECORD_SIZE:
        raise ValueError(f"unsupported record size {record_size}")
    body = lzma.decompress(blob[CODEC_HEADER.size :], format=lzma.FORMAT_XZ)
    expected = n_records * record_size
    if len(body) != expected:
        raise ValueError(f"decoded {len(body)} bytes, expected {expected}")
    columns = np.frombuffer(body, dtype=np.uint8).reshape(record_size, n_records)
    return np.ascontiguousarray(columns.T).tobytes()
