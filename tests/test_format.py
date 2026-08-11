"""The binary layout and the distribution codec."""

from __future__ import annotations

import numpy as np
import pytest

from sstrc7._format import (
    BAND_CENTERS_NM,
    BAND_INDEX,
    BAND_NAMES,
    MAG_ABSENT,
    N_DEC_ZONES,
    RECORD_DTYPE,
    RECORD_SIZE,
    decode_source_flags,
    decode_zone,
    encode_zone,
    interpolate_magnitude,
    magnitudes,
    visual_magnitude,
    zone_dec_range,
    zone_filename,
    zones_for_dec_range,
)

from .conftest import make_records


def test_record_is_sixty_bytes():
    assert RECORD_DTYPE.itemsize == RECORD_SIZE


def test_band_tables_line_up():
    assert len(BAND_NAMES) == len(BAND_CENTERS_NM) == 18
    assert BAND_INDEX["Johnson_V"] == 4
    assert BAND_INDEX["Gaia_G"] == 0


def test_zone_geometry_covers_the_sky_exactly():
    assert zone_dec_range(0)[0] == pytest.approx(-90.0)
    assert zone_dec_range(N_DEC_ZONES - 1)[1] == pytest.approx(90.0)
    for zone_id in range(N_DEC_ZONES - 1):
        assert zone_dec_range(zone_id)[1] == pytest.approx(zone_dec_range(zone_id + 1)[0])


def test_zones_for_dec_range_is_clamped_and_inclusive():
    assert list(zones_for_dec_range(-0.05, 0.05)) == [899, 900]
    assert list(zones_for_dec_range(-100.0, -89.95)) == [0]
    assert list(zones_for_dec_range(89.95, 100.0)) == [N_DEC_ZONES - 1]
    # Reversed bounds are accepted rather than silently returning nothing.
    assert list(zones_for_dec_range(0.05, -0.05)) == [899, 900]


def test_zone_filename_rejects_out_of_range():
    assert zone_filename(0) == "s0000.cat"
    assert zone_filename(1799) == "s1799.cat"
    with pytest.raises(ValueError):
        zone_filename(1800)


def test_absent_magnitudes_become_nan():
    records = make_records([10.0], [20.0])
    records["mag"][0, 0] = MAG_ABSENT
    records["mag"][0, 1] = 12345
    mag = magnitudes(records)
    assert np.isnan(mag[0, 0])
    assert mag[0, 1] == pytest.approx(12.345, abs=1e-4)


def test_visual_magnitude_follows_the_priority_order():
    records = make_records([1.0, 2.0], [3.0, 4.0])
    records["mag"][:] = MAG_ABSENT
    records["mag"][0, BAND_INDEX["Johnson_V"]] = 10000
    records["mag"][0, BAND_INDEX["Johnson_R"]] = 11000
    records["mag"][1, BAND_INDEX["Johnson_R"]] = 11000  # no V, falls through
    visual = visual_magnitude(magnitudes(records))
    assert visual[0] == pytest.approx(10.0, abs=1e-4)
    assert visual[1] == pytest.approx(11.0, abs=1e-4)


def test_visual_magnitude_is_nan_without_any_priority_band():
    records = make_records([1.0], [2.0])
    records["mag"][:] = MAG_ABSENT
    records["mag"][0, BAND_INDEX["WISE_W1"]] = 9000
    assert np.isnan(visual_magnitude(magnitudes(records))[0])


def test_interpolation_matches_numpy_interp():
    records = make_records(np.linspace(0, 359, 200), np.linspace(-80, 80, 200), seed=3)
    mag = magnitudes(records)

    for wavelength in (450.0, 622.0, 700.0, 1500.0):
        got = interpolate_magnitude(mag, wavelength)
        for i in range(0, 200, 37):
            pairs = sorted(
                (c, m) for c, m in zip(BAND_CENTERS_NM, mag[i]) if not np.isnan(m)
            )
            want = np.interp(wavelength, [c for c, _ in pairs], [m for _, m in pairs])
            assert got[i] == pytest.approx(want, abs=1e-3)


def test_interpolation_clamps_outside_the_measured_span():
    records = make_records([1.0], [2.0])
    records["mag"][:] = MAG_ABSENT
    records["mag"][0, BAND_INDEX["Johnson_V"]] = 10000  # 548 nm, the only band
    mag = magnitudes(records)
    assert interpolate_magnitude(mag, 300.0)[0] == pytest.approx(10.0, abs=1e-4)
    assert interpolate_magnitude(mag, 20000.0)[0] == pytest.approx(10.0, abs=1e-4)


def test_interpolation_of_a_star_with_no_bands_is_nan():
    records = make_records([1.0], [2.0])
    records["mag"][:] = MAG_ABSENT
    assert np.isnan(interpolate_magnitude(magnitudes(records), 600.0)[0])


def test_source_flags_decode():
    assert decode_source_flags(0) == []
    assert "Gaia Catalog" in decode_source_flags(0x0010)
    both = decode_source_flags(0x0010 | 0x4000)
    assert "Gaia Catalog" in both and "Variable star" in both


@pytest.mark.parametrize("count", [0, 1, 97, 5000])
def test_codec_round_trip_is_byte_identical(count):
    records = make_records(
        np.linspace(0, 359, count) if count else [],
        np.linspace(-10, 10, count) if count else [],
        seed=count,
    )
    raw = records.tobytes()
    assert decode_zone(encode_zone(raw)) == raw


def test_codec_actually_compresses():
    records = make_records(np.linspace(0, 359, 20000), np.linspace(-1, 1, 20000))
    raw = records.tobytes()
    assert len(encode_zone(raw)) < len(raw) / 2


def test_codec_rejects_misaligned_input():
    with pytest.raises(ValueError):
        encode_zone(b"\x00" * 61)


def test_codec_rejects_foreign_data():
    with pytest.raises(ValueError, match="not an sstrc7 zone archive"):
        decode_zone(b"NOTMAGIC" + b"\x00" * 32)
