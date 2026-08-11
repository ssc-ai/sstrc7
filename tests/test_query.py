"""Query correctness against a brute-force scan of the same synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from sstrc7._format import MAS_PER_DEG, RECORD_DTYPE, zone_filename, zones_for_dec_range
from sstrc7.query import Catalog, CatalogNotFound

from .conftest import make_records, write_catalog


def brute_force(directory, ra_min, ra_max, dec_min, dec_max):
    """Reference: read whole zone files and mask, using no index or ordering."""
    kept = []
    for zone_id in zones_for_dec_range(dec_min, dec_max):
        path = directory / zone_filename(zone_id)
        if not path.exists():
            continue
        records = np.fromfile(path, dtype=RECORD_DTYPE)
        ra = records["ra"] / MAS_PER_DEG
        dec = records["dec"] / MAS_PER_DEG
        if ra_min <= ra_max:
            keep = (ra >= ra_min) & (ra <= ra_max)
        else:
            keep = (ra >= ra_min) | (ra <= ra_max)
        keep &= (dec >= dec_min) & (dec <= dec_max)
        kept.append(records[keep])
    return np.concatenate(kept) if kept else np.empty(0, dtype=RECORD_DTYPE)


def sorted_bytes(records):
    return np.sort(records, order=["ra", "dec"]).tobytes()


BOXES = [
    (0.0, 10.0, 0.0, 10.0),
    (359.0, 1.0, -1.0, 1.0),  # wraps through zero
    (5.9, 6.1, 19.9, 20.1),  # RA zone boundary
    (179.9, 180.1, -0.05, 0.05),  # dec zone boundary
    (0.0, 360.0, 89.0, 90.0),  # north cap
    (0.0, 360.0, -90.0, -89.0),  # south cap
    (0.0, 360.0, -90.0, 90.0),  # whole sky
    (300.0, 60.0, -45.0, 45.0),  # wide wrap
    (100.0, 100.0, 0.0, 1.0),  # zero-width RA
    (100.0, 110.0, 5.0, 5.0),  # zero-width dec
]


@pytest.mark.parametrize("box", BOXES)
def test_query_box_matches_brute_force(synthetic_catalog, box):
    catalog = Catalog(synthetic_catalog)
    got = catalog.query_box(*box)
    want = brute_force(synthetic_catalog, *box)
    assert len(got) == len(want)
    assert sorted_bytes(got.records) == sorted_bytes(want)


def test_query_box_accepts_radians(synthetic_catalog):
    catalog = Catalog(synthetic_catalog)
    degrees = catalog.query_box(10.0, 20.0, -5.0, 5.0)
    radians = catalog.query_box(
        np.radians(10.0), np.radians(20.0), np.radians(-5.0), np.radians(5.0), radians=True
    )
    assert sorted_bytes(degrees.records) == sorted_bytes(radians.records)


def test_query_box_reversed_dec_bounds_are_accepted(synthetic_catalog):
    catalog = Catalog(synthetic_catalog)
    forward = catalog.query_box(10.0, 20.0, -5.0, 5.0)
    reverse = catalog.query_box(10.0, 20.0, 5.0, -5.0)
    assert sorted_bytes(forward.records) == sorted_bytes(reverse.records)


def test_query_box_on_empty_region_returns_empty(synthetic_catalog):
    catalog = Catalog(synthetic_catalog)
    field = catalog.query_box(10.0, 10.0, 10.0, 10.0)
    assert len(field) == 0
    assert field.ra.shape == (0,)
    assert field.mag.shape == (0, 18)


@pytest.mark.parametrize(
    "ra,dec,radius",
    [
        (10.0, 0.0, 5.0),
        (0.0, 0.0, 3.0),  # centred on the RA wrap
        (359.5, -2.0, 2.0),
        (30.0, 89.0, 3.0),  # over the north pole
        (200.0, -89.0, 3.0),  # over the south pole
        (120.0, 45.0, 0.5),
    ],
)
def test_query_cone_matches_brute_force(synthetic_catalog, ra, dec, radius):
    catalog = Catalog(synthetic_catalog)
    got = catalog.query_cone(ra, dec, radius)

    everything = brute_force(
        synthetic_catalog, 0.0, 360.0, max(dec - radius, -90.0), min(dec + radius, 90.0)
    )
    ra_r = np.radians(everything["ra"] / MAS_PER_DEG)
    dec_r = np.radians(everything["dec"] / MAS_PER_DEG)
    c_ra, c_dec = np.radians(ra), np.radians(dec)
    cos_sep = np.sin(dec_r) * np.sin(c_dec) + np.cos(dec_r) * np.cos(c_dec) * np.cos(ra_r - c_ra)
    want = everything[cos_sep >= np.cos(np.radians(radius))]

    assert len(got) == len(want)
    assert sorted_bytes(got.records) == sorted_bytes(want)


def test_star_field_columns(synthetic_catalog):
    catalog = Catalog(synthetic_catalog)
    field = catalog.query_box(0.0, 360.0, -10.0, 10.0)
    assert len(field) > 0

    assert field.ra.min() >= 0.0 and field.ra.max() < 360.0
    assert np.all(np.abs(field.dec) <= 10.0)
    assert field.mag.shape == (len(field), 18)
    assert np.array_equal(field.band("Johnson_R"), field.mag[:, 5], equal_nan=True)
    assert field.visual.shape == (len(field),)
    assert field.at_wavelength(700.0).shape == (len(field),)
    assert np.all(field.source_flags == (0x0010 | 0x0040))
    assert "Gaia Catalog" in field.flags(0)
    assert repr(field).startswith("<StarField:")


def test_band_lookup_rejects_unknown_names(synthetic_catalog):
    field = Catalog(synthetic_catalog).query_box(0.0, 1.0, 0.0, 1.0)
    with pytest.raises(KeyError, match="unknown band"):
        field.band("Johnson_Q")


def test_missing_catalog_raises_a_helpful_error(tmp_path):
    with pytest.raises(CatalogNotFound, match="sstrc7 get"):
        Catalog(tmp_path / "nothing-here")


def test_index_of_wrong_size_is_rejected(tmp_path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "sstrc.acc").write_bytes(b"\x00" * 16)
    with pytest.raises(CatalogNotFound):
        Catalog(directory).index


def test_query_box_is_stable_across_repeated_calls(synthetic_catalog):
    """Results must not alias cached memory maps or mutate between calls."""
    catalog = Catalog(synthetic_catalog)
    first = catalog.query_box(0.0, 30.0, -20.0, 20.0)
    snapshot = first.records.copy()
    second = catalog.query_box(0.0, 30.0, -20.0, 20.0)
    assert first.records.tobytes() == snapshot.tobytes()
    assert second.records.tobytes() == snapshot.tobytes()


def test_stars_outside_any_written_zone_are_not_invented(tmp_path):
    records = make_records([100.0, 101.0], [40.0, 40.05])
    directory = write_catalog(tmp_path / "sparse", records)
    catalog = Catalog(directory)
    assert len(catalog.query_box(0.0, 360.0, -90.0, 90.0)) == 2
    assert len(catalog.query_box(0.0, 360.0, -90.0, 0.0)) == 0


def test_query_by_los_projects_onto_the_focal_plane(synthetic_catalog):
    catalog = Catalog(synthetic_catalog)
    rows, cols, mag = catalog.query_by_los(512, 512, 20.0, 20.0, 100.0, 10.0, pad_mult=1.0)
    assert rows.shape == cols.shape == mag.shape
    assert len(rows) > 0
    assert np.all(np.isfinite(rows)) and np.all(np.isfinite(cols))
    # With origin="center" the boresight is at the array centre, so stars from
    # a field this wide must land on both sides of it.
    assert rows.min() < 256 < rows.max()


def test_query_by_los_filter_center_changes_magnitudes(synthetic_catalog):
    catalog = Catalog(synthetic_catalog)
    _, _, plain = catalog.query_by_los(64, 64, 10.0, 10.0, 50.0, 0.0)
    _, _, tuned = catalog.query_by_los(64, 64, 10.0, 10.0, 50.0, 0.0, filter_center=1200.0)
    assert plain.shape == tuned.shape
    assert not np.allclose(plain, tuned)


@pytest.mark.parametrize(
    "angles,expected",
    [
        ([179.5, 180.5], (179.5, 180.5)),  # ordinary span
        ([359.5, 0.5], (359.5, 0.5)),  # wraps through zero
        ([359.519, 0.769, 0.02], (359.519, 0.769)),
        ([42.0], (42.0, 42.0)),  # degenerate
        ([10.0, 20.0, 30.0], (10.0, 30.0)),
        ([350.0, 355.0, 5.0, 10.0], (350.0, 10.0)),
    ],
)
def test_enclosing_ra_arc(angles, expected):
    from sstrc7.query import _enclosing_ra_arc

    assert _enclosing_ra_arc(np.array(angles)) == pytest.approx(expected)


def test_field_bounds_span_the_field_even_across_ra_zero(synthetic_catalog):
    """Regression: min/max over corners describes the wrong arc at the seam.

    Taking min and max of the projected corner RAs puts them on opposite sides
    of RA = 0, so the "box" covered the sky the field does *not* occupy and the
    query silently dropped most of the stars.
    """
    from sstrc7.query import _field_bounds

    fov, pad = 0.5, 1.0
    expected_span = fov * (1.0 + 2.0 * pad)

    for ra in (0.02, 359.98, 0.0, 180.0, 90.0):
        corner_min, corner_max, _ = _field_bounds(
            512, 512, fov / 512, fov / 512, ra, 0.0, 0.0, pad, "center"
        )
        span = (corner_max[0] - corner_min[0]) % 360.0
        assert span == pytest.approx(expected_span, rel=0.1), f"ra={ra} span={span}"


def test_query_by_los_keeps_stars_on_both_sides_of_ra_zero(tmp_path):
    """A field centred on RA = 0 must return the stars either side of the seam."""
    ras = [359.4, 359.7, 0.0, 0.3, 0.6]
    directory = write_catalog(
        tmp_path / "seam", make_records(ras, [0.0] * len(ras), seed=5)
    )
    rows, cols, mag = Catalog(directory).query_by_los(
        512, 512, 4.0, 4.0, 0.0, 0.0, pad_mult=0.0
    )
    assert len(rows) == len(ras)


def test_query_by_los_empty_field_returns_empty_arrays(tmp_path):
    directory = write_catalog(tmp_path / "one", make_records([10.0], [80.0]))
    rows, cols, mag = Catalog(directory).query_by_los(64, 64, 1.0, 1.0, 200.0, -40.0)
    assert len(rows) == len(cols) == len(mag) == 0
