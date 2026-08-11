"""A small synthetic catalog, so the tests need no downloaded data."""

from __future__ import annotations

import numpy as np
import pytest

from sstrc7._format import (
    INDEX_FILENAME,
    MAG_ABSENT,
    MAS_PER_DEG,
    N_DEC_ZONES,
    N_RA_ZONES,
    RECORD_DTYPE,
    ZONE_HEIGHT_DEG,
    ZONE_WIDTH_DEG,
    zone_filename,
)


def make_records(ra_deg, dec_deg, seed=0):
    """Build catalog records at the given positions with plausible magnitudes."""
    rng = np.random.default_rng(seed)
    records = np.zeros(len(ra_deg), dtype=RECORD_DTYPE)
    records["ra"] = np.round(np.asarray(ra_deg) * MAS_PER_DEG).astype(np.int32)
    records["dec"] = np.round(np.asarray(dec_deg) * MAS_PER_DEG).astype(np.int32)
    records["mag"] = MAG_ABSENT
    # Give every star Johnson_R (index 5) and Sloan_r (index 8), and half of
    # them Johnson_V (index 4), mirroring how sparse the real bands are.
    records["mag"][:, 5] = rng.integers(5000, 20000, len(ra_deg))
    records["mag"][:, 8] = rng.integers(5000, 20000, len(ra_deg))
    has_v = rng.random(len(ra_deg)) < 0.5
    records["mag"][has_v, 4] = rng.integers(5000, 20000, int(has_v.sum()))
    records["source_flags"] = 0x0010 | 0x0040
    records["pm_ra"] = rng.integers(-100, 100, len(ra_deg))
    records["pm_dec"] = rng.integers(-100, 100, len(ra_deg))
    records["parallax"] = rng.integers(0, 500, len(ra_deg))
    return records


def write_catalog(directory, records):
    """Write a catalog directory holding ``records``, with a correct index.

    Records are bucketed into their true declination zones and sorted by RA,
    exactly as the real catalog is laid out.
    """
    directory.mkdir(parents=True, exist_ok=True)
    index = np.zeros((N_DEC_ZONES, N_RA_ZONES, 2), dtype="<u4")

    dec_deg = records["dec"] / MAS_PER_DEG
    zone_of = np.clip(((dec_deg + 90.0) / ZONE_HEIGHT_DEG).astype(int), 0, N_DEC_ZONES - 1)

    for zone_id in range(N_DEC_ZONES):
        in_zone = records[zone_of == zone_id]
        in_zone = in_zone[np.argsort(in_zone["ra"], kind="stable")]

        ra_zone = np.clip(
            (in_zone["ra"] / MAS_PER_DEG / ZONE_WIDTH_DEG).astype(int), 0, N_RA_ZONES - 1
        )
        offset = 0
        for r in range(N_RA_ZONES):
            count = int((ra_zone == r).sum())
            index[zone_id, r] = (offset, count)
            offset += count

        if len(in_zone) or zone_id < 3:
            (directory / zone_filename(zone_id)).write_bytes(in_zone.tobytes())

    (directory / INDEX_FILENAME).write_bytes(index.tobytes())
    return directory


@pytest.fixture(scope="session")
def synthetic_catalog(tmp_path_factory):
    """A catalog with a deterministic grid of stars plus awkward edge cases.

    Session-scoped and read-only: laying out 1800 zone files is the slowest
    thing in the suite, and every test that uses it only queries.
    """
    tmp_path = tmp_path_factory.mktemp("sky")
    rng = np.random.default_rng(1234)
    ra = list(rng.uniform(0, 360, 4000))
    dec = list(np.degrees(np.arcsin(rng.uniform(-1, 1, 4000))))

    # Stars sitting exactly on boundaries the query code has to get right.
    for extra_ra, extra_dec in [
        (0.0, 0.0),
        (359.9999, 0.0),
        (6.0, 20.0),  # RA zone edge
        (180.0, 0.0),  # dec zone edge
        (180.0, -0.0001),
        (12.0, 89.99),  # polar
        (12.0, -89.99),
    ]:
        ra.append(extra_ra)
        dec.append(extra_dec)

    records = make_records(ra, dec, seed=7)
    return write_catalog(tmp_path / "catalog", records)
