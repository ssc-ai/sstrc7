"""The manifest that ships in the package must describe a real release."""

from __future__ import annotations

import pytest

from sstrc7._format import INDEX_FILENAME, INDEX_SIZE, N_DEC_ZONES, RECORD_SIZE, zone_filename
from sstrc7.manifest import load


@pytest.fixture(scope="module")
def manifest():
    return load()


def test_describes_every_file(manifest):
    assert len(manifest.files) == N_DEC_ZONES + 1
    names = {entry.name for entry in manifest.files}
    assert INDEX_FILENAME in names
    assert all(zone_filename(z) in names for z in range(N_DEC_ZONES))


def test_index_entry_is_the_right_size(manifest):
    index = next(e for e in manifest.files if e.name == INDEX_FILENAME)
    assert index.size == INDEX_SIZE
    # The index is published uncompressed, so both hashes describe one file.
    assert index.asset == index.name
    assert index.asset_sha256 == index.sha256


def test_every_zone_is_record_aligned_and_compressed(manifest):
    for entry in manifest.files:
        if entry.name == INDEX_FILENAME:
            continue
        assert entry.size % RECORD_SIZE == 0, entry.name
        assert entry.asset.endswith(".catz"), entry.name
        assert entry.asset_size > 0, entry.name


def test_hashes_are_well_formed_and_distinct(manifest):
    for entry in manifest.files:
        assert len(entry.sha256) == 64
        assert len(entry.asset_sha256) == 64
    # 1801 files of real data should not collide.
    assert len({e.sha256 for e in manifest.files}) == len(manifest.files)


def test_totals_match_the_published_catalog(manifest):
    assert manifest.n_stars == 294_222_203
    assert manifest.total_size == sum(e.size for e in manifest.files)
    assert (manifest.total_size - INDEX_SIZE) // RECORD_SIZE == manifest.n_stars
    # The whole point of the codec: assets must be far smaller than the data.
    assert manifest.download_size < manifest.total_size / 2


def test_asset_urls_point_at_the_release_holding_that_asset(manifest):
    for entry in (manifest.files[0], manifest.files[900], manifest.files[-1]):
        url = manifest.asset_url(entry)
        assert url == (
            f"https://github.com/{manifest.repo}/releases/download/{entry.tag}/{entry.asset}"
        )


def test_assets_are_split_to_respect_githubs_thousand_per_release_limit(manifest):
    """GitHub rejects the 1001st asset on a release, and there are 1801 files."""
    per_tag: dict[str, int] = {}
    for entry in manifest.files:
        per_tag[entry.tag] = per_tag.get(entry.tag, 0) + 1

    assert len(per_tag) >= 2
    assert sum(per_tag.values()) == len(manifest.files)
    for tag, count in per_tag.items():
        assert count <= 1000, f"{tag} holds {count} assets"


def test_every_file_has_a_release(manifest):
    assert all(entry.tag for entry in manifest.files)
    assert set(manifest.tags) == {entry.tag for entry in manifest.files}


def test_zones_are_assigned_to_releases_contiguously(manifest):
    """Each release covers one unbroken zone range, so the split is legible."""
    zones = [e for e in manifest.files if e.name != INDEX_FILENAME]
    boundaries = [i for i in range(1, len(zones)) if zones[i].tag != zones[i - 1].tag]
    assert len(boundaries) == len(manifest.tags) - 1
