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


def test_asset_urls_point_at_the_release(manifest):
    entry = manifest.files[900]
    url = manifest.asset_url(entry)
    assert url.startswith(f"https://github.com/{manifest.repo}/releases/download/{manifest.tag}/")
    assert url.endswith(entry.asset)
