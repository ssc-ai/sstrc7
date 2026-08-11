"""Download, verification, and extraction, against a local HTTP server."""

from __future__ import annotations

import functools
import hashlib
import http.server
import threading

import numpy as np
import pytest

from sstrc7 import fetch
from sstrc7._format import CODEC_SUFFIX, INDEX_FILENAME, encode_zone
from sstrc7.fetch import DownloadError, get, sha256_file, status
from sstrc7.manifest import FileEntry, Manifest

from .conftest import make_records, write_catalog


class LocalManifest(Manifest):
    """A manifest whose assets are served by the test HTTP server."""

    base_url = ""

    def asset_url(self, entry: FileEntry) -> str:
        return f"{self.base_url}/{entry.asset}"


@pytest.fixture
def release(tmp_path):
    """A tiny published catalog: source files, encoded assets, and a manifest."""
    records = make_records(
        np.linspace(0.5, 359.5, 40), np.linspace(-60.0, 60.0, 40), seed=11
    )
    source = write_catalog(tmp_path / "source", records)
    assets = tmp_path / "assets"
    assets.mkdir()

    entries = []
    for path in sorted(source.iterdir()):
        raw = path.read_bytes()
        if path.name == INDEX_FILENAME:
            (assets / path.name).write_bytes(raw)
            blob, asset_name = raw, path.name
        else:
            blob = encode_zone(raw)
            asset_name = path.stem + CODEC_SUFFIX
            (assets / asset_name).write_bytes(blob)
        entries.append(
            FileEntry(
                name=path.name,
                size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                asset=asset_name,
                asset_size=len(blob),
                asset_sha256=hashlib.sha256(blob).hexdigest(),
            )
        )

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(assets))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    manifest = LocalManifest(
        repo="test/test", tag="v0", n_stars=len(records), files=tuple(entries)
    )
    manifest.base_url = f"http://127.0.0.1:{server.server_address[1]}"

    yield manifest, source, tmp_path
    server.shutdown()


@pytest.fixture
def published(release, monkeypatch):
    manifest, source, tmp_path = release
    monkeypatch.setattr(fetch._manifest, "load", lambda: manifest)
    return manifest, source, tmp_path


def assert_matches_source(destination, source):
    for path in sorted(source.iterdir()):
        assert (destination / path.name).read_bytes() == path.read_bytes(), path.name


def test_get_downloads_and_extracts_byte_identical_files(published):
    manifest, source, tmp_path = published
    destination = tmp_path / "download"

    get(destination, progress=False)

    assert_matches_source(destination, source)
    assert status(destination).complete
    assert not (destination / fetch.TEMP_DIRNAME).exists()


def test_get_is_a_no_op_when_already_complete(published, monkeypatch):
    manifest, source, tmp_path = published
    destination = tmp_path / "download"
    get(destination, progress=False)

    def explode(*args, **kwargs):
        raise AssertionError("should not re-download a complete catalog")

    monkeypatch.setattr(fetch, "_download_asset", explode)
    get(destination, progress=False)


def test_get_repairs_a_truncated_file(published):
    manifest, source, tmp_path = published
    destination = tmp_path / "download"
    get(destination, progress=False)

    victim = destination / "s0900.cat"
    if not victim.exists():
        victim = next(p for p in destination.iterdir() if p.suffix == ".cat")
    victim.write_bytes(b"\x00" * 12)

    report = status(destination)
    assert not report.complete
    assert victim.name in report.corrupt

    get(destination, progress=False)
    assert_matches_source(destination, source)


def test_get_repairs_a_file_with_the_right_size_but_wrong_bytes(published):
    manifest, source, tmp_path = published
    destination = tmp_path / "download"
    get(destination, progress=False)

    victim = next(p for p in destination.iterdir() if p.suffix == ".cat" and p.stat().st_size)
    original = victim.read_bytes()
    victim.write_bytes(b"\xff" * len(original))

    # A size-only check cannot see this; a hash check must.
    assert status(destination).complete
    assert not status(destination, verify_hashes=True).complete

    get(destination, verify_hashes=True, progress=False)
    assert victim.read_bytes() == original


def test_get_resumes_a_partial_asset(published):
    manifest, source, tmp_path = published
    destination = tmp_path / "download"
    destination.mkdir()
    temp_dir = destination / fetch.TEMP_DIRNAME
    temp_dir.mkdir()

    entry = next(e for e in manifest.files if e.asset.endswith(CODEC_SUFFIX) and e.asset_size > 40)
    full = (tmp_path / "assets" / entry.asset).read_bytes()
    (temp_dir / entry.asset).write_bytes(full[: len(full) // 2])

    get(destination, progress=False)
    assert_matches_source(destination, source)


def test_get_discards_an_oversized_partial_asset(published):
    manifest, source, tmp_path = published
    destination = tmp_path / "download"
    destination.mkdir()
    temp_dir = destination / fetch.TEMP_DIRNAME
    temp_dir.mkdir()

    entry = next(e for e in manifest.files if e.asset.endswith(CODEC_SUFFIX))
    (temp_dir / entry.asset).write_bytes(b"\x00" * (entry.asset_size + 500))

    get(destination, progress=False)
    assert_matches_source(destination, source)


def test_corrupted_asset_on_the_server_is_rejected(published):
    manifest, source, tmp_path = published
    entry = next(e for e in manifest.files if e.asset.endswith(CODEC_SUFFIX))
    served = tmp_path / "assets" / entry.asset
    served.write_bytes(b"\x00" * entry.asset_size)

    with pytest.raises(DownloadError):
        get(tmp_path / "download", progress=False, workers=2)


def test_missing_asset_reports_the_file_that_failed(published):
    manifest, source, tmp_path = published
    entry = next(e for e in manifest.files if e.asset.endswith(CODEC_SUFFIX))
    (tmp_path / "assets" / entry.asset).unlink()

    with pytest.raises(DownloadError, match=entry.asset):
        get(tmp_path / "download", progress=False, workers=2)


def test_partial_failure_keeps_the_files_that_did_succeed(published):
    manifest, source, tmp_path = published
    entry = next(e for e in manifest.files if e.asset.endswith(CODEC_SUFFIX))
    (tmp_path / "assets" / entry.asset).unlink()
    destination = tmp_path / "download"

    with pytest.raises(DownloadError):
        get(destination, progress=False, workers=2)

    report = status(destination)
    assert report.needed == [entry.name]
    assert len(report.present) == len(manifest.files) - 1


def test_dec_range_fetches_only_the_zones_that_overlap(published):
    manifest, source, tmp_path = published
    destination = tmp_path / "download"

    get(destination, dec_range=(-61.0, -55.0), progress=False)

    assert (destination / INDEX_FILENAME).exists()
    report = status(destination, dec_range=(-61.0, -55.0))
    assert report.complete
    # The whole-sky view is still incomplete, and says so.
    assert not status(destination).complete


def test_status_on_a_directory_that_does_not_exist(published, tmp_path):
    manifest, _, _ = published
    report = status(tmp_path / "absent")
    assert not report.complete
    assert len(report.missing) == len(manifest.files)
    assert "incomplete" in str(report)


def test_status_string_when_complete(published):
    manifest, source, tmp_path = published
    destination = tmp_path / "download"
    get(destination, progress=False)
    assert "complete" in str(status(destination))


def test_force_redownloads_valid_files(published, monkeypatch):
    manifest, source, tmp_path = published
    destination = tmp_path / "download"
    get(destination, progress=False)

    calls = []
    original = fetch._download_asset
    monkeypatch.setattr(
        fetch,
        "_download_asset",
        lambda url, target, entry: (calls.append(entry.name), original(url, target, entry))[1],
    )
    get(destination, force=True, progress=False)
    assert len(calls) == len(manifest.files)


def test_zones_and_dec_range_are_mutually_exclusive(published, tmp_path):
    with pytest.raises(ValueError, match="not both"):
        status(tmp_path / "x", zones=[1], dec_range=(0.0, 1.0))


def test_sha256_file(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"hello")
    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()
