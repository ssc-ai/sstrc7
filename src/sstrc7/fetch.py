"""Download, verify, and extract the catalog.

The catalog is distributed as one release asset per declination zone, so a
fetch is 1801 independent ~5 MB transfers rather than a handful of multi-
gigabyte archives. Each one resumes on its own, is checksummed on its own, and
is skipped entirely if the extracted file is already correct -- which makes
:func:`get` safe and cheap to call unconditionally at program start.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import manifest as _manifest
from ._format import decode_zone, zone_filename, zones_for_dec_range
from ._progress import progress_bar
from .manifest import FileEntry, Manifest
from .paths import catalog_path

TEMP_DIRNAME = ".sstrc7-download"
MAX_ATTEMPTS = 4
CHUNK_SIZE = 1 << 20


class DownloadError(RuntimeError):
    """A file could not be downloaded or failed verification."""


# --- Inspecting a local catalog -------------------------------------------


@dataclass
class CatalogStatus:
    """The result of comparing a directory against the manifest."""

    path: Path
    expected: int
    present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    corrupt: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """True when every expected file is present and valid."""
        return not self.missing and not self.corrupt

    @property
    def needed(self) -> list[str]:
        """Files that a fetch would have to download."""
        return self.missing + self.corrupt

    def __str__(self) -> str:
        if self.complete:
            return f"catalog complete at {self.path} ({self.expected} files)"
        return (
            f"catalog incomplete at {self.path}: {len(self.present)}/{self.expected} files "
            f"({len(self.missing)} missing, {len(self.corrupt)} corrupt)"
        )


def sha256_file(path: str | os.PathLike[str], chunk_size: int = CHUNK_SIZE) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_ok(directory: Path, entry: FileEntry, verify_hashes: bool) -> bool:
    target = directory / entry.name
    try:
        if target.stat().st_size != entry.size:
            return False
    except OSError:
        return False
    if verify_hashes and sha256_file(target) != entry.sha256:
        return False
    return True


def status(
    path: str | os.PathLike[str] | None = None,
    *,
    zones: list[int] | None = None,
    dec_range: tuple[float, float] | None = None,
    verify_hashes: bool = False,
    workers: int = 8,
) -> CatalogStatus:
    """Report which catalog files are present and valid.

    Args:
        path: catalog directory; None resolves from the environment.
        zones: declination zone indices to check, or None for all.
        dec_range: ``(dec_min, dec_max)`` in degrees, an alternative to ``zones``.
        verify_hashes: also check SHA-256 of every file. Correct but slow --
            it reads the full 17.6 GB. Size checks alone are near-instant.
        workers: threads used when hashing.

    Returns:
        A :class:`CatalogStatus`.
    """
    directory = catalog_path(path)
    entries = _resolve_entries(_manifest.load(), zones=zones, dec_range=dec_range)
    result = CatalogStatus(path=directory, expected=len(entries))

    if not directory.is_dir():
        result.missing = [entry.name for entry in entries]
        return result

    def check(entry: FileEntry) -> tuple[FileEntry, bool, bool]:
        exists = (directory / entry.name).exists()
        return entry, exists, _entry_ok(directory, entry, verify_hashes)

    with ThreadPoolExecutor(max_workers=workers if verify_hashes else 1) as pool:
        for entry, exists, ok in pool.map(check, entries):
            if ok:
                result.present.append(entry.name)
            elif exists:
                result.corrupt.append(entry.name)
            else:
                result.missing.append(entry.name)

    return result


def _resolve_entries(
    mf: Manifest,
    *,
    zones: list[int] | None,
    dec_range: tuple[float, float] | None,
) -> tuple[FileEntry, ...]:
    if zones is not None and dec_range is not None:
        raise ValueError("pass zones or dec_range, not both")
    if zones is None and dec_range is None:
        return mf.files
    if dec_range is not None:
        zones = list(zones_for_dec_range(*dec_range))
    return mf.select([zone_filename(z) for z in zones])


# --- Downloading -----------------------------------------------------------


def _open(url: str, offset: int = 0, timeout: float = 60.0):
    request = Request(url, headers={"User-Agent": "sstrc7"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    return urlopen(request, timeout=timeout)


def _download_asset(url: str, target: Path, entry: FileEntry) -> bytes:
    """Fetch one asset to ``target``, resuming a partial file if present.

    Returns the asset bytes. Raises :class:`DownloadError` if the content does
    not match the manifest after the last attempt.
    """
    last_error: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        offset = target.stat().st_size if target.exists() else 0
        if offset > entry.asset_size:
            target.unlink()
            offset = 0

        try:
            if offset < entry.asset_size:
                with _open(url, offset) as response:
                    # A server that ignores Range replies 200 with the whole
                    # body, in which case the partial file must be discarded.
                    mode = "ab" if response.status == 206 and offset else "wb"
                    with open(target, mode) as handle:
                        shutil.copyfileobj(response, handle, CHUNK_SIZE)

            blob = target.read_bytes()
            if len(blob) == entry.asset_size and hashlib.sha256(blob).hexdigest() == entry.asset_sha256:
                return blob

            last_error = DownloadError(f"{entry.asset}: content did not match the manifest")
            target.unlink(missing_ok=True)

        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code == 404:
                raise DownloadError(f"{entry.asset}: not found at {url}") from exc

        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(2**attempt)

    raise DownloadError(f"{entry.asset}: giving up after {MAX_ATTEMPTS} attempts") from last_error


def _install(entry: FileEntry, blob: bytes, directory: Path, temp_dir: Path) -> None:
    """Decode an asset and move it into place atomically."""
    data = blob if entry.asset == entry.name else decode_zone(blob)

    if len(data) != entry.size:
        raise DownloadError(f"{entry.name}: extracted {len(data)} bytes, expected {entry.size}")
    if hashlib.sha256(data).hexdigest() != entry.sha256:
        raise DownloadError(f"{entry.name}: extracted content did not match the manifest")

    staged = temp_dir / (entry.name + ".partial")
    staged.write_bytes(data)
    os.replace(staged, directory / entry.name)


def get(
    path: str | os.PathLike[str] | None = None,
    *,
    zones: list[int] | None = None,
    dec_range: tuple[float, float] | None = None,
    workers: int = 8,
    verify_hashes: bool = False,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Ensure the catalog is present on disk, downloading only what is missing.

    Does nothing if the catalog is already complete, so this is safe to call on
    every run. A full catalog is 7.2 GB to download and 17.6 GB on disk.

    Args:
        path: destination directory; None resolves from the environment
            (``$SSTRC7_PATH``, then ``~/.sstrc7``).
        zones: declination zone indices to fetch, or None for the whole sky.
        dec_range: ``(dec_min, dec_max)`` in degrees, an alternative to ``zones``.
        workers: concurrent downloads.
        verify_hashes: checksum existing local files instead of trusting their
            size. Downloads are always checksummed regardless.
        force: re-download even if the local file is already valid.
        progress: show a progress bar.

    Returns:
        The catalog directory.

    Raises:
        DownloadError: if any file could not be fetched or verified.
    """
    mf = _manifest.load()
    directory = catalog_path(path)
    entries = _resolve_entries(mf, zones=zones, dec_range=dec_range)

    directory.mkdir(parents=True, exist_ok=True)
    temp_dir = directory / TEMP_DIRNAME
    temp_dir.mkdir(exist_ok=True)

    if force:
        todo = list(entries)
    else:
        todo = [e for e in entries if not _entry_ok(directory, e, verify_hashes)]

    if not todo:
        if progress:
            print(f"sstrc7: catalog already complete at {directory} ({len(entries)} files)")
        _cleanup(temp_dir)
        return directory

    total = sum(entry.asset_size for entry in todo)
    label = f"sstrc7: {len(todo)} of {len(entries)} files"
    failures: list[str] = []

    with progress_bar(total, label, enabled=progress) as bar:

        def fetch_one(entry: FileEntry) -> None:
            try:
                blob = _download_asset(mf.asset_url(entry), temp_dir / entry.asset, entry)
                _install(entry, blob, directory, temp_dir)
                (temp_dir / entry.asset).unlink(missing_ok=True)
            except DownloadError as exc:
                failures.append(str(exc))
            finally:
                bar.update(entry.asset_size)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch_one, todo))

    if failures:
        shown = "\n  ".join(failures[:10])
        more = f"\n  ... and {len(failures) - 10} more" if len(failures) > 10 else ""
        raise DownloadError(
            f"{len(failures)} of {len(todo)} files failed; re-run get() to retry "
            f"(completed files are kept):\n  {shown}{more}"
        )

    _cleanup(temp_dir)
    return directory


def _cleanup(temp_dir: Path) -> None:
    try:
        next(temp_dir.iterdir())
    except StopIteration:
        temp_dir.rmdir()
    except OSError:
        pass
