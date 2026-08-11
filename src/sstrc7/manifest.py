"""The expected contents of a complete catalog.

``manifest.json`` ships inside the package and records, for every file, both
the size and SHA-256 of the extracted ``.cat`` and of the compressed release
asset it comes from. That makes a local catalog verifiable with no network
access, and makes a download verifiable before it is decoded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from ._format import INDEX_FILENAME, zone_asset_name, zone_filename

MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class FileEntry:
    """One distributed file: its extracted form and its release asset."""

    name: str
    size: int
    sha256: str
    asset: str
    asset_size: int
    asset_sha256: str
    tag: str


@dataclass(frozen=True)
class Manifest:
    """Everything needed to fetch and verify a catalog release."""

    repo: str
    tag: str
    n_stars: int
    files: tuple[FileEntry, ...]

    @property
    def total_size(self) -> int:
        """Bytes on disk once extracted."""
        return sum(f.size for f in self.files)

    @property
    def download_size(self) -> int:
        """Bytes transferred for a complete download."""
        return sum(f.asset_size for f in self.files)

    def asset_url(self, entry: FileEntry) -> str:
        """Public download URL for one release asset."""
        return f"https://github.com/{self.repo}/releases/download/{entry.tag}/{entry.asset}"

    @property
    def release_url(self) -> str:
        """Human-facing release page."""
        return f"https://github.com/{self.repo}/releases/tag/{self.tag}"

    @property
    def tags(self) -> tuple[str, ...]:
        """Every release tag the assets are spread across, in order."""
        seen: dict[str, None] = {}
        for entry in self.files:
            seen.setdefault(entry.tag, None)
        return tuple(seen)

    def select(self, names: list[str]) -> tuple[FileEntry, ...]:
        """Return the entries matching ``names``, always including the index."""
        wanted = set(names) | {INDEX_FILENAME}
        return tuple(f for f in self.files if f.name in wanted)


def _tag_for_zone(releases: list[dict], zone_id: int) -> str:
    """Which release holds a given zone's asset.

    GitHub allows at most 1000 assets on a release, so the 1801 files are
    spread across more than one.
    """
    for release in releases:
        first, last = release["zones"]
        if first <= zone_id <= last:
            return release["tag"]
    raise ValueError(f"no release covers zone {zone_id}")


def _entries_from_json(data: dict) -> tuple[FileEntry, ...]:
    releases = data["releases"]
    index_tag = next(r["tag"] for r in releases if r.get("index"))

    entries = [
        FileEntry(
            name=INDEX_FILENAME,
            size=data["index"]["size"],
            sha256=data["index"]["sha256"],
            asset=INDEX_FILENAME,
            asset_size=data["index"]["size"],
            asset_sha256=data["index"]["sha256"],
            tag=index_tag,
        )
    ]
    for zone_id, zone in enumerate(data["zones"]):
        entries.append(
            FileEntry(
                name=zone_filename(zone_id),
                size=zone["size"],
                sha256=zone["sha256"],
                asset=zone_asset_name(zone_id),
                asset_size=zone["asset_size"],
                asset_sha256=zone["asset_sha256"],
                tag=_tag_for_zone(releases, zone_id),
            )
        )
    return tuple(entries)


@lru_cache(maxsize=1)
def load() -> Manifest:
    """Load the manifest bundled with this package."""
    text = resources.files(__package__).joinpath(MANIFEST_FILENAME).read_text()
    data = json.loads(text)
    if data.get("schema") != 2:
        raise ValueError(f"unsupported manifest schema {data.get('schema')!r}")
    return Manifest(
        repo=data["repo"],
        tag=data["tag"],
        n_stars=data["n_stars"],
        files=_entries_from_json(data),
    )
