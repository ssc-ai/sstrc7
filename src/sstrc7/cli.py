"""Command line interface: ``sstrc7 get|status|info|zones``."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from ._format import BAND_CENTERS_NM, BAND_NAMES, zone_dec_range
from ._progress import human_bytes
from .fetch import DownloadError, get, status
from .manifest import load as load_manifest
from .paths import PATH_ENV_VARS, catalog_path


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--path", help="catalog directory (default: $SSTRC7_PATH, then ~/.sstrc7)")
    parser.add_argument(
        "--dec-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="only this declination band, in degrees, instead of the whole sky",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sstrc7",
        description="Download and inspect the SSTRC7 star catalog.",
    )
    parser.add_argument("--version", action="version", version=f"sstrc7 {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("get", help="download any missing catalog files")
    _add_selection(fetch)
    fetch.add_argument("-j", "--workers", type=int, default=8, help="concurrent downloads")
    fetch.add_argument(
        "--verify-hashes",
        action="store_true",
        help="checksum existing local files instead of trusting their size (slow)",
    )
    fetch.add_argument("--force", action="store_true", help="re-download even if already valid")
    fetch.add_argument("-q", "--quiet", action="store_true", help="no progress bar")

    check = subparsers.add_parser("status", help="report what is present on disk")
    _add_selection(check)
    check.add_argument("--verify-hashes", action="store_true", help="also checksum every file")
    check.add_argument("-j", "--workers", type=int, default=8, help="threads used for hashing")

    subparsers.add_parser("info", help="describe the catalog and this release")
    subparsers.add_parser("zones", help="print the declination range of every zone")

    return parser


def _cmd_get(args: argparse.Namespace) -> int:
    try:
        path = get(
            args.path,
            dec_range=tuple(args.dec_range) if args.dec_range else None,
            workers=args.workers,
            verify_hashes=args.verify_hashes,
            force=args.force,
            progress=not args.quiet,
        )
    except DownloadError as exc:
        print(f"sstrc7: {exc}", file=sys.stderr)
        return 1
    print(f"sstrc7: catalog ready at {path}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    report = status(
        args.path,
        dec_range=tuple(args.dec_range) if args.dec_range else None,
        verify_hashes=args.verify_hashes,
        workers=args.workers,
    )
    print(report)
    if report.complete:
        return 0

    manifest = load_manifest()
    by_name = {entry.name: entry for entry in manifest.files}
    needed = sum(by_name[name].asset_size for name in report.needed if name in by_name)
    print(f"  {human_bytes(needed)} to download")
    for name in report.missing[:5]:
        print(f"  missing: {name}")
    if len(report.missing) > 5:
        print(f"  ... and {len(report.missing) - 5} more missing")
    for name in report.corrupt[:5]:
        print(f"  corrupt: {name}")
    if len(report.corrupt) > 5:
        print(f"  ... and {len(report.corrupt) - 5} more corrupt")
    print("  fix: sstrc7 get")
    return 1


def _cmd_info(_: argparse.Namespace) -> int:
    manifest = load_manifest()
    print(f"sstrc7 {__version__}")
    print(f"  release      {manifest.repo} {manifest.tag}")
    print(f"  stars        {manifest.n_stars:,}")
    print(f"  files        {len(manifest.files)}")
    print(f"  download     {human_bytes(manifest.download_size)}")
    print(f"  on disk      {human_bytes(manifest.total_size)}")
    print(f"  catalog path {catalog_path()}")
    print(f"  path from    {', '.join('$' + v for v in PATH_ENV_VARS)}, else ~/.sstrc7")
    print("  bands        " + ", ".join(f"{n} ({c} nm)" for n, c in zip(BAND_NAMES, BAND_CENTERS_NM)))
    return 0


def _cmd_zones(_: argparse.Namespace) -> int:
    from ._format import N_DEC_ZONES

    for zone_id in range(N_DEC_ZONES):
        low, high = zone_dec_range(zone_id)
        print(f"{zone_id:4d}  s{zone_id:04d}.cat  {low:+8.2f} .. {high:+8.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``sstrc7`` command."""
    args = build_parser().parse_args(argv)
    handlers = {
        "get": _cmd_get,
        "status": _cmd_status,
        "info": _cmd_info,
        "zones": _cmd_zones,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
