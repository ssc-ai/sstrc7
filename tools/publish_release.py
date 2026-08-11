#!/usr/bin/env python3
"""Maintainer tool: turn a local catalog into a GitHub release.

The catalog does not change, so this is expected to run once. It is committed
as the provenance record for how the published assets were produced, and so
that anyone with a copy of the raw catalog can regenerate byte-identical
assets and check them against the shipped manifest.

    # 1. compress every zone and write the manifest (~5 min on 16 cores)
    python tools/publish_release.py build \\
        --catalog /path/to/sstrc7 --staging /tmp/sstrc7-assets --tag v1.0.0

    # 2. create the release and upload all 1801 assets, resumable
    python tools/publish_release.py upload --staging /tmp/sstrc7-assets

    # 3. confirm every asset is on the release at the right size
    python tools/publish_release.py verify

Authentication for `upload` uses $GITHUB_TOKEN, else the token stored by
`gh auth login`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sstrc7._format import (  # noqa: E402
    INDEX_FILENAME,
    INDEX_SIZE,
    N_DEC_ZONES,
    RECORD_SIZE,
    encode_zone,
    zone_asset_name,
    zone_filename,
)
from sstrc7._progress import human_bytes  # noqa: E402

DEFAULT_REPO = "ssc-ai/sstrc7"
#: GitHub rejects the 1001st asset on a release with a 422 "file_count" error,
#: so the 1801 catalog files are spread over more than one release.
ASSETS_PER_RELEASE = 1000
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "src" / "sstrc7" / "manifest.json"
API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"


# --- build ----------------------------------------------------------------


def _encode_one(catalog: Path, staging: Path, zone_id: int) -> dict:
    """Compress one zone file and return its manifest entry."""
    source = catalog / zone_filename(zone_id)
    raw = source.read_bytes()
    if len(raw) % RECORD_SIZE:
        raise SystemExit(f"{source} is not a multiple of {RECORD_SIZE} bytes")

    target = staging / zone_asset_name(zone_id)
    blob = encode_zone(raw)

    # Never publish an asset that does not decode back to the exact input.
    from sstrc7._format import decode_zone

    if decode_zone(blob) != raw:
        raise SystemExit(f"{source}: round-trip check failed, refusing to publish")

    target.write_bytes(blob)
    return {
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "asset_size": len(blob),
        "asset_sha256": hashlib.sha256(blob).hexdigest(),
    }


def _plan_releases(base_tag: str, n_zones: int) -> list[dict]:
    """Split the zones across as many releases as the 1000-asset limit needs.

    The first release also carries the index file, so it gets one fewer zone.
    """
    count = -(-(n_zones + 1) // ASSETS_PER_RELEASE)  # +1 for the index file
    per_release = -(-n_zones // count)  # split evenly, rather than filling to
    # the cap, so a release is never one asset away from rejecting an upload

    releases: list[dict] = []
    for first in range(0, n_zones, per_release):
        last = min(first + per_release, n_zones) - 1
        releases.append(
            {
                "tag": f"{base_tag}-zones-{first:04d}-{last:04d}",
                "zones": [first, last],
                **({"index": True} if not releases else {}),
            }
        )
    return releases


def cmd_build(args: argparse.Namespace) -> int:
    catalog = Path(args.catalog)
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)

    index_file = catalog / INDEX_FILENAME
    index_bytes = index_file.read_bytes()
    if len(index_bytes) != INDEX_SIZE:
        raise SystemExit(f"{index_file} is {len(index_bytes)} bytes, expected {INDEX_SIZE}")
    (staging / INDEX_FILENAME).write_bytes(index_bytes)

    import numpy as np

    counts = np.frombuffer(index_bytes, dtype="<u4").reshape(N_DEC_ZONES, 60, 2)[:, :, 1]
    n_stars = int(counts.sum())

    zones: list[dict] = [{} for _ in range(N_DEC_ZONES)]
    done = 0
    raw_total = 0
    asset_total = 0
    started = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_encode_one, catalog, staging, z): z for z in range(N_DEC_ZONES)
        }
        for future in concurrent.futures.as_completed(futures):
            zone_id = futures[future]
            entry = future.result()
            zones[zone_id] = entry
            raw_total += entry["size"]
            asset_total += entry["asset_size"]
            done += 1
            if done % 25 == 0 or done == N_DEC_ZONES:
                rate = raw_total / (time.monotonic() - started) / 1e6
                print(
                    f"\r  {done}/{N_DEC_ZONES} zones  "
                    f"{human_bytes(raw_total)} -> {human_bytes(asset_total)}  "
                    f"{rate:.0f} MB/s in",
                    end="",
                    flush=True,
                )
    print()

    # Cross-check the index against what the zone files actually contain.
    for zone_id, entry in enumerate(zones):
        expected = int(counts[zone_id].sum()) * RECORD_SIZE
        if entry["size"] != expected:
            raise SystemExit(
                f"zone {zone_id}: file is {entry['size']} bytes but the index implies {expected}"
            )

    releases = _plan_releases(args.tag, N_DEC_ZONES)
    manifest = {
        "schema": 2,
        "repo": args.repo,
        "tag": releases[0]["tag"],
        "releases": releases,
        "record_size": RECORD_SIZE,
        "n_stars": n_stars,
        "index": {
            "size": len(index_bytes),
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
        },
        "zones": zones,
    }
    out = Path(args.manifest)
    out.write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"  stars     {n_stars:,}")
    print(f"  extracted {human_bytes(raw_total + len(index_bytes))}")
    print(f"  assets    {human_bytes(asset_total + len(index_bytes))}")
    print(f"  ratio     {asset_total / raw_total:.3f}")
    print(f"  manifest  {out}")
    return 0


# --- GitHub ---------------------------------------------------------------


def _token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        )
        if out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass

    hosts = Path.home() / ".config" / "gh" / "hosts.yml"
    if hosts.exists():
        for line in hosts.read_text().splitlines():
            key, _, value = line.strip().partition(":")
            if key in ("oauth_token", "token") and value.strip():
                return value.strip()

    raise SystemExit("no GitHub token: set $GITHUB_TOKEN or run `gh auth login`")


def _api(method: str, url: str, token: str, body: bytes | None = None, content_type: str | None = None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "sstrc7-publish",
    }
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(url, data=body, method=method, headers=headers)
    with urlopen(request, timeout=600) as response:
        payload = response.read()
    return json.loads(payload) if payload else {}


def _release(token: str, repo: str, tag: str, create: bool) -> dict:
    try:
        return _api("GET", f"{API}/repos/{repo}/releases/tags/{tag}", token)
    except HTTPError as exc:
        if exc.code != 404 or not create:
            raise
    body = json.dumps(
        {
            "tag_name": tag,
            "name": tag,
            "body": (
                "SSTRC7 star catalog data release.\n\n"
                "Assets are one compressed file per declination zone plus the "
                "`sstrc.acc` index. Do not download these by hand -- install the "
                "package and run `sstrc7 get`, which fetches, verifies, and "
                "extracts them.\n"
            ),
        }
    ).encode()
    return _api("POST", f"{API}/repos/{repo}/releases", token, body, "application/json")


def _existing_assets(token: str, repo: str, release_id: int) -> dict[str, dict]:
    assets: dict[str, dict] = {}
    page = 1
    while True:
        batch = _api(
            "GET",
            f"{API}/repos/{repo}/releases/{release_id}/assets?per_page=100&page={page}",
            token,
        )
        if not batch:
            return assets
        for asset in batch:
            assets[asset["name"]] = asset
        page += 1


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _asset_plan(manifest: dict) -> dict[str, list[tuple[str, int]]]:
    """Group (asset name, expected size) by the release tag that holds it."""
    releases = manifest["releases"]
    plan: dict[str, list[tuple[str, int]]] = {r["tag"]: [] for r in releases}

    index_tag = next(r["tag"] for r in releases if r.get("index"))
    plan[index_tag].append((INDEX_FILENAME, manifest["index"]["size"]))

    for zone_id, zone in enumerate(manifest["zones"]):
        for release in releases:
            first, last = release["zones"]
            if first <= zone_id <= last:
                plan[release["tag"]].append((zone_asset_name(zone_id), zone["asset_size"]))
                break
        else:
            raise SystemExit(f"no release covers zone {zone_id}")

    for tag, assets in plan.items():
        if len(assets) > ASSETS_PER_RELEASE:
            raise SystemExit(
                f"{tag} would hold {len(assets)} assets; GitHub allows "
                f"{ASSETS_PER_RELEASE} per release"
            )
    return plan


def _upload_one_release(
    args: argparse.Namespace,
    token: str,
    repo: str,
    tag: str,
    assets: list[tuple[str, int]],
) -> list[str]:
    """Upload one release's assets. Returns a list of failure descriptions."""
    staging = Path(args.staging)

    release = _release(token, repo, tag, create=True)
    release_id = release["id"]
    print(f"\nrelease {tag} id={release_id} ({release['html_url']})")

    have = _existing_assets(token, repo, release_id)
    expected = {name for name, _ in assets}

    # Assets from an earlier, differently-planned run would count against the
    # 1000 limit and shadow nothing useful, so drop them.
    for name, asset in have.items():
        if name not in expected:
            print(f"  removing stray asset {name}", flush=True)
            _api("DELETE", f"{API}/repos/{repo}/releases/assets/{asset['id']}", token)
    have = {k: v for k, v in have.items() if k in expected}

    todo = []
    for name, size in assets:
        current = have.get(name)
        if current is None:
            todo.append((name, size))
        elif current["size"] != size or current["state"] != "uploaded":
            # A half-finished upload from an earlier run: delete and redo.
            _api("DELETE", f"{API}/repos/{repo}/releases/assets/{current['id']}", token)
            todo.append((name, size))

    if not todo:
        print(f"  all {len(assets)} assets already uploaded")
        return []

    total = sum(size for _, size in todo)
    print(f"  uploading {len(todo)} of {len(assets)} assets ({human_bytes(total)})")

    sent = 0
    failed: list[str] = []
    started = time.monotonic()

    def put(name: str, size: int) -> str | None:
        blob = (staging / name).read_bytes()
        if len(blob) != size:
            return f"{name}: staged file is {len(blob)} bytes, manifest says {size}"
        url = f"{UPLOADS}/repos/{repo}/releases/{release_id}/assets?name={name}"
        for attempt in range(5):
            try:
                _api("POST", url, token, blob, "application/octet-stream")
                return None
            except HTTPError as exc:
                if exc.code == 422:
                    # 422 covers both "this name already exists" (harmless, a
                    # racing retry) and "file_count limited to 1000 assets per
                    # release" (fatal). Treating them alike silently drops real
                    # failures, so read the body and tell them apart.
                    body = exc.read().decode(errors="replace")
                    if "already_exists" in body or "already exists" in body:
                        return None
                    return f"{name}: HTTP 422 {body[:200]}"
                if attempt == 4:
                    return f"{name}: HTTP {exc.code} {exc.reason}"
            except OSError as exc:
                if attempt == 4:
                    return f"{name}: {exc}"
            time.sleep(2**attempt)
        return f"{name}: exhausted retries"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(put, name, size): (name, size) for name, size in todo}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, size = futures[future]
            error = future.result()
            if error:
                failed.append(error)
            sent += size
            elapsed = time.monotonic() - started
            print(
                f"\r  {done}/{len(todo)}  {human_bytes(sent)}/{human_bytes(total)}  "
                f"{sent / elapsed / 1e6:.1f} MB/s  {len(failed)} failed",
                end="",
                flush=True,
            )
    print()
    return failed


def cmd_upload(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    repo = manifest["repo"]
    token = _token()

    failed: list[str] = []
    for tag, assets in _asset_plan(manifest).items():
        failed += _upload_one_release(args, token, repo, tag, assets)

    for error in failed[:20]:
        print(f"  FAILED {error}", file=sys.stderr)
    if failed:
        print(f"{len(failed)} uploads failed; re-run to retry", file=sys.stderr)
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    repo = manifest["repo"]
    token = _token()

    missing, wrong = [], []
    total = 0

    for tag, assets in _asset_plan(manifest).items():
        total += len(assets)
        release = _release(token, repo, tag, create=False)
        have = _existing_assets(token, repo, release["id"])

        tag_missing, tag_wrong = [], []
        for name, size in assets:
            asset = have.get(name)
            if asset is None:
                tag_missing.append(name)
            elif asset["size"] != size or asset["state"] != "uploaded":
                tag_wrong.append(
                    f"{name}: {asset['size']} bytes state={asset['state']}, expected {size}"
                )
        good = len(assets) - len(tag_missing) - len(tag_wrong)
        print(f"{good}/{len(assets)} assets correct on {tag}")
        missing += tag_missing
        wrong += tag_wrong

    print(f"{total - len(missing) - len(wrong)}/{total} assets correct overall")
    for name in missing[:20]:
        print(f"  missing: {name}")
    for entry in wrong[:20]:
        print(f"  wrong:   {entry}")
    if len(missing) > 20 or len(wrong) > 20:
        print("  ...")
    return 1 if (missing or wrong) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="path to manifest.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="compress zones and write the manifest")
    build.add_argument("--catalog", required=True, help="directory holding the raw .cat files")
    build.add_argument("--staging", required=True, help="directory to write release assets into")
    build.add_argument("--repo", default=DEFAULT_REPO)
    build.add_argument("--tag", default="v1.0.0")
    build.add_argument("-j", "--workers", type=int, default=os.cpu_count() or 4)
    build.set_defaults(func=cmd_build)

    upload = subparsers.add_parser("upload", help="create the release and upload assets")
    upload.add_argument("--staging", required=True)
    upload.add_argument("-j", "--workers", type=int, default=4)
    upload.set_defaults(func=cmd_upload)

    verify = subparsers.add_parser("verify", help="check every asset is present at the right size")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
