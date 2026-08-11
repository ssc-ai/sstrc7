#!/usr/bin/env python3
"""Measure SSTRC7's depth and completeness against a Gaia G<=21 reference.

Every SSTRC7 star comes from Gaia, so a local Gaia mirror complete to G = 21 is
an exact reference: any Gaia star inside a field that SSTRC7 does not contain is
a real incompleteness, not a cross-catalog mismatch.

For each sampled field this compares the two catalogs over the identical cone,
bins by Gaia G, and reports the differential and cumulative completeness. It
also positionally cross-matches the two to check that the magnitudes and
positions SSTRC7 stores agree with Gaia's.

    python tools/analyze_depth.py \\
        --catalog /path/to/sstrc7 --gaia /path/to/gaia_g21/mirror \\
        --out docs

Writes depth.json plus the plots used in the README.

The Gaia mirror is a directory of HEALPix tiles with an index.json describing
their dtype and bounding boxes -- the layout written by senpai's gaia_mirror
ingest. Every tile whose box meets a field is read, so the reference is complete
over the field no matter how the diamond-shaped tiles fall across it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sstrc7._format import BAND_INDEX  # noqa: E402
from sstrc7.query import Catalog  # noqa: E402

# Chart palette: categorical slots 1-3 of the reference palette, which pass the
# all-pairs CVD and normal-vision floors in both light and dark modes.
SSTRC7_COLOR = "#2a78d6"
GAIA_COLOR = "#eb6834"
ACCENT_COLOR = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

MAG_BINS = np.arange(8.0, 21.51, 0.25)
MATCH_RADIUS_ARCSEC = 1.0


# --- geometry --------------------------------------------------------------


def unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cos_dec = np.cos(dec)
    return np.stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)], axis=-1)


def galactic_latitude(ra_deg: float, dec_deg: float) -> float:
    """Galactic latitude in degrees (J2000 north galactic pole)."""
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    pole_ra, pole_dec = math.radians(192.85948), math.radians(27.12825)
    sin_b = math.sin(dec) * math.sin(pole_dec) + math.cos(dec) * math.cos(pole_dec) * math.cos(
        ra - pole_ra
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_b))))


def box_meets_cone(box: dict, ra: float, dec: float, radius: float) -> bool:
    """Cheap rejection test between a tile bounding box and a cone."""
    if box["dec_min"] - radius > dec or box["dec_max"] + radius < dec:
        return False
    # Widen the RA tolerance by 1/cos(dec); near a pole accept everything.
    worst = max(abs(box["dec_min"]), abs(box["dec_max"]))
    if worst >= 89.0:
        return True
    pad = min(radius / max(math.cos(math.radians(worst)), 1e-6), 180.0)
    lo, hi = box["ra_min"] - pad, box["ra_max"] + pad
    return any(lo <= candidate <= hi for candidate in (ra, ra + 360.0, ra - 360.0))


# --- the Gaia reference ----------------------------------------------------


class GaiaMirror:
    """A tiled Gaia mirror, queried by cone."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        index = json.loads((self.directory / "index.json").read_text())
        self.dtype = np.dtype([tuple(field) for field in index["dtype"]])
        self.tiles = list(index["tiles"].values())
        self.n_stars = sum(tile["n"] for tile in self.tiles)

    def query_cone(self, ra: float, dec: float, radius: float) -> np.ndarray:
        parts = []
        for tile in self.tiles:
            if not box_meets_cone(tile, ra, dec, radius):
                continue
            records = np.fromfile(self.directory / tile["file"], dtype=self.dtype)
            if records.size == 0:
                continue
            cos_sep = unit_vectors(records["ra"], records["dec"]) @ unit_vectors(
                np.array([ra]), np.array([dec])
            )[0]
            inside = cos_sep >= math.cos(math.radians(radius))
            if inside.any():
                parts.append(records[inside])
        return np.concatenate(parts) if parts else np.empty(0, dtype=self.dtype)


# --- cross-match -----------------------------------------------------------


def cross_match(ra_a, dec_a, ra_b, dec_b, radius_arcsec: float):
    """Nearest-neighbour match from A into B. Returns (index_a, index_b)."""
    if len(ra_a) == 0 or len(ra_b) == 0:
        return np.empty(0, int), np.empty(0, int)

    vectors_a = unit_vectors(ra_a, dec_a)
    vectors_b = unit_vectors(ra_b, dec_b)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(vectors_b)
        limit = 2.0 * math.sin(math.radians(radius_arcsec / 3600.0) / 2.0)
        distance, index_b = tree.query(vectors_a, distance_upper_bound=limit)
        matched = np.isfinite(distance)
        return np.flatnonzero(matched), index_b[matched]
    except ImportError:
        # Fall back to a declination-sorted sweep; slower but dependency-free.
        order = np.argsort(dec_b)
        sorted_dec = dec_b[order]
        tolerance = radius_arcsec / 3600.0
        index_a, index_b = [], []
        for i in range(len(ra_a)):
            lo = np.searchsorted(sorted_dec, dec_a[i] - tolerance)
            hi = np.searchsorted(sorted_dec, dec_a[i] + tolerance)
            if hi <= lo:
                continue
            candidates = order[lo:hi]
            cos_sep = vectors_b[candidates] @ vectors_a[i]
            best = int(np.argmax(cos_sep))
            if cos_sep[best] >= math.cos(math.radians(tolerance)):
                index_a.append(i)
                index_b.append(int(candidates[best]))
        return np.asarray(index_a, int), np.asarray(index_b, int)


# --- fields ----------------------------------------------------------------


def choose_fields(gaia: GaiaMirror, count: int, seed: int) -> list[dict]:
    """Pick tile centres spanning a range of galactic latitude.

    Depth is driven by crowding, so sampling only random sky would under-report
    how much worse the galactic plane is. Fields are drawn evenly across
    |b| bands instead.
    """
    rng = np.random.default_rng(seed)
    centres = []
    for tile in gaia.tiles:
        ra = 0.5 * (tile["ra_min"] + tile["ra_max"])
        dec = 0.5 * (tile["dec_min"] + tile["dec_max"])
        if tile["ra_max"] - tile["ra_min"] > 90.0:  # wraps RA = 0, skip
            continue
        centres.append((ra, dec, abs(galactic_latitude(ra, dec))))

    bands = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 90)]
    per_band = max(count // len(bands), 1)
    chosen = []
    for low, high in bands:
        pool = [c for c in centres if low <= c[2] < high]
        if not pool:
            continue
        picks = rng.choice(len(pool), size=min(per_band, len(pool)), replace=False)
        for i in picks:
            ra, dec, abs_b = pool[int(i)]
            chosen.append({"ra": round(ra, 4), "dec": round(dec, 4), "abs_b": round(abs_b, 2)})
    return chosen


# --- analysis --------------------------------------------------------------


def analyse(catalog: Catalog, gaia: GaiaMirror, fields: list[dict], radius: float) -> dict:
    gaia_column = BAND_INDEX["Gaia_G"]
    total_sstrc7 = np.zeros(len(MAG_BINS) - 1)
    total_gaia = np.zeros(len(MAG_BINS) - 1)
    delta_mag: list[np.ndarray] = []
    separations: list[np.ndarray] = []
    per_field = []

    for i, field in enumerate(fields, 1):
        ra, dec = field["ra"], field["dec"]
        stars = catalog.query_cone(ra, dec, radius)
        reference = gaia.query_cone(ra, dec, radius)

        mag_sstrc7 = stars.mag[:, gaia_column]
        mag_sstrc7 = mag_sstrc7[~np.isnan(mag_sstrc7)]
        mag_gaia = reference["g"]
        mag_gaia = mag_gaia[np.isfinite(mag_gaia)]

        counts_sstrc7, _ = np.histogram(mag_sstrc7, bins=MAG_BINS)
        counts_gaia, _ = np.histogram(mag_gaia, bins=MAG_BINS)
        total_sstrc7 += counts_sstrc7
        total_gaia += counts_gaia

        index_a, index_b = cross_match(
            stars.ra, stars.dec, reference["ra"], reference["dec"], MATCH_RADIUS_ARCSEC
        )
        if len(index_a):
            paired_sstrc7 = stars.mag[index_a, gaia_column]
            paired_gaia = reference["g"][index_b]
            good = np.isfinite(paired_sstrc7) & np.isfinite(paired_gaia)
            delta_mag.append((paired_sstrc7 - paired_gaia)[good])

            cos_sep = np.sum(
                unit_vectors(stars.ra[index_a], stars.dec[index_a])
                * unit_vectors(reference["ra"][index_b], reference["dec"][index_b]),
                axis=1,
            )
            separations.append(np.degrees(np.arccos(np.clip(cos_sep, -1, 1))) * 3600.0)

        per_field.append(
            {
                **field,
                "n_sstrc7": int(len(stars)),
                "n_gaia": int(reference.size),
                "matched": int(len(index_a)),
                "counts_sstrc7": counts_sstrc7.tolist(),
                "counts_gaia": counts_gaia.tolist(),
            }
        )
        print(
            f"  [{i}/{len(fields)}] ra={ra:7.2f} dec={dec:+7.2f} |b|={field['abs_b']:5.1f}  "
            f"sstrc7={len(stars):7d}  gaia={reference.size:8d}  "
            f"matched={len(index_a):7d}",
            flush=True,
        )

    delta = np.concatenate(delta_mag) if delta_mag else np.zeros(0)
    separation = np.concatenate(separations) if separations else np.zeros(0)

    cut = cut_magnitude(total_sstrc7)
    for entry in per_field:
        entry["completeness"] = completeness_below(
            np.asarray(entry["counts_sstrc7"]), np.asarray(entry["counts_gaia"]), cut
        )

    return {
        "radius_deg": radius,
        "bins": MAG_BINS.tolist(),
        "counts_sstrc7": total_sstrc7.tolist(),
        "counts_gaia": total_gaia.tolist(),
        "cut_magnitude": cut,
        "completeness_below_cut": completeness_below(total_sstrc7, total_gaia, cut),
        "total_sstrc7": int(total_sstrc7.sum()),
        "total_gaia": int(total_gaia.sum()),
        "delta_mag": {
            "n": int(delta.size),
            "median": float(np.median(delta)) if delta.size else None,
            "mad": float(np.median(np.abs(delta - np.median(delta)))) if delta.size else None,
            "p16": float(np.percentile(delta, 16)) if delta.size else None,
            "p84": float(np.percentile(delta, 84)) if delta.size else None,
        },
        "separation_arcsec": {
            "n": int(separation.size),
            "median": float(np.median(separation)) if separation.size else None,
            "p99": float(np.percentile(separation, 99)) if separation.size else None,
        },
        "fields": per_field,
    }


def cut_magnitude(counts_sstrc7) -> float | None:
    """The magnitude past which the catalog holds no stars at all.

    SSTRC7 does not fade out -- it stops. This returns the faint edge of the
    last populated bin, which is the applied cut rather than a sensitivity
    limit.
    """
    populated = np.flatnonzero(np.asarray(counts_sstrc7) > 0)
    if populated.size == 0:
        return None
    return float(MAG_BINS[populated[-1] + 1])


def completeness_below(counts_sstrc7, counts_gaia, cut: float | None) -> float | None:
    """Fraction of the Gaia stars brighter than ``cut`` that SSTRC7 contains."""
    if cut is None:
        return None
    inside = MAG_BINS[1:] <= cut
    total_gaia = float(np.asarray(counts_gaia)[inside].sum())
    if total_gaia <= 0:
        return None
    return float(np.asarray(counts_sstrc7)[inside].sum() / total_gaia)


# --- plots -----------------------------------------------------------------


def style_axes(ax) -> None:
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def make_plots(result: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.asarray(result["bins"])
    centres = 0.5 * (bins[:-1] + bins[1:])
    counts_sstrc7 = np.asarray(result["counts_sstrc7"])
    counts_gaia = np.asarray(result["counts_gaia"])

    written = []

    # 1. Magnitude distribution -- two series, direct-labelled, no legend box.
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    style_axes(ax)
    ax.step(centres, np.where(counts_gaia > 0, counts_gaia, np.nan), where="mid",
            color=GAIA_COLOR, linewidth=2.0)
    ax.step(centres, np.where(counts_sstrc7 > 0, counts_sstrc7, np.nan), where="mid",
            color=SSTRC7_COLOR, linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Gaia G magnitude", color=MUTED, fontsize=10)
    ax.set_ylabel("stars per 0.25 mag", color=MUTED, fontsize=10)
    ax.set_title("Magnitude distribution over the sampled fields", color=INK,
                 fontsize=11, loc="left", pad=12)

    peak_gaia = int(np.nanargmax(counts_gaia))
    ax.annotate("Gaia (G < 21)", (centres[peak_gaia], counts_gaia[peak_gaia]),
                textcoords="offset points", xytext=(6, 6), color=GAIA_COLOR,
                fontsize=9.5, fontweight="bold")
    peak_sstrc7 = int(np.nanargmax(counts_sstrc7))
    ax.annotate("SSTRC7", (centres[peak_sstrc7], counts_sstrc7[peak_sstrc7]),
                textcoords="offset points", xytext=(-10, -16), ha="right", va="top",
                color=SSTRC7_COLOR, fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "depth-magnitude-distribution.png"
    fig.savefig(path, transparent=True)
    plt.close(fig)
    written.append(path)

    # 2. Completeness vs magnitude, with the 50% crossing called out.
    with np.errstate(divide="ignore", invalid="ignore"):
        completeness = np.where(counts_gaia >= 20, counts_sstrc7 / counts_gaia, np.nan)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    style_axes(ax)
    ax.axhline(1.0, color=GRID, linewidth=1.0)
    ax.plot(centres, 100.0 * completeness, color=SSTRC7_COLOR, linewidth=2.0)
    ax.set_ylim(-3, 108)
    ax.set_xlabel("Gaia G magnitude", color=MUTED, fontsize=10)
    ax.set_ylabel("completeness (%)", color=MUTED, fontsize=10)
    ax.set_title("SSTRC7 completeness against Gaia G < 21", color=INK,
                 fontsize=11, loc="left", pad=12)

    cut = result["cut_magnitude"]
    if cut is not None:
        ax.axvline(cut, color=ACCENT_COLOR, linewidth=1.6, linestyle="--", alpha=0.95)
        ax.annotate(f"hard cut at G = {cut:.2f}", (cut, 52),
                    textcoords="offset points", xytext=(-8, 0), rotation=90,
                    ha="right", va="bottom", color=ACCENT_COLOR, fontsize=9.5,
                    fontweight="bold")
    below = result["completeness_below_cut"]
    if below is not None:
        # Axes coordinates: the data-space bottom-left is empty, but the x-range
        # is autoscaled to the populated bins so a data-space anchor can fall
        # outside the view.
        ax.text(0.03, 0.12, f"{100 * below:.1f}% complete below the cut",
                transform=ax.transAxes, color=SSTRC7_COLOR, fontsize=10,
                fontweight="bold")
    fig.tight_layout()
    path = out_dir / "depth-completeness.png"
    fig.savefig(path, transparent=True)
    plt.close(fig)
    written.append(path)

    # 3. Completeness by galactic latitude band -- crowding is the whole story.
    fields = result["fields"]
    bands = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 90)]
    labels, limits = [], []
    for low, high in bands:
        members = [f["completeness"] for f in fields
                   if low <= f["abs_b"] < high and f["completeness"] is not None]
        if members:
            labels.append(f"{low}–{high}°")
            limits.append(100.0 * float(np.median(members)))

    if limits:
        # A dot plot, not bars: the spread is only ~2 points, which bars on a
        # zero baseline cannot show and bars on a cut baseline would exaggerate.
        fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        positions = np.arange(len(labels))
        low = math.floor(min(limits) * 2) / 2 - 0.5
        high = math.ceil(max(limits) * 2) / 2 + 0.5

        ax.hlines(positions, low, limits, color=GRID, linewidth=1.4, zorder=2)
        ax.scatter(limits, positions, s=90, color=SSTRC7_COLOR, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.set_yticks(positions, labels)
        # Leave headroom above the first row for its value label.
        ax.set_ylim(len(labels) - 0.5, -1.0)
        ax.set_xlim(low, high)
        ax.set_xlabel("completeness below the cut (%)", color=MUTED, fontsize=10)
        ax.set_ylabel("galactic latitude |b|", color=MUTED, fontsize=10)
        ax.set_title("Completeness below G = 18, by galactic latitude",
                     color=INK, fontsize=11, loc="left", pad=14)
        for y, value in zip(positions, limits):
            ax.annotate(f"{value:.1f}%", (value, y), textcoords="offset points",
                        xytext=(0, 12), ha="center", color=INK, fontsize=9.5,
                        fontweight="bold")
        fig.tight_layout()
        path = out_dir / "depth-by-latitude.png"
        fig.savefig(path, transparent=True)
        plt.close(fig)
        written.append(path)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--catalog", help="SSTRC7 catalog directory")
    parser.add_argument("--gaia", help="Gaia mirror directory (holding index.json)")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="redraw the plots from an existing depth.json without re-querying",
    )
    parser.add_argument("--out", default="docs", help="directory for the JSON and plots")
    parser.add_argument("--fields", type=int, default=20, help="how many fields to sample")
    parser.add_argument("--radius", type=float, default=0.4, help="field radius in degrees")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        result = json.loads((out_dir / "depth.json").read_text())
        for path in make_plots(result, out_dir):
            print(f"  wrote {path}")
        return 0

    if not args.catalog or not args.gaia:
        parser.error("--catalog and --gaia are required unless --replot is given")

    catalog = Catalog(args.catalog)
    gaia = GaiaMirror(args.gaia)
    print(f"Gaia reference: {gaia.n_stars:,} stars in {len(gaia.tiles)} tiles")

    fields = choose_fields(gaia, args.fields, args.seed)
    print(f"sampling {len(fields)} fields of radius {args.radius}deg")

    result = analyse(catalog, gaia, fields, args.radius)
    result["gaia_reference_stars"] = gaia.n_stars

    (out_dir / "depth.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n  SSTRC7 stars   {result['total_sstrc7']:,}")
    print(f"  Gaia stars     {result['total_gaia']:,}")
    print(f"  hard cut       G = {result['cut_magnitude']}")
    below = result["completeness_below_cut"]
    print(f"  completeness   {100 * below:.2f}% below the cut" if below else "  completeness   n/a")
    delta = result["delta_mag"]
    print(f"  matched        {delta['n']:,}")
    if delta["median"] is not None:
        print(f"  dG median      {delta['median']:+.4f}  (MAD {delta['mad']:.4f})")
    separation = result["separation_arcsec"]
    if separation["median"] is not None:
        print(f"  separation     {separation['median']:.4f}\" median, "
              f"{separation['p99']:.4f}\" p99")

    if not args.no_plots:
        for path in make_plots(result, out_dir):
            print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
