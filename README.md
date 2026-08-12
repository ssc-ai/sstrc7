# sstrc7

The SSTRC7 all-sky star catalog, largely complete (against Gaia) down to 18th magnitude — **294,222,203 stars**, 18 photometric bands — packaged in binary format for rapid reads.

```bash
pip install sstrc7          # or: uv pip install sstrc7
sstrc7 get                  # 7.2 GB download, 17.6 GB on disk, resumable
```

```python
import sstrc7

sstrc7.get()                                  # no-op once the data is there
stars = sstrc7.query_cone(83.822, -5.391, 0.5)   # ra, dec, radius in degrees

print(len(stars))                             # 1904
print(stars.ra, stars.dec)                    # degrees
print(stars.band("Johnson_V"))                # NaN where unmeasured
print(stars.visual)                           # best broadband magnitude per star
```

`get()` checks what is already on disk and downloads only what is missing, so calling it at the top of a script every run costs nothing after the first time.

---

## Contents

- [sstrc7](#sstrc7)
  - [Contents](#contents)
  - [What is in the catalog](#what-is-in-the-catalog)
  - [Photometric bands](#photometric-bands)
  - [Depth and completeness](#depth-and-completeness)
  - [Installing and downloading](#installing-and-downloading)
  - [Where the data lives](#where-the-data-lives)
  - [Querying](#querying)
    - [Cone](#cone)
    - [Box](#box)
    - [Focal plane](#focal-plane)
    - [What a query returns](#what-a-query-returns)
    - [Reusing a catalog handle](#reusing-a-catalog-handle)
  - [Command line](#command-line)
  - [Binary format](#binary-format)
  - [How the release is distributed](#how-the-release-is-distributed)
  - [Provenance and citation](#provenance-and-citation)
  - [Development](#development)
  - [License](#license)

---

## What is in the catalog

|                       |                                                         |
| --------------------- | ------------------------------------------------------- |
| Stars                 | 294,222,203                                             |
| Sky coverage          | all-sky, δ = −90° to +90°                               |
| Photometric bands     | 18, from 440 nm to 22 µm                                |
| Depth                 | hard cut at Gaia **G = 18.0**, 97% complete below it    |
| Astrometric precision | positions stored to 1 mas, agreeing with Gaia to 0.003″ |
| Extracted size        | 17.65 GB across 1801 files                              |
| Download size         | ~7.2 GB                                                 |

Each star carries a position, a proper motion, a parallax, up to 18 magnitudes, and a bitmask recording which source catalogs contributed to it. Survey coverage:

| Source                   | Share of stars |
| ------------------------ | -------------- |
| Gaia                     | 100.00%        |
| 2MASS                    | 94.48%         |
| AllWISE                  | 61.02%         |
| Tycho-Gaia (TGAS)        | 0.71%          |
| Henry Draper (HD)        | 0.10%          |
| Hipparcos                | 0.03%          |
| Bright Star Catalog (HR) | 0.004%         |

Stars are also flagged as photometric standards (80.2%), variable (0.21%), extended sources (0.05%), and multiples (0.002%). Use `sstrc7.decode_source_flags()` to expand a star's bitmask, or `field.flags(i)` for one star:

```python
>>> stars.flags(0)
['Gaia Catalog', '2MASS Catalog', 'AllWISE Catalog', 'Photometric Standard']
```

## Photometric bands

Bands are stored in this order, and `field.mag` is an `(N, 18)` array in the same order. A band with no measurement for a given star is `NaN` — never a sentinel you have to remember to filter.

|   # | Band        | λ (nm) | Coverage | Median mag |
| --: | ----------- | -----: | -------: | ---------: |
|   0 | `Gaia_G`    |    600 |   100.0% |      16.88 |
|   1 | `Gaia_BP`   |    500 |   100.0% |      17.52 |
|   2 | `Gaia_RP`   |    800 |   100.0% |      16.06 |
|   3 | `Johnson_B` |    440 |    20.5% |      16.24 |
|   4 | `Johnson_V` |    548 |    20.5% |      15.33 |
|   5 | `Johnson_R` |    700 |    62.3% |      16.64 |
|   6 | `Johnson_I` |    900 |    62.3% |      15.97 |
|   7 | `Sloan_g`   |    477 |    62.4% |      17.66 |
|   8 | `Sloan_r`   |    622 |    62.5% |      16.86 |
|   9 | `Sloan_i`   |    762 |    62.6% |      16.42 |
|  10 | `Sloan_z`   |    913 |    62.5% |      16.18 |
|  11 | `2MASS_J`   |   1235 |    94.5% |      14.92 |
|  12 | `2MASS_H`   |   1662 |    94.5% |      14.40 |
|  13 | `2MASS_Ks`  |   2159 |    94.5% |      14.25 |
|  14 | `WISE_W1`   |   3400 |    61.0% |      14.00 |
|  15 | `WISE_W2`   |   4600 |    61.0% |      14.07 |
|  16 | `WISE_W3`   |  12000 |    61.0% |      12.40 |
|  17 | `WISE_W4`   |  22000 |    61.0% |       8.96 |

Because coverage is uneven, there are two ways to get one magnitude per star:

```python
stars.visual                  # first available of V, R, Sloan r, Gaia G, Sloan g, B
stars.at_wavelength(650.0)    # interpolate each star's SED across its own bands
```

`at_wavelength` interpolates linearly in wavelength across whichever bands that particular star has, and clamps to the nearest measured band outside that span — so it returns a usable number for every star with at least one measurement, which is all of them.

## Depth and completeness

The SSTRC7 cuts out at G = 18 and is 97.1% complete.

![SSTRC7 completeness against Gaia](https://raw.githubusercontent.com/ssc-ai/sstrc7/main/docs/depth-completeness.png)

Comparison to a deeper Gaia cut:

![Magnitude distribution](https://raw.githubusercontent.com/ssc-ai/sstrc7/main/docs/depth-magnitude-distribution.png)

Completeness holds up in crowded fields: the galactic plane loses under two points against the poles.

![Completeness by galactic latitude](https://raw.githubusercontent.com/ssc-ai/sstrc7/main/docs/depth-by-latitude.png)

Cross-matching the two catalogs position by position shows the stored values are Gaia's, carried through faithfully:

| Check                                          | Result                           |
| ---------------------------------------------- | -------------------------------- |
| SSTRC7 stars with a Gaia counterpart within 1″ | 99.99% (91,172 of 91,177)        |
| Median position difference                     | 0.0031″ (99th percentile 0.019″) |
| Median `Gaia_G` difference                     | +0.017 mag (MAD 0.005)           |

**What this means in practice.** Treat G = 18 as the catalog's horizon: for star-field simulation, astrometric solving, or photometric calibration down to 18th magnitude, SSTRC7 is essentially a complete Gaia sample with 17 extra bands attached. Past 18, it is empty — if you need fainter stars, you need a different catalog.

Reproduce any of this with `tools/analyze_depth.py`, which takes a catalog path and a Gaia mirror and writes `docs/depth.json` alongside these plots.

## Installing and downloading

```bash
pip install sstrc7
```

Then either:

```bash
sstrc7 get                              # whole sky
sstrc7 get --dec-range -30 30           # only the declinations you need
sstrc7 get --path /data/sstrc7 -j 16    # explicit location, 16 parallel downloads
```

or from Python:

```python
sstrc7.get()
sstrc7.get(dec_range=(-30, 30))
sstrc7.get("/data/sstrc7", workers=16)
```

The download is safe to interrupt. Every file is checksummed against a manifest, partial transfers resume, and anything that fails is retried the next time you call `get()`. Files that are already correct are skipped.

```bash
sstrc7 status                   # what is present, what is missing
sstrc7 status --verify-hashes   # full SHA-256 check of all 17.6 GB (slow)
```

By default `status` compares file sizes, which is effectively instant. `--verify-hashes` reads every byte and catches silent corruption.

## Where the data lives

`get()` and every query resolve the catalog directory in this order:

1. an explicit path argument (`sstrc7.get("/data/sstrc7")`, `Catalog("/data/sstrc7")`)
2. `$SSTRC7_PATH`
3. `$SDASIM_SSTRC7_PATH`, `$SDASIM_SSTR7_PATH`, `$SATSIM_SSTR7_PATH` — so an existing sdasim or satsim setup is picked up as-is
4. `~/.sstrc7`

If you already have the catalog somewhere, point `$SSTRC7_PATH` at it and `get()` will verify it and do nothing.

## Querying

### Cone

```python
stars = sstrc7.query_cone(ra=266.417, dec=-29.008, radius=0.25)
```

Exact — clipped by true angular separation, not a bounding box — and correct across the RA = 0 wrap and over both poles.

### Box

```python
stars = sstrc7.query_box(ra_min=359.5, ra_max=0.5, dec_min=-1, dec_max=1)
```

`ra_min > ra_max` selects the wrapped interval through 0°. Pass `radians=True` to work in radians.

### Focal plane

```python
rows, cols, mag = sstrc7.query_by_los(
    height=2048, width=2048,
    y_fov=1.5, x_fov=1.5,     # degrees
    ra=83.822, dec=-5.391,
    rot=0.0,
    pad_mult=1.0,             # search this much beyond the sensor
    filter_center=650.0,      # optional: magnitudes at 650 nm
)
```

Projects the catalog through a gnomonic (TAN) WCS onto a sensor and returns pixel row/column and magnitude.

### What a query returns

`query_cone` and `query_box` return a `StarField`, a thin wrapper over the raw records. Columns are computed on access, so pulling only what you need stays cheap:

| Attribute            | Meaning                                                 |
| -------------------- | ------------------------------------------------------- |
| `len(field)`         | number of stars                                         |
| `.ra`, `.dec`        | degrees (`.ra_rad`, `.dec_rad` for radians)             |
| `.mag`               | `(N, 18)` magnitudes, NaN where unmeasured              |
| `.band(name)`        | one named band                                          |
| `.visual`            | best available broadband magnitude                      |
| `.at_wavelength(nm)` | magnitude interpolated to a wavelength                  |
| `.pm_ra`, `.pm_dec`  | proper motion, mas/yr                                   |
| `.parallax`          | parallax, mas                                           |
| `.source_flags`      | raw provenance bitmask                                  |
| `.flags(i)`          | decoded flags for star `i`                              |
| `.records`           | the underlying structured array (`sstrc7.RECORD_DTYPE`) |
| `.to_table()`        | an `astropy.table.Table`                                |

### Reusing a catalog handle

Module-level `query_*` functions open a cached `Catalog` for you. To hold one explicitly — for a different path, or to keep several open:

```python
from sstrc7 import Catalog

catalog = Catalog("/data/sstrc7")
stars = catalog.query_cone(10.0, 41.0, 1.0)
```

Zone files are memory-mapped and cached, so repeated queries over the same part of the sky are served from the page cache.

## Command line

```
sstrc7 get       download whatever is missing
sstrc7 status    report what is present on disk
sstrc7 info      catalog size, release, bands, resolved data path
sstrc7 zones     the declination range of every zone file
```

## Binary format

The catalog is a set of zone files plus one index, and this package reads that format directly — nothing is converted or re-encoded on your disk.

```
sstrc.acc        864,000 bytes: 1800 × 60 pairs of little-endian uint32
s0000.cat        one file per declination zone, 60-byte records
...
s1799.cat
```

**Zones.** Declination is divided into 1800 zones of 0.1°, indexed by south polar distance, so zone _z_ covers `dec ∈ [z/10 − 90, (z+1)/10 − 90)`. Zone 900 is the strip just north of the equator. Within each zone, records are sorted by right ascension.

**Index.** `sstrc.acc` holds, for each of the 1800 declination zones and each of 60 six-degree-wide RA sub-zones, a `(record offset, record count)` pair. That lets a query seek straight to the records it needs instead of scanning a file. The summed counts reproduce every zone file's record count exactly.

**Record.** 60 bytes, little-endian, described by `sstrc7.RECORD_DTYPE`:

| Offset | Type         | Field          | Units                                       |
| -----: | ------------ | -------------- | ------------------------------------------- |
|      0 | `int32`      | `ra`           | milliarcseconds                             |
|      4 | `int32`      | `dec`          | milliarcseconds                             |
|      8 | `int16`      | `pm_ra`        | 0.32 mas/yr per count                       |
|     10 | `int16`      | `pm_dec`       | 0.32 mas/yr per count                       |
|     12 | `int16`      | `parallax`     | 0.032 mas per count                         |
|     14 | `int16 × 18` | `mag`          | millimagnitudes; 32000 means "not measured" |
|     50 | `uint16`     | reserved       |                                             |
|     52 | `uint16`     | `source_flags` | bitmask, see `sstrc7.SOURCE_FLAGS`          |
|     54 | `uint16 × 3` | reserved       |                                             |

The three reserved fields are preserved byte-for-byte but their meaning is not documented here.

Reading records yourself is a one-liner:

```python
import numpy as np, sstrc7
records = np.fromfile("/data/sstrc7/s0900.cat", dtype=sstrc7.RECORD_DTYPE)
```

## How the release is distributed

The data is published as [GitHub release assets](https://github.com/ssc-ai/sstrc7/releases), one asset per zone file plus the index — 1801 assets rather than a handful of multi-gigabyte archives. That means a failed transfer costs one ~4 MB file instead of the whole download, and `--dec-range` can fetch just the sky you care about.

GitHub allows at most 1000 assets on a release, so the files are spread across two, split by declination zone: `v1.0.0-zones-0000-0899` (which also carries the index) and `v1.0.0-zones-0900-1799`. The manifest records which release holds each file, so this is invisible to `get()`.

Each zone is stored as `.catz`: the record block is transposed to a column-major byte layout and then compressed with `lzma`. Grouping byte _i_ of every record together gives the compressor long runs of near-identical values across the 18 magnitude columns, which roughly halves the size — a representative equatorial zone goes from 10.40 MB to 5.15 MB, where plain gzip only reaches 7.24 MB. The transform is exactly invertible, and `get()` verifies the SHA-256 of both the compressed asset and the extracted file, so what lands on your disk is byte-identical to the source catalog.

Decoding uses `lzma` from the Python standard library plus numpy, so no third-party compression library is involved.

To regenerate the release assets from a local catalog — or to verify that the published ones match yours — see `tools/publish_release.py`.

## Provenance and citation

SSTRC7 is a merged reference catalog: every star comes from Gaia, selected to G < 18 and cross-matched against 2MASS, AllWISE, Tycho-Gaia, Hipparcos, Henry Draper, the Bright Star Catalog, and Landolt standards, as recorded per star in `source_flags`. The magnitude columns outside Gaia's own three bands are populated for the subsets listed in the coverage table above. The Gaia selection, the completeness below it, and the fidelity of the stored positions and magnitudes are all measured in [Depth and completeness](#depth-and-completeness) — they are not claims taken from documentation.

> **Note for downstream users:** this repository distributes the catalog data; it is not its origin. If you publish work using SSTRC7, cite the catalog's original source. The reference epoch of the positions is not recorded in the binary format — proper motions are provided so you can propagate positions once you have established the epoch your application needs.

## Development

```bash
git clone https://github.com/ssc-ai/sstrc7
cd sstrc7
pip install -e ".[dev]"
pytest
```

The test suite builds a small synthetic catalog on the fly and checks every query against a brute-force scan of the same data, so it runs in seconds and needs no downloaded files.

## License

The code in this repository is MIT licensed; see [LICENSE](LICENSE).
