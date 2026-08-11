"""SSTRC7: a 294-million-star all-sky catalog, packaged for easy use.

Typical use is two lines -- fetch the data once, then query it::

    import sstrc7

    sstrc7.get()                                    # no-op if already present
    stars = sstrc7.query_cone(83.82, -5.39, 0.5)    # degrees
    print(len(stars), stars.band("Johnson_V"))

The catalog is 17.6 GB extracted (7.2 GB to download). :func:`get` verifies
what is already on disk and downloads only what is missing, so calling it on
every run is cheap.
"""

from __future__ import annotations

import os
from functools import lru_cache

from ._format import (
    BAND_CENTERS_NM,
    BAND_INDEX,
    BAND_NAMES,
    N_DEC_ZONES,
    N_RA_ZONES,
    RECORD_DTYPE,
    RECORD_SIZE,
    SOURCE_FLAGS,
    decode_source_flags,
    zone_dec_range,
    zone_filename,
    zones_for_dec_range,
)
from .fetch import CatalogStatus, DownloadError, get, status
from .manifest import Manifest
from .manifest import load as load_manifest
from .paths import DEFAULT_PATH, PATH_ENV_VARS, catalog_path
from .query import Catalog, CatalogNotFound, StarField

__version__ = "1.0.1"

__all__ = [
    "BAND_CENTERS_NM",
    "BAND_INDEX",
    "BAND_NAMES",
    "Catalog",
    "CatalogNotFound",
    "CatalogStatus",
    "DEFAULT_PATH",
    "DownloadError",
    "Manifest",
    "N_DEC_ZONES",
    "N_RA_ZONES",
    "PATH_ENV_VARS",
    "RECORD_DTYPE",
    "RECORD_SIZE",
    "SOURCE_FLAGS",
    "StarField",
    "__version__",
    "catalog_path",
    "decode_source_flags",
    "get",
    "load_manifest",
    "open_catalog",
    "query_box",
    "query_by_los",
    "query_cone",
    "status",
    "zone_dec_range",
    "zone_filename",
    "zones_for_dec_range",
]


@lru_cache(maxsize=4)
def _cached_catalog(resolved: str) -> Catalog:
    return Catalog(resolved)


def open_catalog(path: str | os.PathLike[str] | None = None) -> Catalog:
    """Open a catalog directory, reusing an already-open one when possible.

    Args:
        path: catalog directory; None resolves from the environment.

    Returns:
        A :class:`~sstrc7.query.Catalog`.
    """
    return _cached_catalog(str(catalog_path(path)))


def query_cone(
    ra: float,
    dec: float,
    radius: float,
    *,
    path: str | os.PathLike[str] | None = None,
    radians: bool = False,
) -> StarField:
    """Select stars within ``radius`` of ``(ra, dec)``. Degrees unless ``radians``."""
    return open_catalog(path).query_cone(ra, dec, radius, radians=radians)


def query_box(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    *,
    path: str | os.PathLike[str] | None = None,
    radians: bool = False,
) -> StarField:
    """Select stars inside an RA/Dec rectangle. ``ra_min > ra_max`` wraps through 0."""
    return open_catalog(path).query_box(
        ra_min, ra_max, dec_min, dec_max, radians=radians
    )


def query_by_los(
    height: int,
    width: int,
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rot: float = 0.0,
    *,
    path: str | os.PathLike[str] | None = None,
    pad_mult: float = 0.0,
    origin: str = "center",
    filter_ob: bool = True,
    filter_center: float | None = None,
):
    """Project stars onto a focal plane. Returns ``(rows, cols, magnitudes)``."""
    return open_catalog(path).query_by_los(
        height,
        width,
        y_fov,
        x_fov,
        ra,
        dec,
        rot,
        pad_mult=pad_mult,
        origin=origin,
        filter_ob=filter_ob,
        filter_center=filter_center,
    )
