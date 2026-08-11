"""Reading and querying a local SSTRC7 catalog.

Records are read through numpy memory maps and a structured dtype, so a query
touches only the pages holding the zones it needs and never builds a Python
object per star. That matters at this scale: the catalog holds 294 million
records, and a wide-field query can select millions of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._format import (
    BAND_INDEX,
    BAND_NAMES,
    INDEX_FILENAME,
    INDEX_SIZE,
    MAS_PER_DEG,
    N_DEC_ZONES,
    N_RA_ZONES,
    PARALLAX_SCALE_MAS,
    PM_SCALE_MAS_PER_YEAR,
    RECORD_DTYPE,
    ZONE_WIDTH_DEG,
    decode_source_flags,
    interpolate_magnitude,
    magnitudes,
    visual_magnitude,
    zone_filename,
    zones_for_dec_range,
)
from .paths import catalog_path, describe_path_source


class CatalogNotFound(FileNotFoundError):
    """The catalog directory is missing or lacks its index file."""


# --- Results ---------------------------------------------------------------


@dataclass
class StarField:
    """Stars returned by a query.

    Holds the raw records; the derived quantities below are computed on access
    and cached, so pulling only the columns you need stays cheap.
    """

    records: np.ndarray

    def __len__(self) -> int:
        return int(self.records.size)

    def __repr__(self) -> str:
        return f"<StarField: {len(self)} stars>"

    @property
    def ra(self) -> np.ndarray:
        """Right ascension in degrees, [0, 360)."""
        return self.records["ra"].astype(np.float64) / MAS_PER_DEG

    @property
    def dec(self) -> np.ndarray:
        """Declination in degrees, [-90, 90]."""
        return self.records["dec"].astype(np.float64) / MAS_PER_DEG

    @property
    def ra_rad(self) -> np.ndarray:
        """Right ascension in radians."""
        return np.radians(self.ra)

    @property
    def dec_rad(self) -> np.ndarray:
        """Declination in radians."""
        return np.radians(self.dec)

    @property
    def pm_ra(self) -> np.ndarray:
        """Proper motion in right ascension, mas/yr (coordinate, not great-circle)."""
        return self.records["pm_ra"].astype(np.float64) * PM_SCALE_MAS_PER_YEAR

    @property
    def pm_dec(self) -> np.ndarray:
        """Proper motion in declination, mas/yr."""
        return self.records["pm_dec"].astype(np.float64) * PM_SCALE_MAS_PER_YEAR

    @property
    def parallax(self) -> np.ndarray:
        """Parallax in mas."""
        return self.records["parallax"].astype(np.float64) * PARALLAX_SCALE_MAS

    @property
    def source_flags(self) -> np.ndarray:
        """Raw provenance bitmask; see :func:`sstrc7.decode_source_flags`."""
        return self.records["source_flags"]

    @property
    def mag(self) -> np.ndarray:
        """``(N, 18)`` magnitudes in :data:`sstrc7.BAND_NAMES` order, NaN where absent."""
        cached = getattr(self, "_mag", None)
        if cached is None:
            cached = magnitudes(self.records)
            self._mag = cached
        return cached

    def band(self, name: str) -> np.ndarray:
        """Magnitudes in one named band, NaN where the star has no measurement."""
        try:
            column = BAND_INDEX[name]
        except KeyError:
            raise KeyError(f"unknown band {name!r}; expected one of {', '.join(BAND_NAMES)}") from None
        return self.mag[:, column]

    @property
    def visual(self) -> np.ndarray:
        """Best available broadband magnitude per star (see VISUAL_BAND_PRIORITY)."""
        return visual_magnitude(self.mag)

    def at_wavelength(self, wavelength_nm: float) -> np.ndarray:
        """Magnitude interpolated across each star's measured bands."""
        return interpolate_magnitude(self.mag, wavelength_nm)

    def flags(self, index: int) -> list[str]:
        """Human-readable provenance flags for one star."""
        return decode_source_flags(int(self.records["source_flags"][index]))

    def to_table(self):
        """Return an :class:`astropy.table.Table` of the common columns."""
        from astropy.table import Table

        table = Table()
        table["ra"] = self.ra
        table["dec"] = self.dec
        table["pm_ra"] = self.pm_ra
        table["pm_dec"] = self.pm_dec
        table["parallax"] = self.parallax
        for i, name in enumerate(BAND_NAMES):
            table[name] = self.mag[:, i]
        table["source_flags"] = self.source_flags
        return table


EMPTY = np.empty(0, dtype=RECORD_DTYPE)


# --- Catalog ---------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_index(path: str, mtime: float) -> np.ndarray:
    """Load ``sstrc.acc`` as an ``(1800, 60, 2)`` array of (offset, count)."""
    size = os.path.getsize(path)
    if size != INDEX_SIZE:
        raise CatalogNotFound(f"{path} is {size} bytes, expected {INDEX_SIZE}")
    return np.fromfile(path, dtype="<u4").reshape(N_DEC_ZONES, N_RA_ZONES, 2)


@lru_cache(maxsize=2048)
def _zone_records(path: str, size: int) -> np.ndarray:
    """Memory-map one zone file as an array of :data:`RECORD_DTYPE`."""
    return np.memmap(path, dtype=RECORD_DTYPE, mode="r")


class Catalog:
    """A local SSTRC7 catalog directory.

    Args:
        path: catalog directory; None resolves from the environment
            (``$SSTRC7_PATH``, then ``~/.sstrc7``).

    Raises:
        CatalogNotFound: if the directory has no usable index file.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = catalog_path(path)
        index_file = self.path / INDEX_FILENAME
        if not index_file.exists():
            raise CatalogNotFound(
                f"no SSTRC7 catalog at {self.path} (from {describe_path_source(path)}): "
                f"{INDEX_FILENAME} is missing. Run `sstrc7 get` or sstrc7.get() to download it."
            )
        self._index_file = index_file

    def __repr__(self) -> str:
        return f"<Catalog {self.path}>"

    @property
    def index(self) -> np.ndarray:
        """The zone index as ``(1800, 60, 2)`` of (record offset, record count)."""
        return _load_index(str(self._index_file), self._index_file.stat().st_mtime)

    def _zone(self, zone_id: int) -> np.ndarray:
        file = self.path / zone_filename(zone_id)
        try:
            return _zone_records(str(file), file.stat().st_size)
        except OSError as exc:
            raise CatalogNotFound(
                f"zone file {file} is missing or unreadable; run sstrc7.get() to repair"
            ) from exc

    # -- Queries ------------------------------------------------------------

    def query_box(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        *,
        radians: bool = False,
    ) -> StarField:
        """Select stars inside a right ascension / declination rectangle.

        ``ra_min > ra_max`` selects the wrapped interval through 0 degrees.

        Args:
            ra_min, ra_max: right ascension bounds.
            dec_min, dec_max: declination bounds.
            radians: inputs are radians rather than degrees.

        Returns:
            A :class:`StarField`.
        """
        if radians:
            ra_min, ra_max = np.degrees(ra_min), np.degrees(ra_max)
            dec_min, dec_max = np.degrees(dec_min), np.degrees(dec_max)

        if dec_min > dec_max:
            dec_min, dec_max = dec_max, dec_min

        if ra_max - ra_min >= 360.0:
            intervals = [(0.0, 360.0)]
        else:
            ra_min %= 360.0
            ra_max %= 360.0
            intervals = (
                [(ra_min, ra_max)] if ra_min <= ra_max else [(ra_min, 360.0), (0.0, ra_max)]
            )

        dec_lo = int(np.floor(dec_min * MAS_PER_DEG))
        dec_hi = int(np.ceil(dec_max * MAS_PER_DEG))

        blocks: list[np.ndarray] = []
        for zone_id in zones_for_dec_range(dec_min, dec_max):
            zone_index = self.index[zone_id]
            records = None
            for lo, hi in intervals:
                first = max(int(lo / ZONE_WIDTH_DEG), 0)
                last = min(int(hi / ZONE_WIDTH_DEG), N_RA_ZONES - 1)
                start = int(zone_index[first, 0])
                stop = int(zone_index[last, 0] + zone_index[last, 1])
                if stop <= start:
                    continue
                if records is None:
                    records = self._zone(zone_id)
                block = records[start:stop]

                keep = (block["ra"] >= lo * MAS_PER_DEG) & (block["ra"] <= hi * MAS_PER_DEG)
                keep &= (block["dec"] >= dec_lo) & (block["dec"] <= dec_hi)
                if keep.any():
                    blocks.append(np.asarray(block[keep]))

        return StarField(np.concatenate(blocks) if blocks else EMPTY.copy())

    def query_cone(
        self,
        ra: float,
        dec: float,
        radius: float,
        *,
        radians: bool = False,
    ) -> StarField:
        """Select stars within an angular radius of a point.

        Args:
            ra, dec: cone centre.
            radius: cone radius.
            radians: inputs are radians rather than degrees.

        Returns:
            A :class:`StarField`, exactly clipped to the cone.
        """
        if radians:
            ra, dec, radius = np.degrees(ra), np.degrees(dec), np.degrees(radius)

        dec_min = dec - radius
        dec_max = dec + radius

        if dec_min <= -90.0 or dec_max >= 90.0:
            ra_min, ra_max = 0.0, 360.0
        else:
            # Widen in RA by 1/cos(dec) at whichever bound is closer to a pole.
            worst = max(abs(dec_min), abs(dec_max))
            half_width = min(np.degrees(np.arcsin(np.sin(np.radians(radius)) / np.cos(np.radians(worst)))), 180.0)
            ra_min, ra_max = ra - half_width, ra + half_width

        field = self.query_box(
            ra_min,
            ra_max,
            max(dec_min, -90.0),
            min(dec_max, 90.0),
        )
        if len(field) == 0:
            return field

        centre = _unit_vector(np.array([ra]), np.array([dec]))
        stars = _unit_vector(field.ra, field.dec)
        cos_sep = stars @ centre[0]
        keep = cos_sep >= np.cos(np.radians(radius))
        return StarField(field.records[keep])

    def query_by_los(
        self,
        height: int,
        width: int,
        y_fov: float,
        x_fov: float,
        ra: float,
        dec: float,
        rot: float = 0.0,
        *,
        pad_mult: float = 0.0,
        origin: str = "center",
        filter_ob: bool = True,
        filter_center: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project the catalog onto a focal plane for a given line of sight.

        Args:
            height, width: sensor size in pixels.
            y_fov, x_fov: field of view in degrees.
            ra, dec: boresight in degrees.
            rot: focal plane rotation in degrees.
            pad_mult: fraction of the sensor size to pad the search by, so stars
                just off the edge are still returned.
            origin: ``"center"`` puts the boresight at the array centre;
                ``"corner"`` puts it at pixel (0, 0).
            filter_ob: drop stars outside the padded field of view.
            filter_center: interpolate magnitudes to this wavelength in nm
                instead of using the best broadband magnitude.

        Returns:
            ``(rows, cols, magnitudes)`` as float arrays.
        """
        y_ifov = y_fov / height
        x_ifov = x_fov / width

        corner_min, corner_max, wcs = _field_bounds(
            height, width, y_ifov, x_ifov, ra, dec, rot, pad_mult, origin
        )

        field = self.query_box(
            corner_min[0], corner_max[0], corner_min[1], corner_max[1]
        )

        if len(field) == 0:
            empty = np.empty(0)
            return empty, empty.copy(), empty.copy()

        mag = field.at_wavelength(filter_center) if filter_center is not None else field.visual
        # Stars with no usable band keep the catalog's "no measurement" value.
        mag = np.where(np.isnan(mag), 32.0, mag).astype(np.float64)

        cols, rows = wcs.wcs_world2pix(field.ra, field.dec, 0)

        if filter_ob:
            pad_h = height * (1 + pad_mult)
            pad_w = width * (1 + pad_mult)
            inside = (
                (rows <= pad_h) & (rows >= -pad_h) & (cols <= pad_w) & (cols >= -pad_w)
            )
            rows, cols, mag = rows[inside], cols[inside], mag[inside]

        if origin == "center":
            rows = rows + height / 2.0
            cols = cols + width / 2.0

        return rows, cols, mag


def _unit_vector(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates in degrees to unit vectors, shape (N, 3)."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cos_dec = np.cos(dec)
    return np.stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)], axis=-1)


def _build_wcs(y_ifov: float, x_ifov: float, ra: float, dec: float, rot: float):
    # Imported here rather than at module scope: astropy is a required
    # dependency, but importing it costs about a second, and the cone and box
    # queries never need it.
    from astropy import wcs as astropy_wcs

    w = astropy_wcs.WCS(naxis=2)
    w.wcs.crpix = [1, 1]
    w.wcs.cdelt = np.array([x_ifov, y_ifov])
    w.wcs.crval = np.array([ra, dec])
    w.wcs.crota = [rot, rot]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _field_bounds(
    height: int,
    width: int,
    y_ifov: float,
    x_ifov: float,
    ra: float,
    dec: float,
    rot: float,
    pad_mult: float,
    origin: str,
):
    """Bounding box in RA/Dec of a padded focal plane, plus its WCS.

    Returns ``(corner_min, corner_max, wcs)`` in degrees. ``corner_min[0]`` may
    exceed ``corner_max[0]``, which means the box wraps through RA = 0.
    """
    w = _build_wcs(y_ifov, x_ifov, ra, dec, rot)

    pad_y = height * pad_mult
    pad_x = width * pad_mult
    pixels = np.array(
        [
            [-pad_x, -pad_y],
            [-pad_x, height * 0.5],
            [-pad_x, height + pad_y],
            [width * 0.5, height + pad_y],
            [width + pad_x, height + pad_y],
            [width + pad_x, height * 0.5],
            [width + pad_x, -pad_y],
            [width * 0.5, -pad_y],
        ],
        np.float64,
    )
    centre = np.array([[width / 2.0, height / 2.0]])

    if origin == "center":
        pixels[:, 0] -= width / 2.0
        pixels[:, 1] -= height / 2.0
        centre[:, 0] -= width / 2.0
        centre[:, 1] -= height / 2.0

    world = w.wcs_pix2world(pixels, 1)

    dec_min = float(np.min(world[:, 1]))
    dec_max = float(np.max(world[:, 1]))

    north = w.wcs_world2pix([[0, 89.99999]], 1)[0]
    south = w.wcs_world2pix([[0, -89.99999]], 1)[0]

    if not np.any(np.isnan(north)) and 0 < north[0] < width and 0 < north[1] < height:
        return [0.0, dec_min], [360.0, 90.0], w
    if not np.any(np.isnan(south)) and 0 < south[0] < width and 0 < south[1] < height:
        return [0.0, -90.0], [360.0, dec_max], w

    ra_min, ra_max = _enclosing_ra_arc(world[:, 0])
    return [ra_min, dec_min], [ra_max, dec_max], w


def _enclosing_ra_arc(ra_deg: np.ndarray) -> tuple[float, float]:
    """Smallest arc of right ascension containing every given angle.

    Taking min and max is wrong whenever a field straddles RA = 0: the extremes
    then sit on opposite sides of the seam and describe the arc the field does
    *not* occupy. Instead, find the widest empty gap between neighbouring
    angles; the field is everything else. The returned pair may wrap, meaning
    ``ra_min > ra_max``, which is what :meth:`Catalog.query_box` expects.
    """
    angles = np.sort(np.asarray(ra_deg, dtype=np.float64) % 360.0)
    if angles.size == 1:
        return float(angles[0]), float(angles[0])

    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    widest = int(np.argmax(gaps))
    return float(angles[(widest + 1) % angles.size]), float(angles[widest])
