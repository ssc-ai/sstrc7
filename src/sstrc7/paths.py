"""Where the catalog lives on disk."""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variables consulted in order when no path is given. All but the
#: first exist so an installation already configured for sdasim or satsim is
#: picked up without changing anything.
PATH_ENV_VARS: tuple[str, ...] = (
    "SSTRC7_PATH",
    "SDASIM_SSTRC7_PATH",
    "SDASIM_SSTR7_PATH",
    "SATSIM_SSTR7_PATH",
)

#: Used when nothing else is configured.
DEFAULT_PATH = Path.home() / ".sstrc7"


def catalog_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the catalog directory.

    Resolution order: the explicit argument, then each variable in
    :data:`PATH_ENV_VARS`, then :data:`DEFAULT_PATH`. The directory is not
    required to exist -- this only decides where to look or download to.

    Args:
        path: explicit directory, or None to use the environment.
    """
    if path is not None:
        return Path(path).expanduser()

    for var in PATH_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return Path(value).expanduser()

    return DEFAULT_PATH


def describe_path_source(path: str | os.PathLike[str] | None = None) -> str:
    """Explain which setting :func:`catalog_path` used, for error messages."""
    if path is not None:
        return "explicit path argument"
    for var in PATH_ENV_VARS:
        if os.environ.get(var):
            return f"${var}"
    return "default location"
