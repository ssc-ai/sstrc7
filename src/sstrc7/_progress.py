"""Byte progress reporting for downloads."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol


class Bar(Protocol):
    def update(self, amount: int) -> None: ...


class _NullBar:
    """Used when the caller asked for no progress output."""

    def update(self, amount: int) -> None:
        pass


def human_bytes(size: float) -> str:
    """Format a byte count with a sensible unit."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024.0
    return f"{size:.1f} TB"


@contextmanager
def progress_bar(total: int, label: str, enabled: bool = True) -> Iterator[Bar]:
    """Yield a progress bar over ``total`` bytes."""
    if not enabled:
        yield _NullBar()
        return

    from tqdm import tqdm

    with tqdm(total=total, desc=label, unit="B", unit_scale=True, unit_divisor=1024) as bar:
        yield bar
