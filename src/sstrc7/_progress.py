"""A byte progress bar that uses tqdm when it is installed, and otherwise not.

Keeping this behind a tiny interface means the download path has no required
third-party dependency beyond numpy.
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Protocol


class Bar(Protocol):
    def update(self, amount: int) -> None: ...


class _NullBar:
    def update(self, amount: int) -> None:
        pass


class _PlainBar:
    """Minimal stderr bar: one refresh per second, one final line."""

    def __init__(self, total: int, label: str) -> None:
        self._total = max(total, 1)
        self._label = label
        self._done = 0
        self._last = 0.0
        self._start = time.monotonic()
        self._lock = threading.Lock()
        self._draw(force=True)

    def update(self, amount: int) -> None:
        with self._lock:
            self._done += amount
            self._draw()

    def _draw(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < 1.0:
            return
        self._last = now
        pct = 100.0 * self._done / self._total
        elapsed = now - self._start
        rate = self._done / elapsed / 1e6 if elapsed > 0 else 0.0
        sys.stderr.write(
            f"\r{self._label}: {pct:5.1f}%  "
            f"{human_bytes(self._done)} / {human_bytes(self._total)}  {rate:.1f} MB/s"
        )
        sys.stderr.flush()

    def close(self) -> None:
        with self._lock:
            self._draw(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()


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

    try:
        from tqdm import tqdm
    except ImportError:
        bar = _PlainBar(total, label)
        try:
            yield bar
        finally:
            bar.close()
        return

    with tqdm(total=total, desc=label, unit="B", unit_scale=True, unit_divisor=1024) as bar:
        yield bar
