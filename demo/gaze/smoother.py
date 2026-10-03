"""
Gaze coordinate smoother using Exponential Moving Average (EMA)
and optional multi-frame sliding-window averaging.
"""

from __future__ import annotations

import collections
from typing import Tuple


class GazeSmoother:
    """Smooth raw gaze coordinates to eliminate jitter.

    Primary method: Exponential Moving Average (EMA).
    Fallback / complementary: sliding-window average over the last *window_size* frames.
    """

    def __init__(self, ema_alpha: float = 0.3, window_size: int = 5):
        """
        Parameters
        ----------
        ema_alpha : float
            Blend factor for new samples.  0 → infinite smoothing (laggy),
            1 → no smoothing (raw / jittery).  0.3 is a good default for
            ~30 fps webcam input.
        window_size : int
            Number of recent frames kept for the sliding-window average.
            Only used when :meth:`smooth_windowed` is called explicitly.
        """
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        self.alpha = ema_alpha
        self._prev: tuple[float, float] | None = None

        self._window_size = max(1, window_size)
        self._history: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=self._window_size,
        )

    # ------------------------------------------------------------------
    # Primary: EMA
    # ------------------------------------------------------------------
    def smooth(self, x: float, y: float) -> Tuple[float, float]:
        """Apply EMA smoothing and return the filtered (x, y)."""
        if self._prev is None:
            self._prev = (x, y)
            self._history.append((x, y))
            return (x, y)

        sx = self.alpha * x + (1.0 - self.alpha) * self._prev[0]
        sy = self.alpha * y + (1.0 - self.alpha) * self._prev[1]
        self._prev = (sx, sy)
        self._history.append((sx, sy))
        return (sx, sy)

    # ------------------------------------------------------------------
    # Fallback: sliding-window average
    # ------------------------------------------------------------------
    def smooth_windowed(self, x: float, y: float) -> Tuple[float, float]:
        """Sliding-window average over the last *window_size* frames."""
        self._history.append((x, y))
        avg_x = sum(p[0] for p in self._history) / len(self._history)
        avg_y = sum(p[1] for p in self._history) / len(self._history)
        return (avg_x, avg_y)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all state (e.g. between calibration runs)."""
        self._prev = None
        self._history.clear()
