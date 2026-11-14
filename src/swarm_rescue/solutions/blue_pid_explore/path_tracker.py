"""Continuous lookahead tracking that outputs velocity, never final thrust."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PathTrackOutput:
    target_velocity: np.ndarray
    lookahead_point: np.ndarray
    path_index: int
    reached: bool


class LookaheadPathTracker:
    def __init__(self, *, base_lookahead: float = 18.0, speed_gain: float = 2.0,
                 min_lookahead: float = 12.0, max_lookahead: float = 55.0,
                 max_speed: float = 8.0):
        self.base_lookahead = float(base_lookahead); self.speed_gain = float(speed_gain)
        self.min_lookahead = float(min_lookahead); self.max_lookahead = float(max_lookahead)
        self.max_speed = float(max_speed); self.index = 0

    def reset(self) -> None:
        self.index = 0

    def compute(self, position: Sequence[float], velocity: Sequence[float], path: Sequence[Sequence[float]]) -> PathTrackOutput:
        p = np.asarray(position, dtype=float); v = np.asarray(velocity, dtype=float)
        points = np.asarray(path, dtype=float)
        if len(points) == 0:
            return PathTrackOutput(np.zeros(2), p.copy(), self.index, True)
        distances = np.linalg.norm(points - p, axis=1)
        self.index = max(self.index, int(np.argmin(distances)))
        lookahead = float(np.clip(self.base_lookahead + self.speed_gain * np.linalg.norm(v),
                                  self.min_lookahead, self.max_lookahead))
        chosen = self.index
        accumulated = 0.0
        for i in range(self.index, len(points) - 1):
            accumulated += float(np.linalg.norm(points[i + 1] - points[i]))
            chosen = i + 1
            if accumulated >= lookahead:
                break
        target = points[chosen]
        direction = target - p; distance = float(np.linalg.norm(direction))
        reached = chosen == len(points) - 1 and distance <= 5.0
        if distance <= 1e-9 or reached:
            desired = np.zeros(2)
        else:
            # Slow for a sharp upcoming corner and for the terminal point.
            speed = self.max_speed
            if chosen < len(points) - 1:
                a = target - p; b = points[chosen + 1] - target
                cosine = float(np.dot(a, b) / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b)))
                speed *= float(np.clip(0.45 + 0.55 * (cosine + 1.0) * 0.5, 0.4, 1.0))
            else:
                speed = min(speed, max(1.0, 0.18 * distance))
            desired = speed * direction / distance
        self.index = chosen
        return PathTrackOutput(desired, target.copy(), chosen, reached)

