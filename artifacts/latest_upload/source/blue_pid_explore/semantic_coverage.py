"""Semantic-ray coverage, intentionally separate from LiDAR occupancy."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .occupancy_map import OccupancyMap, bresenham


class SemanticCoverageMap:
    def __init__(self, occupancy: OccupancyMap):
        self.occupancy = occupancy
        self.visits = np.zeros((occupancy.height, occupancy.width), dtype=np.uint16)

    def update(self, position, heading: float, ray_angles, *, max_range: float = 130.0,
               blocking_cells: np.ndarray | None = None) -> None:
        start = self.occupancy.world_to_grid(position)
        for angle in np.asarray(ray_angles, dtype=float).reshape(-1):
            endpoint = np.asarray(position, dtype=float) + max_range * np.array([
                math.cos(heading + angle), math.sin(heading + angle)])
            end = self.occupancy.world_to_grid(endpoint)
            for x, y in bresenham(*start, *end):
                if not self.occupancy.inside(x, y):
                    break
                self.visits[y, x] = min(np.iinfo(np.uint16).max, int(self.visits[y, x]) + 1)
                if blocking_cells is not None and blocking_cells[y, x] > 0:
                    break

    @property
    def covered(self) -> np.ndarray:
        return self.visits > 0


