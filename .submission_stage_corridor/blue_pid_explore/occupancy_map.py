"""Legal-pose LiDAR log-odds occupancy grid with explicit unknown/free/occupied."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Sequence

import numpy as np


class Cell(IntEnum):
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1
    INFLATED = 2


def bresenham(x0: int, y0: int, x1: int, y1: int):
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy; x0 += sx
        if twice <= dx:
            error += dx; y0 += sy


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    resolution: float
    origin: np.ndarray


class OccupancyMap:
    def __init__(self, size_world: Sequence[float], *, resolution: float = 8.0,
                 origin: Sequence[float] | None = None):
        size = np.asarray(size_world, dtype=float)
        self.resolution = float(resolution)
        self.width = int(math.ceil(size[0] / self.resolution))
        self.height = int(math.ceil(size[1] / self.resolution))
        self.origin = np.asarray(origin if origin is not None else -0.5 * size, dtype=float)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.observed = np.zeros((self.height, self.width), dtype=bool)
        self.free_update = -0.45
        self.hit_update = 0.9
        self.minimum, self.maximum = -4.0, 4.0
        self._cell_cache: dict[tuple[float, float, float], np.ndarray] = {}

    @property
    def spec(self) -> GridSpec:
        return GridSpec(self.width, self.height, self.resolution, self.origin.copy())

    def world_to_grid(self, point: Sequence[float]) -> tuple[int, int]:
        cell = np.floor((np.asarray(point, dtype=float) - self.origin) / self.resolution).astype(int)
        return int(cell[0]), int(cell[1])

    def grid_to_world(self, cell: Sequence[int]) -> np.ndarray:
        return self.origin + self.resolution * (np.asarray(cell, dtype=float) + 0.5)

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def update_scan(self, position, heading: float, distances, angles, *, max_range: float = 230.0) -> None:
        if distances is None:
            return
        self._cell_cache.clear()
        start = self.world_to_grid(position)
        d = np.asarray(distances, dtype=float).reshape(-1)
        a = np.asarray(angles, dtype=float).reshape(-1)
        for distance, angle in zip(d, a):
            if not np.isfinite(distance) or distance <= 0.0:
                continue
            hit = distance < max_range - 10.0
            ray_length = min(float(distance), max_range)
            endpoint = np.asarray(position, dtype=float) + ray_length * np.array([
                math.cos(heading + angle), math.sin(heading + angle)])
            end = self.world_to_grid(endpoint)
            cells = list(bresenham(*start, *end))
            free_cells = cells[:-1] if hit else cells
            for x, y in free_cells:
                if self.inside(x, y):
                    self.log_odds[y, x] = max(self.minimum, self.log_odds[y, x] + self.free_update)
                    self.observed[y, x] = True
            if hit and cells:
                x, y = cells[-1]
                if self.inside(x, y):
                    self.log_odds[y, x] = min(self.maximum, self.log_odds[y, x] + self.hit_update)
                    self.observed[y, x] = True

    def cells(self, *, occupied_threshold: float = 0.8, free_threshold: float = -0.4,
              inflation_radius: float = 0.0) -> np.ndarray:
        key = (float(occupied_threshold), float(free_threshold), float(inflation_radius))
        cached = self._cell_cache.get(key)
        if cached is not None:
            return cached
        result = np.full(self.log_odds.shape, Cell.UNKNOWN, dtype=np.int8)
        result[self.observed & (self.log_odds <= free_threshold)] = Cell.FREE
        occupied = self.observed & (self.log_odds >= occupied_threshold)
        result[occupied] = Cell.OCCUPIED
        radius = int(math.ceil(inflation_radius / self.resolution))
        if radius > 0:
            ys, xs = np.nonzero(occupied)
            for y, x in zip(ys, xs):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx * dx + dy * dy > radius * radius:
                            continue
                        nx, ny = x + dx, y + dy
                        if self.inside(nx, ny) and result[ny, nx] != Cell.OCCUPIED:
                            result[ny, nx] = Cell.INFLATED
        # Callers treat grids as immutable snapshots.  Caching is important:
        # frontier selection, planning and semantic occlusion otherwise each
        # rebuild and inflate the full map in the same control frame.
        self._cell_cache[key] = result
        return result


