"""Orientation-aware convex drone footprint and swept collision checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def angle_delta(target: float, current: float) -> float:
    return wrap_angle(float(target) - float(current))


@dataclass(frozen=True)
class Pose2:
    x: float
    y: float
    theta: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


class OrientedFootprint:
    def __init__(self, vertices_local: Sequence[Sequence[float]], margin: float = 0.0):
        vertices = np.asarray(vertices_local, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("vertices_local must have shape (N,2), N>=3")
        self.vertices = vertices
        self.margin = float(margin)
        self.radius = float(np.max(np.linalg.norm(vertices, axis=1))) + self.margin

    def transformed(self, pose: Pose2) -> np.ndarray:
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        rotation = np.array([[c, s], [-s, c]])
        return self.vertices @ rotation + pose.position

    def projection_width(self, theta: float, normal: Sequence[float], *, extra_margin: float = 0.0) -> float:
        n = np.asarray(normal, dtype=float)
        n /= max(1e-12, np.linalg.norm(n))
        points = self.transformed(Pose2(0.0, 0.0, theta))
        projection = points @ n
        return float(np.max(projection) - np.min(projection) + 2.0 * (self.margin + extra_margin))

    def best_corridor_heading(self, normal: Sequence[float], bins: int = 32) -> tuple[float, float]:
        headings = np.linspace(-math.pi, math.pi, bins, endpoint=False)
        widths = [self.projection_width(theta, normal) for theta in headings]
        index = int(np.argmin(widths))
        return float(headings[index]), float(widths[index])

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        sign = None
        for i in range(len(polygon)):
            edge = polygon[(i + 1) % len(polygon)] - polygon[i]
            rel = point - polygon[i]
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if abs(cross) <= 1e-9:
                continue
            current = cross > 0.0
            if sign is None:
                sign = current
            elif current != sign:
                return False
        return True

    def collides_grid(self, pose: Pose2, occupied: np.ndarray, origin: Sequence[float], resolution: float) -> bool:
        polygon = self.transformed(pose)
        origin = np.asarray(origin, dtype=float)
        h, w = occupied.shape
        minimum = np.floor((np.min(polygon, axis=0) - origin) / resolution).astype(int)
        maximum = np.ceil((np.max(polygon, axis=0) - origin) / resolution).astype(int)
        for gy in range(minimum[1], maximum[1] + 1):
            for gx in range(minimum[0], maximum[0] + 1):
                if gx < 0 or gy < 0 or gx >= w or gy >= h:
                    return True
                if not occupied[gy, gx]:
                    continue
                center = origin + resolution * (np.array([gx, gy]) + 0.5)
                half = 0.5 * resolution + self.margin
                # Conservative polygon/cell overlap: center, polygon vertices,
                # and bounding-box separation cover the small planning cells.
                if self._point_in_polygon(center, polygon):
                    return True
                if np.any(np.all(np.abs(polygon - center) <= half, axis=1)):
                    return True
                if not (np.max(polygon[:, 0]) < center[0] - half or
                        np.min(polygon[:, 0]) > center[0] + half or
                        np.max(polygon[:, 1]) < center[1] - half or
                        np.min(polygon[:, 1]) > center[1] + half):
                    # Edge/cell intersection, sampled at sub-cell spacing.
                    for i in range(len(polygon)):
                        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
                        count = max(2, int(math.ceil(np.linalg.norm(b - a) / max(1.0, resolution * 0.4))))
                        samples = a + np.linspace(0, 1, count)[:, None] * (b - a)
                        if np.any(np.all(np.abs(samples - center) <= half, axis=1)):
                            return True
        return False

    def swept_collides(
        self, start: Pose2, end: Pose2, occupied: np.ndarray,
        origin: Sequence[float], resolution: float,
        *, translation_step: float | None = None, angle_step: float = math.radians(4),
    ) -> bool:
        distance = float(np.linalg.norm(end.position - start.position))
        rotation = abs(angle_delta(end.theta, start.theta))
        translation_step = translation_step or max(1.0, resolution * 0.4)
        count = max(1, int(math.ceil(distance / translation_step)), int(math.ceil(rotation / angle_step)))
        delta_theta = angle_delta(end.theta, start.theta)
        for alpha in np.linspace(0.0, 1.0, count + 1):
            position = start.position + alpha * (end.position - start.position)
            pose = Pose2(float(position[0]), float(position[1]), wrap_angle(start.theta + alpha * delta_theta))
            if self.collides_grid(pose, occupied, origin, resolution):
                return True
        return False

