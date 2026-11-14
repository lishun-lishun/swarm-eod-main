"""Robust LiDAR point geometry for predictive collision constraints.

The official LiDAR measures from the drone body origin.  Consequently the
vehicle footprint is deducted exactly once, after a wall normal has been
estimated; individual ray directions are never treated as wall normals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LidarBoundary:
    normal: np.ndarray
    clearance: float
    surface_distance: float
    distance_sigma: float
    geometry_error: float
    point_count: int
    tangent: np.ndarray
    centroid: np.ndarray
    dynamic_velocity: np.ndarray | None = None


def dynamic_drone_boundary(
    own_position: Sequence[float],
    other_position: Sequence[float],
    other_velocity: Sequence[float],
    own_footprint: Sequence[Sequence[float]],
    other_footprint: Sequence[Sequence[float]],
    own_heading: float,
    other_heading: float,
    *,
    geometry_error: float = 2.0,
) -> LidarBoundary:
    """Construct a relative-motion boundary from communicated legal estimates."""
    relative = np.asarray(other_position, dtype=float) - np.asarray(own_position, dtype=float)
    distance = float(np.linalg.norm(relative))
    if distance <= 1e-9:
        normal = np.array([1.0, 0.0])
    else:
        normal = relative / distance
    clearance = (
        distance
        - footprint_support(own_footprint, normal, own_heading)
        - footprint_support(other_footprint, -normal, other_heading)
    )
    tangent = np.array([-normal[1], normal[0]])
    return LidarBoundary(
        normal, clearance, distance, 0.0, float(geometry_error), 1,
        tangent, np.asarray(other_position, dtype=float),
        np.asarray(other_velocity, dtype=float),
    )


def footprint_vertices_from_body(body) -> np.ndarray:
    """Return the actual local Pymunk convex-hull vertices of a drone body."""
    vertices: list[tuple[float, float]] = []
    for shape in getattr(body, "_pm_shapes", ()):  # official DroneBase
        getter = getattr(shape, "get_vertices", None)
        if getter is None:
            continue
        vertices.extend((float(v.x), float(v.y)) for v in getter())
    if not vertices:
        raise ValueError("body exposes no polygonal Pymunk collision hull")
    return np.asarray(vertices, dtype=float)


def footprint_support(vertices_local: Sequence[Sequence[float]], normal_world, heading: float) -> float:
    vertices = np.asarray(vertices_local, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("footprint vertices must have shape (N, 2), N >= 3")
    n = np.asarray(normal_world, dtype=float)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        raise ValueError("normal must be non-zero")
    n /= n_norm
    c, s = math.cos(float(heading)), math.sin(float(heading))
    world = vertices @ np.array([[c, s], [-s, c]])
    return float(np.max(world @ n))


class LidarGeometryEstimator:
    """Cluster adjacent returns and fit local wall lines with temporal filtering."""

    def __init__(
        self,
        *,
        max_range: float = 230.0,
        min_cluster_points: int = 3,
        cluster_base_gap: float = 7.0,
        cluster_angular_scale: float = 2.5,
        max_fit_rms: float = 7.5,
        no_return_guard_sigma: float = 4.0,
        history_length: int = 5,
        release_alpha: float = 0.25,
        match_angle_deg: float = 18.0,
    ):
        self.max_range = float(max_range)
        self.min_cluster_points = int(min_cluster_points)
        self.cluster_base_gap = float(cluster_base_gap)
        self.cluster_angular_scale = float(cluster_angular_scale)
        self.max_fit_rms = float(max_fit_rms)
        self.no_return_guard = float(no_return_guard_sigma) * 2.5
        self.history_length = int(history_length)
        self.release_alpha = float(release_alpha)
        self.match_cos = math.cos(math.radians(match_angle_deg))
        self._tracks: list[dict] = []

    def reset(self) -> None:
        self._tracks.clear()

    def points_from_scan(self, distances, angles, position, heading: float) -> tuple[np.ndarray, np.ndarray]:
        if distances is None:
            return np.empty((0, 2)), np.empty(0, dtype=int)
        d = np.asarray(distances, dtype=float).reshape(-1)
        a = np.asarray(angles, dtype=float).reshape(-1)
        if d.shape != a.shape:
            raise ValueError("LiDAR distances and angles must have equal length")
        # The engine adds Gaussian noise after ray casting.  A max-range miss
        # can therefore become 227 px and must not turn into a phantom wall.
        valid = np.isfinite(d) & (d > 0.0) & (d < self.max_range - self.no_return_guard)
        idx = np.flatnonzero(valid)
        if not len(idx):
            return np.empty((0, 2)), idx
        world_angles = a[idx] + float(heading)
        origin = np.asarray(position, dtype=float)
        points = origin + d[idx, None] * np.column_stack((np.cos(world_angles), np.sin(world_angles)))
        return points, idx

    def _clusters(self, points: np.ndarray, indices: np.ndarray, ray_count: int,
                  origin: np.ndarray) -> list[np.ndarray]:
        if len(points) == 0:
            return []
        groups: list[list[int]] = [[0]]
        for k in range(1, len(points)):
            consecutive = indices[k] == indices[k - 1] + 1
            radius = max(np.linalg.norm(points[k] - origin), np.linalg.norm(points[k - 1] - origin))
            allowed = self.cluster_base_gap + self.cluster_angular_scale * radius * (2.0 * math.pi / max(1, ray_count))
            if consecutive and np.linalg.norm(points[k] - points[k - 1]) <= allowed:
                groups[-1].append(k)
            else:
                groups.append([k])
        # Join the +/-pi seam when both end rays are valid and geometrically adjacent.
        if len(groups) > 1 and indices[0] == 0 and indices[-1] == ray_count - 1:
            if np.linalg.norm(points[0] - points[-1]) <= self.cluster_base_gap * 2.0:
                groups[0] = groups[-1] + groups[0]
                groups.pop()
        return [points[g] for g in groups if len(g) >= self.min_cluster_points]

    @staticmethod
    def _fit_line(cluster: np.ndarray, origin: np.ndarray):
        centroid = np.mean(cluster, axis=0)
        centered = cluster - centroid
        covariance = centered.T @ centered / max(1, len(cluster))
        values, vectors = np.linalg.eigh(covariance)
        tangent = vectors[:, int(np.argmax(values))]
        normal = np.array([-tangent[1], tangent[0]])
        if np.dot(normal, centroid - origin) < 0.0:
            normal = -normal
        signed = centered @ normal
        rms = float(np.sqrt(np.mean(signed**2)))
        # Median line offset avoids the severe negative bias of taking the
        # minimum across dozens of independent noisy rays.
        surface = float(np.median((cluster - origin) @ normal))
        return centroid, tangent, normal, surface, rms

    def _split_cluster(self, cluster: np.ndarray) -> list[np.ndarray]:
        """Split connected L/corner returns before fitting individual walls."""
        if len(cluster) < 2 * self.min_cluster_points:
            return [cluster]
        chord = cluster[-1] - cluster[0]
        length = float(np.linalg.norm(chord))
        if length <= 1e-9:
            return [cluster]
        chord /= length
        normal = np.array([-chord[1], chord[0]])
        deviations = np.abs((cluster - cluster[0]) @ normal)
        index = int(np.argmax(deviations))
        if deviations[index] <= self.max_fit_rms or index < self.min_cluster_points - 1 \
                or len(cluster) - index < self.min_cluster_points:
            return [cluster]
        left = self._split_cluster(cluster[:index + 1])
        right = self._split_cluster(cluster[index:])
        return left + right

    def _filter(self, raw: list[LidarBoundary]) -> list[LidarBoundary]:
        used: set[int] = set()
        new_tracks: list[dict] = []
        output: list[LidarBoundary] = []
        for boundary in sorted(raw, key=lambda b: b.clearance):
            best = None
            best_score = -1.0
            for i, track in enumerate(self._tracks):
                if i in used:
                    continue
                score = float(np.dot(boundary.normal, track["normal"]))
                if score > self.match_cos and score > best_score:
                    best, best_score = i, score
            history = deque(maxlen=self.history_length)
            clearance = boundary.clearance
            if best is not None:
                used.add(best)
                old = self._tracks[best]
                history.extend(old["history"])
                # Tighten immediately; release slowly to avoid noisy thrust chatter.
                if clearance > old["clearance"]:
                    clearance = old["clearance"] + self.release_alpha * (clearance - old["clearance"])
            history.append(boundary.surface_distance)
            sigma = max(boundary.distance_sigma, float(np.std(history)) if len(history) > 1 else 0.0)
            filtered = LidarBoundary(
                boundary.normal, clearance, boundary.surface_distance, sigma,
                boundary.geometry_error, boundary.point_count, boundary.tangent,
                boundary.centroid, boundary.dynamic_velocity,
            )
            new_tracks.append({"normal": boundary.normal, "clearance": clearance, "history": history})
            output.append(filtered)
        self._tracks = new_tracks
        return output

    def estimate(
        self,
        distances,
        angles,
        position,
        heading: float,
        footprint_vertices: Sequence[Sequence[float]],
        *,
        base_geometry_margin: float = 3.0,
    ) -> list[LidarBoundary]:
        origin = np.asarray(position, dtype=float)
        points, indices = self.points_from_scan(distances, angles, origin, heading)
        raw: list[LidarBoundary] = []
        clusters = self._clusters(points, indices, len(np.asarray(angles).reshape(-1)), origin)
        for connected in clusters:
            for cluster in self._split_cluster(connected):
                centroid, tangent, normal, surface, rms = self._fit_line(cluster, origin)
                if rms > self.max_fit_rms or surface <= 0.0:
                    continue
                support = footprint_support(footprint_vertices, normal, heading)
                geometry_error = float(base_geometry_margin + rms)
                raw.append(LidarBoundary(
                    normal=normal,
                    clearance=surface - support,
                    surface_distance=surface,
                    distance_sigma=2.5,
                    geometry_error=geometry_error,
                    point_count=len(cluster),
                    tangent=tangent,
                    centroid=centroid,
                ))
        return self._filter(raw)

