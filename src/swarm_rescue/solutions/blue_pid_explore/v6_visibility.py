"""V6: geometrically grounded semantic-ray evidence and next-best-view selection.

Input angles are the OFFICIAL semantic ray angles (35 values from
compute_ray_angles), not a downsampled continuous 181-beam approximation.
Only legal pose/heading/LiDAR readings may be passed by a competition controller.
The observed mask is evidence of ray passage, not proof of target absence.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def official_semantic_angles(count: int = 35) -> np.ndarray:
    if count < 2:
        raise ValueError('count must exceed 1')
    # Matches simulation/ray_sensors/distance_sensor.py::compute_ray_angles
    return np.linspace(-math.pi, math.pi, count, dtype=np.float64)


@dataclass(frozen=True)
class ViewProposal:
    x: float
    y: float
    heading: float
    score: float
    expected_new_cells: float
    path_cost_px: float
    cell: tuple[int, int]


class SemanticRayEvidence:
    """Records real 35-beam traversals; does NOT infer a region is bomb-free."""

    def __init__(self, shape: tuple[int, int], *, cell_px: float = 10.0,
                 origin_xy: tuple[float, float] = (-1000.0, -1000.0)):
        self.seen_at = np.zeros(shape, dtype=np.int32)
        self.cell_px = float(cell_px)
        self.origin = tuple(float(x) for x in origin_xy)
        if self.cell_px <= 0:
            raise ValueError('positive cell size required')

    @property
    def shape(self) -> tuple[int, int]:
        return self.seen_at.shape

    def cell(self, p: tuple[float, float]) -> tuple[int, int] | None:
        ix = math.floor((float(p[0]) - self.origin[0]) / self.cell_px)
        iy = math.floor((float(p[1]) - self.origin[1]) / self.cell_px)
        if iy < 0 or ix < 0 or iy >= self.shape[0] or ix >= self.shape[1]:
            return None
        return (iy, ix)

    def world(self, cell: tuple[int, int]) -> tuple[float, float]:
        iy, ix = cell
        return (self.origin[0] + (ix + 0.5) * self.cell_px,
                self.origin[1] + (iy + 0.5) * self.cell_px)

    def update(self, *, pose_xy, heading: float, lidar_ranges, lidar_angles,
               semantic_angles, step: int, semantic_range_px: float = 130.0,
               occlusion_margin_px: float = 6.0) -> int:
        """Add cells crossed by actual semantic rays up to the matched LiDAR hit.

        This is intentionally conservative. LiDAR can hit a wall, bomb, or drone;
        nothing behind the first hit is ever marked. Only centerline evidence is
        stored: don't mark a bomb-free disk around the ray without proof.
        """
        ranges = np.asarray(lidar_ranges, dtype=float)
        angles = np.asarray(lidar_angles, dtype=float)
        sem = np.asarray(semantic_angles, dtype=float)
        if (ranges.ndim != 1 or ranges.shape != angles.shape
                or sem.ndim != 1 or sem.size != 35 or not np.isfinite(heading)):
            raise ValueError('invalid official sensor arrays')
        if not np.isfinite(np.asarray(pose_xy, float)).all():
            return 0
        marked = 0
        for a in sem:
            idx = int(np.argmin(np.abs((angles - a + math.pi) % (2*math.pi) - math.pi)))
            d = ranges[idx]
            if not np.isfinite(d):
                continue
            limit = min(float(semantic_range_px), max(0.0, float(d) - occlusion_margin_px))
            phi = float(heading) + float(a)
            cp, sp = math.cos(phi), math.sin(phi)
            for length in np.arange(0.0, limit + 1e-9, self.cell_px * 0.5):
                cell = self.cell((float(pose_xy[0]) + length * cp,
                                  float(pose_xy[1]) + length * sp))
                if cell is None:
                    break
                if self.seen_at[cell] == 0:
                    marked += 1
                self.seen_at[cell] = max(self.seen_at[cell], int(step) + 1)
        return marked

    def predicted_information(self, xy, heading: float, occ: np.ndarray,
                              *, occupied_threshold: float = 1.0,
                              free_threshold: float = -0.6,
                              range_px: float = 130.0) -> float:
        """Utility only. Ray tracing stops at first unknown or occupied cell.

        Unlike 'verified coverage', predicted information can be uncertain and
        must never grant a collision-free route or mark search completed.
        """
        if occ.shape != self.shape:
            raise ValueError('grid shape mismatch')
        gain = 0.0
        for a in official_semantic_angles():
            phi = heading + float(a)
            cp, sp = math.cos(phi), math.sin(phi)
            last = None
            for length in np.arange(self.cell_px, range_px + 1e-9, self.cell_px):
                c = self.cell((float(xy[0]) + length * cp,
                               float(xy[1]) + length * sp))
                if c is None:
                    break
                if c == last:
                    continue
                last = c
                value = float(occ[c])
                if value > occupied_threshold:
                    break
                if value >= free_threshold:
                    gain += 2.0  # first unknown cell, not hidden cells beyond it
                    break
                if self.seen_at[c] == 0:
                    gain += 0.3
        return gain


def choose_next_view(evidence: SemanticRayEvidence, occ: np.ndarray,
                     clearance_px: np.ndarray, route_cost_px: np.ndarray,
                     *, position: tuple[float, float],
                     min_clearance_px: float = 19.0,
                     max_candidates: int = 24,
                     min_gain: float = 1.0,
                     last_targets: tuple[tuple[float, float], ...] = (),
                     max_route_cost_px: float = 1600.0) -> ViewProposal | None:
    """Sample KNOWN-FREE, route-reachable viewpoints; orient for real beams.

    The caller must additionally verify a true contiguous path and current
    hull clearance before executing the returned proposal. The distance field
    must be freshly computed from the current legal estimated pose.
    """
    if occ.shape != evidence.shape or clearance_px.shape != occ.shape or route_cost_px.shape != occ.shape:
        raise ValueError('grid shape mismatch')
    walkable = ((occ < -0.6) & (clearance_px >= min_clearance_px)
                & np.isfinite(route_cost_px)
                & (route_cost_px <= max_route_cost_px))
    # Useful regions sit at the known/unknown boundary or beside a wall face
    # that the real semantic rays have not traversed.
    from scipy.ndimage import binary_dilation, maximum_filter
    unknown = (occ >= -0.6) & (occ <= 1.0)
    boundary = binary_dilation(unknown, iterations=2) & walkable
    unseen = (evidence.seen_at == 0) & walkable
    near_unseen = binary_dilation(unseen, iterations=2) & walkable
    candidates = boundary | near_unseen
    if not candidates.any():
        return None
    # Cheap pre-ranking, expensive 35-ray orientation search only for top K.
    novelty = maximum_filter((unknown | unseen).astype(np.float32), size=9)
    iy, ix = np.where(candidates)
    dist = route_cost_px[iy, ix]
    priority = 200.0 * novelty[iy, ix] - 0.12 * dist
    if last_targets:
        xx = evidence.origin[0] + (ix + .5) * evidence.cell_px
        yy = evidence.origin[1] + (iy + .5) * evidence.cell_px
        for x, y in last_targets:
            priority -= 90.0 * (np.hypot(xx - x, yy - y) < 50.0)
    k = min(max_candidates, len(iy))
    indices = np.argpartition(priority, -k)[-k:]
    best = None
    for j in indices:
        cell = (int(iy[j]), int(ix[j]))
        xy = evidence.world(cell)
        # 35 rays cover 360 degrees; small yaw offsets fill angular gaps.
        for delta in np.linspace(0.0, 2.0 * math.pi / 35.0, 4, endpoint=False):
            h = float(delta)
            gain = evidence.predicted_information(xy, h, occ)
            score = gain - 0.025 * float(route_cost_px[cell])
            if gain < min_gain:
                continue
            if best is None or score > best.score:
                best = ViewProposal(xy[0], xy[1], h, score, gain,
                                    float(route_cost_px[cell]), cell)
    return best
