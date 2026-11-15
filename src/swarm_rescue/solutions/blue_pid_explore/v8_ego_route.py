"""GPS-outage continuity of the *last valid route* (not blind retreat).

Uses ONLY previously trusted legal world pose and incremental odometry.
No simulator truth.  A short live LiDAR + cargo-footprint check remains
mandatory for EVERY proposed move.  Stale routes always fall back to baseline.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


def wrap(angle):
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class LocalContinuation:
    ok: bool
    reason: str
    bearing_body: float = 0.0
    distance_px: float = 0.0
    waypoint_index: int = 0
    estimated_drift_px: float = math.inf


class EgoRouteContinuity:
    def __init__(self, *, max_age_frames=180, max_drift_px=12.0,
                 lookahead_px=38.0, dist_noise_px=0.35,
                 angle_noise_rad=math.radians(0.9)):
        self.max_age = int(max_age_frames)
        self.max_drift = float(max_drift_px)
        self.lookahead = float(lookahead_px)
        self.dist_noise = float(dist_noise_px)
        self.angle_noise = float(angle_noise_rad)
        self.reset()

    def reset(self):
        self.x = self.y = self.heading = 0.0
        self.anchor_step = None
        self.last_odom_step = -1
        self.travel = 0.0
        self.route = ()
        self.index = 0
        self.path_id = None

    def odometry_update(self, step, odometry):
        if step <= self.last_odom_step or odometry is None or len(odometry) != 3:
            return False
        try:
            d, alpha, dyaw = map(float, odometry)
        except (ValueError, TypeError):
            return False
        if not all(map(math.isfinite, (d, alpha, dyaw))):
            return False
        self.x += d * math.cos(self.heading + alpha)
        self.y += d * math.sin(self.heading + alpha)
        self.heading = wrap(self.heading + dyaw)
        self.travel += abs(d)
        self.last_odom_step = int(step)
        return True

    def capture(self, *, step, legal_pose_xy, legal_heading,
                world_path, path_id=None, gps_valid=True):
        """Capture only with current valid GPS and an existing route.

        World path is converted ONCE to the present body-aligned ego frame;
        later world-map changes must NOT silently rotate or translate it.
        """
        if not gps_valid or legal_pose_xy is None or not world_path:
            return False
        try:
            px, py = map(float, legal_pose_xy)
            yaw = float(legal_heading)
            pts = [tuple(map(float, v)) for v in world_path]
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(z) for z in [px, py, yaw, *[w for p in pts for w in p]]):
            return False
        c, s = math.cos(yaw), math.sin(yaw)
        self.route = tuple((c * (x - px) + s * (y - py),
                            -s * (x - px) + c * (y - py)) for x, y in pts)
        self.x = self.y = self.heading = self.travel = 0.0
        self.anchor_step = int(step)
        self.index = 0
        self.path_id = path_id
        return True

    def drift_bound(self, now):
        if self.anchor_step is None:
            return math.inf
        elapsed = max(0, int(now) - self.anchor_step)
        # Simple conservative proxy; must be calibrated against legal
        # odometry, and NOT presented as statistical coverage guarantee.
        return (math.sqrt(elapsed) * self.dist_noise
                + self.travel * math.sqrt(elapsed) * self.angle_noise)

    def propose(self, *, step, gps_valid, carrying, stalled_or_missing,
                local_segment_safe, peer_blocked=False):
        if gps_valid or not carrying or not stalled_or_missing:
            return LocalContinuation(False, 'NOT_REQUIRED')
        if self.anchor_step is None or not self.route:
            return LocalContinuation(False, 'NO_VALID_ANCHOR')
        age = int(step) - self.anchor_step
        drift = self.drift_bound(step)
        if age > self.max_age or drift > self.max_drift:
            return LocalContinuation(False, 'ANCHOR_UNCERTAIN', estimated_drift_px=drift)
        if peer_blocked:
            return LocalContinuation(False, 'DYNAMIC_PEER_BLOCK', estimated_drift_px=drift)
        if local_segment_safe is None:
            return LocalContinuation(False, 'NO_LIVE_GEOMETRY_CHECK', estimated_drift_px=drift)
        here = (self.x, self.y)
        idx = self.index
        # Consume only successive reached points; do not snap to a distant
        # unrelated section of the polyline across a wall.
        while idx < len(self.route) - 1 and math.dist(here, self.route[idx]) < 9.0:
            idx += 1
        best = None
        for j in range(idx, min(len(self.route), idx + 7)):
            target = self.route[j]
            dx, dy = target[0] - self.x, target[1] - self.y
            distance = math.hypot(dx, dy)
            if distance > self.lookahead:
                break
            if distance < 3.0:
                continue
            bearing = wrap(math.atan2(dy, dx) - self.heading)
            if not local_segment_safe(bearing, distance):
                break
            best = (j, bearing, distance)
        if best is None:
            return LocalContinuation(False, 'NO_VERIFIED_FORWARD_SEGMENT',
                                     waypoint_index=idx, estimated_drift_px=drift)
        self.index = best[0]
        return LocalContinuation(True, 'SHORT_FORWARD_CONTINUITY',
                                 best[1], best[2], best[0], drift)
