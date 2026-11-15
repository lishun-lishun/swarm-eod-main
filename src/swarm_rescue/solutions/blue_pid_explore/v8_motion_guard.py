"""Conservative local LiDAR clearance for one short ego-relative movement.

A proxy, not a formal proof of collision freedom: moving peers and pivot-joint
swing are not fully observable.  Never bypass the original predictive filter.
"""
from __future__ import annotations
import math
import numpy as np


def lidar_short_segment_safe(*, bearing_body, distance_px, lidar_ranges,
                             lidar_angles, hull_radius_px,
                             carrying=False, peer_nearby=False,
                             cargo_reach_px=33.0, cargo_radius_px=12.0,
                             noise_margin_px=5.0):
    if peer_nearby or not np.isfinite([bearing_body, distance_px, hull_radius_px]).all():
        return False
    if distance_px < 0 or distance_px > 42:
        return False
    distances = np.asarray(lidar_ranges, dtype=float)
    angles = np.asarray(lidar_angles, dtype=float)
    if distances.ndim != 1 or angles.shape != distances.shape or distances.size < 35:
        return False
    radius = max(float(hull_radius_px),
                 float(cargo_reach_px) + float(cargo_radius_px) if carrying else 0.0)
    required = distance_px + radius + float(noise_margin_px)
    half = math.atan2(radius, max(distance_px, 4.0))
    errors = np.abs((angles - bearing_body + math.pi) % (2 * math.pi) - math.pi)
    mask = (errors <= half) & np.isfinite(distances)
    if int(np.sum(mask)) < 3:
        return False
    return bool(np.min(distances[mask]) >= required)
