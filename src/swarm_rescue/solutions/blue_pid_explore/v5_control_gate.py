"""V5: bounded shadow-to-canary transition for EMPTY model thrust only.

Does not replace the baseline controller or final predictive safety filter.
Default blending factor is zero. No cargo or contact-mode authorization.
"""
from __future__ import annotations
import math
import numpy as np


def unit_disk(value):
    a = np.asarray(value, dtype=float).reshape(2)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite thrust')
    size = float(np.linalg.norm(a))
    return a / size if size > 1.0 else a.copy()


def world_to_body(u, heading):
    x, y = np.asarray(u, dtype=float).reshape(2)
    c, s = math.cos(heading), math.sin(heading)
    return np.array([c*x + s*y, -s*x + c*y])


def body_to_world(u, heading):
    x, y = np.asarray(u, dtype=float).reshape(2)
    c, s = math.cos(heading), math.sin(heading)
    return np.array([c*x - s*y, s*x + c*y])


class BoundedModelBlend:
    def __init__(self, alpha: float = 0.0, delta_max: float = 0.15):
        if not 0.0 <= alpha <= 1.0 or delta_max < 0.0:
            raise ValueError('invalid blend parameters')
        self.alpha = alpha
        self.delta_max = delta_max

    def apply(self, *, baseline_body, model_world, heading: float,
              state: str, carrying: bool, lidar_min_px: float,
              gps_valid: bool, peer_nearby: bool):
        base = unit_disk(baseline_body)
        # The caller still runs the original final predictive safety filter.
        if (self.alpha == 0 or carrying or state != 'EXPLORE'
                or not gps_valid or peer_nearby or lidar_min_px <= 110.0):
            return base, False
        original_world = body_to_world(base, heading)
        candidate = unit_disk(model_world)
        # Avoid a candidate that reverses original task intention.
        if float(original_world @ candidate) < -1e-9:
            return base, False
        delta = candidate - original_world
        mag = float(np.linalg.norm(delta))
        if mag > self.delta_max and mag > 0:
            delta *= self.delta_max / mag
        blended = unit_disk(original_world + self.alpha * delta)
        return unit_disk(world_to_body(blended, heading)), True
