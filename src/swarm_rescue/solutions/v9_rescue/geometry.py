from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

TAU = 2.0 * math.pi

def wrap(x: float) -> float:
    return (float(x) + math.pi) % TAU - math.pi

def norm(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))

def normalize(x, default=(0.0, 0.0)):
    z = np.asarray(x, dtype=float).reshape(2)
    n = norm(z)
    return (z / n) if n > 1e-8 else np.asarray(default, dtype=float)

def rotate(v, angle: float):
    x, y = np.asarray(v, dtype=float).reshape(2)
    c, s = math.cos(angle), math.sin(angle)
    return np.array((c*x - s*y, s*x + c*y), dtype=float)

def to_body(world_vec, heading):
    return rotate(world_vec, -heading)

def clip_disk(v):
    v = np.asarray(v, dtype=float).reshape(2)
    if not np.all(np.isfinite(v)):
        return np.zeros(2, dtype=float)
    n = norm(v)
    return v / n if n > 1 else v

@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)
    def plus(self, dx, dy, dyaw=0):
        return Pose(self.x+float(dx), self.y+float(dy), wrap(self.yaw+dyaw))

@dataclass(frozen=True)
class RelativeDetection:
    kind: str  # bomb | center | drone
    distance: float
    angle: float  # body-frame
    grasped: bool = False

def valid_odom(odom):
    if odom is None:
        return False
    try:
        v = np.asarray(odom, dtype=float)
        return v.shape == (3,) and np.all(np.isfinite(v))
    except (TypeError, ValueError):
        return False

def stable_min(values, quantile=0.15):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a) & (a > 0.0)]
    return float(np.quantile(a, quantile)) if a.size else math.inf
