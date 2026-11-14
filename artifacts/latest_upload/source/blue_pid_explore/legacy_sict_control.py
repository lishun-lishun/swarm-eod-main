"""Deterministic local-control primitives used by the v2 rescue controller.

The functions in this module are deliberately stateless.  They only consume
sensor-derived clearance, velocity, and pose values, so they are safe to use
in competition mode and easy to unit test without the simulator.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def braking_speed(clearance: float, desired: float, *, carrying: bool,
                  body_radius: float = 16.4, pad: float | None = None) -> float:
    """Return a conservative speed cap from one-dimensional braking distance."""
    clearance = max(0.0, float(clearance) - body_radius - (8.0 if pad is None else pad))
    # The loaded model is intentionally gentler: a carried bomb widens the
    # collision envelope and takes longer to settle after a command change.
    decel = 0.30 if carrying else 0.45
    floor = 0.55 if carrying else 0.8
    return min(max(0.0, float(desired)), max(floor, math.sqrt(2.0 * decel * clearance)))


def heading_command(target_angle: float, current_angle: float,
                    *, gain: float = 0.62, deadband: float = 0.08,
                    max_command: float = 1.0) -> float:
    """Proportional yaw command with wrap-safe error and a small deadband."""
    err = (float(target_angle) - float(current_angle) + math.pi) % (2.0 * math.pi) - math.pi
    if abs(err) <= deadband:
        return 0.0
    return max(-max_command, min(max_command, gain * err / math.pi))


def grasp_candidates(bomb: Sequence[float], ring: float, headings: int = 24
                     ) -> Iterable[tuple[float, float, float]]:
    """Yield candidate hull centres and approach bearings around a bomb."""
    n = max(8, int(headings))
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        yield (float(bomb[0]) + ring * math.cos(angle),
               float(bomb[1]) + ring * math.sin(angle), angle)


def collision_priority(my_id: int, carrying: bool, peer_id: int,
                       peer_carrying: bool) -> tuple[int, int]:
    """Stable right-of-way rank: loaded carriers, then lower identifier."""
    return (0 if carrying else 1, int(my_id)) if carrying != peer_carrying else (
        0 if carrying else 1, min(int(my_id), int(peer_id)))


