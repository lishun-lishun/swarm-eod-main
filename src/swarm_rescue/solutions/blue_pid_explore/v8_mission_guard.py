"""Conservative scout preemption: protect fetchers, carriers, and deliveries.

Only accepts local legal state / currently received peer messages.  Do not
invent offline bomb positions, and never infer global current team state.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


def point_segment_distance(p, a, b):
    px, py = map(float, p)
    ax, ay = map(float, a)
    bx, by = map(float, b)
    dx, dy = bx - ax, by - ay
    d2 = dx * dx + dy * dy
    if d2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / d2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str


class MissionPreservationGuard:
    def __init__(self, *, min_remaining_frames=350,
                 carrier_protect_px=110.0,
                 max_peer_age_frames=35):
        self.min_remaining = int(min_remaining_frames)
        self.carrier_protect = float(carrier_protect_px)
        self.max_peer_age = int(max_peer_age_frames)

    def check(self, *, now, deadline, state, carrying, has_claim,
              visible_ungrasped_bomb, own_pose, proposed_xy,
              known_peers=(), emergency=False):
        if emergency:
            return GuardResult(False, 'SAFETY_OR_RESUME')
        if carrying or state not in ('EXPLORE', 'CORRIDOR'):
            return GuardResult(False, 'NOT_IDLE_EXPLORER')
        if has_claim or visible_ungrasped_bomb:
            return GuardResult(False, 'BOMB_TASK_TAKES_PRIORITY')
        if deadline > 0 and int(deadline) - int(now) < self.min_remaining:
            return GuardResult(False, 'NO_USEFUL_EXPLORATION_WINDOW')
        if own_pose is None or proposed_xy is None:
            return GuardResult(False, 'NO_LOCAL_GOAL')
        for peer in known_peers:
            # peers are only fresh, received legal radio messages.  The
            # absence of messages does not assert that a corridor is empty.
            if not bool(peer.get('carry', False)):
                continue
            age = now - int(peer.get('last_seen', -1000000))
            if age < 0 or age > self.max_peer_age:
                continue
            p = peer.get('pos')
            if p is None:
                continue
            if point_segment_distance(p, own_pose, proposed_xy) < self.carrier_protect:
                return GuardResult(False, 'CARRIER_ROUTE_PRIORITY')
            g = peer.get('goal')
            if g is not None and point_segment_distance(own_pose, p, g) < self.carrier_protect:
                return GuardResult(False, 'CARRIER_GOAL_PRIORITY')
        return GuardResult(True, 'SCOUT_ELIGIBLE')
