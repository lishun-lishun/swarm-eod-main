"""V5: conservative, deadline-aware task ranking using legal local estimates.

Pure Python; no simulator state or global radio access. Default-off component.
ETA is in control frames; older radio claims in px must NOT be compared to it.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import dist, inf, isfinite
from typing import Iterable, Optional


@dataclass(frozen=True)
class TaskCandidate:
    key: object                    # preserve Knowledge key type (often tuple)
    position: tuple[float, float]  # legal observed/received position
    approach_px: float             # legal reachable Dijkstra cost, px
    loaded_return_px: Optional[float]  # None means unknown center/route
    failures: int = 0
    blocked_approach: bool = False
    queue_frames: float = 20.0


@dataclass(frozen=True)
class TaskBidV5:
    drone_id: int
    position: tuple[float, float]
    cost_frames: float
    sent_step: int
    ttl_frames: int = 25
    protocol: int = 5

    def valid(self, step: int) -> bool:
        return (self.protocol == 5 and self.sent_step <= step
                < self.sent_step + self.ttl_frames
                and isfinite(self.cost_frames) and self.cost_frames >= 0)


@dataclass(frozen=True)
class TaskDecision:
    key: object
    position: tuple[float, float]
    completion_frames: float
    score_frames: float
    expected_finish_step: float
    reason: str


@dataclass(frozen=True)
class SchedulerSettings:
    empty_effective_speed: float = 6.0  # calibrate on route progress, not top speed
    loaded_effective_speed: float = 2.2
    empty_path_factor: float = 1.20
    loaded_path_factor: float = 1.30
    pickup_frames: float = 40.0
    unknown_return_px: float = 650.0
    unknown_uncertainty_frames: float = 140.0
    failure_penalty: float = 60.0
    blocked_penalty: float = 45.0
    recent_pickup_penalty: float = 90.0
    deadline_penalty: float = 2.0
    min_switch_gain: float = 50.0
    fractional_switch_gain: float = 0.18
    peer_bid_match_px: float = 20.0
    peer_bid_margin_frames: float = 25.0


class DeadlineTaskScheduler:
    def __init__(self, drone_id: int, settings: SchedulerSettings | None = None):
        self.drone_id = int(drone_id)
        self.cfg = settings or SchedulerSettings()

    def eta(self, candidate: TaskCandidate) -> float:
        c = self.cfg
        if not isfinite(candidate.approach_px) or candidate.approach_px < 0:
            return inf
        if candidate.loaded_return_px is None or not isfinite(candidate.loaded_return_px):
            return_px = c.unknown_return_px
            uncertainty = c.unknown_uncertainty_frames
        elif candidate.loaded_return_px < 0:
            return inf
        else:
            return_px = candidate.loaded_return_px
            uncertainty = 0.0
        return (c.empty_path_factor * candidate.approach_px / c.empty_effective_speed
                + c.pickup_frames
                + c.loaded_path_factor * return_px / c.loaded_effective_speed
                + max(0.0, candidate.queue_frames)
                + c.failure_penalty * max(0, candidate.failures)
                + (c.blocked_penalty if candidate.blocked_approach else 0.0)
                + uncertainty)

    def evaluate(self, candidate: TaskCandidate, *, now: int, horizon: int,
                 own_recent_pickups: int) -> tuple[float, float]:
        eta = self.eta(candidate)
        if not isfinite(eta):
            return inf, inf
        late = max(0.0, now + eta - horizon)
        # Fairness never makes a task impossible: if all others are worse,
        # the caller can still choose this reachable bomb.
        rank = (eta + self.cfg.deadline_penalty * late
                + self.cfg.recent_pickup_penalty * max(0, own_recent_pickups))
        return rank, eta

    def choose(self, candidates: Iterable[TaskCandidate], *, now: int,
               horizon: int, incumbent_key: object | None,
               own_recent_pickups: int = 0,
               peer_bids: Iterable[TaskBidV5] = ()) -> TaskDecision | None:
        choices = []
        bids = tuple(b for b in peer_bids if b.drone_id != self.drone_id and b.valid(now))
        for task in candidates:
            rank, eta = self.evaluate(task, now=now, horizon=horizon,
                                      own_recent_pickups=own_recent_pickups)
            if not isfinite(rank):
                continue
            # Compare only V5 frame-based bids, never the old px claims.
            better_peer = any(
                dist(task.position, b.position) <= self.cfg.peer_bid_match_px
                and ((b.cost_frames < rank - self.cfg.peer_bid_margin_frames)
                     or (abs(b.cost_frames - rank) <= self.cfg.peer_bid_margin_frames
                         and b.drone_id < self.drone_id))
                for b in bids
            )
            if better_peer:
                continue
            choices.append((rank, eta, repr(task.key), task))
        if not choices:
            return None
        choices.sort(key=lambda t: (t[0], t[2]))
        selected = choices[0]
        incumbent = next((c for c in choices if c[3].key == incumbent_key), None)
        reason = 'new_claim' if incumbent is None else 'same_claim'
        if incumbent is not None and selected[3].key != incumbent[3].key:
            improvement = incumbent[0] - selected[0]
            need = max(self.cfg.min_switch_gain,
                       self.cfg.fractional_switch_gain * max(incumbent[1], 1.0))
            if improvement < need:
                selected = incumbent
                reason = 'hold_incumbent'
            else:
                reason = 'better_finish_time'
        rank, eta, key, task = selected
        return TaskDecision(task.key, task.position, eta, rank, now + eta, reason)

    def make_bid(self, decision: TaskDecision, *, now: int) -> TaskBidV5:
        return TaskBidV5(self.drone_id, decision.position,
                         decision.score_frames, now)
