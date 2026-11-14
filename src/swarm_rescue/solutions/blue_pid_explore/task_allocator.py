"""Distributed time-to-complete auction with expiring bomb claims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TaskBid:
    bomb_id: str
    drone_id: int
    arrival_time: float
    pickup_time: float
    return_time: float
    queue_time: float
    step: int

    @property
    def total_time(self) -> float:
        return self.arrival_time + self.pickup_time + self.return_time + self.queue_time


@dataclass(frozen=True)
class Claim:
    bomb_id: str
    drone_id: int
    bid_time: float
    version: int
    expires: int


class DistributedTaskAllocator:
    def __init__(self, drone_id: int, *, lease_steps: int = 180):
        self.drone_id = int(drone_id); self.lease_steps = int(lease_steps)
        self.claims: dict[str, Claim] = {}; self._versions: dict[str, int] = {}

    @staticmethod
    def winner(bids: Iterable[TaskBid]) -> TaskBid | None:
        bids = list(bids)
        return min(bids, key=lambda bid: (bid.total_time, bid.drone_id)) if bids else None

    def claim(self, bid: TaskBid, step: int) -> Claim:
        version = self._versions.get(bid.bomb_id, 0) + 1
        result = Claim(bid.bomb_id, bid.drone_id, bid.total_time, version, int(step) + self.lease_steps)
        self._versions[bid.bomb_id] = version; self.claims[bid.bomb_id] = result
        return result

    def merge(self, incoming: Iterable[Claim], step: int) -> None:
        for claim in incoming:
            if claim.expires <= step:
                continue
            current = self.claims.get(claim.bomb_id)
            key = (claim.version, -claim.bid_time, -claim.drone_id)
            current_key = (-1, float("-inf"), float("-inf")) if current is None else (
                current.version, -current.bid_time, -current.drone_id)
            if key > current_key:
                self.claims[claim.bomb_id] = claim
                self._versions[claim.bomb_id] = max(self._versions.get(claim.bomb_id, 0), claim.version)
        self.expire(step)

    def expire(self, step: int) -> None:
        self.claims = {bomb: claim for bomb, claim in self.claims.items() if claim.expires > step}

    def owns(self, bomb_id: str, step: int) -> bool:
        self.expire(step); claim = self.claims.get(bomb_id)
        return claim is not None and claim.drone_id == self.drone_id


