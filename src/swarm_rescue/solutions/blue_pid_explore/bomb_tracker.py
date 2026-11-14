"""Geometric bomb identities, lifecycle state and communication merges."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Iterable, Sequence

import numpy as np


class BombStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    CLAIMED = "CLAIMED"
    PICKED = "PICKED"
    DISPOSED = "DISPOSED"
    LOST = "LOST"


@dataclass(frozen=True)
class BombTrack:
    bomb_id: str
    position: np.ndarray
    status: BombStatus
    version: int
    observed_step: int
    owner: int | None = None
    lease_until: int = 0


class BombTracker:
    def __init__(self, *, identity_resolution: float = 18.0, merge_distance: float = 22.0):
        self.identity_resolution = float(identity_resolution); self.merge_distance = float(merge_distance)
        self.tracks: dict[str, BombTrack] = {}

    def _identity(self, position: Sequence[float]) -> str:
        p = np.asarray(position, dtype=float)
        cell = np.round(p / self.identity_resolution).astype(int)
        return f"B{cell[0]}:{cell[1]}"

    def observe(self, position: Sequence[float], step: int) -> BombTrack:
        p = np.asarray(position, dtype=float)
        candidates = [track for track in self.tracks.values()
                      if track.status != BombStatus.DISPOSED and np.linalg.norm(track.position - p) <= self.merge_distance]
        if candidates:
            track = min(candidates, key=lambda item: np.linalg.norm(item.position - p))
            updated = replace(track, position=0.7 * track.position + 0.3 * p,
                              observed_step=int(step), version=track.version + 1)
        else:
            identity = self._identity(p)
            suffix = 0; candidate = identity
            while candidate in self.tracks:
                suffix += 1; candidate = f"{identity}:{suffix}"
            updated = BombTrack(candidate, p.copy(), BombStatus.DISCOVERED, 1, int(step))
        self.tracks[updated.bomb_id] = updated
        return updated

    def update_status(self, bomb_id: str, status: BombStatus, step: int,
                      *, owner: int | None = None, lease_until: int = 0) -> BombTrack:
        current = self.tracks[bomb_id]
        updated = replace(current, status=status, version=current.version + 1,
                          observed_step=int(step), owner=owner, lease_until=int(lease_until))
        self.tracks[bomb_id] = updated; return updated

    def merge(self, remote: Iterable[BombTrack]) -> None:
        precedence = {BombStatus.DISCOVERED: 0, BombStatus.CLAIMED: 1,
                      BombStatus.LOST: 1, BombStatus.PICKED: 2, BombStatus.DISPOSED: 3}
        for incoming in remote:
            current = self.tracks.get(incoming.bomb_id)
            if current is None or incoming.version > current.version or (
                incoming.version == current.version and precedence[incoming.status] > precedence[current.status]
            ):
                self.tracks[incoming.bomb_id] = incoming


