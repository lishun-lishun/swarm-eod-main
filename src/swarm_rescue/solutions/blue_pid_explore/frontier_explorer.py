"""Reachable frontier clustering with gain, risk, distance and history score."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Sequence

import numpy as np

from .occupancy_map import Cell


@dataclass(frozen=True)
class Frontier:
    cells: tuple[tuple[int, int], ...]
    representative: tuple[int, int]
    gain: float
    distance: float
    risk: float
    history: float
    score: float


class FrontierExplorer:
    def __init__(self, *, gain_weight=1.0, distance_weight=0.7,
                 risk_weight=1.5, history_weight=2.0, blacklist_frames=120):
        self.weights = tuple(map(float, (gain_weight, distance_weight, risk_weight, history_weight)))
        self.blacklist_frames = int(blacklist_frames)
        self._visits: dict[tuple[int, int], int] = {}
        self._blacklist: dict[tuple[int, int], int] = {}
        self._held: tuple[int, int] | None = None

    def tick(self) -> None:
        self._blacklist = {k: v - 1 for k, v in self._blacklist.items() if v > 1}

    def blacklist(self, cell: Sequence[int]) -> None:
        self._blacklist[tuple(map(int, cell))] = self.blacklist_frames

    def mark_selected(self, cell: Sequence[int]) -> None:
        key = tuple(map(int, cell)); self._visits[key] = self._visits.get(key, 0) + 1; self._held = key

    def frontiers(self, grid: np.ndarray, start: Sequence[int], distance_field: np.ndarray,
                  risk_field: np.ndarray | None = None) -> list[Frontier]:
        free = grid == int(Cell.FREE); unknown = grid == int(Cell.UNKNOWN)
        neighbor_unknown = np.zeros_like(free)
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            neighbor_unknown |= np.roll(np.roll(unknown, dy, axis=0), dx, axis=1)
        mask = free & neighbor_unknown
        seen = np.zeros_like(mask); clusters = []
        h, w = mask.shape
        for y, x in zip(*np.nonzero(mask)):
            if seen[y, x]: continue
            queue=deque([(x,y)]); seen[y,x]=True; cells=[]
            while queue:
                cx,cy=queue.popleft(); cells.append((cx,cy))
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx,ny=cx+dx,cy+dy
                    if 0<=nx<w and 0<=ny<h and mask[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=True; queue.append((nx,ny))
            clusters.append(cells)
        result=[]; wg,wd,wr,wh=self.weights
        for cells in clusters:
            representative=min(cells,key=lambda c: distance_field[c[1],c[0]])
            if representative in self._blacklist or not np.isfinite(distance_field[representative[1], representative[0]]):
                continue
            gain=float(len(cells)); distance=float(distance_field[representative[1],representative[0]])
            risk=0.0 if risk_field is None else float(np.mean([risk_field[y,x] for x,y in cells]))
            history=float(self._visits.get(representative,0)); score=wg*gain-wd*distance-wr*risk-wh*history
            result.append(Frontier(tuple(cells), representative, gain, distance, risk, history, score))
        return sorted(result, key=lambda f: f.score, reverse=True)

