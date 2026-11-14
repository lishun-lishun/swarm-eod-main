"""Distributed disposal-center entry reservations and queue positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DisposalReservation:
    drone_id: int
    start_step: int
    end_step: int
    priority: int
    version: int


class DisposalTrafficManager:
    def __init__(self, drone_id: int, *, slot_steps: int = 70, timeout: int = 140):
        self.drone_id=int(drone_id); self.slot_steps=int(slot_steps); self.timeout=int(timeout)
        self.version=0; self.reservations: dict[int,DisposalReservation]={}

    def merge(self, reservations: Iterable[DisposalReservation], step: int) -> None:
        for item in reservations:
            current=self.reservations.get(item.drone_id)
            if current is None or item.version>current.version:
                self.reservations[item.drone_id]=item
        self.reservations={k:v for k,v in self.reservations.items() if v.end_step+self.timeout>step}

    def reserve(self, earliest: int, *, health: int = 100) -> DisposalReservation:
        intervals=sorted(self.reservations.values(),key=lambda r:(r.start_step,-r.priority,r.drone_id))
        start=int(earliest)
        for item in intervals:
            if start+self.slot_steps<=item.start_step: break
            if start<item.end_step: start=item.end_step
        self.version+=1
        reservation=DisposalReservation(self.drone_id,start,start+self.slot_steps,100-health,self.version)
        self.reservations[self.drone_id]=reservation; return reservation

    def admitted(self, step: int) -> bool:
        mine=self.reservations.get(self.drone_id)
        return mine is not None and mine.start_step<=step<mine.end_step


