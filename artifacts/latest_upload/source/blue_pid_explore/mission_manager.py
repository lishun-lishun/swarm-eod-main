"""Single authoritative high-level mission state; it never writes final thrust."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class MissionState(str, Enum):
    EXPLORE="EXPLORE"; FETCH="FETCH"; PICKUP="PICKUP"; CARRY="CARRY"
    QUEUE="QUEUE"; DELIVER="DELIVER"; RECOVER="RECOVER"; COMPLETE="COMPLETE"


@dataclass(frozen=True)
class MissionGoal:
    state: MissionState
    target: np.ndarray | None
    bomb_id: str | None
    carrying: bool


class MissionManager:
    def __init__(self):
        self.state=MissionState.EXPLORE; self.bomb_id=None; self.target=None

    def assign_bomb(self,bomb_id:str,position:Sequence[float])->None:
        self.bomb_id=bomb_id; self.target=np.asarray(position,float); self.state=MissionState.FETCH

    def update(self,*,near_target:bool,grasped:bool,disposed_confirmed:bool,
               disposal_target:Sequence[float]|None,exploration_target:Sequence[float]|None,
               queue_admitted:bool=True)->MissionGoal:
        if disposed_confirmed:
            self.bomb_id=None; self.state=MissionState.EXPLORE
        elif grasped and self.state in (MissionState.FETCH,MissionState.PICKUP):
            self.state=MissionState.CARRY
        elif self.state==MissionState.FETCH and near_target:
            self.state=MissionState.PICKUP
        if self.state==MissionState.CARRY:
            self.target=None if disposal_target is None else np.asarray(disposal_target,float)
            if near_target and not queue_admitted: self.state=MissionState.QUEUE
            elif near_target and queue_admitted: self.state=MissionState.DELIVER
        elif self.state==MissionState.QUEUE and queue_admitted:
            self.state=MissionState.DELIVER
        elif self.state==MissionState.EXPLORE:
            self.target=None if exploration_target is None else np.asarray(exploration_target,float)
        return MissionGoal(self.state,None if self.target is None else self.target.copy(),self.bomb_id,grasped)

