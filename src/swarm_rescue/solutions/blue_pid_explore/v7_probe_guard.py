"""Persistence gate for *ego-visible* very short unknown-space probes.

Single noisy LiDAR returns never authorize passing a mapped conflict.
Requires repeated clear readings and an external final safety layer.
Only for EMPTY exploration; not a collision guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .v7_gateway import Probe, short_unknown_probe


@dataclass(frozen=True)
class ProbeObservation:
    confirmed: bool
    consecutive: int
    proposal: Probe | None
    reason: str


class ProbePersistenceGate:
    def __init__(self, *, required_clear_frames: int = 3,
                 max_step_px: float = 8.0):
        self.required = int(required_clear_frames)
        self.max_step = float(max_step_px)
        if self.required<2 or not 0<self.max_step<=15:
            raise ValueError('invalid probe safety gate')
        self.anchor=None
        self.count=0

    def reset(self):
        self.anchor=None;self.count=0

    def observe(self, *, gateway_cell, position, heading, target_heading,
                lidar_ranges, lidar_angles, body_radius_px: float,
                carrying:bool=False, dynamic_obstruction:bool=False,
                local_segment_safe=None)->ProbeObservation:
        if carrying or dynamic_obstruction:
            self.reset()
            return ProbeObservation(False,0,None,'CARGO_OR_DYNAMIC_BLOCK')
        key=tuple(gateway_cell)
        if self.anchor!=key:
            self.anchor=key;self.count=0
        probe=short_unknown_probe(current_xy=position,heading=heading,
                                  lidar_ranges=lidar_ranges,lidar_angles=lidar_angles,
                                  target_heading=target_heading,hull_radius_px=body_radius_px,
                                  max_length_px=self.max_step,
                                  current_segment_safe=local_segment_safe)
        if probe is None:
            self.count=0
            return ProbeObservation(False,0,None,'LIVE_LIDAR_OR_GEOMETRY_NOT_CLEAR')
        self.count+=1
        if self.count<self.required:
            return ProbeObservation(False,self.count,None,'WAIT_FOR_REPEAT_SCAN')
        self.count=0  # Require multiple frames again before each further probe
        return ProbeObservation(True,self.required,probe,'SHORT_PROBE_VERIFIED_LOCALLY')
