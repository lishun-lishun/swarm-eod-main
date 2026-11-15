"""team861 v3 鈥?blue-team bomb disposal swarm controller.

Pipeline per step:
  1. pose filter (odometry prediction + GPS / compass correction)
  2. radio: merge peers' bombs, tombstones, claims, map snapshots
  3. perception: semantic bombs / drones / disposal centre; lidar mapping with
     drones and bombs masked out
  4. task: CARRY (have a bomb) > FETCH (claimed bomb) > EXPLORE (scan coverage)
  5. global plan: Dijkstra on the fused cost map, look-ahead waypoint
  6. local control: velocity PD with lidar braking, wall and drone repulsion,
     yaw control that keeps the carried bomb trailing / facing the centre
  7. stuck monitor: escape manoeuvre + temporary cost penalty, then replan

Only sensor data and radio messages are used (no ground truth).
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque

import numpy as np
from scipy import ndimage

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
# Helper modules ship flat next to this file (submission zip is root-level).
# Absolute import works once the files are installed into solutions/; the
# bare-import fallback covers environments that load this file stand-alone.
from .legacy_sict_knowledge import Knowledge, MATCH
from .legacy_sict_mapping import CELL, GridMap
from .legacy_sict_control import braking_speed, grasp_candidates, heading_command
from .legacy_sict_coordination import auction_utility
from .predictive_task_eta import CompletionTimeEstimator, AssignmentHysteresis
from .model_velocity_control import PolePlacementVelocityController, world_to_body
from .v5_deadline_allocator import DeadlineTaskScheduler, TaskCandidate
from .v5_control_gate import BoundedModelBlend
from .velocity_estimator import LegalVelocityEstimator
from .v6_adapter import V6Adapter
from .v6_task_policy import KnownTask, PeerCommitment

TYPE = DroneSemanticSensor.TypeEntity

import os as _os
from collections import Counter
FAR_WEIGHT = float(_os.environ.get("V3_FAR_WEIGHT", "1.0"))
EXPLORE_KIND = _os.environ.get("V3_EXPLORE_KIND", "0") == "1"
EXP_GAIN = float(_os.environ.get("V3_EXP_GAIN", "2500.0"))
EXP_DIST = float(_os.environ.get("V3_EXP_DIST", "1.0"))
EXP_NARROW = 0.0  # tested: the narrow bonus did not help
# Corner exploration enhancement (team861_v3 improved "conservative" set)
CORNER_BONUS = float(_os.environ.get("V3_CORNER_BONUS", "400.0"))
EDGE_BONUS = float(_os.environ.get("V3_EDGE_BONUS", "3000.0"))
EDGE_START = int(_os.environ.get("V3_EDGE_START", "1200"))
EDGE_RANGE = float(_os.environ.get("V3_EDGE_RANGE", "150.0"))
EDGE_SCOUTS = {
    int(value) for value in _os.environ.get("V3_EDGE_SCOUTS", "9").split(",")
    if value.strip()
}
# Generic second-pass coverage.  A long-range LiDAR ray proves free space but
# does not prove that a semantic target hidden behind a wall lip was observed.
# Once normal frontier coverage is nearly exhausted, value reachable standing
# cells near free space that has never been observed from close range.
VERIFY_GAIN = float(_os.environ.get("V3_VERIFY_GAIN", "0.0"))
VERIFY_START = int(_os.environ.get("V3_VERIFY_START", "1200"))
VERIFY_START_LEFT = int(_os.environ.get("V3_VERIFY_START_LEFT", "80"))
VERIFY_WINDOW = int(_os.environ.get("V3_VERIFY_WINDOW", "13"))
OBJECT_GAIN = float(_os.environ.get("V3_OBJECT_GAIN", "0.0"))
OBJECT_MIN_STEP = int(_os.environ.get("V3_OBJECT_MIN_STEP", "300"))
OBJECT_MAX_CELLS = int(_os.environ.get("V3_OBJECT_MAX_CELLS", "24"))
WALL_INSPECT_GAIN = float(_os.environ.get("V3_WALL_INSPECT_GAIN", "0.0"))
WALL_INSPECT_START = int(_os.environ.get("V3_WALL_INSPECT_START", "900"))
WALL_INSPECT_LEFT = int(_os.environ.get("V3_WALL_INSPECT_LEFT", "300"))
WALL_INSPECT_SEPARATION = float(
    _os.environ.get("V3_WALL_INSPECT_SEPARATION", "75.0"))
SEMANTIC_SHADOW_VIEWS = _os.environ.get("ENABLE_SEMANTIC_SHADOW_VIEWS", "0") == "1"
SEMANTIC_SHADOW_START_STEP = int(_os.environ.get("SEMANTIC_SHADOW_START_STEP", "1500"))
SEMANTIC_SWEEP_ROTATION = float(
    _os.environ.get("V3_SEMANTIC_SWEEP_ROTATION", "0.0")
)
RESCAN_AGE_RATIO = float(_os.environ.get("V3_RESCAN_AGE_RATIO", "0.55"))
RESCAN_MIN_AGE = int(_os.environ.get("V3_RESCAN_MIN_AGE", "130"))
SECTOR_FLEXIBILITY = float(_os.environ.get("V3_SECTOR_FLEX", "0.15"))
CENTER_DEAD_N = int(_os.environ.get("V3_CENTER_DEAD_N", "20"))
CENTER_STRICT_ARRIVAL = _os.environ.get("ENABLE_CENTER_STRICT_ARRIVAL", "0") == "1"
SCOUTS = {int(v) for v in _os.environ.get("V3_SCOUTS", "").split(",") if v.strip()}
SCOUT_UNTIL = int(_os.environ.get("V3_SCOUT_UNTIL", "3000"))
SCOUT_MIN = int(_os.environ.get("V3_SCOUT_MIN", "40"))
EXP_HYST = float(_os.environ.get("V3_EXP_HYST", "2000.0"))
GOAL_MAX = int(_os.environ.get("V3_GOAL_MAX", "300"))
UNK_W = float(_os.environ.get("V3_UNK_W", "0.0"))
FAN_W = float(_os.environ.get("V3_FAN_W", "0.0"))
FAN_STEPS = int(_os.environ.get("V3_FAN_STEPS", "300"))
RA_RETURN = _os.environ.get("V3_RA_RETURN", "1") == "1"
EXPECTED_BOMBS = int(_os.environ.get("V3_EXPECTED_BOMBS", "10"))
RA_IDLE_RESCAN = int(_os.environ.get("V3_RA_IDLE_RESCAN", "400"))
RA_RESCAN_LEN = int(_os.environ.get("V3_RA_RESCAN_LEN", "800"))
LAUNCH_STEPS = int(_os.environ.get("V3_LAUNCH_STEPS", "0"))
LAUNCH_BLEND = float(_os.environ.get("V3_LAUNCH_BLEND", "0.75"))
REGION_EXCLUSIVE = _os.environ.get("V3_REGION_EXCLUSIVE", "0") == "1"
REGION_EXCLUSIVE_START = int(
    _os.environ.get("V3_REGION_EXCLUSIVE_START", "1200"))
REGION_BLOCK = float(_os.environ.get("V3_REGION_BLOCK", "120.0"))
GOAL_STICKY = int(_os.environ.get("V3_GOAL_STICKY", "40"))
CLAIM_DELIVERY = float(_os.environ.get("V3_CLAIM_DELIVERY", "0.0"))
V4_PREDICTIVE_ASSIGNMENT = _os.environ.get("V4_PREDICTIVE_ASSIGNMENT", "0") == "1"
V4_MODEL_EMPTY = _os.environ.get("V4_MODEL_EMPTY", "0") == "1"
V4_MODEL_LEGAL_OBSERVER = _os.environ.get("V4_MODEL_LEGAL_OBSERVER", "0") == "1"
V5_DEADLINE_ASSIGNMENT = _os.environ.get("V5_DEADLINE_ASSIGNMENT", "0") == "1"
V5_TIME_HORIZON = int(_os.environ.get("V5_TIME_HORIZON", "0"))
V5_MODEL_BLEND = _os.environ.get("V5_MODEL_BLEND", "0") == "1"
V5_MODEL_SHADOW = _os.environ.get("V5_MODEL_SHADOW", "0") == "1"
V6_RAY_AUDIT = _os.environ.get("V6_RAY_AUDIT", "0") == "1"
V6_VIEW_PLAN = _os.environ.get("V6_VIEW_PLAN", "0") == "1"
V6_ROUTE_AUDIT = _os.environ.get("V6_ROUTE_AUDIT", "0") == "1"
V6_CONNECTED_REPAIR = _os.environ.get("V6_CONNECTED_REPAIR", "0") == "1"
V6_TASK_POLICY = _os.environ.get("V6_TASK_POLICY", "0") == "1"
V6_BRAKE_SHADOW = _os.environ.get("V6_BRAKE_SHADOW", "0") == "1"
V7_RAY_AUDIT = any(_os.environ.get(name, "0") == "1" for name in (
    "V7_RAY_AUDIT", "V7_GATEWAY_SHADOW", "V7_GATEWAY", "V7_PROBE",
    "V8_MACRO_SHADOW", "V8_MACRO_ACTIVE"))
EXP_DIST_POW = float(_os.environ.get("V3_EXP_DIST_POW", "1.0"))
NARROW_CAP = int(_os.environ.get("V3_NARROW_CAP", "10"))
FRONTIER = _os.environ.get("V3_FRONTIER", "0") == "1"  # tested: worse
FRONTIER_D = float(_os.environ.get("V3_FRONTIER_D", "70.0"))
NARROW_PEN = float(_os.environ.get("V3_NARROW_PEN", "4000.0"))
PEER_GOAL_R = float(_os.environ.get("V3_PEER_GOAL_R", "260.0"))
DEPART_GAP = float(_os.environ.get("V3_DEPART_GAP", "0.0"))

# ---- deep exploration (endgame): after the normal coverage is exhausted and
# fewer than EXPECTED_BOMBS bombs were picked up, visit every concave pocket
# (corner, wedge, dead end, wall bump) that no drone has seen from close range.
DEEP = _os.environ.get("V3_DEEP", "1") == "1"
DEEP_START_LEFT = int(_os.environ.get("V3_DEEP_START_LEFT", "60"))
DEEP_MIN_STEP = int(_os.environ.get("V3_DEEP_MIN_STEP", "300"))
DEEP_MIN_FREE = int(_os.environ.get("V3_DEEP_MIN_FREE", "200"))
DEEP_ENCLOSE = int(_os.environ.get("V3_DEEP_ENCLOSE", "9"))     # of 16 rays
DEEP_RAY_CELLS = int(_os.environ.get("V3_DEEP_RAY_CELLS", "4"))  # 40 px
DEEP_GAIN = float(_os.environ.get("V3_DEEP_GAIN", "2500.0"))
DEEP_MARK = float(_os.environ.get("V3_DEEP_MARK", "55.0"))
DEEP_STAND = float(_os.environ.get("V3_DEEP_STAND", "18.0"))
WIDE_CLAIM = _os.environ.get("V3_WIDE_CLAIM", "1") == "1"
WIDE_RADIUS = int(_os.environ.get("V3_WIDE_RADIUS", "6"))
WIDE_PEN = float(_os.environ.get("V3_WIDE_PEN", "200.0"))
SAME_SIDE_COS = float(_os.environ.get("V3_SAME_SIDE_COS", "0.6"))
VIEW_FINAL_R = float(_os.environ.get("V3_VIEW_FINAL_R", "140.0"))
# v2 local safety layer.  It is intentionally mild and can be disabled for an
# ablation with V2_SAFE_CONTROL=0 without changing task allocation.
V2_SAFE_CONTROL = _os.environ.get("V2_SAFE_CONTROL", "0") == "1"
V2_HEADING_ALIGN = _os.environ.get("V2_HEADING_ALIGN", "0") == "1"
V2_GRASP_HEADINGS = int(_os.environ.get("V2_GRASP_HEADINGS", "16"))
AUCTION_WEIGHT = float(_os.environ.get("V2_AUCTION_WEIGHT", "2.0"))

# Process-local registry used only by the benchmark harness.  It contains
# controller-owned counters and never exposes simulator truth to control().
_ACTIVE_DRONES = []


def diagnostics_snapshot():
    """Return aggregate controller telemetry for the current simulation."""
    drones = list(_ACTIVE_DRONES)
    calls = sum(int(getattr(d, "stats", {}).get("control_calls", 0)) for d in drones)
    total_us = sum(float(getattr(d, "stats", {}).get("control_us", 0.0)) for d in drones)
    max_us = max((float(getattr(d, "stats", {}).get("control_max_us", 0.0)) for d in drones), default=0.0)
    samples = [v for d in drones for v in getattr(d, "stats", {}).get("control_samples_us", ())]
    corridor_stats = [getattr(getattr(d, "corridor_explorer", None), "stats", {})
                      for d in drones]
    return {
        "drone_count": len(drones),
        "control_calls": calls,
        "control_mean_ms": (total_us / calls / 1000.0) if calls else 0.0,
        "control_max_ms": max_us / 1000.0,
        "control_p95_ms": (float(np.percentile(samples, 95)) / 1000.0) if samples else 0.0,
        "stuck_events": sum(int(getattr(d, "stats", {}).get("stuck", 0)) for d in drones),
        "grasp_failures": sum(int(getattr(d, "stats", {}).get("grasp_failures", 0)) for d in drones),
        "grasp_time_steps": sum(int(getattr(d, "stats", {}).get("grasp_steps", 0)) for d in drones),
        "grasp_successes": sum(int(getattr(d, "stats", {}).get("grasps", 0)) for d in drones),
        "recovery_events": sum(1 for d in drones for e in getattr(d, "task_events", []) if e.get("kind") == "stuck"),
        "collision_damage_events": sum(max(0, 25 - int(getattr(d, "drone_health", 25))) for d in drones),
        "event_counts": dict(Counter(e.get("kind") for d in drones for e in getattr(d, "task_events", []))),
        "all_discovered_step": min((int(d.stats["all_discovered_step"]) for d in drones
                                     if d.stats.get("all_discovered_step") is not None), default=None),
        "explore_goal_selections": sum(int(d.stats.get("explore_goal_selections", 0)) for d in drones),
        "repeated_exploration_ratio": (
            sum(int(d.stats.get("repeated_explore_goals", 0)) for d in drones)
            / max(1, sum(int(d.stats.get("explore_goal_selections", 0)) for d in drones))
        ),
        "corridor_entered": sum(int(s.get("entered", 0)) for s in corridor_stats),
        "corridor_ends_reached": sum(int(s.get("ends_reached", 0)) for s in corridor_stats),
        "corridor_closed_ends": sum(int(s.get("closed_ends", 0)) for s in corridor_stats),
        "corridor_branches_found": sum(int(s.get("branches_found", 0)) for s in corridor_stats),
        "corridor_branches_entered": sum(int(s.get("branches_entered", 0)) for s in corridor_stats),
        "corridor_completed": sum(int(s.get("completed", 0)) for s in corridor_stats),
        "corridor_approach_failures": sum(int(s.get("approach_failures", 0)) for s in corridor_stats),
        "unexplained_reversals": sum(len(getattr(d, "corridor_reversal_traces", ()))
                                     for d in drones),
    }


def controller_state_snapshot():
    """Controller-owned final state for official-run failure diagnosis."""
    rows = []
    for drone in list(_ACTIVE_DRONES):
        if not getattr(drone, "ready", False):
            continue
        events = []
        for event in list(getattr(drone, "task_events", ()))[-16:]:
            events.append({key: value for key, value in event.items()
                           if isinstance(value, (str, int, float, bool)) or value is None})
        safety = getattr(drone, "safety_stats", {})
        corridor = getattr(getattr(drone, "corridor_explorer", None), "stats", {})
        traces = []
        for trace in list(getattr(drone, "corridor_reversal_traces", ()))[:12]:
            traces.append({key: value for key, value in trace.items()
                           if key != "remaining"})
        rows.append({
            "id": int(drone.me),
            "state": str(drone.state),
            "carrying": bool(drone.grasped_bombs()),
            "accounted": int(drone.know.accounted()),
            "known_bombs": int(len(drone.know.bombs)),
            "known_disposed": int(drone.know.disposed),
            "claim": None if drone.claim_key is None else str(drone.claim_key),
            "pose": None if drone.pose is None else [float(v) for v in drone.pose],
            "goal": None if drone.goal is None else [float(v) for v in drone.goal],
            "goal_kind": drone.goal_kind,
            "path_goal": (None if drone.path_goal is None
                          else str(drone.path_goal)),
            "path_points": int(len(drone.path)),
            "path_index": int(drone.path_i),
            "path_end": (None if not drone.path else
                         [float(v) for v in drone.path[-1]]),
            "stuck_count": int(drone.stuck_count),
            "grasps": int(drone.stats.get("grasps", 0)),
            "drops": int(drone.stats.get("drops", 0)),
            "final_timeouts": int(drone.stats.get("final_timeouts", 0)),
            "push_retries": int(drone.stats.get("push_retries", 0)),
            "explore_goal_selections": int(
                drone.stats.get("explore_goal_selections", 0)),
            "repeated_explore_goals": int(
                drone.stats.get("repeated_explore_goals", 0)),
            "safety": {key: int(value) for key, value in safety.items()},
            "v5_path_counts": dict(getattr(drone, "v5_path_counts", {})),
            "v6_route_counts": dict(getattr(drone, "v6_route_counts", {})),
            "v6_real_ray_cells": int(getattr(drone, "_v6_real_ray_cells", 0)),
            "v7_counts": dict(getattr(drone, "v7_counts", {})),
            "v8_counts": dict(getattr(drone, "v8_counts", {})),
            "v5_blend_frames": int(getattr(drone, "v5_blend_frames", 0)),
            "v5_shadow_frames": int(getattr(drone, "v5_shadow_frames", 0)),
            "v5_bid": (None if drone.know.v5_outgoing_bid is None
                       else {"protocol": 5,
                             "cost_frames": float(drone.know.v5_outgoing_bid.cost_frames),
                             "sent_step": int(drone.know.v5_outgoing_bid.sent_step)}),
            "cargo_model": (drone.cargo_identifier.diagnostics()
                            if getattr(drone, "cargo_identifier", None)
                            is not None else None),
            "corridor": {
                key: (None if isinstance(value, float) and not np.isfinite(value)
                      else int(value) if isinstance(value, (int, np.integer))
                      else float(value))
                for key, value in corridor.items()
            },
            "corridor_reasons": dict(getattr(drone, "corridor_reason_counts", {})),
            "reversal_traces": traces,
            "bomb_trace": list(getattr(drone, "bomb_trace", ()))[:96],
            "carry_trace": list(getattr(drone, "carry_trace", ())),
            "branch_trace": list(getattr(drone, "branch_trace", ())),
            "pose_trace": list(getattr(drone, "pose_trace", ())),
            "last_events": events,
        })
    return sorted(rows, key=lambda row: row["id"])

V_EMPTY = 11.5      # terminal speed at full force, empty (px / step)
V_CARRY = 3.8       # terminal speed at full force, carrying a bomb
DRONE_R = 16.4
GRASP_REACH = 33.0  # grasper radius 21.4 + bomb radius 12

# Loaded braking model. A carrier is ~3x slower than an empty drone, so a
# carrier that stops for every wall in front of it discharges the whole
# makespan at 1.5 px/step. These are the only two knobs of that model.
CARRY_DECEL = float(_os.environ.get("V3_CARRY_DECEL", "0.30"))
CARRY_GAP_PAD = float(_os.environ.get("V3_CARRY_PAD", "8.0"))
# Clearance-adaptive braking: below CARRY_TIGHT px of free space in front the
# carrier brakes as cautiously as before, above CARRY_OPEN px it uses the full
# relaxed value, linearly in between.
ADAPTIVE_CARRY = _os.environ.get("V3_ADAPTIVE_CARRY", "0") == "1"
# How far from the disposal centre the final push may start. Semantic rays
# reach 200 px and the push is steered on relative hits, so starting from
# the edge of that range is safe; too small it and a carrier whose arrival
# cell is recorded further out never engages and oscillates instead.
PUSH_RANGE = float(_os.environ.get("V3_PUSH_RANGE", "170.0"))
# How far short of the visible centre face a stalled carrier aims when the
# delivery field's arrival cells sit too far out.
CLOSE_IN = float(_os.environ.get("V3_CLOSE_IN", "70.0"))
CARRY_DECEL_TIGHT = float(_os.environ.get("V3_CARRY_DECEL_TIGHT", "0.18"))
CARRY_TIGHT = float(_os.environ.get("V3_CARRY_TIGHT", "45.0"))
CARRY_OPEN = float(_os.environ.get("V3_CARRY_OPEN", "90.0"))
# While the disposal centre is still unknown a carrier cannot deliver, and a
# fully loaded team explores at carry speed. Cap the number of simultaneous
# carriers so the rest keep the empty-drone speed (0 = no cap).
CARRIER_CAP = int(_os.environ.get("V3_CARRIER_CAP", "0"))
UNKNOWN_DRONE_CARRIER_PRIORITY = (
    _os.environ.get("V3_UNKNOWN_DRONE_CARRIER_PRIORITY", "0") == "1")
# While the disposal centre is unknown a carrier must find it before it can
# deliver. Lowering the travel price looked like the fix but drove carriers
# into cluttered wings (Map13: 30 -> 107 collisions), so the hunt runs on
# the normal price and this stays at 1.0. Kept as a measured, disabled knob.
HUNT_DIST = float(_os.environ.get("V3_HUNT_DIST", "1.0"))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v, lo=-1.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


class SictRescueDrone(DroneAbstract):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ready = False

    # ================================================================= setup
    def _setup(self):
        self.me = int(self.identifier or 0)
        self.grid = GridMap(self.size_area, self.me)
        self.v6_adapter = (V6Adapter(self.me, self.grid) if
                           (V6_RAY_AUDIT or V6_VIEW_PLAN or V6_ROUTE_AUDIT
                            or V6_CONNECTED_REPAIR or V6_TASK_POLICY
                            or V7_RAY_AUDIT) else None)
        self._v6_gps_outage = 0
        self._v6_real_ray_cells = 0
        self._v6_known_task_keys = frozenset()
        self._v6_failure_snapshot = ()
        self._v6_disposed_snapshot = 0
        self._v6_claim_snapshot = None
        self._v6_slot_snapshot = None
        self.know = Knowledge(self.me)
        self.pose = None
        self.heading = None
        self.vel = (0.0, 0.0)
        self.start_pose = None
        self.start_step = 0
        self.started_in_return = None
        self.state = "EXPLORE"
        self.goal = None
        self.goal_kind = None
        self.goal_step = -1000
        self.path = []
        self.path_i = 0
        self.path_goal = None
        self.path_step = -1000
        self.cum = None
        self.mcp = None
        self.cum_step = -1000
        self.center_mcp = None
        self.center_cum = None
        self.center_step = -1000
        self.claim_key = None
        self.claim_pos = None
        self.claim_cost = None
        self.claim_since = 0
        self._eta_estimator = CompletionTimeEstimator()
        self.v5_scheduler = DeadlineTaskScheduler(self.me)
        self._eta_hysteresis = AssignmentHysteresis()
        self._last_claim_eta = None
        self.model_empty_controller = PolePlacementVelocityController(
            closed_loop_pole=0.75)
        self.v5_blend = BoundedModelBlend(
            alpha=min(0.2, max(0.0, float(_os.environ.get("V5_BLEND_ALPHA", "0.1"))))
            if V5_MODEL_BLEND else 0.0)
        self.v5_blend_frames = 0
        self.v5_shadow_frames = 0
        self._v4_velocity_estimator = (
            LegalVelocityEstimator()
            if V4_MODEL_EMPTY and V4_MODEL_LEGAL_OBSERVER else None)
        self.v_control = (0.0, 0.0)
        self.v_identification = (0.0, 0.0)
        self.velocity_sigma = math.inf
        self.measurement_valid = False
        self.repick_until = -1
        self.final_since = None
        self.unseen_close = 0
        self.cargo_bearing = None
        self.cargo_try = 0
        self.push_since = None
        self.backoff_until = -1
        self.backoff_dir = None
        self.history = deque(maxlen=10)
        self.escape_until = -1
        self.escape_dir = None
        self.stuck_count = 0
        self.was_carrying = False
        self.last_pickup = None
        self.visible_bombs = []
        self.visible_drones = []
        self.center_hits = []
        self.prev_force = (0.0, 0.0)
        self.stats = {"stuck": 0, "grasps": 0, "drops": 0, "tombs_absent": 0,
                      "final_timeouts": 0, "push_retries": 0,
                      "control_calls": 0, "control_us": 0.0,
                      "control_samples_us": deque(maxlen=2048),
                      "all_discovered_step": None,
                      "explore_goal_selections": 0,
                      "repeated_explore_goals": 0,
                      "control_max_us": 0.0, "grasp_failures": 0,
                      "grasp_steps": 0}
        self.task_events = []      # read by the benchmark diagnostics only
        self._last_health = int(self.drone_health)
        self.prog_best = None
        self.prog_step = 0
        self.interest_left = None
        self.goal_eval_step = -1000
        self.carry_goal = None
        self.carry_goal_step = -1000
        # Legal odometry breadcrumbs collected only while approaching a
        # claimed bomb.  A bomb is often tucked behind a wall lip: the
        # loaded planner cannot immediately rediscover the narrow route by
        # which the unloaded drone entered, so first retrace that route.
        self.fetch_breadcrumbs = deque(maxlen=90)
        self.pickup_exit_until = -1
        self.pickup_exit_origin = None
        self.center_dead = 0
        self.last_team_busy = 0
        self.rescan_until = -1
        self.bad_goals = []
        self.cocarry_steps = 0
        self.arm_owner_since = None      # step the unique-owner check started
        self.arm_target = None
        self.no_grasp_until = -1
        self.yielded_cargo = False
        self.pickup_info = None
        self.goal_deep = False
        self.goal_object = False
        self.object_pick = False
        self.checked_objects = []
        self._explore_goal_history = deque(maxlen=32)
        self.claim_view = None
        self.claim_wide = False
        self.view_reached = False
        self.deep_pick = False
        self.deep_left = 0
        self.deep_on = False
        self.reach_free = 0
        self._pocket = None
        self._pocket_step = -1000
        self.ready = True
        _ACTIVE_DRONES.append(self)

    # ================================================================ radio
    def define_message_for_all(self):
        if not self.ready or self.pose is None:
            return None
        if self.state != "FETCH":
            self.know.v5_outgoing_bid = None
        claim = ((self.claim_pos, self.claim_cost)
                 if self.claim_pos is not None and self.state == "FETCH" else None)
        return self.know.message(int(self.elapsed_timestep), self.pose, self.vel,
                                 self.was_carrying, self.state, self.goal, claim,
                                 self.grid, int(self.drone_health),
                                 self.pickup_info if self.was_carrying else None)

    # ================================================================= pose
    def _update_pose(self):
        gps = self.measured_gps_position()
        compass = self.measured_compass_angle()
        odom = self.odometer_values()
        if getattr(self, "v8_route", None) is not None:
            self.v8_route.odometry_update(int(self.elapsed_timestep), odom)
        if getattr(self, "v7_ego_trail", None) is not None:
            self.v7_ego_trail.update(
                step=int(self.elapsed_timestep), odometry=odom,
                gps_valid=gps is not None)
        if self._v4_velocity_estimator is not None:
            legal_estimate = self._v4_velocity_estimator.update(
                odometry=odom, compass=compass, gps_position=gps)
            self.v_control = tuple(float(v) for v in legal_estimate.velocity)
        self.measurement_valid = False
        self.velocity_sigma = math.inf
        if self.pose is None:
            if gps is None:
                return
            self.pose = (float(gps[0]), float(gps[1]))
            self.heading = float(compass) if compass is not None else 0.0
            return
        x, y = self.pose
        h = self.heading
        if odom is not None:
            dist, alpha, theta = float(odom[0]), float(odom[1]), float(odom[2])
            vx, vy = dist * math.cos(h + alpha), dist * math.sin(h + alpha)
            if all(math.isfinite(value) for value in (dist, alpha, theta, vx, vy)):
                # This is the previous control interval's noisy odometry
                # displacement, used only for shadow identification.
                self.v_identification = (vx, vy)
                self.velocity_sigma = math.hypot(0.2, abs(dist) * math.radians(8.0))
                self.measurement_valid = True
            x, y = x + vx, y + vy
            h = wrap(h + theta)
            self.vel = (0.5 * self.vel[0] + 0.5 * vx, 0.5 * self.vel[1] + 0.5 * vy)
            if self._v4_velocity_estimator is None:
                self.v_control = self.vel
        if gps is not None:
            x += 0.12 * (float(gps[0]) - x)
            y += 0.12 * (float(gps[1]) - y)
        if compass is not None:
            h = wrap(h + 0.15 * wrap(float(compass) - h))
        self.pose, self.heading = (x, y), h

    # ========================================================== perception
    def _perceive(self, step):
        self.visible_bombs, self.visible_drones, self.center_hits = [], [], []
        dynamic_objects = []   # (relative angle, distance) of non-wall returns
        hits = self.semantic_values() or []
        lid, lid_a = self._lidar()
        bomb_raw = []
        for hit in hits:
            d, a = float(hit.distance), float(hit.angle)
            if not math.isfinite(d) or not math.isfinite(a) or d <= 0:
                continue
            world_a = self.heading + a
            ux, uy = math.cos(world_a), math.sin(world_a)
            if hit.entity_type == TYPE.BOMB:
                dynamic_objects.append((a, d, 14.0))
                if not hit.grasped:
                    bomb_raw.append(((self.pose[0] + (d + 11.0) * ux,
                                      self.pose[1] + (d + 11.0) * uy), d))
            elif hit.entity_type == TYPE.DRONE:
                dynamic_objects.append((a, d, 20.0))
                self.visible_drones.append((self.pose[0] + (d + DRONE_R) * ux,
                                            self.pose[1] + (d + DRONE_R) * uy, d))
            elif hit.entity_type == TYPE.DISPOSAL_CENTER:
                # The centre is a physical body, so the lidar must see it at
                # the same range. A shorter lidar return means a wall is in
                # between (a semantic ray slipped through a wall joint).
                if lid is not None:
                    k = int(np.argmin(np.abs((lid_a - a + math.pi) % (2 * math.pi) - math.pi)))
                    if lid[k] < d - 12.0:
                        continue
                self.center_hits.append((self.pose[0] + (d + 4.0) * ux,
                                         self.pose[1] + (d + 4.0) * uy, d))
        # Cluster bomb rays (one bomb is often hit by 2-3 rays).
        clusters = []
        for pos, d in sorted(bomb_raw, key=lambda item: item[1]):
            for c in clusters:
                if math.dist(c[0], pos) < 20.0:
                    n = c[2] + 1
                    c[0] = ((c[0][0] * c[2] + pos[0]) / n, (c[0][1] * c[2] + pos[1]) / n)
                    c[2] = n
                    break
            else:
                clusters.append([pos, d, 1])
        for pos, d, _ in clusters:
            self.visible_bombs.append((pos, d))
            if d < 190:
                before = set(self.know.bombs)
                key = self.know.observe_bomb(
                    pos, step, 1.0 / (1.0 + d / 40.0),
                    view=(self.pose[0], self.pose[1], d),
                )
                if key is not None and key not in before:
                    if not hasattr(self, "bomb_trace"):
                        self.bomb_trace = []
                    self.bomb_trace.append({
                        "kind": "first_seen", "step": int(step),
                        "key": str(key),
                        "bomb": [round(float(v), 2) for v in pos],
                        "view": [round(float(v), 2) for v in self.pose],
                        "distance": round(float(d), 2),
                    })
        # Arrival points: own poses from which the centre was seen up close.
        # They have line of sight to the centre and are reachable by
        # construction (unlike surface points, which can sit behind a wall).
        if self.center_hits:
            self.grid.add_seen_points([(x, y) for x, y, _ in self.center_hits])
        close = [h for h in self.center_hits
                 if h[2] < (55.0 if CENTER_STRICT_ARRIVAL else 100.0)]
        if close and any(self._center_passable((h[0], h[1])) for h in close):
            self.grid.add_center_points([self.pose])
        # Peers known by radio but not seen semantically are also not walls.
        for peer in self.know.peers.values():
            p = peer.get("pos")
            if p is None:
                continue
            dx, dy = p[0] - self.pose[0], p[1] - self.pose[1]
            d = math.hypot(dx, dy)
            if 5.0 < d < 300.0:
                dynamic_objects.append((wrap(math.atan2(dy, dx) - self.heading),
                                        d - DRONE_R, 24.0))
        return dynamic_objects

    def _center_passable(self, hit, stop=22.0, min_clear=13.0):
        """Can a drone body fly straight from here to ``hit`` (a centre point)?

        The last ``stop`` px are excluded (the centre itself is an obstacle on
        the map). A view through a slit narrower than the hull fails.
        """
        dx, dy = hit[0] - self.pose[0], hit[1] - self.pose[1]
        d = math.hypot(dx, dy)
        if d <= stop + 8.0 or self.grid.clearance is None:
            return True
        ux, uy = dx / d, dy / d
        a = (self.pose[0] + 8.0 * ux, self.pose[1] + 8.0 * uy)
        b = (self.pose[0] + (d - stop) * ux, self.pose[1] + (d - stop) * uy)
        return self.grid.segment_clear(a, b, min_clear)

    def _lidar(self):
        values = self.lidar_values()
        angles = self.lidar_rays_angles()
        if values is None or angles is None:
            return None, None
        return np.asarray(values, float), np.asarray(angles, float)

    def _map_update(self, step, dynamic_objects, lidar, angles):
        if lidar is None:
            return
        mask = np.zeros(lidar.shape, bool)
        for a, d, half in dynamic_objects:
            width = math.atan2(half + 6.0, max(d, 1.0)) + 0.04
            rel = np.abs((angles - a + math.pi) % (2 * math.pi) - math.pi)
            mask |= (rel < width) & (lidar > d - 25.0) & (lidar < d + 45.0)
        self.grid.integrate(self.pose, self.heading, lidar, angles, mask, step)
        if (V6_RAY_AUDIT or V6_VIEW_PLAN or V7_RAY_AUDIT) and self.v6_adapter is not None:
            gps = self.measured_gps_position()
            self._v6_gps_outage = 0 if gps is not None else self._v6_gps_outage + 1
            pose_ok = (self.pose is not None and self.heading is not None
                       and all(math.isfinite(v) for v in (*self.pose, self.heading))
                       and ((gps is not None and math.dist(self.pose, gps) < 30.0)
                            or (gps is None and self._v6_gps_outage <= 40
                                and self.odometer_values() is not None)))
            new_cells = self.v6_adapter.observe_semantic_rays(
                step=step, pose=self.pose, heading=self.heading,
                lidar_distances=lidar, lidar_angles=angles,
                pose_trustworthy=pose_ok)
            if new_cells:
                self._v6_real_ray_cells += new_cells
                self._event("v6_real_ray_cells", step, new_cells=new_cells,
                            gps_valid=gps is not None)

    # ============================================================= planning
    def _plan_kind(self):
        if self.was_carrying:
            return "carry"
        return "explore" if (EXPLORE_KIND and self.state == "EXPLORE") else "tight"

    def _traffic_spots(self):
        """Peers' bodies and their next route segment as soft obstacles."""
        spots = []
        if not self.cum_step:
            pass
        for pid, peer in self.know.peers.items():
            p = peer.get("pos")
            if p is None:
                continue
            carrier = bool(peer.get("carry"))
            # Yield to carriers: their bomb hangs off the hull and a bump
            # costs them far more than a detour costs us.
            weight = 3.0 if carrier else 1.5
            radius = 24.0 if carrier else 18.0
            spots.append((p[0], p[1], radius, weight))
            goal = peer.get("goal")
            if goal is None:
                continue
            dx, dy = goal[0] - p[0], goal[1] - p[1]
            n = math.hypot(dx, dy)
            if n < 1.0:
                continue
            reach = min(n, 140.0)
            for k in range(1, 5):
                t = reach * k / 4.0
                spots.append((p[0] + dx / n * t, p[1] + dy / n * t,
                              radius, weight * 0.6))
        return spots

    def _refresh_dijkstra(self, step, force=False):
        if self.grid.costs is None:
            return
        # Event-driven: only replan when costs changed (fused_step), or periodically as fallback.
        if not force and step - self.cum_step < 10:
            return
        if not force and step - self.cum_step < 40 and self.grid.fused_step <= self.cum_step:
            return
        self.grid.set_traffic(self._traffic_spots())
        self.grid._rebuild_costs(step)
        kind = self._plan_kind()
        start = self.grid.free_start(self.grid.costs[kind], *self.pose)
        if start is None:
            self.cum, self.mcp = None, None
            return
        self.mcp, self.cum = self.grid.dijkstra(kind, [start])
        self.cum_step = step

    def _cost_to(self, pos, radius_cells=3):
        """Path cost (px) to reach within ``radius_cells`` of ``pos``."""
        if self.cum is None:
            return math.inf, None
        iy, ix = self.grid.cell(*pos)
        y0, y1 = max(0, iy - radius_cells), min(self.grid.ny, iy + radius_cells + 1)
        x0, x1 = max(0, ix - radius_cells), min(self.grid.nx, ix + radius_cells + 1)
        window = self.cum[y0:y1, x0:x1]
        if not np.isfinite(window).any():
            return math.inf, None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        total = window + np.hypot(yy - iy, xx - ix) * 0.8
        k = np.unravel_index(int(np.argmin(total)), total.shape)
        return float(total[k]) * CELL, (int(yy[k]), int(xx[k]))

    def _set_path_to(self, goal, step, end_cell=None, wide=False):
        if self.mcp is None:
            self.path = []
            return
        if end_cell is None:
            _, end_cell = self._cost_to(goal)
        if end_cell is None and wide:
            # Blocked bomb claims only: carriers and explorers keep the base
            # behaviour (an empty path), a 60 px end cell can lie behind a
            # wall and a loaded detour costs hundreds of steps (Map13).
            _, end_cell = self._cost_to(goal, WIDE_RADIUS)
        if end_cell is None:
            self.path = []
            return
        self.path = self.grid.path_from(self.mcp, end_cell)
        if self.path:
            self.path[-1] = goal if math.dist(self.path[-1], goal) < 40 else self.path[-1]
        self.path_i = 0
        self.path_goal = goal
        self.path_step = step

    def _refresh_center_field(self, step):
        if self.grid.costs is None or not (self.grid.center_points or self.grid.seen_points):
            self.center_mcp, self.center_cum = None, None
            return
        if self.center_cum is not None and step - self.center_step < 20:
            return
        cost = self.grid.costs["carry"]
        # Arrival zone only: the final push is steered by live (relative)
        # semantic hits, so global pose error near the centre does not matter.
        zone = self.grid.center_mask
        goals = np.argwhere(zone & np.isfinite(cost))
        if goals.size == 0 and zone.any():
            zone = ndimage.binary_dilation(zone, iterations=2)
            goals = np.argwhere(zone & np.isfinite(cost))
        if goals.size == 0 and self.grid.seen_points:
            # Only far sightings so far: aim at known-free cells near them.
            # Getting there brings the centre into close range, which creates
            # real (passable) arrival points.
            near_seen = ndimage.binary_dilation(
                self.grid.seen_mask,
                iterations=4 if CENTER_STRICT_ARRIVAL else 9)
            bad = np.zeros_like(near_seen)
            for c in self.grid.center_bad:
                if 0 <= c[0] < bad.shape[0] and 0 <= c[1] < bad.shape[1]:
                    bad[c] = True
            goals = np.argwhere(near_seen & ~bad & (self.grid.occ < -0.6) & np.isfinite(cost))
        if goals.size == 0:
            self.center_mcp, self.center_cum = None, None
            return
        self.center_mcp, self.center_cum = self.grid.dijkstra(
            "carry", [tuple(g) for g in goals])
        self.center_step = step

    def _center_path(self, step):
        self._refresh_center_field(step)
        if self.center_mcp is None:
            return []
        start = self.grid.free_start(self.grid.costs["carry"], *self.pose, radius=5)
        if start is None or not np.isfinite(self.center_cum[start]):
            return []
        cells = self.grid.path_from(self.center_mcp, start)
        return list(reversed(cells))

    def _waypoint(self, min_clear=14.0, reach=110.0):
        if not self.path:
            return None
        n = len(self.path)
        best_i, best_d = self.path_i, math.inf
        for i in range(self.path_i, min(n, self.path_i + 40)):
            d = math.dist(self.pose, self.path[i])
            if d < best_d:
                best_i, best_d = i, d
        self.path_i = best_i
        # Inside a squeeze, accept segments as tight as where we already are.
        here = self.grid.clearance_at(*self.pose)
        min_clear = min(min_clear, max(9.0, here - 2.0))
        target = self.path[min(n - 1, best_i + 2)]
        for i in range(best_i, n):
            p = self.path[i]
            if math.dist(self.pose, p) > reach:
                break
            if self.grid.segment_clear(self.pose, p, min_clear):
                target = p
        return target

    # ===================================================== deep exploration
    def _pockets(self):
        """Concave pockets of the fused map: corners, wedges, dead ends.

        Sixteen 40 px rays are cast from every lidar-free cell. A cell is a
        pocket when most rays hit a wall (>= DEEP_ENCLOSE of 16) and the open
        rays form a single arc: a corridor has two opposite open arcs and is
        swept by anyone flying through it, a pocket is only resolved by a
        drone that looks *into* it. A bomb that lidar mistook for a wall bump
        turns its neighbourhood into a pocket as well.
        """
        g = self.grid
        if self._pocket is not None and self._pocket_step == g.fused_step:
            return self._pocket
        obst = g.obstacle
        free = g.occ < -0.6
        ny, nx = obst.shape
        pad = DEEP_RAY_CELLS + 1
        padded = np.pad(obst, pad, constant_values=True)
        rays = np.zeros((16, ny, nx), bool)
        for k in range(16):
            a = 2.0 * math.pi * k / 16.0
            hit = rays[k]
            for r in range(1, DEEP_RAY_CELLS + 1):
                dy, dx = int(round(r * math.sin(a))), int(round(r * math.cos(a)))
                hit |= padded[pad + dy:pad + dy + ny, pad + dx:pad + dx + nx]
        hits = rays.sum(axis=0)
        open_ = ~rays
        arcs = (open_ & ~np.roll(open_, 1, axis=0)).sum(axis=0)
        pocket = free & (hits >= DEEP_ENCLOSE) & (hits < 16) & (arcs == 1)
        self._pocket, self._pocket_step = pocket, g.fused_step
        return pocket

    def _picked_count(self):
        return sum(1 for key in self.know.pickups if key not in self.know.drops)

    def _deep_wanted(self, cand_any, step):
        if not DEEP or self.start_pose is None:
            return False
        # Only once the normal sweep is over: an empty map at the start also
        # has "no interest left", so require time and a mapped area.
        if step - self.start_step < DEEP_MIN_STEP or (self.reach_free or 0) < DEEP_MIN_FREE:
            return False
        if EXPECTED_BOMBS > 0 and self._picked_count() >= EXPECTED_BOMBS:
            return False
        if not cand_any:
            return True
        return self.interest_left is not None and self.interest_left <= DEEP_START_LEFT

    def _deep_candidates(self):
        """Standing cells that look into a pocket nobody has seen up close."""
        g = self.grid
        todo = self._pockets() & (g.near_f == 0)
        if not todo.any():
            return None, None
        count = ndimage.uniform_filter(todo.astype(np.float32), size=9) * 81.0
        clear = np.nan_to_num(g.clearance, posinf=99.0)
        cand = ((count > 0.5) & np.isfinite(self.cum) & (g.occ < -0.6)
                & (clear >= DEEP_STAND))
        gain = DEEP_GAIN * np.minimum(1.0, count / 4.0)
        return cand, gain

    def _object_candidates(self, step):
        """Safe viewpoints around small unclassified LiDAR obstacle blobs.

        An unseen bomb is initially indistinguishable from a wall in LiDAR.
        Small isolated components are therefore inspection targets until a
        drone has reached their observation ring.  Large wall components are
        excluded, and checked viewpoints are remembered to prevent loops.
        """
        if (OBJECT_GAIN <= 0.0 or step - self.start_step < OBJECT_MIN_STEP
                or self.grid.obstacle is None):
            return None, None
        g = self.grid
        labels, count = ndimage.label(g.obstacle)
        if count <= 0:
            return None, None
        sizes = np.bincount(labels.ravel())
        small_ids = np.flatnonzero(
            (sizes >= 1) & (sizes <= OBJECT_MAX_CELLS))
        small_ids = small_ids[small_ids != 0]
        if small_ids.size == 0:
            return None, None
        objects = np.isin(labels, small_ids)
        outer = ndimage.binary_dilation(objects, iterations=7)
        inner = ndimage.binary_dilation(objects, iterations=2)
        free = g.occ < -0.6
        clear = np.nan_to_num(g.clearance, posinf=99.0)
        cand = outer & ~inner & free & np.isfinite(self.cum) & (clear >= 18.0)
        for checked in self.checked_objects:
            cand &= np.hypot(g.X - checked[0], g.Y - checked[1]) >= 90.0
        if not cand.any():
            return None, None
        distance = ndimage.distance_transform_edt(~objects) * CELL
        gain = OBJECT_GAIN * np.clip(1.0 - np.abs(distance - 42.0) / 45.0,
                                     0.0, 1.0)
        return cand, gain

    def _wall_inspection_candidates(self, step):
        """Unvisited safe standing cells along obstacle faces.

        This is the second semantic pass for targets whose LiDAR body merged
        into a wall component.  It measures actual drone approach coverage,
        not ray coverage, and therefore also works with sparse semantic rays.
        """
        if (WALL_INSPECT_GAIN <= 0.0
                or step - self.start_step < WALL_INSPECT_START
                or self.interest_left is None
                or self.interest_left > WALL_INSPECT_LEFT):
            return None, None
        g = self.grid
        free = g.occ < -0.6
        clear = np.nan_to_num(g.clearance, posinf=999.0)
        distance_to_visit = ndimage.distance_transform_edt(
            g.visit_f == 0) * CELL
        cand = (free & np.isfinite(self.cum)
                & (clear >= 24.0) & (clear <= 65.0)
                & (distance_to_visit >= WALL_INSPECT_SEPARATION))
        if not cand.any():
            return None, None
        gain = WALL_INSPECT_GAIN * np.clip(
            distance_to_visit / (2.0 * WALL_INSPECT_SEPARATION), 0.0, 1.0)
        return cand, gain

    def _shadow_view_candidates(self, step):
        """Reachable fresh viewpoints beside nearby unscanned wall edges."""
        if (not SEMANTIC_SHADOW_VIEWS
                or step - self.start_step < SEMANTIC_SHADOW_START_STEP
                or self.pose is None or self.grid.clearance is None):
            return None, None
        g = self.grid
        free = g.occ < -0.6
        boundary = g.obstacle & ndimage.binary_dilation(free, iterations=1)
        shadow = boundary & (g.near_f == 0)
        if not shadow.any():
            return None, None
        mass = ndimage.uniform_filter(shadow.astype(np.float32), size=11) * 121.0
        goal_distance = np.full(g.occ.shape, np.inf, dtype=np.float32)
        for cell in self._explore_goal_history:
            point = g.world(*cell)
            goal_distance = np.minimum(
                goal_distance, np.hypot(g.X - point[0], g.Y - point[1]))
        clearance = np.nan_to_num(g.clearance, posinf=999.0)
        local_distance = np.hypot(g.X - self.pose[0], g.Y - self.pose[1])
        cand = (free & np.isfinite(self.cum) & (clearance >= 20.0)
                & (clearance <= 75.0) & (goal_distance >= 35.0)
                & (local_distance >= 28.0) & (local_distance <= 125.0)
                & (mass >= 0.5))
        if not cand.any():
            return None, None
        gain = 3000.0 * np.minimum(1.0, mass / 3.0)
        return cand, gain

    def _deep_goal_done(self, goal):
        g = self.grid
        iy, ix = g.cell(*goal)
        y0, y1 = max(0, iy - 4), min(g.ny, iy + 5)
        x0, x1 = max(0, ix - 4), min(g.nx, ix + 5)
        todo = self._pockets()[y0:y1, x0:x1] & (g.near_f[y0:y1, x0:x1] == 0)
        return not todo.any()

    # ========================================================== exploration
    def _explore_target(self, step, keep_cell=None, dist_scale=1.0):
        if self.cum is None:
            return None
        g = self.grid
        # Interest = lidar-confirmed free space not yet semantically scanned,
        # plus real frontiers (unknown next to known free). Unknown cells that
        # never border free space (inside solid blocks) are never targets.
        free = g.occ < -0.6
        unknown = ~g.obstacle & ~free
        frontier = unknown & ndimage.binary_dilation(free)
        free_like = free
        interest = ((g.scan_f == 0) & free) | frontier
        if FRONTIER:
            # Distance to the nearest cell that still needs a semantic look.
            if interest.any():
                to_interest = ndimage.distance_transform_edt(~interest) * CELL
                reach = np.where(np.isfinite(self.cum), to_interest, 9999.0)
                gain = 1.0 / (1.0 + reach / FRONTIER_D)
                cand = np.isfinite(self.cum) & (free | frontier)
            else:
                gain = np.zeros_like(self.cum)
                cand = np.zeros_like(self.cum, dtype=bool)
        else:
            gain = ndimage.uniform_filter(interest.astype(np.float32), size=9)
            cand = (gain > 0.10) & np.isfinite(self.cum) & (free | frontier)
        self.interest_left = int((interest & np.isfinite(self.cum)).sum())
        self.reach_free = int((free & np.isfinite(self.cum)).sum())
        if UNK_W > 0 and not FRONTIER:
            # A frontier that opens onto a large unknown region is worth more
            # than its thin line of frontier cells: value the unknown mass
            # behind it. Solid-block interiors never border free space, so
            # they cannot attract drones this way.
            unk_mass = ndimage.uniform_filter(unknown.astype(np.float32), size=15)
            near_front = ndimage.binary_dilation(frontier, iterations=2)
            gain = gain + UNK_W * np.where(near_front, unk_mass, 0.0)
        score_gain = EXP_GAIN * gain
        deep_cells = None
        self.deep_on = self._deep_wanted(bool(cand.any()), step)
        self.deep_left = 0
        if self.deep_on:
            dcand, dgain = self._deep_candidates()
            if dcand is not None and dcand.any():
                self.deep_left = int(dcand.sum())
                score_gain = np.where(dcand, np.maximum(score_gain, dgain), score_gain)
                cand = cand | dcand
                deep_cells = dcand
        object_cells = None
        if EXPECTED_BOMBS <= 0 or self._picked_count() < EXPECTED_BOMBS:
            ocand, ogain = self._object_candidates(step)
            if ocand is not None and ocand.any():
                score_gain = np.where(
                    ocand, np.maximum(score_gain, ogain), score_gain)
                cand = cand | ocand
                object_cells = ocand
        if EXPECTED_BOMBS <= 0 or self._picked_count() < EXPECTED_BOMBS:
            wcand, wgain = self._wall_inspection_candidates(step)
            if wcand is not None and wcand.any():
                score_gain = np.where(
                    wcand, np.maximum(score_gain, wgain), score_gain)
                cand = cand | wcand
            scand, sgain = self._shadow_view_candidates(step)
            if scand is not None and scand.any():
                score_gain = np.where(scand, np.maximum(score_gain, sgain), score_gain)
                cand = cand | scand
        verify_on = (
            VERIFY_GAIN > 0.0
            and step - self.start_step >= VERIFY_START
            and self.interest_left is not None
            and self.interest_left <= VERIFY_START_LEFT
            and (EXPECTED_BOMBS <= 0 or self._picked_count() < EXPECTED_BOMBS)
        )
        if verify_on:
            close_unseen = free & (g.near_f == 0)
            close_mass = ndimage.uniform_filter(
                close_unseen.astype(np.float32), size=VERIFY_WINDOW
            ) * float(VERIFY_WINDOW * VERIFY_WINDOW)
            verify_cand = (
                (close_mass > 0.5) & free & np.isfinite(self.cum)
            )
            if verify_cand.any():
                verify_gain = VERIFY_GAIN * np.minimum(1.0, close_mass / 4.0)
                score_gain = np.where(
                    verify_cand, np.maximum(score_gain, verify_gain), score_gain
                )
                cand = cand | verify_cand
        if not cand.any():
            # Everything scanned once but the round is still running, so a
            # bomb is still out there (missed between rays, or moved).
            # Re-scan the areas whose last scan is oldest *relative* to the
            # oldest one, so revisits sweep the whole map, not the start area.
            age = step - g.scan_f.astype(np.int32)
            reach = free_like & (g.scan_f > 0) & np.isfinite(self.cum)
            if not reach.any():
                return None
            oldest = int(age[reach].max())
            stale = reach & (age >= max(RESCAN_MIN_AGE, int(RESCAN_AGE_RATIO * oldest)))
            gain = ndimage.uniform_filter(stale.astype(np.float32), size=9)
            score_gain = 0.6 * EXP_GAIN * gain + 1.0 * age
            if CORNER_BONUS > 0 and g.clearance is not None:
                clearance_val = np.nan_to_num(g.clearance, nan=0.0, posinf=99.0)
                narrow_rescan = (clearance_val < 35.0) & stale
                score_gain = np.where(narrow_rescan, score_gain + CORNER_BONUS * 0.5, score_gain)
            cand = (gain > 0.10) & np.isfinite(self.cum)
            if not cand.any():
                return None
        if EXP_DIST_POW == 1.0:
            travel = dist_scale * EXP_DIST * self.cum * CELL
        else:
            # Concave distance term: a distant area is still worth going to,
            # which is what stops the team from milling around the dock.
            travel = (dist_scale * 45.0 *
                      np.power(np.maximum(0.0, self.cum * CELL), EXP_DIST_POW))
        score = np.where(cand, score_gain - travel, -np.inf)
        if NARROW_CAP < 10:
            # Corridors narrower than ~2 hulls cannot pass two drones. Count
            # peers already inside such space and stop sending more there.
            busy = sum(1 for peer in self.know.peers.values()
                       if peer.get("pos") is not None
                       and self.grid.clearance_at(*peer["pos"]) < 16.0)
            narrow_cells = np.nan_to_num(g.clearance, posinf=99.0) < 16.0
            if busy >= NARROW_CAP:
                score = np.where(narrow_cells, score - NARROW_PEN, score)
        if EXP_NARROW:
            # Bombs are laid against walls: a cell that a drone must approach
            # from within 100 px to resolve is worth more than open floor.
            narrow = np.clip((34.0 - np.nan_to_num(g.clearance, posinf=99.0)) / 12.0, 0.0, 1.0)
            score = score + EXP_NARROW * narrow
        if CORNER_BONUS > 0 and g.clearance is not None:
            clearance_val = np.nan_to_num(g.clearance, nan=0.0, posinf=99.0)
            is_narrow = clearance_val < 40.0
            obstacle_density = ndimage.uniform_filter(g.obstacle.astype(np.float32), size=7)
            corner_metric = np.where(is_narrow, obstacle_density * (40.0 - clearance_val) / 40.0, 0.0)
            corner_metric = np.clip(corner_metric, 0.0, 1.0)
            score = score + CORNER_BONUS * corner_metric
        if (EDGE_BONUS > 0 and self.me in EDGE_SCOUTS and self.start_pose is not None
                and step-self.start_step>EDGE_START
                and (EXPECTED_BOMBS<=0 or self._picked_count()<EXPECTED_BOMBS)):
            # Bombs may legally sit against the outer wall. A distant semantic
            # sweep can mark nearby open floor while a wall hides the bomb in
            # a shallow pocket. Value reachable standing cells that are close
            # to still-unchecked boundary cells; this is geometry-derived and
            # has no map- or bomb-coordinate special case.
            edge_dist=np.minimum.reduce((g.X+g.width/2.,g.width/2.-g.X,
                                         g.Y+g.height/2.,g.height/2.-g.Y))
            edge_todo=((edge_dist<EDGE_RANGE)&(g.near_f==0)&~g.obstacle)
            edge_mass=ndimage.uniform_filter(edge_todo.astype(np.float32),size=17)*289.0
            score=score+EDGE_BONUS*np.minimum(1.0,edge_mass/3.0)
        # Far-first: a bomb far from the disposal centre needs the longest
        # (slow, loaded) delivery, so finding it early shortens the makespan.
        self._refresh_center_field(step)
        if self.center_cum is not None:
            far = np.where(np.isfinite(self.center_cum), self.center_cum * CELL, 0.0)
            score = score + 3000.0 * FAR_WEIGHT * (np.minimum(far, 3000.0) / 3000.0)
        # Sector ownership: drone i prefers the bearing 2*pi*i/10 seen from
        # the start area, so every wing (the maze for some ids) has an owner
        # from the first decision instead of being found last.
        # Hard sector ownership. Ten drones, ten equal 36-degree wedges seen
        # from the start area. A soft alignment bonus still let everyone pick
        # the big open hall; a hard mask guarantees the maze wing is searched
        # by its owner in the first pass.
        if self.start_pose is not None:
            sector = 2.0 * math.pi * (self.me % 10) / 10.0
            dx, dy = g.X - self.start_pose[0], g.Y - self.start_pose[1]
            norm = np.maximum(1.0, np.hypot(dx, dy))
            align = (dx * math.cos(sector) + dy * math.sin(sector)) / norm
            fan = 320.0
            if FAN_W > 0 and step - self.start_step < FAN_STEPS:
                # Opening fan-out: during the first steps the travel penalty
                # and the far-first bonus cancel out, so direction is decided
                # by this term. Make it strong, then fade it out.
                fan += FAN_W * (1.0 - (step - self.start_step) / FAN_STEPS)
            sector_bonus = fan * align
            if SECTOR_FLEXIBILITY > 0 and step > SCOUT_UNTIL:
                for offset in [-1, 1]:
                    adj_sector = (sector + offset * 2.0 * math.pi / 10.0) % (2.0 * math.pi)
                    adj_align = (dx * math.cos(adj_sector) + dy * math.sin(adj_sector)) / norm
                    sector_bonus = np.maximum(sector_bonus, fan * adj_align * SECTOR_FLEXIBILITY)
            score = score + sector_bonus
        if (REGION_EXCLUSIVE and step - self.start_step >= REGION_EXCLUSIVE_START
                and (EXPECTED_BOMBS <= 0 or self._picked_count() < EXPECTED_BOMBS)):
            # Communication can disappear exactly where the difficult final
            # pockets are.  Assign coarse spatial regions deterministically,
            # so independent controllers do not converge on the same frontier
            # merely because they cannot hear one another's goal lease.  This
            # is a geometry hash, not a map identity or hard-coded location.
            bx = np.floor((g.X + g.width / 2.0) / REGION_BLOCK).astype(np.int64)
            by = np.floor((g.Y + g.height / 2.0) / REGION_BLOCK).astype(np.int64)
            owner = np.mod(bx * 3 + by * 7, 10)
            owned = cand & (owner == self.me)
            # Never strand a drone when its share is temporarily exhausted.
            if owned.any():
                score = np.where(owned, score, -np.inf)
        for peer in self.know.peers.values():
            for key, weight, radius in (("goal", 700.0, PEER_GOAL_R), ("pos", 250.0, 150.0)):
                p = peer.get(key)
                if p is None:
                    continue
                d = np.hypot(g.X - p[0], g.Y - p[1])
                score -= weight * np.clip(1.0 - d / radius, 0.0, 1.0)
        self.bad_goals = [(p, until) for p, until in self.bad_goals if until > step]
        for (bx, by), _ in self.bad_goals:
            d = np.hypot(g.X - bx, g.Y - by)
            score = np.where(d < 80.0, score - 5000.0, score)
        if AUCTION_WEIGHT > 0.0 and self.pose is not None:
            # Apply a small deterministic CBBA-style tie break to the best
            # local candidates.  The image score remains dominant, so this
            # cannot pull a drone across the map merely to satisfy a lease.
            finite = np.isfinite(score)
            if finite.any():
                flat = score.ravel()
                top_n = min(12, int(finite.sum()))
                mask_flat = finite.ravel()
                top_idx = np.argpartition(np.where(mask_flat, flat, -np.inf), -top_n)[-top_n:]
                for flat_i in top_idx:
                    iy, ix = np.unravel_index(int(flat_i), score.shape)
                    p = g.world(int(iy), int(ix))
                    score[iy, ix] += AUCTION_WEIGHT * auction_utility(
                        self.me, p, self.pose, 0.0, self.know.peers)
        k = np.unravel_index(int(np.argmax(score)), score.shape)
        if not np.isfinite(score[k]):
            return None
        self.deep_pick = bool(deep_cells is not None and deep_cells[k])
        self.object_pick = bool(object_cells is not None and object_cells[k])
        if keep_cell is not None and EXP_HYST > 0:
            # Hysteresis: the score landscape shifts every time a peer moves
            # its goal, and re-picking the argmax made explorers turn around
            # every ~50 steps (0.5 px/step average). Keep the committed goal
            # unless something clearly better appeared.
            kept = score[keep_cell]
            if np.isfinite(kept) and kept + EXP_HYST >= score[k]:
                self.deep_pick = bool(deep_cells is not None and deep_cells[keep_cell])
                self.object_pick = bool(
                    object_cells is not None and object_cells[keep_cell])
                return g.world(*keep_cell), keep_cell
        return g.world(int(k[0]), int(k[1])), (int(k[0]), int(k[1]))

    # ============================================================ bomb tasks
    def _carrier_load(self):
        """Carriers known to the team right now (own body + fresh gossip).

        Claimed-but-not-yet-held bombs count too: every drone chooses its
        bomb on the first steps, long before anybody has grasped one, so a
        cap that only counted carrying bodies would let all ten claim first
        and then all ten end up loaded.
        """
        load = 1 if self.grasped_bombs() else 0
        for pid, peer in self.know.peers.items():
            if peer.get("carry") or peer.get("state") in ("FETCH", "CARRY"):
                load += 1
        for pid in self.know.claims:
            if pid in self.know.peers:
                continue
            load += 1
        return load

    def _carrier_slot_free(self):
        """May this drone pick up another bomb?

        While the disposal centre is unknown a carrier cannot deliver, and a
        fully loaded team explores at 3.8 px/step instead of 11.5. Keeping a
        few drones empty keeps the search fast. Once the centre is known the
        cap is lifted: delivery, not search, is then the bottleneck.
        """
        if CARRIER_CAP <= 0:
            return True
        if self.grid.center_points or self.grid.seen_points:
            return True
        # Endgame: with only a couple of bombs left the cap would leave them
        # unclaimed (the fast empty drones are the ones the cap holds back,
        # and a single late bomb costs the whole makespan).
        if EXPECTED_BOMBS > 0 and self.know.accounted() >= EXPECTED_BOMBS - 2:
            return True
        return self._carrier_load() < CARRIER_CAP

    def _choose_bomb(self, step):
        previous_claim = self.claim_key
        if V6_TASK_POLICY:
            task_keys = frozenset(self.know.bombs)
            slot_open = self._carrier_slot_free()
            failures = tuple(sorted((repr(k), int(v))
                                    for k, v in self.know.failures.items()))
            lease_conflict = (previous_claim in self.know.bombs and any(
                cpos is not None
                and math.dist(cpos, self.know.bombs[previous_claim]["pos"]) <= MATCH + 6
                for cpos, _, _ in self.know.claims.values()))
            task_event = (task_keys != self._v6_known_task_keys
                          or failures != self._v6_failure_snapshot
                          or self.know.disposed != self._v6_disposed_snapshot
                          or previous_claim != self._v6_claim_snapshot
                          or slot_open != self._v6_slot_snapshot
                          or lease_conflict)
            self._v6_known_task_keys = task_keys
            self._v6_failure_snapshot = failures
            self._v6_disposed_snapshot = self.know.disposed
            self._v6_claim_snapshot = previous_claim
            self._v6_slot_snapshot = slot_open
            if not task_event:
                return
        if V5_DEADLINE_ASSIGNMENT:
            self.know.v5_outgoing_bid = None
        if self.cum is None:
            return
        if not self._carrier_slot_free():
            # Another carrier slot is not available: keep scanning instead of
            # joining a convoy that cannot deliver yet.
            if self.claim_key is not None:
                self._drop_claim()
            return
        if (V4_PREDICTIVE_ASSIGNMENT or V5_DEADLINE_ASSIGNMENT) and (self.grid.center_points
                                         or self.grid.seen_points):
            self._refresh_center_field(step)
        best = None
        v5_options = []
        v6_options = []
        incumbent = None
        incumbent_eta = None
        for key, track in self.know.bombs.items():
            pos = track["pos"]
            if math.dist(pos, self.pose) > 30 and self.know.failures.get(key, 0) >= 3:
                continue
            cost, cell = self._cost_to(pos)
            view = None
            blocked = False
            if WIDE_CLAIM:
                # Blocked claim: no planned cell within 30 px (the bomb sits
                # in a slot the 10 px map draws as wall), or the best cell is
                # on the far side of a mapped wall (bomb wedged behind it).
                blocked = not math.isfinite(cost)
                seen_from = track.get("view")
                if not blocked and cell is not None and seen_from is not None:
                    # Reachable cell, but on the far side of a real wall
                    # from where the bomb was seen (thin wall, wedged bomb).
                    # A corner bomb approached from its open side passes.
                    end = self.grid.world(*cell)
                    blocked = (self._segment_blocked(end, pos, min_hits=2)
                               and not self._same_side_pts(seen_from, end, pos))
            if blocked:
                # Go to the spot it was sighted from (reachable, with real
                # line of sight), else to the nearest reachable cell within
                # 60 px, and let the live-sensor final approach do the rest.
                seen_from = track.get("view")
                if seen_from is not None and seen_from[2] < 130.0:
                    c_view, cell = self._cost_to(seen_from[:2], 2)
                    if math.isfinite(c_view):
                        cost = c_view + seen_from[2] + WIDE_PEN
                        view = (seen_from[0], seen_from[1])
                if view is None and not math.isfinite(cost):
                    cost, cell = self._cost_to(pos, WIDE_RADIUS)
                    cost += WIDE_PEN
            if not math.isfinite(cost):
                continue
            approach_px = float(cost)
            home_px = None
            if cell is not None and self.center_cum is not None:
                home_cell = self.center_cum[cell]
                if np.isfinite(home_cell):
                    home_px = float(home_cell) * CELL
            eta_rank = (self._eta_estimator.estimate(
                approach_path_px=approach_px,
                loaded_home_path_px=home_px,
                failure_count=self.know.failures.get(key, 0),
                approach_risk_frames=35.0 if blocked else 0.0,
            ).total_frames if V4_PREDICTIVE_ASSIGNMENT else None)
            # The wireless claim protocol continues to compare pixel costs.
            # ETA is strictly local ranking and never enters claim_cost.
            if CLAIM_DELIVERY and self.center_cum is not None and cell is not None:
                # Total work, not just the approach: a taxi driver chooses by
                # trip length, and a maze bomb costs ~1000 loaded steps to
                # bring home. Longest-processing-time first shortens the
                # makespan, because the slow trips start earliest.
                home = self.center_cum[cell]
                if np.isfinite(home):
                    cost += CLAIM_DELIVERY * float(home) * CELL
            cost += 120.0 * self.know.failures.get(key, 0)
            if key == self.claim_key:
                cost -= 40.0
                if step < self.repick_until:
                    # This drone dropped the bomb moments ago and is standing
                    # beside it: picking it up again is far cheaper than any
                    # teammate flying here, so keep it out of the auction.
                    cost -= 400.0
            matching_v5_ids = (
                {pid for pid, bid in self.know.v5_incoming_bids.items()
                 if bid.valid(step) and math.dist(pos, bid.position) <= 20.0}
                if V5_DEADLINE_ASSIGNMENT else set()
            )
            if self.know.claimed_by_better_peer(
                    pos, cost, ignore_peer_ids=matching_v5_ids):
                continue
            if V5_DEADLINE_ASSIGNMENT:
                v5_options.append((TaskCandidate(
                    key=key, position=tuple(pos), approach_px=approach_px,
                    loaded_return_px=home_px,
                    failures=self.know.failures.get(key, 0),
                    blocked_approach=blocked,
                ), cost, view, blocked))
            if V6_TASK_POLICY:
                estimate = self._eta_estimator.estimate(
                    approach_path_px=approach_px,
                    loaded_home_path_px=home_px,
                    failure_count=self.know.failures.get(key, 0),
                    approach_risk_frames=35.0 if blocked else 0.0)
                v6_options.append((KnownTask(
                    key=key, xy=tuple(pos),
                    approach_frames=estimate.approach_frames,
                    return_frames=estimate.return_frames + estimate.queue_frames,
                    grasp_frames=estimate.grasp_frames,
                    failure_risk_frames=estimate.risk_frames),
                    cost, view, blocked))
            rank = eta_rank if V4_PREDICTIVE_ASSIGNMENT else cost
            candidate = (rank, cost, key, pos, view, blocked)
            if key == previous_claim:
                incumbent = candidate
                incumbent_eta = eta_rank
            if best is None or rank < best[0]:
                best = candidate
        v5_decision = None
        if V6_TASK_POLICY and not V5_DEADLINE_ASSIGNMENT:
            peers = []
            for pid, (cpos, _, received_step) in self.know.claims.items():
                if cpos is None:
                    continue
                bid = self.know.v5_incoming_bids.get(pid)
                frames = (float(bid.cost_frames) if bid is not None
                          and bid.valid(step) and math.dist(cpos, bid.position) <= 20.0
                          else None)
                peers.append(PeerCommitment(
                    drone_id=pid, xy=tuple(cpos), bid_frames=frames,
                    observed_step=received_step,
                    carrying=bool(self.know.peers.get(pid, {}).get("carry"))))
            choice = self.v6_adapter.assigner.choose(
                [row[0] for row in v6_options], now=step,
                current_key=previous_claim,
                current_feasible=any(row[0].key == previous_claim
                                     for row in v6_options),
                peers=peers, allow_switch=True)
            if choice.task is None:
                best = None
            else:
                selected = next(row for row in v6_options
                                if row[0].key == choice.task.key)
                _, cost, view, blocked = selected
                best = (choice.total_frames, cost, choice.task.key,
                        choice.task.xy, view, blocked)
                if choice.task.key != previous_claim:
                    self._event("v6_task_choice", step,
                                reason=choice.reason,
                                eta_frames=round(choice.total_frames, 1))
        if V5_DEADLINE_ASSIGNMENT:
            recent_pickups = sum(
                event["kind"] == "grasp" and step - event["step"] <= 1200
                for event in self.task_events
            )
            # Public round limit can be supplied by the official test plan.
            # If unavailable to the competition controller, omit lateness
            # penalty rather than guessing a map-specific hidden limit.
            horizon = V5_TIME_HORIZON if V5_TIME_HORIZON > 0 else 10**9
            v5_decision = self.v5_scheduler.choose(
                [item[0] for item in v5_options], now=step,
                horizon=horizon, incumbent_key=previous_claim,
                own_recent_pickups=recent_pickups,
                peer_bids=self.know.v5_incoming_bids.values(),
            )
            if v5_decision is not None:
                selected = next(item for item in v5_options
                                if item[0].key == v5_decision.key)
                candidate, cost, view, blocked = selected
                best = (v5_decision.score_frames, cost, candidate.key,
                        candidate.position, view, blocked)
                self.know.v5_outgoing_bid = self.v5_scheduler.make_bid(
                    v5_decision, now=step)
                if v5_decision.key != previous_claim:
                    self._event("v5_task_switch", step,
                                reason=v5_decision.reason,
                                eta=round(v5_decision.completion_frames, 1),
                                finish=round(v5_decision.expected_finish_step, 1),
                                recent_pickups=recent_pickups)
            else:
                best = None
        if best is None:
            if self.claim_key is not None:
                self._drop_claim()
            return
        if (V4_PREDICTIVE_ASSIGNMENT and not V5_DEADLINE_ASSIGNMENT and incumbent is not None
                and best[2] != previous_claim
                and not self._eta_hysteresis.should_switch(
                    incumbent_eta, best[0])):
            best = incumbent
        eta_rank, cost, key, pos, view, blocked = best
        if V4_PREDICTIVE_ASSIGNMENT and key != previous_claim:
            reason = ("new_claim" if previous_claim is None else
                      "old_invalid" if incumbent is None else "eta_gain")
            self._event("task_switch", step, task_switch_reason=reason,
                        old_eta=(None if incumbent_eta is None else
                                 round(float(incumbent_eta), 2)),
                        new_eta=round(float(eta_rank), 2),
                        current_claim_age=(0 if previous_claim is None else
                                           step - self.claim_since))
        if key != self.claim_key:
            self.fetch_breadcrumbs.clear()
            self.claim_view = None
            self.claim_key = key
            self.claim_since = step
            self._reset_progress()
            self.final_since = None
            self.unseen_close = 0
            self.view_reached = False
        self.claim_pos, self.claim_cost = pos, cost
        if V4_PREDICTIVE_ASSIGNMENT:
            self._last_claim_eta = float(eta_rank)
        self.claim_wide = blocked
        if key != previous_claim:
            if not hasattr(self, "bomb_trace"):
                self.bomb_trace = []
            self.bomb_trace.append({
                "kind": "claim", "step": int(step), "key": str(key),
                "bomb": [round(float(v), 2) for v in pos],
                "cost": round(float(cost), 2), "blocked": bool(blocked),
            })
        if view is not None and self.claim_view is None:
            self.claim_view = view

    def unique_grasp_owner(self, target, step):
        """True only if this drone is the single best candidate for ``target``.

        Deterministic and symmetric: the same inputs give the same answer on
        every drone in radio range, so two graspers cannot both arm.
        """
        if self.pose is None:
            return False
        mine = math.dist(self.pose, target)
        if mine > 40.0:
            return False
        best = (mine, self.me)
        for pid, peer in self.know.peers.items():
            p = peer.get("pos")
            if p is None or peer.get("carry"):
                continue
            d = math.dist(p, target)
            if d <= 40.0 and (d, pid) < best:
                best = (d, pid)
        if best[1] != self.me:
            return False
        # A peer that already holds this bomb is the owner, not us.
        for pid, peer in self.know.peers.items():
            pickup = peer.get("pickup")
            if (peer.get("carry") and pickup is not None
                    and math.dist(pickup[:2], target) < 25.0):
                return False
        return True

    def grasp_armed(self, target, step):
        """Grasper gate: unique owner, stable for a few steps, bomb in reach."""
        if self.pose is None or step < self.no_grasp_until:
            return False
        if math.dist(self.pose, target) > 34.0:
            self.arm_owner_since = None
            return False
        if not self.unique_grasp_owner(target, step):
            self.arm_owner_since = None
            return False
        if self.arm_target is None or math.dist(self.arm_target, target) > 15.0:
            self.arm_target = target
            self.arm_owner_since = step
        elif self.arm_owner_since is None:
            self.arm_owner_since = step
        return step - self.arm_owner_since >= 3

    def _drop_claim(self):
        self.claim_view = None
        self.claim_wide = False
        self.view_reached = False
        self.claim_key = None
        self.claim_pos = None
        self.claim_cost = None
        self.final_since = None
        self.unseen_close = 0

    def _line_of_sight(self, target, lidar, angles, tol=22.0):
        if lidar is None:
            return True
        dx, dy = target[0] - self.pose[0], target[1] - self.pose[1]
        d = math.hypot(dx, dy)
        rel = wrap(math.atan2(dy, dx) - self.heading)
        idx = np.abs((angles - rel + math.pi) % (2 * math.pi) - math.pi) < 0.06
        if not idx.any():
            return True
        return bool(np.min(lidar[idx]) >= d - tol)

    # ======================================================= local control
    def _velocity_to_command(self, v_des, lidar, angles, carrying, careful=False,
                             wall_margin=None, ignore_target=None):
        """Velocity PD + lidar braking + wall and drone repulsion."""
        if (V4_MODEL_EMPTY and not (V5_MODEL_BLEND or V5_MODEL_SHADOW)
                and not carrying and not careful
                and self.state == "EXPLORE" and lidar is not None):
            finite = np.asarray(lidar, float)
            finite = finite[np.isfinite(finite)]
            peers_close = any(distance < 90.0
                              for _, _, distance in self.visible_drones)
            if finite.size and float(np.min(finite)) > 90.0 and not peers_close:
                try:
                    command_world = self.model_empty_controller.compute(
                        v_des, self.v_control)
                    command_body = world_to_body(command_world, self.heading)
                    return float(command_body[0]), float(command_body[1])
                except (ValueError, TypeError):
                    pass
        vt = V_CARRY if carrying else V_EMPTY
        vdx, vdy = v_des
        speed = math.hypot(vdx, vdy)
        # Braking: limit speed toward the nearest obstacle in the travel cone.
        if lidar is not None and speed > 0.05:
            ang = wrap(math.atan2(vdy, vdx) - self.heading)
            rel = (angles - ang + math.pi) % (2 * math.pi) - math.pi
            corridor = (np.abs(rel) < math.pi / 2) & (np.abs(lidar * np.sin(rel)) < DRONE_R + 4)
            corridor &= np.abs(rel) < 0.9
            if ignore_target is not None:
                corridor &= ~ignore_target
            ahead = float(np.min(lidar[corridor])) if corridor.any() else 300.0
            decel = CARRY_DECEL if carrying else 0.45
            if carrying and ADAPTIVE_CARRY:
                # The braking distance that matters is the FORWARD corridor,
                # not the nearest wall anywhere: in a corridor the side walls
                # are always ~25 px away, so min(lidar) kept every carrier in
                # the cautious branch for the whole round (Map15: 1647 ->
                # 7200 steps). Open space ahead allows the relaxed value, a
                # tight squeeze ahead keeps the cautious one.
                span = max(1.0, CARRY_OPEN - CARRY_TIGHT)
                share = min(1.0, max(0.0, (ahead - CARRY_TIGHT) / span))
                decel = CARRY_DECEL_TIGHT + share * (CARRY_DECEL - CARRY_DECEL_TIGHT)
            if careful:
                pad = 3.0
            elif carrying:
                pad = CARRY_GAP_PAD
            else:
                pad = 8.0
            gap = max(0.0, ahead - DRONE_R - pad)
            if V2_SAFE_CONTROL:
                # Keep the official braking law but route it through the
                # standalone helper so the loaded envelope is explicit and
                # unit-testable.  Preserve the previous minimum command to
                # avoid freezing in a narrow but traversable corridor.
                v_allow = max(0.6 if careful else 1.0,
                              braking_speed(ahead, speed, carrying=carrying,
                                            body_radius=DRONE_R, pad=pad))
            else:
                v_allow = max(0.6 if careful else 1.0, math.sqrt(2.0 * decel * gap))
            if speed > v_allow:
                vdx, vdy = vdx * v_allow / speed, vdy * v_allow / speed
        fx = vdx / vt + 0.35 * (vdx - self.vel[0])
        fy = vdy / vt + 0.35 * (vdy - self.vel[1])
        # Wall repulsion from close lidar returns.
        if lidar is not None:
            margin = wall_margin if wall_margin is not None else (34.0 if not careful else 24.0)
            close = lidar < margin
            if ignore_target is not None:
                close &= ~ignore_target
            if close.any():
                w = (margin - lidar[close]) / margin
                world = angles[close] + self.heading
                fx -= 0.9 * float(np.sum(w * np.cos(world))) / max(1.0, close.sum() ** 0.5)
                fy -= 0.9 * float(np.sum(w * np.sin(world))) / max(1.0, close.sum() ** 0.5)
        # Drone avoidance with priority (carriers first, then lower id).
        my_rank = (0 if carrying else 1, self.me)
        others = []
        for pid, peer in self.know.peers.items():
            p = peer.get("pos")
            if p is not None:
                others.append((p, (0 if peer.get("carry") else 1, pid), peer.get("vel")))
        for x, y, _ in self.visible_drones:
            if not any(math.dist((x, y), o[0]) < 30 for o in others):
                unknown_rank = ((0, -1) if UNKNOWN_DRONE_CARRIER_PRIORITY
                                else (1, -1))
                others.append(((x, y), unknown_rank, None))
        for p, rank, pvel in others:
            dx, dy = p[0] - self.pose[0], p[1] - self.pose[1]
            d = math.hypot(dx, dy)
            if d < 1.0 or d > 75.0:
                continue
            ux, uy = dx / d, dy / d
            strength = (75.0 - d) / 75.0
            yield_factor = 0.35 if my_rank < rank else 1.0
            fx -= 0.8 * yield_factor * strength * ux
            fy -= 0.8 * yield_factor * strength * uy
            # Keep right when closing in head-on.
            pv = pvel or (0.0, 0.0)
            closing = (self.vel[0] - pv[0]) * ux + (self.vel[1] - pv[1]) * uy
            if closing > 0.5 and d < 70.0:
                fx += 0.3 * yield_factor * uy
                fy -= 0.3 * yield_factor * ux
        norm = math.hypot(fx, fy)
        if norm > 1.0:
            fx, fy = fx / norm, fy / norm
        ca, sa = math.cos(self.heading), math.sin(self.heading)
        baseline_body = (ca * fx + sa * fy, -sa * fx + ca * fy)
        if ((V5_MODEL_SHADOW or V5_MODEL_BLEND) and not carrying
                and not careful and self.state == "EXPLORE"):
            try:
                model_world = self.model_empty_controller.compute(v_des, self.v_control)
                self.v5_shadow_frames += 1
                lidar_min = (float(np.min(lidar)) if lidar is not None
                             else 0.0)
                blended, enabled = self.v5_blend.apply(
                    baseline_body=baseline_body, model_world=model_world,
                    heading=self.heading, state=self.state, carrying=False,
                    lidar_min_px=lidar_min,
                    gps_valid=self.measured_gps_position() is not None,
                    peer_nearby=any(d < 90.0 for _, _, d in self.visible_drones),
                )
                self.v5_blend_frames += int(enabled)
                return float(blended[0]), float(blended[1])
            except (ValueError, TypeError):
                pass
        return baseline_body

    def _toward(self, target, speed_cap, stop_dist=0.0):
        dx, dy = target[0] - self.pose[0], target[1] - self.pose[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return (0.0, 0.0)
        # Old rule (0.12*d) capped a 10 px look-ahead waypoint at ~1.6 px/step,
        # i.e. a crawl through long corridors. Stop only near the true goal.
        speed = speed_cap if d - stop_dist > 6.0 else min(speed_cap, 0.5 * d)
        return (speed * dx / d, speed * dy / d)

    def _yaw_for_cargo(self, cargo_world_dir):
        """Rotation command that puts the carried bomb at ``cargo_world_dir``."""
        if self.cargo_bearing is None:
            return 0.0
        desired_heading = wrap(cargo_world_dir - self.cargo_bearing)
        err = wrap(desired_heading - self.heading)
        return clamp(err / 0.17 * 0.6, -0.6, 0.6)

    # ============================================================ stuck logic
    def _stuck_monitor(self, step, goal):
        if step % 10 == 0:
            self.history.append(self.pose)
        if (goal is None or len(self.history) < self.history.maxlen
                or step < self.escape_until):
            return False
        span = max(math.dist(self.history[0], p) for p in self.history)
        if span > 35.0 or math.dist(self.pose, goal) < 45.0:
            return False
        return True

    def _event(self, kind, step, **extra):
        self.task_events.append({"kind": kind, "step": step,
                                 "pos": tuple(round(v) for v in self.pose), **extra})

    def _progress_stalled(self, step, value, window=150, need=25.0):
        """True when the remaining path cost ``value`` (px) stopped shrinking.

        Jittering in place (a carrier hooked on a wall end, two drones
        blocking each other) moves the body but never reduces the cost to go.
        """
        if value is None or not math.isfinite(value):
            return False
        if self.prog_best is None or value < self.prog_best - need:
            self.prog_best, self.prog_step = value, step
            return False
        return step - self.prog_step > window

    def _reset_progress(self):
        self.prog_best = None

    def _start_escape(self, step, lidar, angles, away_from=None):
        self._event("stuck", step, state=self.state)
        self._reset_progress()
        self.stats["stuck"] += 1
        self.stuck_count += 1
        self.history.clear()
        if self.was_carrying:
            # A carrier is often inside a narrow corridor that is the only
            # way out; a large penalty would wall off its own exit.
            self.grid.add_penalty(self.pose[0], self.pose[1], 22.0, step + 220, 2.5)
        else:
            self.grid.add_penalty(self.pose[0], self.pose[1], 40.0, step + 500, 8.0)
        self.grid.dirty = True
        if lidar is not None:
            score = lidar.copy()
            if away_from is not None:
                # Prefer backing off from the blocked direction.
                back = math.atan2(self.pose[1] - away_from[1],
                                  self.pose[0] - away_from[0]) - self.heading
                rel = np.abs((angles - back + math.pi) % (2 * math.pi) - math.pi)
                score = np.where(rel < 1.8, score, 0.3 * score)
            k = int(np.argmax(score))
            a = self.heading + float(angles[k])
        else:
            a = self.heading + math.pi
        self.escape_dir = (math.cos(a), math.sin(a))
        self.escape_until = step + (40 if self.was_carrying else 30)
        self.path = []
        self.cum_step = -1000
        self.center_step = -1000

    # ================================================================ control
    def control(self):
        started = time.perf_counter()
        try:
            cmd = self._control_body()
            if not self.ready:
                return cmd
            step = int(self.elapsed_timestep)
            if (self.stats.get("all_discovered_step") is None
                    and self.know.accounted() >= EXPECTED_BOMBS):
                self.stats["all_discovered_step"] = step
            if (not self.grasped_bombs() and self.start_pose is not None
                    and self.pose is not None and step - self.start_step < LAUNCH_STEPS):
                cmd = self._launch_blend(step, cmd)
            return cmd
        finally:
            # Keep lightweight latency telemetry on the drone for benchmark
            # tooling.  Do not print from control(), since the official
            # evaluator may suppress solution output and printing is costly.
            if getattr(self, "ready", False):
                elapsed_us = (time.perf_counter() - started) * 1e6
                self.stats["control_calls"] += 1
                self.stats["control_us"] += elapsed_us
                self.stats["control_max_us"] = max(self.stats["control_max_us"], elapsed_us)
                self.stats["control_samples_us"].append(elapsed_us)

    def _control_body(self):
        if not self.ready:
            self._setup()
        step = int(self.elapsed_timestep)
        cmd = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0,
               "grasper": 1 if self.grasped_bombs() else 0}
        self._update_pose()
        if self.pose is None:
            return cmd
        if self.start_pose is None:
            self.start_pose = self.pose
            self.start_step = step
            self.started_in_return = bool(self.is_inside_return_area)
        self.know.receive(self.communicator.received_messages
                          if not self.communicator_is_disabled() else (), step, self.grid)
        dynamic = self._perceive(step)
        lidar, angles = self._lidar()
        health = int(self.drone_health)
        if health < self._last_health:
            self._event(
                "damage", step, loss=int(self._last_health - health),
                state=self.state, carrying=bool(self.grasped_bombs()),
                visible_drones=len(self.visible_drones),
                lidar_min=(None if lidar is None else
                           round(float(np.nanmin(lidar)), 1)),
                clearance=round(float(self.grid.clearance_at(*self.pose)), 1),
                gps_valid=self.measured_gps_position() is not None,
                comm_disabled=bool(self.communicator_is_disabled()),
            )
        self._last_health = health
        if step % 2 == self.me % 2:
            self._map_update(step, dynamic, lidar, angles)
        if step % 5 == self.me % 5:
            self.grid.make_snapshot(step)
        if step - self.grid.fused_step >= 4 or self.grid.costs is None:
            if self.grid.fuse(step):
                pass
        carrying = bool(self.grasped_bombs())
        self._transitions(step, carrying)

        if step < self.escape_until and self.escape_dir is not None:
            v = (self.escape_dir[0] * 4.0, self.escape_dir[1] * 4.0)
            cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                v, lidar, angles, carrying)
            cmd["grasper"] = 1 if carrying and step >= self.no_grasp_until else 0
            return cmd

        if (not carrying and self.start_pose is not None
                and step - self.start_step < self.me * DEPART_GAP):
            # Ten drones spawn in a 40 px grid, i.e. hulls almost touching.
            # Leaving together costs ~3 health each; take a slot instead.
            cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                (0.0, 0.0), lidar, angles, False, wall_margin=20.0)
            return cmd

        if (carrying or self.state in ("FETCH", "CARRY")
                or any(peer.get("carry") or peer.get("state") in ("FETCH", "CARRY")
                       for peer in self.know.peers.values())):
            self.last_team_busy = step

        if carrying:
            self.state = "CARRY"
            return self._carry(step, cmd, lidar, angles)
        if step % 5 == self.me % 5 or self.claim_key is None:
            self._refresh_dijkstra(step)
            self._choose_bomb(step)
        if (self.me in SCOUTS and step < SCOUT_UNTIL
                and (self.interest_left is None or self.interest_left > SCOUT_MIN)):
            # Dedicated scout: keep mapping while the shared map still has
            # reachable space no semantic ray has covered. No assumption on
            # the number of bombs: the stop condition is coverage only.
            self.state = "EXPLORE"
            self._drop_claim()
            return self._explore(step, cmd, lidar, angles)
        if self.claim_key is not None and self.claim_key in self.know.bombs:
            self.state = "FETCH"
            return self._fetch(step, cmd, lidar, angles)
        if self.claim_key is not None:
            self._drop_claim()
        self.state = "EXPLORE"
        return self._explore(step, cmd, lidar, angles)

    def _launch_blend(self, step, cmd):
        """Bias the first steps away from the packed spawn without losing them.

        ``cmd`` already holds the task command in the body frame. Mixing a
        radial escape into the *body-frame* command keeps the task heading
        while separating the hulls.
        """
        if LAUNCH_STEPS <= 0 or step - self.start_step >= LAUNCH_STEPS:
            return cmd
        peers = [peer["pos"] for peer in self.know.peers.values()
                 if peer.get("pos") is not None]
        peers.append(self.pose)
        cx = sum(p[0] for p in peers) / len(peers)
        cy = sum(p[1] for p in peers) / len(peers)
        dx, dy = self.pose[0] - cx, self.pose[1] - cy
        n = math.hypot(dx, dy)
        if n < 6.0:
            a = 2.0 * math.pi * self.me / 10.0
            dx, dy, n = math.cos(a), math.sin(a), 1.0
        weight = LAUNCH_BLEND * (1.0 - (step - self.start_step) / LAUNCH_STEPS)
        ca, sa = math.cos(self.heading), math.sin(self.heading)
        ax, ay = ca * cmd["forward"] + sa * cmd["lateral"], -sa * cmd["forward"] + ca * cmd["lateral"]
        ax += weight * 0.55 * dx / n
        ay += weight * 0.55 * dy / n
        norm = math.hypot(ax, ay)
        if norm > 1.0:
            ax, ay = ax / norm, ay / norm
        cmd["forward"], cmd["lateral"] = ca * ax + sa * ay, -sa * ax + ca * ay
        return cmd

    def _transitions(self, step, carrying):
        if carrying and not self.was_carrying:
            self.stats["grasps"] += 1
            # Body-frame bearing of the bomb at the grasp instant.
            ref = self.claim_pos
            near = [p for p, d in self.visible_bombs if d < 45]
            if near:
                ref = min(near, key=lambda p: math.dist(p, self.pose))
            if ref is not None and math.dist(ref, self.pose) < 50:
                self.cargo_bearing = wrap(math.atan2(ref[1] - self.pose[1],
                                                     ref[0] - self.pose[0]) - self.heading)
                self.last_pickup = ref
            else:
                self.cargo_bearing = 0.0
                self.last_pickup = self.pose
            self.know.tombstone(self.last_pickup, step)
            self.pickup_info = (self.last_pickup[0], self.last_pickup[1], step)
            self.pickup_key = f"{self.me}:{step}"
            self.know.pickups[self.pickup_key] = (round(self.last_pickup[0], 1),
                                                  round(self.last_pickup[1], 1))
            self._drop_claim()
            # Reverse the actual legal approach, rather than asking the
            # newly enlarged cargo footprint to plan from inside a pickup
            # pocket.  Downsample so waypoint look-ahead cannot cut across
            # the wall lip that made the pocket difficult in the first place.
            exit_path = []
            # A bomb enlarges the moving footprint immediately after grasp.
            # In low-clearance pockets, first reverse the already traversed
            # legal approach instead of asking the enlarged footprint to plan
            # a new route from inside the pocket.  The legacy edge specialist
            # remains covered; the generic branch depends only on measured
            # map clearance, never on a map id or target coordinate.
            if ((EDGE_BONUS > 0 and self.me in EDGE_SCOUTS
                 and step - self.start_step > EDGE_START)):
                for pose in reversed(self.fetch_breadcrumbs):
                    if not exit_path or math.dist(exit_path[-1], pose) >= 12.0:
                        exit_path.append(tuple(pose))
            self.path = exit_path
            self.path_i = 0
            self.path_goal = "pickup_exit" if len(exit_path) >= 2 else None
            self.path_step = step
            self.pickup_exit_until = step + 260
            self.pickup_exit_origin = tuple(self.pose)
            self.fetch_breadcrumbs.clear()
            self.cum_step = -1000
            self.push_since = None
            self.cargo_try = 0
            self.history.clear()
            self._reset_progress()
            self._event("grasp", step)
        if self.was_carrying and not carrying:
            near_center = any(d < 60 for _, _, d in self.center_hits)
            if near_center:
                self.know.disposed += 1
            elif self.yielded_cargo:
                # Handed over to a co-carrier: its own pickup accounts for it.
                if getattr(self, "pickup_key", None):
                    self.know.drops.add(self.pickup_key)
            else:
                if getattr(self, "pickup_key", None):
                    self.know.drops.add(self.pickup_key)
                # Dropped on the way: put it back on the task list.
                self.stats["drops"] += 1
                if self.cargo_bearing is not None:
                    a = self.heading + self.cargo_bearing
                    drop = (self.pose[0] + 22 * math.cos(a), self.pose[1] + 22 * math.sin(a))
                    key = self.know.observe_bomb(drop, step + 1, 0.3)
                    if key is not None:
                        # Keep the bomb as this drone's own task: the drone
                        # that dropped it is standing next to it, and a track
                        # left unclaimed was observed to sit untaken for the
                        # rest of the round (Map01 user layout: 9/10).
                        self.claim_key = key
                        self.claim_pos = self.know.bombs[key]["pos"]
                        self.claim_cost = None
                        self.claim_since = step
                        self.repick_until = step + 200
            self._event("release", step, near_center=near_center)
            self.yielded_cargo = False
            self.cargo_bearing = None
            self.path = []
            self.cum_step = -1000
            self.history.clear()
            self._reset_progress()
        self.was_carrying = carrying

    # ----------------------------------------------------------------- CARRY
    def _carry(self, step, cmd, lidar, angles):
        cmd["grasper"] = 1
        # Two drones that grasped the same bomb pull against each other.
        # The higher id lets go; the bomb stays with the other carrier.
        partner = None
        mine = self.pickup_info
        for pid, peer in self.know.peers.items():
            theirs = peer.get("pickup")
            if (peer.get("carry") and theirs is not None and mine is not None
                    and math.dist(theirs[:2], mine[:2]) < 25
                    and abs(theirs[2] - mine[2]) < 80):
                partner = pid if partner is None else min(partner, pid)
        self.cocarry_steps = self.cocarry_steps + 1 if partner is not None else 0
        if partner is not None and partner < self.me:
            self._event("cocarry_release", step, partner=partner)
            self.yielded_cargo = True
            self.no_grasp_until = step + 60
            cmd["grasper"] = 0
            peer_pos = self.know.peers[partner]["pos"]
            dx, dy = self.pose[0] - peer_pos[0], self.pose[1] - peer_pos[1]
            n = max(1.0, math.hypot(dx, dy))
            self.escape_dir = (dx / n, dy / n)
            self.escape_until = step + 20
            return cmd
        near = min(self.center_hits, key=lambda h: h[2]) if self.center_hits else None
        field_here = None
        if self.center_cum is not None:
            st = self.grid.free_start(self.grid.costs["carry"], *self.pose, radius=5)
            if st is not None:
                field_here = float(self.center_cum[st]) * CELL
        # "Arrived" means the delivery field says we are at the centre, OR the
        # planned path has run out: an arrival cell recorded from a distant
        # sighting can sit further out than the push radius, and a carrier
        # that stops there oscillates forever instead of closing the last
        # ~170 px (measured on Map01: 900 steps lost at 181 px from centre).
        arrived = ((field_here is not None and field_here < 60.0)
                   or (self.path_goal == "center" and not self.path))
        if (near is not None and step >= self.backoff_until
                and (near[2] < 110 or (arrived and near[2] < PUSH_RANGE))
                and self._center_passable((near[0], near[1]))):
            # Final push. Touching the disposal centre costs no health, and
            # the bomb hangs 11-28 px from the hull on a swinging pivot, so:
            # press the hull against the face, then spin so the bomb sweeps
            # across it. Semantic hits are relative, so pose error is moot.
            if self.push_since is None:
                self.push_since = step
            if self.cargo_try % 2 == 0 or len(self.center_hits) < 3:
                target = (near[0], near[1])
            else:
                target = (sum(h[0] for h in self.center_hits) / len(self.center_hits),
                          sum(h[1] for h in self.center_hits) / len(self.center_hits))
            a = math.atan2(target[1] - self.pose[1], target[0] - self.pose[0])
            ignore = None
            if lidar is not None:
                rel = np.abs((angles - (a - self.heading) + math.pi) % (2 * math.pi) - math.pi)
                ignore = rel < 0.7
            if near[2] < 40:
                cmd["rotation"] = 1.0 if (self.cargo_try % 2 == 0) else -1.0
                v = self._toward(target, 2.0)
            else:
                cmd["rotation"] = 0.6 * self._yaw_for_cargo(a)
                v = self._toward(target, 3.5)
            cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                v, lidar, angles, True, careful=True, wall_margin=14.0,
                ignore_target=ignore)
            if step - self.push_since > 110:
                self.stats["push_retries"] += 1
                self.cargo_try += 1
                if self.cargo_try >= 3:
                    # Repeated failure from here: this side is not usable
                    # (slit, wrong side of a wall). Tell the whole team.
                    self.grid.mark_center_bad(self.pose[0], self.pose[1], 70.0)
                    self._event("center_bad", step)
                    self.center_step = -1000
                    self.cargo_try = 0
                self.push_since = None
                self.backoff_until = step + 30
                self.backoff_dir = (-math.cos(a), -math.sin(a))
            return cmd
        if step < self.backoff_until and self.backoff_dir is not None:
            v = (self.backoff_dir[0] * 2.5, self.backoff_dir[1] * 2.5)
            cmd["forward"], cmd["lateral"] = self._velocity_to_command(v, lidar, angles, True)
            return cmd
        self.push_since = None
        if self.path_goal == "pickup_exit":
            travelled = (math.dist(self.pose, self.pickup_exit_origin)
                         if self.pickup_exit_origin is not None else 0.0)
            nearest_end = math.dist(self.pose, self.path[-1]) if self.path else 0.0
            done = (step >= self.pickup_exit_until or not self.path
                    or nearest_end < 28.0 or travelled >= 135.0)
            if done:
                self._event("pickup_exit_complete", step,
                            distance=round(travelled, 1))
                self.path = []
                self.path_i = 0
                self.path_goal = None
                self.center_step = -1000
                self._reset_progress()
            else:
                # Keep look-ahead short while reversing a narrow entry.
                waypoint = self._waypoint(min_clear=18.0, reach=34.0)
                if waypoint is not None:
                    velocity = self._toward(waypoint, min(V_CARRY, 3.2), 0.0)
                    cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                        velocity, lidar, angles, True, careful=True,
                        wall_margin=17.0)
                    if lidar is None or float(np.min(lidar)) >= 36.0:
                        move = math.atan2(velocity[1], velocity[0])
                        cmd["rotation"] = 0.45 * self._yaw_for_cargo(move + math.pi)
                return cmd
        have_center = bool(self.grid.center_points or self.grid.seen_points)
        if have_center and (not self.path or step - self.path_step > 25
                            or self.path_goal != "center"):
            center_path = self._center_path(step)
            if center_path or self.path_goal == "center":
                self.path = center_path
                self.path_i = 0
                self.path_goal = "center"
                self.path_step = step
        if self.path_goal == "center" and len(self.path) <= 1:
            self.path_goal = None
            # A carrier whose centre path keeps running out, with no live
            # centre hit to close in on, has no target and circles on the spot
            # (Map15 mix0 s1: 4000 steps, 81 pts). Drop the centre goal and
            # hunt for the centre again through loaded exploration.
            self.center_dead += 1
            if self.center_dead > CENTER_DEAD_N:
                self.center_dead = 0
                self.carry_goal = None
                self.center_step = -1000
                self._event("center_goal_drop", step)
            if (near is not None and 110.0 <= near[2] <= 210.0
                    and self._center_passable((near[0], near[1]))):
                # The delivery field can only aim at cells recorded when the
                # centre was seen, and a cell recorded from 200 px away is a
                # vantage point, not an arrival point: the carrier walks there,
                # the path runs out, and it oscillates 180 px from the centre
                # (measured on Map01: 900 steps lost). Close the gap on the
                # live relative hit instead, which no pose error can spoil.
                d = near[2]
                ux = (near[0] - self.pose[0]) / max(1e-6, d)
                uy = (near[1] - self.pose[1]) / max(1e-6, d)
                goal = (self.pose[0] + ux * (d - CLOSE_IN), self.pose[1] + uy * (d - CLOSE_IN))
                self._refresh_dijkstra(step)
                self._set_path_to(goal, step)
                self.carry_goal = goal
                self.carry_goal_step = step
                self.path_goal = "approach"
        else:
            self.center_dead = 0
        if self.path_goal not in ("center", "approach"):
            # Centre unknown (or unreachable on the current map): search for
            # it while loaded instead of waiting at the spawn.
            need = (self.carry_goal is None
                    or math.dist(self.pose, self.carry_goal) < 30
                    or step - self.carry_goal_step > 200)
            if need:
                self._refresh_dijkstra(step, force=True)
                # Loaded centre hunt: a far wing is exactly where the centre
                # is likely to be on the big maps (it sits in the corner
                # opposite the spawn), so travel must not price it out.
                choice = self._explore_target(
                    step, dist_scale=HUNT_DIST if not have_center else 1.0)
                self.carry_goal = choice[0] if choice else None
                self.carry_goal_step = step
                if choice:
                    self._set_path_to(self.carry_goal, step, end_cell=choice[1])
            elif self.carry_goal is not None and (step - self.path_step > 25 or not self.path):
                self._refresh_dijkstra(step)
                self._set_path_to(self.carry_goal, step)
            self.path_goal = "carry_explore"
            if not self.path:
                cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                    (0.0, 0.0), lidar, angles, True)
                return cmd
        # The bomb can hang up to ~40 px from the hull centre: shortcuts must
        # keep that much room or the bomb hooks on wall ends in tight turns.
        wp = self._waypoint(min_clear=32.0, reach=42.0)
        goal_pt = self.path[-1] if self.path else self.start_pose
        field = None
        if self.center_cum is not None:
            st = self.grid.free_start(self.grid.costs["carry"], *self.pose, radius=5)
            if st is not None:
                field = float(self.center_cum[st]) * CELL
        if (self._progress_stalled(step, field, window=160)
                or self._stuck_monitor(step, goal_pt)):
            self._start_escape(step, lidar, angles, away_from=wp)
            return cmd
        if wp is None:
            return cmd
        v = self._toward(wp, V_CARRY, 0.0)
        cmd["forward"], cmd["lateral"] = self._velocity_to_command(
            v, lidar, angles, True, wall_margin=22.0)
        tight = lidar is not None and float(np.min(lidar)) < 38.0
        if not tight:
            # Rotate so the bomb trails behind; never swing it in tight spots.
            move = math.atan2(v[1], v[0])
            cmd["rotation"] = 0.6 * self._yaw_for_cargo(move + math.pi)
        return cmd

    def _grasp_pose(self, bomb):
        """Best SE(2) standing point from which the grasper reaches ``bomb``.

        The grasper is concentric with the hull (radius 21.4 px) and the bomb
        radius is 12, so a grasp needs the hull CENTRE within 33 px of the
        bomb. Corridor corners leave a thin legal arc: driving straight at the
        bomb pushes the centre against the wall, where the braking model stops
        the drone ~25 px short for hundreds of steps. Search a ring of
        candidate centres instead and take the one with the most wall
        clearance, tie-broken toward the current heading so the drone does not
        spin on the spot.
        """
        if self.grid.clearance is None or self.pose is None:
            return bomb
        best = None
        ring = GRASP_REACH * 0.72
        for px, py, a in grasp_candidates(bomb, ring, V2_GRASP_HEADINGS):
            p = (px, py)
            if math.dist(p, bomb) > GRASP_REACH - 2.0:
                continue
            if not self.grid.segment_clear(
                    (self.pose[0], self.pose[1]), p, 14.0) and \
                    math.dist(self.pose, p) > 40.0:
                continue
            clr = self.grid.clearance_at(*p)
            if clr < DRONE_R + 1.0:
                continue
            heading_pen = abs(wrap(a - self.heading)) * 6.0
            cost = -1.6 * min(clr, 60.0) + heading_pen
            if best is None or cost < best[0]:
                best = (cost, p)
        return best[1] if best is not None else bomb

    def _same_side(self, track, target):
        seen_from = track.get("view")
        if seen_from is None:
            return True
        ax, ay = seen_from[0] - target[0], seen_from[1] - target[1]
        bx, by = self.pose[0] - target[0], self.pose[1] - target[1]
        na, nb = math.hypot(ax, ay), math.hypot(bx, by)
        if na < 1.0 or nb < 1.0:
            return True
        return (ax * bx + ay * by) / (na * nb) > SAME_SIDE_COS

    def _same_side_pts(self, view, p, target):
        ax, ay = view[0] - target[0], view[1] - target[1]
        bx, by = p[0] - target[0], p[1] - target[1]
        na, nb = math.hypot(ax, ay), math.hypot(bx, by)
        if na < 1.0 or nb < 1.0:
            return True
        return (ax * bx + ay * by) / (na * nb) > SAME_SIDE_COS

    def _segment_blocked(self, a, b, skip_end=15.0, min_hits=1):
        """Mapped wall cell on segment a-b, ignoring the last ``skip_end`` px
        (the bomb itself is drawn as wall when lidar saw it unlabelled)."""
        g = self.grid
        if g.obstacle is None:
            return False
        d = math.dist(a, b)
        if d <= skip_end + 5.0:
            return False
        ux, uy = (b[0] - a[0]) / d, (b[1] - a[1]) / d
        hits = 0
        for t in np.arange(0.0, d - skip_end, 5.0):
            iy, ix = g.cell(a[0] + t * ux, a[1] + t * uy)
            if g.obstacle[iy, ix]:
                hits += 1
                if hits >= min_hits:
                    return True
            else:
                hits = 0
        return False

    def _wall_between(self, target, d):
        """Does the straight line to ``target`` cross a mapped wall cell?

        The last 22 px are skipped: a bomb seen by lidar alone is itself
        drawn as a wall cell.
        """
        g = self.grid
        if g.obstacle is None or d <= 30.0:
            return False
        ux, uy = (target[0] - self.pose[0]) / d, (target[1] - self.pose[1]) / d
        for t in np.arange(10.0, d - 22.0, 5.0):
            iy, ix = g.cell(self.pose[0] + t * ux, self.pose[1] + t * uy)
            if g.obstacle[iy, ix]:
                return True
        return False

    # ----------------------------------------------------------------- FETCH
    def _fetch(self, step, cmd, lidar, angles):
        if step % 6 == self.me % 6 and self.pose is not None:
            pose = tuple(self.pose)
            if (not self.fetch_breadcrumbs
                    or math.dist(self.fetch_breadcrumbs[-1], pose) >= 5.0):
                self.fetch_breadcrumbs.append(pose)
        track = self.know.bombs[self.claim_key]
        target = track["pos"]
        # Live semantic sighting refines the target every step.
        live = [p for p, d in self.visible_bombs if math.dist(p, target) < MATCH + 8]
        if live:
            target = min(live, key=lambda p: math.dist(p, target))
            self.unseen_close = 0
        d = math.dist(self.pose, target)
        los = self._line_of_sight(target, lidar, angles)
        # Absent only when looking from the side the bomb was sighted from:
        # a thin wall right in front of a wedged bomb passes the 22 px lidar
        # test, the semantic sensor cannot see through it, and the bomb was
        # wrongly tombstoned from behind that wall (Map10 wedge).
        absent_view = los and (not self.claim_wide or self._same_side(track, target))
        if (self.claim_wide and self.claim_view is None and not live and d < 95
                and track.get("view") is not None and not self._same_side(track, target)):
            # Close, blind, and on the far side from where the bomb was
            # seen: a wall is in between. Go round to the sighting point.
            self.claim_view = (track["view"][0], track["view"][1])
            self.view_reached = False
            self.final_since = None
            self._event("use_view", step)
            self.path = []
        view = self.claim_view
        if view is not None and not self.view_reached:
            at_path_end = (bool(self.path) and math.dist(self.pose, self.path[-1]) < 20.0
                           and math.dist(self.pose, view) < 60.0)
            if (math.dist(self.pose, view) < 22.0 or at_path_end
                    or (live and d < VIEW_FINAL_R and absent_view)):
                self.view_reached = True
        use_view = view is not None and not self.view_reached
        if d < 70 and absent_view and not live:
            self.unseen_close += 1
            if self.unseen_close > 25:
                # Nothing there any more (moved or already taken).
                self.stats["tombs_absent"] += 1
                self.know.tombstone(target, step)
                self._drop_claim()
                return cmd
        final_r = 95.0
        near_final = d < final_r and (los or live) and not use_view
        if near_final and self.claim_wide and d > 40 and self.cum is not None:
            # A thin wall between us and the bomb also passes the lidar test
            # (the bomb hugs it). If the planned route is a long detour,
            # keep following the route instead of pushing into the wall.
            if self._wall_between(target, d):
                route, _ = self._cost_to(target, WIDE_RADIUS)
                if math.isfinite(route) and route > 1.5 * d + 60.0:
                    near_final = False
        if near_final:
            if self.final_since is None:
                self.final_since = step
            if step - self.final_since > 220:
                self.stats["final_timeouts"] += 1
                self.stats["grasp_failures"] += 1
                self.stats["grasp_steps"] += step - self.final_since
                self.know.failures[self.claim_key] = self.know.failures.get(self.claim_key, 0) + 1
                self.grid.add_penalty(self.pose[0], self.pose[1], 20.0, step + 300, 3.0)
                self._drop_claim()
                self._start_escape(step, lidar, angles)
                return cmd
            cmd["grasper"] = 1 if self.grasp_armed(target, step) else 0
            # Aim at the roomiest standing point that still reaches the
            # grasper, not straight at the bomb: in a corner the latter
            # presses the hull into the wall and the braking model stalls.
            if not cmd["grasper"]:
                target = self._grasp_pose(target)
            v = self._toward(target, 3.2, stop_dist=0.0)
            ignore = None
            if lidar is not None:
                rel = wrap(math.atan2(target[1] - self.pose[1], target[0] - self.pose[0])
                           - self.heading)
                dr = np.abs((angles - rel + math.pi) % (2 * math.pi) - math.pi)
                ignore = (dr < math.atan2(16.0, max(d, 1.0)) + 0.05) & (lidar > d - 30)
            cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                v, lidar, angles, False, careful=True, wall_margin=21.0,
                ignore_target=ignore)
            if V2_HEADING_ALIGN and d < 65.0:
                # Rotate toward the bomb while approaching instead of relying
                # on lateral thrust alone.  The command is proportional and
                # dead-banded, so it does not induce constant spin in open
                # space or fight the corner standing-point search.
                desired = math.atan2(target[1] - self.pose[1],
                                     target[0] - self.pose[0])
                cmd["rotation"] = heading_command(desired, self.heading,
                                                   gain=0.9, deadband=0.10,
                                                   max_command=0.65)
            return cmd
        self.final_since = None
        if view is not None and self.view_reached and (los or live) and d < VIEW_FINAL_R:
            # Seen from the viewpoint, still beyond the final radius: the
            # straight line is the open side (the route planner would send
            # us round to the blocked side again). Close in at cruise speed.
            v = self._toward(target, 6.0, stop_dist=90.0)
            cmd["forward"], cmd["lateral"] = self._velocity_to_command(v, lidar, angles, False)
            return cmd
        if (not self.path or self.path_goal != self.claim_key
                or step - self.path_step > 15):
            self._refresh_dijkstra(step, force=step - self.cum_step > 5)
            self._set_path_to(view if use_view else target, step, wide=self.claim_wide)
            self.path_goal = self.claim_key
        wp = self._waypoint()
        if (self._progress_stalled(step, self.claim_cost, window=200, need=30.0)
                or self._stuck_monitor(step, view if use_view else target)):
            self.know.failures[self.claim_key] = self.know.failures.get(self.claim_key, 0) + 1
            self._drop_claim()
            self._start_escape(step, lidar, angles, away_from=wp)
            return cmd
        if wp is None:
            wp = view if use_view else target
        v = self._toward(wp, 9.0)
        cmd["forward"], cmd["lateral"] = self._velocity_to_command(v, lidar, angles, False)
        return cmd

    # --------------------------------------------------------------- EXPLORE
    def _should_return(self, step):
        if not (RA_RETURN and self.started_in_return and self.start_pose is not None):
            return False
        if step < self.rescan_until:
            return False
        if EXPECTED_BOMBS > 0 and self._picked_count() >= EXPECTED_BOMBS:
            return True
        if self.interest_left is None:
            return False
        if self.deep_on and self.deep_left > 0:
            # Bombs are still missing and some pocket was never looked into.
            return False
        return self.interest_left <= SCOUT_MIN

    def _go_return(self, step, cmd, lidar, angles):
        """Wait at the own spawn spot (inside the return area by construction)."""
        self.state = "RETURN"
        home = self.start_pose
        if (step - self.last_team_busy > RA_IDLE_RESCAN
                and self.is_inside_return_area):
            # The round is still running although nobody has anything to do:
            # some bomb is undiscovered. Go out and re-scan for a while.
            self.rescan_until = step + RA_RESCAN_LEN
            self.last_team_busy = step
            self._event("rescan", step)
            self.goal = None
            return self._explore(step, cmd, lidar, angles)
        d = math.dist(self.pose, home)
        if d > 30:
            if (self.path_goal != "return" or not self.path
                    or step - self.path_step > 30):
                self._refresh_dijkstra(step)
                self._set_path_to(home, step)
                self.path_goal = "return"
            wp = self._waypoint() or home
            v = self._toward(wp, 7.0)
        else:
            v = self._toward(home, 2.0, stop_dist=3.0) if d > 4 else (0.0, 0.0)
        cmd["forward"], cmd["lateral"] = self._velocity_to_command(v, lidar, angles, False)
        return cmd

    def _explore(self, step, cmd, lidar, angles):
        g = self.grid
        corridor_goal_lock = bool(getattr(self, "_corridor_goal_lock", False))
        if self._should_return(step):
            if step - self.goal_eval_step > 150:
                # Refresh coverage now and then (new frontiers may appear).
                self._refresh_dijkstra(step, force=True)
                self._explore_target(step)
                self.goal_eval_step = step
            if self._should_return(step):
                return self._go_return(step, cmd, lidar, angles)
        # Goal stickiness: re-picking every 40 steps made explorers ping-pong
        # between two viewpoints (~6 switches per 600 steps) and walk further
        # than a single committed sweep. Keep a chosen goal until it is
        # reached, becomes useless, or the sticky window expires.
        if (self.goal is not None and self.goal_kind == "explore" and self.goal_deep
                and math.dist(self.pose, self.goal) < 25):
            # Arrived at a pocket viewpoint: whatever stays hidden from here
            # (behind a slot too narrow for the hull) is given up, so the
            # team never loops on the same pocket.
            g.mark_near(self.goal[0], self.goal[1], DEEP_MARK, step)
            self._event("deep_visit", step)
        if (self.goal is not None and self.goal_kind == "explore"
                and self.goal_object and math.dist(self.pose, self.goal) < 30):
            self.checked_objects.append(tuple(self.goal))
            self.checked_objects = self.checked_objects[-80:]
            self._event("object_inspect", step)
        need_new = (self.goal is None or self.goal_kind != "explore"
                    or (not corridor_goal_lock
                        and step - self.goal_eval_step > GOAL_STICKY)
                    or math.dist(self.pose, self.goal) < 25)
        if self.goal is not None and self.goal_kind == "explore" and not need_new:
            if self.goal_deep:
                if self._deep_goal_done(self.goal):
                    need_new = True
            elif not corridor_goal_lock:
                iy, ix = g.cell(*self.goal)
                if g.scan_f[iy, ix] > 0 and g.scan_f[iy, ix] > self.goal_step:
                    need_new = True
        if need_new and step >= self.me:      # staggered first decisions
            self._refresh_dijkstra(step, force=True)
            keep = None
            if (self.goal is not None and self.goal_kind == "explore"
                    and math.dist(self.pose, self.goal) >= 25):
                if step - self.goal_step > GOAL_MAX:
                    # Committed too long without arriving: the goal is
                    # probably unreachable from here. Drop it for a while.
                    self.bad_goals.append((self.goal, step + 400))
                    self._event("explore_goal_timeout", step)
                    if self.goal_deep:
                        g.mark_near(self.goal[0], self.goal[1], DEEP_MARK, step)
                else:
                    keep = self.grid.cell(*self.goal)
            choice = self._explore_target(step, keep_cell=keep)
            self.goal_eval_step = step
            if choice is None:
                self.goal = None
            else:
                self.stats["explore_goal_selections"] += 1
                goal_cell = tuple(choice[1]) if choice[1] is not None else None
                if goal_cell is not None and goal_cell in self._explore_goal_history:
                    self.stats["repeated_explore_goals"] += 1
                if goal_cell is not None:
                    self._explore_goal_history.append(goal_cell)
                same = (self.goal is not None and math.dist(choice[0], self.goal) < 1.0)
                self.goal, cell = choice
                self.goal_kind = "explore"
                self.goal_deep = self.deep_pick
                self.goal_object = self.object_pick
                if not same:
                    self.goal_step = step
                self._set_path_to(self.goal, step, end_cell=cell)
                self.path_goal = "explore"
        if self.goal is None:
            v = (0.0, 0.0)
            cmd["forward"], cmd["lateral"] = self._velocity_to_command(v, lidar, angles, False)
            return cmd
        if not self.path or step - self.path_step > 20:
            self._refresh_dijkstra(step)
            self._set_path_to(self.goal, step)
            self.path_goal = "explore"
        if self._stuck_monitor(step, self.goal):
            self.goal = None
            self._start_escape(step, lidar, angles)
            return cmd
        wp = self._waypoint() or self.goal
        v = self._toward(wp, 9.0)
        cmd["forward"], cmd["lateral"] = self._velocity_to_command(v, lidar, angles, False)
        if SEMANTIC_SWEEP_ROTATION > 0.0:
            # Translation is holonomic, so a slow yaw sweep fills the angular
            # gaps between the 35 body-fixed semantic rays without changing
            # the planned path.  Disable it near walls, during fetch, and
            # while carrying so it cannot compromise tight navigation.
            direction = 1.0 if self.me % 2 == 0 else -1.0
            clearance_scale = (1.0 if lidar is None or float(np.min(lidar)) >= 38.0
                               else 0.55)
            cmd["rotation"] = (direction * SEMANTIC_SWEEP_ROTATION
                               * clearance_scale)
        return cmd






