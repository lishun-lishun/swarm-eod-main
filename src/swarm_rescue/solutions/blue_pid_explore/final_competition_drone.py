"""Competition controller: proven SICT mission stack plus predictive safety."""

from __future__ import annotations

from collections import Counter, deque
import math
import os

import numpy as np
from scipy.spatial import ConvexHull

from .pose_fusion_drone import FusionRescueDrone
from .pose_sict_drone import (EXPECTED_BOMBS, GOAL_STICKY, CELL,
                              GRASP_REACH, V5_TIME_HORIZON, wrap)
from .corridor_explorer import CorridorExplorer, CorridorExploreState
from .lidar_geometry import footprint_vertices_from_body
from .lidar_safety import LidarPredictiveSafetyLayer
from .oriented_footprint import OrientedFootprint, Pose2
from .v5_path_supervisor import PathProgressSupervisor
from .predictive_brake import BrakingMode, PredictiveBrakingController
from .velocity_controller import body_to_world, world_to_body
from .model_velocity_control import ShadowCargoIdentifier
from .v6_local_validator import make_short_segment_validator
from .v6_braking_audit import stopping_distance_upper
from .v7_gateway import GatewayPlanner
from .v7_probe_guard import ProbePersistenceGate
from .v7_egotrail import LocalOdomTrail, retreat_desired_world_velocity
from .v7_yaw_sweep import YawPhaseSweep
from .v8_macro_frontier import MacroFrontierPlanner
from .v8_mission_guard import MissionPreservationGuard
from .v8_ego_route import EgoRouteContinuity
from .v8_motion_guard import lidar_short_segment_safe


class FinalCompetitionDrone(FusionRescueDrone):
    """Keep the measured winning strategy and filter its final 2-D thrust.

    The legacy stack remains authoritative for mapping, gossip, allocation,
    pickup and cargo delivery.  The filter is deliberately downstream, so it
    cannot create a second mission state machine.  Semantic task targets are
    cut out of the LiDAR obstacle scan only at contact range; otherwise a bomb
    or the disposal centre would be indistinguishable from a wall and safe
    pickup/delivery would be impossible.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predictive_safety = LidarPredictiveSafetyLayer()
        self.corridor_brake = PredictiveBrakingController()
        self.branch_brake = PredictiveBrakingController()
        self.v5_path_watchdog = (os.environ.get("V5_PATH_WATCHDOG", "0") == "1"
                                 or os.environ.get("V5_PATH_RECOVER", "0") == "1")
        self.v5_path_recover = os.environ.get("V5_PATH_RECOVER", "0") == "1"
        self.v5_path_supervisor = PathProgressSupervisor()
        self.v5_path_counts = Counter()
        self.v6_route_audit = os.environ.get("V6_ROUTE_AUDIT", "0") == "1"
        self.v6_connected_repair = os.environ.get("V6_CONNECTED_REPAIR", "0") == "1"
        self.v6_view_plan = os.environ.get("V6_VIEW_PLAN", "0") == "1"
        self.v6_brake_shadow = os.environ.get("V6_BRAKE_SHADOW", "0") == "1"
        self.v7_gateway_shadow = os.environ.get("V7_GATEWAY_SHADOW", "0") == "1"
        self.v7_gateway_enabled = os.environ.get("V7_GATEWAY", "0") == "1"
        self.v7_probe_enabled = os.environ.get("V7_PROBE", "0") == "1"
        self.v7_sweep_enabled = os.environ.get("V7_SWEEP", "0") == "1"
        self.v7_retreat_shadow = os.environ.get("V7_EGO_SHADOW", "0") == "1"
        self.v7_retreat_enabled = os.environ.get("V7_EGO_RETREAT", "0") == "1"
        self.v7_scout_id = int(os.environ.get("V7_SCOUT_ID", "9"))
        self.v7_ego_target_id = int(os.environ.get("V7_EGO_TARGET_ID", "-1"))
        self.v7_gateway_start = int(os.environ.get("V7_GATEWAY_START", "900"))
        self.v7_ego_trail = (LocalOdomTrail() if
                             (self.v7_retreat_shadow or self.v7_retreat_enabled)
                             else None)
        self.v7_gateway = None  # GridMap is constructed by _setup, not __init__.
        self.v7_probe = ProbePersistenceGate(required_clear_frames=3, max_step_px=8.0)
        self.v7_yaw_sweep = YawPhaseSweep(ray_count=35, phases=3)
        self._v7_gateway_goal = None
        self._v7_gateway_index = 1
        self._v7_gateway_start_step = -1
        self._v7_gateway_best_distance = math.inf
        self._v7_gateway_progress_step = -1
        self._v7_last_gateway_eval = -10000
        self._v7_probe_goal = None
        self._v7_probe_steps = 0
        self._v7_probe_in_motion = False
        self._v7_missing_carry_frames = 0
        self._v7_carry_progress = deque(maxlen=75)
        self.v7_counts = Counter()
        self.v8_macro_shadow = os.environ.get("V8_MACRO_SHADOW", "0") == "1"
        self.v8_macro_active = os.environ.get("V8_MACRO_ACTIVE", "0") == "1"
        self.v8_ego_shadow = os.environ.get("V8_EGO_SHADOW", "0") == "1"
        self.v8_ego_active = os.environ.get("V8_EGO_ACTIVE", "0") == "1"
        self.v8_scout_id = int(os.environ.get("V8_SCOUT_ID", "9"))
        self.v8_macro = MacroFrontierPlanner()
        self.v8_gate = MissionPreservationGuard()
        self.v8_route = (EgoRouteContinuity() if
                         (self.v8_ego_shadow or self.v8_ego_active) else None)
        self._v8_macro_goal = None
        self._v8_macro_index = 1
        self._v8_last_macro_eval = -100000
        self._v8_last_ego_anchor = -100000
        self._v8_last_route_id = None
        self._v8_carry_progress = deque(maxlen=70)
        self._v8_ego_last_active = -100000
        self._v8_ego_active_streak = 0
        self._v8_ego_cooldown_until = -1
        self.v8_counts = Counter()
        self._v7_action_step = -1
        self._v7_action_source = None
        self._v7_action_nominal = None
        self._v6_last_view_eval = -100000
        self._v6_pending_view = None
        self._v6_view_goal = None
        self._v6_view_seen_at_start = 0
        self._v6_view_planned_step = -1
        self._v6_view_sweep_until = -1
        self._v6_last_issue = None
        self._v6_last_issue_step = -100000
        self._v6_last_plan_step = -100000
        self.v6_route_counts = Counter()
        self._v5_last_recovery_step = -100000
        self.cargo_identifier = (
            ShadowCargoIdentifier()
            if os.environ.get("V4_CARGO_IDENTIFICATION", "0") == "1"
            else None
        )
        self._safety_footprint = footprint_vertices_from_body(self.base)
        self.corridor_explorer = CorridorExplorer(
            OrientedFootprint(self._safety_footprint, margin=1.0),
            advance_speed=float(os.environ.get("CORRIDOR_ADVANCE_SPEED", "2.8")),
        )
        self.corridor_wall_margin = float(
            os.environ.get("CORRIDOR_WALL_MARGIN", "10.0"))
        specialist_text = os.environ.get("CORRIDOR_SPECIALISTS", "9").strip()
        corridor_specialists = (
            None if specialist_text == "*" else
            {int(value) for value in specialist_text.split(",") if value.strip()}
        )
        self.corridor_enabled = (
            os.environ.get("CORRIDOR_EXPLORER", "1") == "1"
            and (corridor_specialists is None
                 or int(self.identifier or 0) in corridor_specialists)
        )
        self.corridor_start_step = int(os.environ.get("CORRIDOR_START_STEP", "3000"))
        self.corridor_degraded_start = int(
            os.environ.get("CORRIDOR_DEGRADED_START", "-1"))
        self.corridor_owner_mod = int(
            os.environ.get("CORRIDOR_OWNER_MOD", "0"))
        self.degraded_pickup_slots = int(
            os.environ.get("DEGRADED_PICKUP_SLOTS", "0"))
        self.degraded_pickup_slot_frames = int(
            os.environ.get("DEGRADED_PICKUP_SLOT_FRAMES", "120"))
        self.corridor_trace_enabled = os.environ.get("CORRIDOR_TRACE", "0") == "1"
        self.carry_trace_enabled = os.environ.get("CARRY_TRACE", "0") == "1"
        self.branch_trace_enabled = os.environ.get("BRANCH_TRACE", "0") == "1"
        self.pose_trace_enabled = os.environ.get("POSE_TRACE", "0") == "1"
        self.pose_trace = deque(maxlen=6000)
        self.branch_trace = deque(maxlen=900)
        self._branch_trace_row = None
        self.carry_trace = deque(maxlen=360)
        self._trace_ring = deque(maxlen=30)
        self._trace_pending = []
        self.corridor_reversal_traces = []
        self.corridor_reason_counts = Counter()
        self._trace_last_world_command = None
        self._trace_last_goal = None
        self._corridor_control_state = CorridorExploreState.IDLE
        self._corridor_end_gap = math.inf
        self._corridor_goal_lock = False
        self._corridor_branch_goal = None
        self.branch_local_enabled = os.environ.get("ENABLE_BRANCH_LOCAL_EXECUTOR", "0") == "1"
        self._branch_local_best = -math.inf
        self._branch_local_progress_step = -1
        self._branch_lidar_range = None
        self._last_blocked_reselect_step = -1000
        self._last_degraded_state = None
        self._gps_outage_frames = 0
        self._gps_recovery_frames = 60
        self._comm_outage_frames = 0
        self._comm_recovery_frames = 60
        self._stable_gps_degraded = False
        self._stable_comm_degraded = False
        self.safety_stats = {
            "filtered_frames": 0,
            "predictive_brake_frames": 0,
            "infeasible_frames": 0,
        }

    def _setup(self):
        super()._setup()
        if self.v7_gateway_enabled or self.v7_gateway_shadow:
            self.v7_gateway = GatewayPlanner(
                origin_xy=(-self.grid.width / 2.0, -self.grid.height / 2.0),
                cell_px=CELL, min_clearance_px=16.0, max_expansions=8000)

    def _v7_known_free_segment(self, a, b):
        """Check the complete link in the fused free belief, not just clearance."""
        distance = math.dist(a, b)
        count = max(2, int(math.ceil(distance / (CELL * 0.4))) + 1)
        for fraction in np.linspace(0.0, 1.0, count):
            point = (a[0] + fraction * (b[0] - a[0]),
                     a[1] + fraction * (b[1] - a[1]))
            iy, ix = self.grid.cell(*point)
            if (self.grid.occ[iy, ix] >= -0.6
                    or self.grid.clearance[iy, ix] < 16.0
                    or self.grid.border[iy, ix]):
                return False
        return True

    def _v7_live_segment(self, a, b, lidar, angles, hull_radius):
        if lidar is None or angles is None or self.heading is None:
            return False
        distance = math.dist(a, b)
        if distance < 1e-6:
            return True
        bearing = wrap(math.atan2(b[1] - a[1], b[0] - a[0]) - self.heading)
        ranges = np.asarray(lidar, float)
        rays = np.asarray(angles, float)
        difference = np.abs((rays - bearing + math.pi) % (2 * math.pi) - math.pi)
        nearest = np.argsort(difference)[:3]
        return (nearest.size == 3 and np.all(np.isfinite(ranges[nearest]))
                and float(np.min(ranges[nearest])) > distance + hull_radius + 4.0)

    def _v7_end_gateway(self, step, reason, *, failed=False):
        gateway = self._v7_gateway_goal
        if gateway is not None:
            self._event("v7_gateway_end", step, reason=reason,
                        gateway=list(gateway.xy),
                        reached=reason == "REACHED")
            if self.v7_gateway is not None:
                self.v7_gateway.mark_failed(
                    gateway.cell, step, cooldown=350 if failed else 550)
        self._v7_gateway_goal = None
        self._v7_probe_goal = None
        self._v7_probe_steps = 0
        self._v7_probe_in_motion = False
        self.v7_probe.reset()
        self.v7_yaw_sweep.reset()

    def _v7_note_action(self, step, command, source):
        self._v7_action_step = step
        self._v7_action_source = source
        self._v7_action_nominal = {
            key: float(command[key]) for key in ("forward", "lateral", "rotation")}
        return command

    def _v7_gateway_explore(self, step, cmd, lidar, angles):
        """One-scout, bounded known-free route with an optional short probe."""
        if self.v7_gateway is None or self.me != self.v7_scout_id:
            return None
        eligible = (step >= self.v7_gateway_start
                    and self.grid.clearance is not None
                    and self.grid.fused_step >= 0
                    and self.pose is not None and self.heading is not None
                    and self.measured_gps_position() is not None
                    and not self.grasped_bombs() and self.claim_key is None
                    and not self.visible_bombs
                    and not self.corridor_explorer.committed
                    and step - self.last_team_busy > 120
                    and self.know.accounted() < EXPECTED_BOMBS)
        if not eligible:
            if self._v7_gateway_goal is not None:
                self._v7_end_gateway(step, "INTERRUPTED")
            return None
        if (self._v7_gateway_goal is None
                and step - self._v7_last_gateway_eval >= 60):
            self._v7_last_gateway_eval = step
            radius = self.corridor_explorer.footprint.radius
            proposal = self.v7_gateway.choose(
                occ=self.grid.occ, clearance_px=self.grid.clearance,
                own_evidence=self.grid.ev,
                semantic_seen_at=(self.v6_adapter.visibility.seen_at
                                  if self.v6_adapter is not None else None),
                current_xy=self.pose, now=step,
                start_link_safe=lambda a, b: (
                    self._v7_known_free_segment(a, b)
                    and self._v7_live_segment(a, b, lidar, angles, radius)),
                peer_goals=[peer["goal"] for peer in self.know.peers.values()
                            if peer.get("goal") is not None])
            if proposal is not None:
                self.v7_counts["planned"] += 1
                self._event("v7_gateway_planned", step,
                            target=list(proposal.xy),
                            route_cost_px=round(proposal.route_cost_px, 1),
                            route_points=len(proposal.route),
                            semantic_gap=proposal.semantic_gap)
                if self.v7_gateway_enabled:
                    self._v7_gateway_goal = proposal
                    self._v7_gateway_index = 1
                    self._v7_gateway_start_step = step
                    self._v7_gateway_best_distance = math.inf
                    self._v7_gateway_progress_step = step
        if not self.v7_gateway_enabled or self._v7_gateway_goal is None:
            return None
        gateway = self._v7_gateway_goal
        if step - self._v7_gateway_start_step > 340:
            self._v7_end_gateway(step, "TIMEOUT", failed=True)
            return None
        if self._v7_probe_goal is not None:
            target = self._v7_probe_goal
            if math.dist(self.pose, target) < 3.0:
                self.v7_counts["probed"] += 1
                self._event("v7_probe_reached", step, target=list(target))
                self._v7_probe_goal = None
                self._v7_probe_in_motion = False
            else:
                self._v7_probe_in_motion = True
                velocity = self._toward(target, 1.2, stop_dist=1.0)
                cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                    velocity, lidar, angles, False, careful=True)
                cmd["rotation"] = 0.0
                return self._v7_note_action(step, cmd, "V7_PROBE")
        while (self._v7_gateway_index < len(gateway.route)
               and math.dist(self.pose, gateway.route[self._v7_gateway_index]) < 10.0):
            self._v7_gateway_index += 1
        if self._v7_gateway_index < len(gateway.route):
            target = gateway.route[self._v7_gateway_index]
            if not self._v7_known_free_segment(self.pose, target):
                self._v7_end_gateway(step, "MAP_LINK_BLOCKED", failed=True)
                return None
            if self._v7_gateway_index <= 2 and not self._v7_live_segment(
                    self.pose, target, lidar, angles,
                    self.corridor_explorer.footprint.radius):
                self._v7_end_gateway(step, "LIVE_LINK_BLOCKED", failed=True)
                return None
            remaining = math.dist(self.pose, gateway.xy)
            if remaining < self._v7_gateway_best_distance - 2.0:
                self._v7_gateway_best_distance = remaining
                self._v7_gateway_progress_step = step
            if step - self._v7_gateway_progress_step > 75:
                self._v7_end_gateway(step, "NO_PROGRESS", failed=True)
                return None
            velocity = self._toward(target, 3.0)
            cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                velocity, lidar, angles, False, careful=True)
            cmd["rotation"] = 0.0
            return self._v7_note_action(step, cmd, "V7_GATEWAY")
        if math.dist(self.pose, gateway.xy) > 16.0:
            self._v7_end_gateway(step, "NOT_ACTUALLY_REACHED", failed=True)
            return None
        if self._v7_probe_steps == 0:
            self.v7_counts["reached"] += 1
            self._event("v7_gateway_reached", step, target=list(gateway.xy),
                        route_points=len(gateway.route))
            self._v7_probe_steps = 1
        cmd["forward"], cmd["lateral"] = self._velocity_to_command(
            (0.0, 0.0), lidar, angles, False, careful=True)
        if self.v7_sweep_enabled:
            sweep = self.v7_yaw_sweep.update(
                current_heading=self.heading, carrying=False,
                visible_target=bool(self.visible_bombs), stable_standpoint=True)
            if not sweep.done:
                cmd["rotation"] = sweep.rotation
                return self._v7_note_action(step, cmd, "V7_SWEEP")
        if self.v7_probe_enabled and self._v7_probe_steps <= 4:
            observation = self.v7_probe.observe(
                gateway_cell=gateway.cell, position=self.pose,
                heading=self.heading, target_heading=gateway.heading,
                lidar_ranges=lidar, lidar_angles=angles,
                body_radius_px=self.corridor_explorer.footprint.radius,
                carrying=False, dynamic_obstruction=bool(self.visible_drones))
            if observation.confirmed:
                target = observation.proposal.xy
                cell = self.grid.cell(*target)
                if (self.grid.border[cell] or self.grid.occ[cell] > 1.0):
                    self._v7_end_gateway(step, "KNOWN_OBSTACLE", failed=True)
                    return None
                self._v7_probe_goal = target
                self._v7_probe_steps += 1
                self.v7_counts["probe_confirmed"] += 1
                self._event("v7_probe_confirmed", step, target=list(target),
                            lidar_frames=observation.consecutive)
                return self._v7_note_action(step, cmd, "V7_PROBE")
            if observation.reason == "WAIT_FOR_REPEAT_SCAN":
                return self._v7_note_action(step, cmd, "V7_PROBE")
        self._v7_end_gateway(step, "REACHED")
        return None

    def _v8_end_macro(self, step, reason, *, reached=False, blocked=False):
        goal = self._v8_macro_goal
        if goal is None:
            return
        semantic_now = (int(np.count_nonzero(self.v6_adapter.visibility.seen_at))
                        if self.v6_adapter is not None else 0)
        gain = max(0, semantic_now - self._v8_semantic_at_start)
        self.v8_macro.mark_result(goal.region, step, reached=reached,
                                  blocked=blocked, new_semantic_cells=gain)
        self._event("v8_macro_end", step, reason=reason, target=list(goal.xy),
                    new_semantic_cells=gain, reached=bool(reached))
        self._v8_macro_goal = None

    def _v8_macro_explore(self, step, cmd, lidar, angles):
        if not (self.v8_macro_shadow or self.v8_macro_active):
            return None
        if self.me != self.v8_scout_id or self.v6_view_plan or self.v7_gateway_enabled:
            return None
        eligible = (self.state == "EXPLORE" and not self.corridor_explorer.committed
                    and self._v7_gateway_goal is None and self.grid.clearance is not None
                    and step >= 900 and self.pose is not None and self.heading is not None
                    and self.measured_gps_position() is not None
                    and lidar is not None and angles is not None
                    and not self.grasped_bombs() and self.claim_key is None
                    and not self.visible_bombs and self.know.accounted() < EXPECTED_BOMBS)
        if not eligible:
            self._v8_end_macro(step, "MISSION_PREEMPTED")
            return None
        peers = [dict(pos=peer.get("pos"), goal=peer.get("goal"),
                      carry=bool(peer.get("carry")),
                      last_seen=int(peer.get("step", -999999)))
                 for peer in self.know.peers.values()]
        deadline = V5_TIME_HORIZON if V5_TIME_HORIZON > 0 else 0
        if (self._v8_macro_goal is None
                and step - self._v8_last_macro_eval >= 90):
            self._v8_last_macro_eval = step
            radius = self.corridor_explorer.footprint.radius
            import time as _time
            started = _time.perf_counter()
            proposal = self.v8_macro.select(
                occ=self.grid.occ, clearance_px=self.grid.clearance,
                current_xy=self.pose,
                map_origin_xy=(-self.grid.width / 2, -self.grid.height / 2),
                now=step, own_evidence=self.grid.ev,
                semantic_seen_at=(self.v6_adapter.visibility.seen_at
                                  if self.v6_adapter is not None else None),
                peer_goal_xy=[p["goal"] for p in peers if p["goal"] is not None],
                local_start_safe=lambda a, b: (
                    self._v7_known_free_segment(a, b)
                    and self._v7_live_segment(a, b, lidar, angles, radius)))
            elapsed_ms = (_time.perf_counter() - started) * 1000.0
            self.v8_counts["planner_calls"] += 1
            self.v8_counts["planner_ms_total"] += elapsed_ms
            self.v8_counts["planner_ms_max"] = max(
                self.v8_counts["planner_ms_max"], elapsed_ms)
            if proposal is not None:
                gate = self.v8_gate.check(
                    now=step, deadline=deadline, state=self.state,
                    carrying=False, has_claim=False,
                    visible_ungrasped_bomb=False, own_pose=self.pose,
                    proposed_xy=proposal.xy, known_peers=peers,
                    emergency=bool(self.visible_drones or step < self.escape_until))
                self.v8_counts["planned"] += 1
                self._event("v8_macro_planned", step, target=list(proposal.xy),
                            region=list(proposal.region),
                            route_cost_px=round(proposal.route_cost_px, 1),
                            unknown_nearby=round(proposal.unknown_nearby, 1),
                            far_unknown=round(proposal.far_unknown, 1),
                            expansions=proposal.expansions,
                            planner_ms=round(elapsed_ms, 2),
                            guard=gate.reason)
                if self.v8_macro_active and gate.allowed:
                    self._v8_macro_goal = proposal
                    self._v8_macro_index = 1
                    self._v8_macro_start = step
                    self._v8_macro_best = math.inf
                    self._v8_macro_progress_step = step
                    self._v8_semantic_at_start = (
                        int(np.count_nonzero(self.v6_adapter.visibility.seen_at))
                        if self.v6_adapter is not None else 0)
                    self.v8_counts["active"] += 1
        goal = self._v8_macro_goal
        if goal is None or not self.v8_macro_active:
            return None
        gate = self.v8_gate.check(
            now=step, deadline=deadline, state=self.state,
            carrying=False, has_claim=False, visible_ungrasped_bomb=False,
            own_pose=self.pose, proposed_xy=goal.xy, known_peers=peers,
            emergency=bool(self.visible_drones or step < self.escape_until))
        if not gate.allowed:
            self._v8_end_macro(step, gate.reason, blocked=True)
            return None
        if step - self._v8_macro_start > 600:
            self._v8_end_macro(step, "TIMEOUT", blocked=True)
            return None
        while (self._v8_macro_index < len(goal.route)
               and math.dist(self.pose, goal.route[self._v8_macro_index]) < 8.0):
            self._v8_macro_index += 1
        if self._v8_macro_index >= len(goal.route):
            if math.dist(self.pose, goal.xy) <= 16.0:
                self.v8_counts["reached"] += 1
                self._v8_end_macro(step, "REACHED", reached=True)
            else:
                self._v8_end_macro(step, "FALSE_ARRIVAL", blocked=True)
            return None
        target = goal.route[self._v8_macro_index]
        if not self._v7_known_free_segment(self.pose, target):
            self._v8_end_macro(step, "MAP_LINK_BLOCKED", blocked=True)
            return None
        if not self._v7_live_segment(
                self.pose, target, lidar, angles,
                self.corridor_explorer.footprint.radius):
            self._v8_end_macro(step, "LIVE_LINK_BLOCKED", blocked=True)
            return None
        remaining = math.dist(self.pose, goal.xy)
        if remaining < self._v8_macro_best - 2.0:
            self._v8_macro_best = remaining
            self._v8_macro_progress_step = step
        if step - self._v8_macro_progress_step > 80:
            self._v8_end_macro(step, "NO_PROGRESS", blocked=True)
            return None
        velocity = self._toward(target, 2.6)
        cmd["forward"], cmd["lateral"] = self._velocity_to_command(
            velocity, lidar, angles, False, careful=True)
        cmd["rotation"] = 0.0
        self._v7_note_action(step, cmd, "V8_MACRO")
        return cmd

    @staticmethod
    def _sector_min(distances, angles, center, half=0.18):
        d = np.asarray(distances, float)
        a = np.asarray(angles, float)
        error = np.abs((a - center + math.pi) % (2.0 * math.pi) - math.pi)
        values = d[np.isfinite(d) & (d > 0.0) & (error <= half)]
        return None if values.size == 0 else round(float(np.min(values)), 2)

    def _trace_explore(self, command, lidar, angles, reason):
        if not self.corridor_trace_enabled or self.pose is None:
            return
        step = int(self.elapsed_timestep)
        world = body_to_world((float(command["forward"]), float(command["lateral"])),
                              self.heading)
        goal = None if self.goal is None else [round(float(v), 2) for v in self.goal]
        sample = {
            "step": step,
            "mode": str(self.state),
            "reason": str(reason),
            "pose": [round(float(v), 2) for v in self.pose],
            "heading": round(float(self.heading), 4),
            "velocity": [round(float(v), 3) for v in self.vel],
            "command_world": [round(float(v), 3) for v in world],
            "goal": goal,
            "path_points": int(len(self.path)),
            "path_index": int(getattr(self, "path_i", 0)),
            "interest_left": None if self.interest_left is None else int(self.interest_left),
            "gps_valid": self.measured_gps_position() is not None,
            "lidar_min": round(float(np.nanmin(lidar)), 2),
            "lidar_front": self._sector_min(lidar, angles, 0.0),
            "lidar_left": self._sector_min(lidar, angles, math.pi / 2.0),
            "lidar_right": self._sector_min(lidar, angles, -math.pi / 2.0),
            "corridor_state": self.corridor_explorer.state.value,
        }
        for pending in list(self._trace_pending):
            pending["after"].append(sample)
            pending["remaining"] -= 1
            if pending["remaining"] <= 0:
                self._trace_pending.remove(pending)
        previous = self._trace_last_world_command
        previous_goal = self._trace_last_goal
        norm = float(np.linalg.norm(world))
        prev_norm = 0.0 if previous is None else float(np.linalg.norm(previous))
        reversed_motion = (
            previous is not None and norm > 0.18 and prev_norm > 0.18
            and float(np.dot(world, previous) / (norm * prev_norm)) < -0.35
        )
        goal_changed = previous_goal is not None and goal is not None and math.dist(
            previous_goal, goal) > 35.0
        if reversed_motion and len(self.corridor_reversal_traces) < 12:
            cause = ("GOAL_RESELECT" if goal_changed else
                     "RETURN" if self.state == "RETURN" else
                     "PATH_EMPTY" if not self.path else "LOCAL_CONTROL")
            record = {
                "drone": int(self.me), "step": step, "cause": cause,
                "before": list(self._trace_ring), "after": [sample], "remaining": 30,
            }
            self.corridor_reversal_traces.append(record)
            self._trace_pending.append(record)
        self._trace_ring.append(sample)
        self._trace_last_world_command = world.copy()
        self._trace_last_goal = None if goal is None else tuple(goal)

    def _corridor_travel_heading(self):
        target = self.goal
        if self.path:
            index = min(len(self.path) - 1, max(0, int(getattr(self, "path_i", 0))) + 3)
            target = self.path[index]
        if target is None or math.dist(self.pose, target) < 2.0:
            speed = math.hypot(*self.vel)
            return self.heading if speed < 0.2 else math.atan2(self.vel[1], self.vel[0])
        return math.atan2(target[1] - self.pose[1], target[0] - self.pose[0])

    def _fetch(self, step, cmd, lidar, angles):
        self._v8_end_macro(step, "BOMB_TASK_PREEMPTED")
        if self._v7_gateway_goal is not None:
            self._v7_end_gateway(step, "FETCH_INTERRUPTED")
        self.corridor_explorer.pause()
        self._corridor_control_state = CorridorExploreState.PAUSED
        return super()._fetch(step, cmd, lidar, angles)

    def _carrier_slot_free(self):
        if not super()._carrier_slot_free():
            return False
        slots = self.degraded_pickup_slots
        if (slots <= 1
                or not (self._stable_gps_degraded
                        or self._stable_comm_degraded)):
            return True
        phase = (int(self.elapsed_timestep)
                 // max(1, self.degraded_pickup_slot_frames)) % slots
        return self.me % slots == phase

    def _carry(self, step, cmd, lidar, angles):
        self._v8_end_macro(step, "CARRY_PREEMPTED")
        if self._v7_gateway_goal is not None:
            self._v7_end_gateway(step, "CARRY_INTERRUPTED")
        self.corridor_explorer.pause()
        self._corridor_control_state = CorridorExploreState.PAUSED
        command = super()._carry(step, cmd, lidar, angles)
        if self.v8_route is not None:
            command = self._v8_route_carry(step, command, lidar, angles)
        if self.v7_ego_trail is not None:
            command = self._v7_ego_carry(step, command, lidar, angles)
        if (self.v6_adapter is not None and self.pose is not None
                and (self.v6_route_audit or self.v6_connected_repair)):
            self._v6_observe_carry_route(step, command, lidar, angles)
        if self.v5_path_watchdog and self.pose is not None:
            self._v5_observe_carry_path(step, lidar, angles)
        if self.carry_trace_enabled and self.pose is not None:
            world = body_to_world(
                (float(command["forward"]), float(command["lateral"])), self.heading,
            )
            waypoint = None
            if self.path:
                index = min(len(self.path) - 1, max(0, int(getattr(self, "path_i", 0))))
                waypoint = [round(float(v), 2) for v in self.path[index]]
            center_cost = None
            if self.center_cum is not None:
                start = self.grid.free_start(self.grid.costs["carry"], *self.pose, radius=5)
                if start is not None:
                    center_cost = round(float(self.center_cum[start]) * 10.0, 2)
            nearest_center = min(self.center_hits, key=lambda h: h[2]) if self.center_hits else None
            self.carry_trace.append({
                "step": int(step),
                "pose": [round(float(v), 2) for v in self.pose],
                "heading": round(float(self.heading), 4),
                "velocity": [round(float(v), 3) for v in self.vel],
                "gps_valid": self.measured_gps_position() is not None,
                "comm_disabled": bool(self.communicator_is_disabled()),
                "command_body": [round(float(command[key]), 3)
                                 for key in ("forward", "lateral", "rotation")],
                "command_world": [round(float(v), 3) for v in world],
                "path_goal": self.path_goal,
                "path_points": int(len(self.path)),
                "path_index": int(getattr(self, "path_i", 0)),
                "waypoint": waypoint,
                "center_cost": center_cost,
                "center_hits": int(len(self.center_hits)),
                "center_nearest_distance": None if nearest_center is None else round(float(nearest_center[2]), 2),
                "center_nearest_hit": None if nearest_center is None else [
                    round(float(nearest_center[0]), 2),
                    round(float(nearest_center[1]), 2)],
                "center_nearest_passable": None if nearest_center is None else bool(
                    self._center_passable((nearest_center[0], nearest_center[1]))),
                "map_center_points_near": [
                    [round(float(v), 1) for v in point]
                    for point in self.grid.center_points
                    if math.dist(point, self.pose) < 140.0
                ][:8],
                "escape_until": int(self.escape_until),
                "escape_dir": None if self.escape_dir is None else [
                    round(float(v), 3) for v in self.escape_dir
                ],
                "lidar_min": None if lidar is None else round(float(np.nanmin(lidar)), 2),
                "lidar_front": None if lidar is None else self._sector_min(lidar, angles, 0.0),
                "lidar_back": None if lidar is None else self._sector_min(lidar, angles, math.pi),
                "lidar_left": None if lidar is None else self._sector_min(lidar, angles, math.pi / 2.0),
                "lidar_right": None if lidar is None else self._sector_min(lidar, angles, -math.pi / 2.0),
            })
        return command

    def _v7_ego_carry(self, step, command, lidar, angles):
        """Observe a short local retreat; never merge ego coordinates into GridMap."""
        if self.v7_ego_target_id >= 0 and self.me != self.v7_ego_target_id:
            return command
        gps_invalid = self.measured_gps_position() is None
        semantic_near = any(hit[2] < 150.0 for hit in self.center_hits)
        no_route = not self.path or self.path_i >= len(self.path)
        waypoint = (self.path[min(self.path_i, len(self.path) - 1)]
                    if self.path else None)
        if gps_invalid and waypoint is not None:
            self._v7_carry_progress.append((
                step, math.dist(self.pose, waypoint), tuple(self.pose)))
        else:
            self._v7_carry_progress.clear()
        stalled = False
        if len(self._v7_carry_progress) >= 45:
            old_step, old_distance, old_pose = self._v7_carry_progress[-45]
            stalled = (step - old_step >= 44
                       and old_distance - self._v7_carry_progress[-1][1] < 8.0
                       and math.dist(old_pose, self.pose) < 25.0)
        candidate = (gps_invalid and (no_route or stalled) and not semantic_near
                     and step >= self.escape_until and step >= self.backoff_until
                     and self.push_since is None and lidar is not None
                     and angles is not None and not self.visible_drones)
        self._v7_missing_carry_frames = (
            self._v7_missing_carry_frames + 1 if candidate else 0)
        if self._v7_missing_carry_frames < 8:
            return command
        # PivotJoint cargo can swing around the airframe. A full grasp-reach
        # envelope is deliberately conservative until a loaded sweep model is
        # validated against official Pymunk. Only current local rays are used.
        loaded_radius = max(self.corridor_explorer.footprint.radius,
                            GRASP_REACH + 19.0)
        def local_safe(bearing, length):
            distances = np.asarray(lidar, float)
            rays = np.asarray(angles, float)
            delta = np.abs((rays - bearing + math.pi) % (2 * math.pi) - math.pi)
            nearby = np.argsort(delta)[:5]
            return (nearby.size == 5 and np.all(np.isfinite(distances[nearby]))
                    and float(np.min(distances[nearby])) >
                    length + loaded_radius + 5.0)
        request = self.v7_ego_trail.retreat_target(
            now=step, path_missing=bool(no_route or stalled), carrying=True,
            local_segment_safe=local_safe)
        if request is None:
            self.v7_counts["ego_rejected"] += 1
            if step % 50 == self.me % 50:
                self._event("v7_ego_no_safe_retreat", step,
                            gps_valid=False, path_points=len(self.path),
                            loaded_radius=round(loaded_radius, 1))
            return command
        self.v7_counts["ego_verified"] += 1
        if step % 20 == self.me % 20:
            self._event("v7_ego_retreat_candidate", step,
                        distance_px=round(request.distance_px, 1),
                        age_frames=request.age_frames,
                        drift_px=round(request.estimated_drift_px, 1),
                        active=bool(self.v7_retreat_enabled))
        if not self.v7_retreat_enabled:
            return command
        # This remains the legacy cargo controller's nominal action and is
        # subsequently passed through its unchanged final safety chain.
        original = {key: float(command[key]) for key in
                    ("forward", "lateral", "rotation")}
        velocity = retreat_desired_world_velocity(
            request, self.heading, speed_px_per_frame=1.3)
        command["forward"], command["lateral"] = self._velocity_to_command(
            velocity, lidar, angles, True, careful=True, wall_margin=18.0)
        command["rotation"] = 0.0
        command["grasper"] = 1
        self.v7_counts["ego_active"] += 1
        self._v7_note_action(step, command, "V7_EGO_RETREAT")
        self._event("v7_ego_retreat_action", step, source="V7_EGO_RETREAT",
                    original=original,
                    nominal={key: float(command[key]) for key in
                             ("forward", "lateral", "rotation")},
                    gps_valid=False, lidar_valid=True)
        return command

    def _v8_route_carry(self, step, command, lidar, angles):
        if not (self.v8_ego_shadow or self.v8_ego_active):
            return command
        gps_valid = self.measured_gps_position() is not None
        carrying = bool(self.grasped_bombs())
        if not carrying:
            self.v8_route.reset()
            self._v8_carry_progress.clear()
            return command
        if gps_valid:
            self._v8_carry_progress.clear()
            route_id = (self.path_goal, self.path_step)
            if (self.path and self.path_goal == "center"
                    and (step - self._v8_last_ego_anchor >= 40
                         or route_id != self._v8_last_route_id)):
                if self.v8_route.capture(
                        step=step, legal_pose_xy=self.pose,
                        legal_heading=self.heading,
                        world_path=self.path[self.path_i:],
                        path_id=route_id, gps_valid=True):
                    self._v8_last_ego_anchor = step
                    self._v8_last_route_id = route_id
                    self.v8_counts["ego_anchor"] += 1
                    self._event("v8_route_anchor", step,
                                route_points=len(self.v8_route.route))
            return command
        waypoint = (self.path[min(self.path_i, len(self.path) - 1)]
                    if self.path and self.path_i < len(self.path) else None)
        if waypoint is None:
            stalled = True
        else:
            self._v8_carry_progress.append((
                step, math.dist(self.pose, waypoint), tuple(self.pose)))
            stalled = False
            if len(self._v8_carry_progress) >= 55:
                old_step, old_distance, old_pose = self._v8_carry_progress[-55]
                stalled = (step - old_step >= 54
                           and old_distance - self._v8_carry_progress[-1][1] < 8.0
                           and math.dist(old_pose, self.pose) < 25.0)
        if (not stalled or any(hit[2] < 150 for hit in self.center_hits)
                or self.push_since is not None
                or step < self.escape_until or step < self.backoff_until
                or self.visible_drones or lidar is None or angles is None):
            self._v8_ego_active_streak = 0
            return command
        safe = lambda bearing, distance: lidar_short_segment_safe(
            bearing_body=bearing, distance_px=distance,
            lidar_ranges=lidar, lidar_angles=angles,
            hull_radius_px=self.corridor_explorer.footprint.radius,
            carrying=True, peer_nearby=bool(self.visible_drones))
        proposal = self.v8_route.propose(
            step=step, gps_valid=False, carrying=True,
            stalled_or_missing=True, local_segment_safe=safe,
            peer_blocked=False)
        self.v8_counts[f"ego_{proposal.reason}"] += 1
        if step % 20 == self.me % 20:
            self._event("v8_route_candidate", step, reason=proposal.reason,
                        age=None if self.v8_route.anchor_step is None
                        else step - self.v8_route.anchor_step,
                        distance_px=round(proposal.distance_px, 1),
                        drift_px=(round(proposal.estimated_drift_px, 2)
                                  if math.isfinite(proposal.estimated_drift_px) else None),
                        path_points=len(self.path), active=self.v8_ego_active)
        if not proposal.ok or not self.v8_ego_active:
            return command
        if step < self._v8_ego_cooldown_until:
            return command
        if self._v8_ego_active_streak >= 24:
            self._v8_ego_active_streak = 0
            self._v8_ego_cooldown_until = step + 50
            return command
        self._v8_ego_active_streak += 1
        self._v8_ego_last_active = step
        original = {key: float(command[key]) for key in
                    ("forward", "lateral", "rotation")}
        bearing = self.heading + proposal.bearing_body
        velocity = (math.cos(bearing), math.sin(bearing))
        command["forward"], command["lateral"] = self._velocity_to_command(
            velocity, lidar, angles, True, careful=True, wall_margin=18.0)
        command["grasper"] = 1
        self.v8_counts["ego_active"] += 1
        self._v7_note_action(step, command, "V8_EGO_ROUTE")
        self._event("v8_route_action", step, source="V8_EGO_ROUTE",
                    original=original,
                    nominal={key: float(command[key]) for key in
                             ("forward", "lateral", "rotation")})
        return command

    def _v6_observe_carry_route(self, step, command, lidar, angles):
        waypoint = (self.path[min(max(0, self.path_i), len(self.path) - 1)]
                    if self.path else None)
        cost = None
        if self.center_cum is not None and self.grid.costs is not None:
            cell = self.grid.free_start(self.grid.costs["carry"], *self.pose, radius=5)
            if cell is not None and math.isfinite(float(self.center_cum[cell])):
                cost = float(self.center_cum[cell]) * CELL
        nearest = min((hit[2] for hit in self.center_hits), default=None)
        issue = self.v6_adapter.monitor_carrier(
            step=step, route_tag=str(self.path_goal), pose=self.pose,
            waypoint=waypoint, route_count=len(self.path),
            commanded_speed=math.hypot(command["forward"], command["lateral"]),
            center_cost_px=cost, center_surface_distance_px=nearest,
            carrying=bool(self.grasped_bombs()))
        if issue.kind in ("OK", "COOLDOWN", "LOW_COMMAND", "YIELD"):
            return
        self.v6_route_counts[issue.kind] += 1
        if (issue.kind != self._v6_last_issue
                or step - self._v6_last_issue_step >= 40):
            self._event("v6_carry_issue", step, issue=issue.kind,
                        reason=issue.reason, center_cost=cost,
                        center_semantic=nearest, path_points=len(self.path),
                        waypoint_progress=round(issue.waypoint_progress_px, 2))
            self._v6_last_issue = issue.kind
            self._v6_last_issue_step = step
        if (not self.v6_connected_repair or issue.kind != "MISSING_ROUTE"
                or cost is None or cost < 90.0
                or (nearest is not None and nearest < 110.0)
                or lidar is None or angles is None
                or self.measured_gps_position() is None
                or step - self._v6_last_plan_step < 120):
            return
        self._v6_last_plan_step = step
        if not (self.grid.center_points or self.grid.seen_points):
            self.v6_route_counts["no_center_witness"] += 1
            return
        # Candidate routes are local beliefs. The official disposal event is
        # the only delivery confirmation; a center map cell is not arrival.
        validate = make_short_segment_validator(
            grid=self.grid, position=self.pose, heading=self.heading,
            lidar_ranges=lidar, lidar_angles=angles,
            footprint_vertices=self._safety_footprint, carrying=True)
        targets = sorted(self.grid.center_points or self.grid.seen_points,
                         key=lambda p: math.dist(p, self.pose))[:4]
        for target in targets:
            route = self.v6_adapter.propose_reconnected_loaded_path(
                grid=self.grid, current_pose=self.pose,
                center_witness_xy=target, segment_safe=validate,
                min_clearance_px=18.0)
            self.v6_route_counts[route.status] += 1
            if route.status != "MAP_CONNECTED" or len(route.points) < 3:
                continue
            first = route.points[1]
            if not validate(self.pose, first):
                self.v6_route_counts["first_lidar_reject"] += 1
                continue
            if not all(self._v5_loaded_segment_safe(a, b)
                       for a, b in zip(route.points[:3], route.points[1:4])):
                self.v6_route_counts["first_hull_reject"] += 1
                continue
            # Commit only after both map connection and live first-leg checks.
            self.path = list(route.points[1:])
            self.path_i = 0
            self.path_goal = "center"
            self.path_step = step
            self._reset_progress()
            self.v6_route_counts["verified_replan"] += 1
            self._event("v6_connected_repair", step,
                        path_points=len(self.path), expansions=route.expansions,
                        length_px=round(route.length_px, 1))
            break

    def _explore_target(self, step, keep_cell=None, dist_scale=1.0):
        baseline = super()._explore_target(step, keep_cell=keep_cell,
                                            dist_scale=dist_scale)
        if (not self.v6_view_plan or self.v6_adapter is None
                or self.pose is None or self.cum is None
                or self.grasped_bombs() or self.claim_key is not None
                or self.know.accounted() >= EXPECTED_BOMBS
                or step - self.start_step < 300
                or np.count_nonzero(self.grid.occ < -0.6) < 250
                or self.interest_left is None or self.interest_left > 80
                or step - self._v6_last_view_eval < 30):
            return baseline
        self._v6_last_view_eval = step
        lidar, angles = self._lidar()
        if (lidar is None or angles is None
                or self.measured_gps_position() is None):
            return baseline
        view = self.v6_adapter.propose_explore_view(
            grid=self.grid, current_pose=self.pose,
            fresh_cum_cost=self.cum)
        if view is None or view.score <= 0:
            return baseline
        validator = make_short_segment_validator(
            grid=self.grid, position=self.pose, heading=self.heading,
            lidar_ranges=lidar, lidar_angles=angles,
            footprint_vertices=self._safety_footprint, carrying=False)
        route = self.v6_adapter.propose_reconnected_loaded_path(
            grid=self.grid, current_pose=self.pose,
            center_witness_xy=(view.x, view.y),
            segment_safe=validator, min_clearance_px=19.0)
        if (route.status != "MAP_CONNECTED" or len(route.points) < 3
                or not validator(self.pose, route.points[1])):
            self._event("v6_view_reject", step, reason=route.status,
                        expansions=route.expansions)
            return baseline
        goal = (view.x, view.y)
        self._v6_pending_view = (goal, tuple(route.points[1:]))
        self._v6_view_goal = goal
        self._v6_view_seen_at_start = int(np.count_nonzero(
            self.v6_adapter.visibility.seen_at))
        self._v6_view_planned_step = step
        self.deep_pick = False
        self.object_pick = False
        self._event("v6_view_planned", step, goal=list(goal),
                    expected_new=round(view.expected_new_cells, 1),
                    path_cost=round(view.path_cost_px, 1),
                    expansions=route.expansions)
        return goal, view.cell

    def _set_path_to(self, goal, step, end_cell=None, wide=False):
        pending = self._v6_pending_view
        if (pending is not None and step == self._v6_view_planned_step
                and math.dist(tuple(goal), pending[0]) < 1.0):
            self.path = list(pending[1])
            self.path_i = 0
            self.path_goal = "explore"
            self.path_step = step
            self._v6_pending_view = None
            return
        if (self.v6_view_plan and self._v6_view_goal is not None
                and math.dist(tuple(goal), self._v6_view_goal) < 1.0
                and self.path and self.path_goal == "explore"):
            # The current verified path remains the transaction's fallback.
            self.path_step = step
            return
        return super()._set_path_to(goal, step, end_cell=end_cell, wide=wide)

    def _v5_loaded_segment_safe(self, start, end):
        """Check a short map connection with the present loaded hull and swing envelope."""
        if (self.measured_gps_position() is None or self.grid.obstacle is None
                or getattr(self.grid, "unknown", None) is None):
            return False
        vertices = [tuple(v) for v in self._safety_footprint]
        # The pivot permits the bomb to swing around the drone. Include the
        # full grasp-reach disk plus the 12 px bomb radius and a 5 px buffer;
        # a current-bearing-only footprint would miss the swept load.
        loaded_radius = GRASP_REACH + 12.0 + 5.0
        for phase in np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False):
            vertices.append((loaded_radius * math.cos(phase),
                             loaded_radius * math.sin(phase)))
        points = np.asarray(vertices, dtype=float)
        hull = OrientedFootprint(points[ConvexHull(points).vertices], margin=2.0)
        occupied = np.asarray(self.grid.obstacle | self.grid.unknown, dtype=bool)
        origin = (-self.grid.width / 2.0, -self.grid.height / 2.0)
        return not hull.swept_collides(
            Pose2(float(start[0]), float(start[1]), float(self.heading)),
            Pose2(float(end[0]), float(end[1]), float(self.heading)),
            occupied, origin, CELL,
        )

    def _v5_observe_carry_path(self, step, lidar, angles):
        cost = None
        if self.center_cum is not None and self.grid.costs is not None:
            cell = self.grid.free_start(self.grid.costs["carry"], *self.pose, radius=5)
            if cell is not None and math.isfinite(float(self.center_cum[cell])):
                cost = float(self.center_cum[cell]) * CELL
        waypoint = (self.path[min(len(self.path) - 1, self.path_i)]
                    if self.path else None)
        nearest = min((hit[2] for hit in self.center_hits), default=None)
        signal = self.v5_path_supervisor.observe(
            step=step, tag="CARRY", position=self.pose,
            path_cost_px=cost,
            waypoint_distance_px=(math.dist(self.pose, waypoint)
                                  if waypoint is not None else None),
            path_points=len(self.path), is_carrying=bool(self.grasped_bombs()),
            live_center_distance_px=nearest,
        )
        if signal.action == "CONTINUE":
            return
        self.v5_path_counts[signal.reason] += 1
        self._event("v5_path_signal", step, action=signal.action,
                    reason=signal.reason,
                    progress=round(signal.progress_px, 1),
                    displacement=round(signal.displacement_px, 1),
                    center_cost=None if cost is None else round(cost, 1),
                    center_semantic=nearest,
                    path_points=len(self.path))
        if not self.v5_path_recover:
            return
        endgame_fraction = float(os.environ.get("V5_PATH_ENDGAME_FRACTION", "0.8"))
        if (V5_TIME_HORIZON <= 0 or not 0.0 <= endgame_fraction <= 1.0
                or step < endgame_fraction * V5_TIME_HORIZON):
            self.v5_path_counts["pre_endgame_diagnostic"] += 1
            return
        if signal.reason != "nonzero_cost_empty_path":
            # Changing center-field evidence can make a moving carrier's
            # apparent map cost rise. Treat that as diagnostic only until a
            # route truly disappears; the first canary over-recovered here.
            self.v5_path_counts["diagnostic_only"] += 1
            return
        if step - self._v5_last_recovery_step < 120:
            self.v5_path_counts["recovery_rate_limited"] += 1
            return
        self._v5_last_recovery_step = step
        # Regenerate the existing legal carry-cost route. A new path is used
        # only when the first links are connected for the present load hull.
        if signal.action in ("REPLAN", "RECOVER"):
            self.center_step = -1000
            new_path = self._center_path(step)
            first_links = [self.pose] + new_path[:3]
            connected = (len(new_path) >= 2 and all(
                self._v5_loaded_segment_safe(a, b)
                for a, b in zip(first_links, first_links[1:])
            ))
            if connected:
                self.path = new_path
                self.path_i = 0
                self.path_goal = "center"
                self.path_step = step
                self._reset_progress()
                self.v5_path_counts["verified_replan"] += 1
            elif signal.action == "RECOVER":
                self._start_escape(step, lidar, angles, away_from=waypoint)
                self.v5_path_counts["safe_recovery"] += 1
            else:
                self.v5_path_counts["replan_rejected"] += 1
        elif signal.action == "TERMINAL_CONFLICT":
            # No live, connected loaded path was proven. Keep the baseline
            # controller's legal recovery; never treat map-cost zero as disposal.
            self.v5_path_counts["terminal_unresolved"] += 1

    def _explore(self, step, cmd, lidar, angles):
        if (self.v6_view_plan and self._v6_view_goal is not None
                and self.pose is not None and self.goal is not None
                and math.dist(self.goal, self._v6_view_goal) < 1.0
                and math.dist(self.pose, self._v6_view_goal) < 25.0
                and not self.corridor_explorer.committed):
            if self._v6_view_sweep_until < self._v6_view_planned_step:
                self._v6_view_sweep_until = step + 8
                self._event("v6_view_reached", step,
                            goal=list(self._v6_view_goal))
            if step <= self._v6_view_sweep_until:
                cmd["grasper"] = 0
                cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                    (0.0, 0.0), lidar, angles, False, careful=True)
                if lidar is None or float(np.nanmin(lidar)) >= 38.0:
                    cmd["rotation"] = 0.22 if self.me % 2 == 0 else -0.22
                return cmd
            new_cells = int(np.count_nonzero(
                self.v6_adapter.visibility.seen_at)) - self._v6_view_seen_at_start
            self.v6_adapter.note_view_execution(
                self._v6_view_goal, reached=True,
                new_observed_cells=new_cells)
            self._event("v6_view_complete", step, new_ray_cells=new_cells)
            self._v6_view_goal = None
        reason = "BASELINE"
        self._corridor_goal_lock = False
        bombs_missing = EXPECTED_BOMBS <= 0 or self.know.accounted() < EXPECTED_BOMBS
        team_idle = step - self.last_team_busy > 120
        existing_commitment = self.corridor_explorer.committed
        corridor_heading = self._corridor_travel_heading()
        raw_gps_degraded = self.measured_gps_position() is None
        raw_comm_degraded = self.communicator_is_disabled()
        if raw_gps_degraded:
            self._gps_outage_frames += 1
            self._gps_recovery_frames = 0
        else:
            self._gps_outage_frames = 0
            self._gps_recovery_frames += 1
        if raw_comm_degraded:
            self._comm_outage_frames += 1
            self._comm_recovery_frames = 0
        else:
            self._comm_outage_frames = 0
            self._comm_recovery_frames += 1
        if self._gps_outage_frames >= 8:
            self._stable_gps_degraded = True
        elif self._gps_recovery_frames >= 60:
            self._stable_gps_degraded = False
        if self._comm_outage_frames >= 4:
            self._stable_comm_degraded = True
        elif self._comm_recovery_frames >= 60:
            self._stable_comm_degraded = False
        gps_degraded = self._stable_gps_degraded
        comm_degraded = self._stable_comm_degraded
        degraded = gps_degraded or comm_degraded
        degraded_state = (gps_degraded, comm_degraded)
        if degraded_state != self._last_degraded_state:
            self._event(
                "zone_state", step, gps_degraded=bool(gps_degraded),
                comm_degraded=bool(comm_degraded))
            self._last_degraded_state = degraded_state
        degraded_trigger = (
            self.corridor_degraded_start >= 0
            and step >= self.corridor_degraded_start
            and degraded
        )
        if (degraded_trigger and not existing_commitment
                and self.corridor_owner_mod > 1):
            owner = int(math.floor(
                (wrap(corridor_heading) + math.pi)
                * self.corridor_owner_mod / (2.0 * math.pi)
            )) % self.corridor_owner_mod
            degraded_trigger = self.me % self.corridor_owner_mod == owner
        corridor_available = (
            self.corridor_enabled and bombs_missing
            and (existing_commitment
                 or degraded_trigger
                 or (step >= self.corridor_start_step and team_idle))
        )
        if not bombs_missing and self.corridor_explorer.state != CorridorExploreState.IDLE:
            self.corridor_explorer.reset()
            self._corridor_branch_goal = None
        if corridor_available and lidar is not None and angles is not None:
            previous_corridor_state = self.corridor_explorer.state
            if self.corridor_explorer.state == CorridorExploreState.PAUSED:
                self.corridor_explorer.resume(self.pose, step)
            output = self.corridor_explorer.update(
                step=step,
                position=self.pose,
                velocity=self.vel,
                heading=self.heading,
                travel_heading=corridor_heading,
                lidar_distances=lidar,
                lidar_angles=angles,
            )
            self._corridor_control_state = output.state
            if output.reason != "BRANCH_ENTER":
                self._branch_lidar_range = None
            self.corridor_reason_counts[output.reason] += 1
            if self.branch_trace_enabled and output.reason in (
                    "BRANCH_ALIGN", "BRANCH_ENTER", "BRANCH_CLEARANCE"):
                direction = output.branch_angle
                origin = self.corridor_explorer._branch_start
                progress = None if origin is None or direction is None else float(
                    (np.asarray(self.pose) - origin) @ np.array([
                        math.cos(direction), math.sin(direction)]))
                self._branch_trace_row = {
                    "step": int(step), "reason": output.reason,
                    "pose": [float(v) for v in self.pose],
                    "heading": float(self.heading),
                    "velocity": [float(v) for v in self.vel],
                    "gps_valid": self.measured_gps_position() is not None,
                    "comm_disabled": bool(self.communicator_is_disabled()),
                    "branch_angle": None if direction is None else float(direction),
                    "branch_origin": None if origin is None else origin.tolist(),
                    "branch_progress": progress,
                    "target_velocity": output.target_velocity.tolist(),
                    "corridor_width": float(output.corridor_width),
                    "end_gap": float(output.end_gap),
                    "lidar_front": self._sector_min(lidar, angles, 0.0),
                    "lidar_left": self._sector_min(lidar, angles, math.pi / 2),
                    "lidar_right": self._sector_min(lidar, angles, -math.pi / 2),
                    "visible_drones": int(len(self.visible_drones)),
                }
                if direction is not None:
                    unit = np.array([math.cos(direction), math.sin(direction)])
                    for horizon in (20.0, 40.0):
                        destination = np.asarray(self.pose) + horizon * unit
                        self._branch_trace_row[f"map_clear_{int(horizon)}"] = bool(
                            self.grid.segment_clear(self.pose, destination, 14.0))
                        self._branch_trace_row[f"endpoint_clearance_{int(horizon)}"] = float(
                            self.grid.clearance_at(*destination))
                    self._branch_trace_row["lidar_branch"] = self._sector_min(
                        lidar, angles, wrap(direction - self.heading), half=0.14)
            self._corridor_end_gap = output.end_gap
            if (previous_corridor_state in (CorridorExploreState.IDLE,
                                            CorridorExploreState.COMPLETE)
                    and output.active):
                self.corridor_brake.reset()
            reason = output.reason
            if output.active:
                # A real bomb task bypasses _explore entirely and pauses this
                # commitment.  While active, frontier rescoring, scan_f and
                # interest_left are intentionally not consulted.
                if output.state == CorridorExploreState.ADVANCE:
                    # Preserve the semantic target and mature map path.  The
                    # corridor commitment only suppresses periodic frontier
                    # rescoring and scan_f invalidation; it does not replace
                    # the proven flight controller.
                    self._corridor_goal_lock = True
                    blocked_reselect = step - self.goal_eval_step > GOAL_STICKY
                    if self.goal is not None and self.goal_kind == "explore":
                        iy, ix = self.grid.cell(*self.goal)
                        scan_block = (
                            self.grid.scan_f[iy, ix] > 0
                            and self.grid.scan_f[iy, ix] > self.goal_step
                        )
                        blocked_reselect = blocked_reselect or scan_block
                    if (blocked_reselect
                            and step - self._last_blocked_reselect_step > GOAL_STICKY):
                        self.corridor_explorer.stats["midway_reversals_blocked"] += 1
                        self._last_blocked_reselect_step = step
                    command = super()._explore(step, cmd, lidar, angles)
                    if self._branch_trace_row is not None:
                        self._branch_trace_row["nominal_action"] = {
                            key: float(command[key]) for key in
                            ("forward", "lateral", "rotation")}
                    self._trace_explore(command, lidar, angles, reason)
                    return command
                if (output.state == CorridorExploreState.BRANCH
                        and output.reason == "BRANCH_ENTER"):
                    if self.branch_local_enabled:
                        axis = np.array([math.cos(output.branch_angle),
                                         math.sin(output.branch_angle)])
                        origin = self.corridor_explorer._branch_start
                        progress = float((np.asarray(self.pose) - origin) @ axis)
                        if self._branch_local_progress_step < 0:
                            self._branch_local_best = progress
                            self._branch_local_progress_step = step
                            self.branch_brake.reset()
                        if progress >= self._branch_local_best + 2.0:
                            self._branch_local_best = progress
                            self._branch_local_progress_step = step
                        if step - self._branch_local_progress_step > 45:
                            self._event("branch_local_stall", step,
                                        progress=round(progress, 1))
                            self.corridor_explorer.completed_endpoints.append(
                                (np.asarray(self.pose, float), step + 300))
                            self.corridor_explorer.reset()
                            self._corridor_branch_goal = None
                            self._branch_local_progress_step = -1
                            self.goal = None
                            return super()._explore(step, cmd, lidar, angles)
                        lookahead = np.asarray(self.pose) + 20.0 * axis
                        bearing = wrap(output.branch_angle - self.heading)
                        visible = self._sector_min(lidar, angles, bearing, half=0.14)
                        map_clear = self.grid.segment_clear(self.pose, lookahead, 14.0)
                        if (not map_clear or visible is None
                                or visible <= self.corridor_explorer.footprint.radius + 7.0):
                            target_velocity = (0.0, 0.0)
                        else:
                            target_velocity = tuple(output.target_velocity)
                        self.state = "CORRIDOR"
                        cmd["grasper"] = 0
                        cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                            target_velocity, lidar, angles, False, careful=True,
                            wall_margin=self.corridor_wall_margin)
                        cmd["rotation"] = 0.0
                        self._branch_lidar_range = visible
                        self._trace_explore(cmd, lidar, angles, "BRANCH_LOCAL")
                        return cmd
                    if self._corridor_branch_goal is None:
                        branch_forward = np.array([
                            math.cos(output.branch_angle), math.sin(output.branch_angle)
                        ])
                        target = np.asarray(self.pose, float) + 70.0 * branch_forward
                        self._corridor_branch_goal = tuple(float(v) for v in target)
                        self.goal = self._corridor_branch_goal
                        self.goal_kind = "explore"
                        self.goal_deep = False
                        self.goal_step = step
                        self.goal_eval_step = step
                        self._refresh_dijkstra(step, force=True)
                        self._set_path_to(self.goal, step)
                        self.path_goal = "explore"
                    self._corridor_goal_lock = True
                    command = super()._explore(step, cmd, lidar, angles)
                    if self._branch_trace_row is not None:
                        self._branch_trace_row["nominal_action"] = {
                            key: float(command[key]) for key in
                            ("forward", "lateral", "rotation")}
                    self._trace_explore(command, lidar, angles, reason)
                    return command
                self.state = "CORRIDOR"
                cmd["grasper"] = 0
                cmd["forward"], cmd["lateral"] = self._velocity_to_command(
                    tuple(output.target_velocity), lidar, angles, False,
                    careful=output.state in (
                        CorridorExploreState.END_APPROACH,
                        CorridorExploreState.INSPECT,
                        CorridorExploreState.BRANCH,
                    ),
                    wall_margin=self.corridor_wall_margin,
                )
                if output.state != CorridorExploreState.INSPECT:
                    cmd["rotation"] = max(-0.55, min(
                        0.55, 0.8 * wrap(output.target_heading - self.heading)
                    ))
                if self._branch_trace_row is not None:
                    self._branch_trace_row["nominal_action"] = {
                        key: float(cmd[key]) for key in
                        ("forward", "lateral", "rotation")}
                self._trace_explore(cmd, lidar, angles, reason)
                return cmd
            if reason in ("CLOSED", "COMPLETE"):
                self.goal = None
                self.path = []
                self._corridor_branch_goal = None
                self._branch_local_progress_step = -1
        else:
            self._corridor_control_state = CorridorExploreState.IDLE
        v7_command = self._v7_gateway_explore(step, cmd, lidar, angles)
        if v7_command is not None:
            return v7_command
        v8_command = self._v8_macro_explore(step, cmd, lidar, angles)
        if v8_command is not None:
            return v8_command
        command = super()._explore(step, cmd, lidar, angles)
        self._trace_explore(command, lidar, angles, reason)
        return command

    def _task_open_scan(self, distances: np.ndarray, angles: np.ndarray) -> np.ndarray:
        scan = distances.copy()
        target = None
        half_angle = 0.0
        if self.grasped_bombs() and self.center_hits:
            nearest = min(self.center_hits, key=lambda hit: hit[2])
            if nearest[2] < 185.0:
                target = nearest[:2]
                half_angle = 0.72
        elif self.visible_bombs:
            position, distance = min(self.visible_bombs, key=lambda item: item[1])
            if distance < 58.0 and self.state == "FETCH":
                target = position
                half_angle = 0.48
        if target is None:
            return scan
        bearing = math.atan2(target[1] - self.pose[1], target[0] - self.pose[0])
        relative = np.abs(np.asarray([wrap(float(a) - (bearing - self.heading)) for a in angles]))
        scan[relative < half_angle] = max(230.0, float(np.max(scan)))
        return scan

    def control(self):
        command = self._control_with_safety()
        if self._v7_action_step == int(self.elapsed_timestep):
            final_action = {
                key: float(command[key]) for key in
                ("forward", "lateral", "rotation")}
            self._event("v7_final_action", int(self.elapsed_timestep),
                        source=self._v7_action_source,
                        nominal=self._v7_action_nominal,
                        final=final_action,
                        gps_valid=self.measured_gps_position() is not None)
        if (self.v6_brake_shadow and self.pose is not None
                and self.heading is not None and not self.grasped_bombs()
                and int(self.elapsed_timestep) % 10 == 0):
            speed = math.hypot(*self.vel)
            if math.isfinite(speed) and speed > 0.3:
                lidar, angles = self._lidar()
                if lidar is not None and angles is not None:
                    bearing = wrap(math.atan2(self.vel[1], self.vel[0])
                                   - self.heading)
                    gap = self._sector_min(lidar, angles, bearing, half=0.08)
                    if gap is not None:
                        stop = stopping_distance_upper(speed)
                        if stop + 8.0 > gap:
                            self._event("v6_brake_shadow", int(self.elapsed_timestep),
                                        speed=round(speed, 2), gap=gap,
                                        empty_stop_upper=round(stop, 2))
        if self.v5_path_watchdog and not self.grasped_bombs():
            self.v5_path_supervisor.reset()
        if (getattr(self, "_v4_velocity_estimator", None) is not None
                and self.heading is not None):
            self._v4_velocity_estimator.set_previous_command(
                body_to_world(
                    (float(command["forward"]), float(command["lateral"])),
                    self.heading,
                ),
                float(command["rotation"]),
            )
        if self.cargo_identifier is not None and self.pose is not None:
            step = int(self.elapsed_timestep)
            carrying = bool(self.grasped_bombs())
            gps_valid = self.measured_gps_position() is not None
            clearance = float(self.grid.clearance_at(*self.pose))
            peer_clear = all(distance > 75.0 for _, _, distance
                             in self.visible_drones)
            health = int(self.drone_health)
            previous_health = getattr(self, "_cargo_shadow_health", health)
            eligible = (carrying and self.state == "CARRY"
                        and self.measurement_valid and gps_valid
                        and math.isfinite(clearance) and clearance > 45.0
                        and peer_clear and health == previous_health)
            applied = body_to_world(
                (float(command["forward"]), float(command["lateral"])),
                self.heading,
            )
            self.cargo_identifier.advance(
                step, self.v_identification, applied, eligible=eligible,
            )
            self._cargo_shadow_health = health
        if self.pose_trace_enabled and self.pose is not None:
            self.pose_trace.append({
                "step": int(self.elapsed_timestep),
                "pose": [float(v) for v in self.pose],
                "heading": float(self.heading),
                "state": str(self.state),
                "gps_valid": self.measured_gps_position() is not None,
                "comm_disabled": bool(self.communicator_is_disabled()),
                "goal": None if self.goal is None else [float(v) for v in self.goal],
                "path_points": int(len(self.path)),
            })
        if self.branch_trace_enabled:
            previous = self.branch_trace[-1] if self.branch_trace else None
            if previous is not None and previous.get("next_pose") is None and self.pose is not None:
                previous["next_pose"] = [float(v) for v in self.pose]
                previous["actual_displacement"] = [
                    float(self.pose[i] - previous["pose"][i]) for i in range(2)]
            row = self._branch_trace_row
            if row is not None:
                row["goal"] = None if self.goal is None else [float(v) for v in self.goal]
                row["path_goal"] = None if self.path_goal is None else str(self.path_goal)
                row["path_points"] = int(len(self.path))
                row["path_index"] = int(self.path_i)
                row["path_end"] = None if not self.path else [float(v) for v in self.path[-1]]
                row["next_waypoint"] = None if not self.path else [float(v) for v in self.path[
                    min(len(self.path) - 1, self.path_i)]]
                row["final_action"] = {key: float(command[key]) for key in
                                       ("forward", "lateral", "rotation")}
                row["safety_filtered_total"] = int(self.safety_stats["filtered_frames"])
                self.branch_trace.append(row)
                self._branch_trace_row = None
        return command

    def _control_with_safety(self):
        command = super().control()
        if (not self.ready or self.pose is None or self.heading is None
                or int(self.elapsed_timestep) < 20):
            return command
        # The calibrated empty-airframe filter is used for its intended
        # regime: high-speed scouting.  Fetch and cargo dynamics are different
        # (the bomb is a swinging second body), and the historical controller's
        # measured load-specific braking / semantic contact logic is the valid
        # model there.  Applying the empty model to those modes reduced the
        # same-layout result from 10/10 to 3/10 despite higher health.
        corridor_mode = self.state == "CORRIDOR"
        if (self.state != "EXPLORE" and not corridor_mode) or self.grasped_bombs():
            return command
        distances = self.lidar_values()
        angles = self.lidar_rays_angles()
        if distances is None or angles is None:
            return command
        # The baseline already brakes from its cargo/empty identified models.
        # Invoke the costlier nonlinear guard only as the last emergency layer
        # for a fast scout with an actually imminent return.  Earlier use at
        # the full stopping horizon was safe but changed frontier arrival
        # timing enough to lose one late bomb on the fixed 2000-step budget.
        finite = np.asarray(distances, float)
        finite = finite[np.isfinite(finite)]
        if corridor_mode and self._corridor_control_state not in (
                CorridorExploreState.END_APPROACH,
                CorridorExploreState.INSPECT,
                *( (CorridorExploreState.BRANCH,) if self.branch_local_enabled else () )):
            return command
        if (corridor_mode and self.branch_local_enabled
                and self._corridor_control_state == CorridorExploreState.BRANCH):
            if self._branch_lidar_range is None:
                return command
            axis = np.array([math.cos(self.corridor_explorer.branch_angle),
                             math.sin(self.corridor_explorer.branch_angle)])
            nominal_world = body_to_world(
                (float(command["forward"]), float(command["lateral"])), self.heading)
            nominal_normal = float(nominal_world @ axis)
            lidar_range = self._branch_lidar_range
            remaining = max(0.0, (lidar_range or 0.0)
                            - self.corridor_explorer._support(
                                self.heading, self.corridor_explorer.branch_angle))
            brake = self.branch_brake.compute(
                speed=max(0.0, float(np.asarray(self.vel) @ axis)),
                nominal_thrust=nominal_normal, remaining_distance=remaining,
                safety_margin=4.0, velocity_uncertainty=0.35,
                velocity_confidence=1.0)
            tangent = nominal_world - nominal_normal * axis
            tangent_limit = math.sqrt(max(0.0, 1.0 - brake.safe_thrust ** 2))
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > tangent_limit and tangent_norm > 1e-12:
                tangent *= tangent_limit / tangent_norm
            body = world_to_body(brake.safe_thrust * axis + tangent, self.heading)
            command["forward"], command["lateral"] = float(body[0]), float(body[1])
            self.safety_stats["filtered_frames"] += 1
            if brake.mode == BrakingMode.PREDICTIVE_BRAKE:
                self.safety_stats["predictive_brake_frames"] += 1
            elif brake.mode == BrakingMode.INFEASIBLE_BRAKING:
                self.safety_stats["infeasible_frames"] += 1
            return command
        if corridor_mode:
            # Endpoint braking is a single half-plane normal to the end wall.
            # Solving it as a scalar preserves the validated nonlinear
            # stopping model without inventing mutually opposed constraints
            # from the two corridor side walls.
            axis = np.array([
                math.cos(self.corridor_explorer.direction),
                math.sin(self.corridor_explorer.direction),
            ])
            nominal_world = body_to_world(
                (float(command["forward"]), float(command["lateral"])), self.heading,
            )
            nominal_normal = float(nominal_world @ axis)
            robust_margin = (
                self.corridor_explorer.geometry_margin
                + self.corridor_explorer.sigma_gain * self.corridor_explorer.lidar_sigma
            )
            brake = self.corridor_brake.compute(
                speed=max(0.0, float(np.asarray(self.vel) @ axis)),
                nominal_thrust=nominal_normal,
                remaining_distance=float(self._corridor_end_gap),
                safety_margin=robust_margin,
                velocity_uncertainty=0.35,
                velocity_confidence=1.0,
            )
            tangent = nominal_world - nominal_normal * axis
            tangent_limit = math.sqrt(max(0.0, 1.0 - brake.safe_thrust ** 2))
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > tangent_limit and tangent_norm > 1e-12:
                tangent *= tangent_limit / tangent_norm
            # Preserve the safety-critical normal component and fit the
            # lateral component into the remaining unit-circle budget.
            safe_world = brake.safe_thrust * axis + tangent
            body = world_to_body(safe_world, self.heading)
            command["forward"] = float(body[0])
            command["lateral"] = float(body[1])
            self.safety_stats["filtered_frames"] += 1
            if brake.mode == BrakingMode.PREDICTIVE_BRAKE:
                self.safety_stats["predictive_brake_frames"] += 1
            elif brake.mode == BrakingMode.INFEASIBLE_BRAKING:
                self.safety_stats["infeasible_frames"] += 1
            return command
        # Corridor commitment by itself is not a near-wall condition.  Running
        # the joint half-plane guard against both side walls for every frame of
        # a narrow corridor makes the empty-airframe stopping model report
        # mutually infeasible constraints and can command needless reversals.
        # Keep the guard at *all* speeds in corridor mode, but only once an
        # obstacle is actually close.  This preserves the required low-speed
        # end inspection without filtering ordinary corridor cruise.
        close_obstacle = finite.size > 0 and float(np.min(finite)) < 30.0
        if (not close_obstacle or
                (not corridor_mode and not self._v7_probe_in_motion
                 and float(np.linalg.norm(self.vel)) < 5.0)):
            return command
        distances = self._task_open_scan(np.asarray(distances, float), np.asarray(angles, float))
        nominal_world = body_to_world(
            (float(command["forward"]), float(command["lateral"])), self.heading,
        )
        output = self.predictive_safety.compute(
            lidar_distances=distances,
            lidar_angles=angles,
            estimated_position=self.pose,
            estimated_heading=self.heading,
            estimated_velocity=self.vel,
            nominal_thrust=nominal_world,
            footprint_vertices=self._safety_footprint,
            velocity_sigma=0.35,
            position_sigma=0.0,
        )
        body = world_to_body(output.command_world, self.heading)
        command["forward"] = float(body[0])
        command["lateral"] = float(body[1])
        self.safety_stats["filtered_frames"] += 1
        status = output.joint.status.value
        if status == "PREDICTIVE_BRAKE":
            self.safety_stats["predictive_brake_frames"] += 1
        elif status == "INFEASIBLE_BRAKING":
            self.safety_stats["infeasible_frames"] += 1
        return command

