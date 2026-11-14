"""V6 adapters for current FinalCompetitionDrone (no simulator modifications).

These helpers intentionally do not take action/control away from the baseline.
Codex wires calls only at documented hook points after a baseline A/B is saved.
"""
from __future__ import annotations
import math
import numpy as np
from .v6_visibility import SemanticRayEvidence, official_semantic_angles, choose_next_view
from .v6_connected_route import KnownFreeAStar, verified_lookahead
from .v6_route_monitor import RouteIntegrityMonitor
from .v6_task_policy import EventTaskPolicy


class V6Adapter:
    def __init__(self, drone_id:int, grid):
        self.visibility=SemanticRayEvidence(
            grid.occ.shape,cell_px=10.0,
            origin_xy=(-float(grid.width)/2.0,-float(grid.height)/2.0))
        self.monitor=RouteIntegrityMonitor()
        self.assigner=EventTaskPolicy(drone_id)
        self.last_view_goals=[]
        self.last_scan_step=-1

    def observe_semantic_rays(self, *, step, pose, heading, lidar_distances,
                              lidar_angles, pose_trustworthy=True):
        if not pose_trustworthy or step-self.last_scan_step<5:
            return 0
        self.last_scan_step=int(step)
        return self.visibility.update(
            pose_xy=pose, heading=heading,
            lidar_ranges=lidar_distances, lidar_angles=lidar_angles,
            semantic_angles=official_semantic_angles(), step=step)

    def propose_explore_view(self, *, grid, current_pose, fresh_cum_cost,
                             max_route_cost_px=1600.0):
        if (grid.clearance is None or fresh_cum_cost is None
                or self.visibility.seen_at.shape!=grid.occ.shape):
            return None
        v=choose_next_view(
            self.visibility,np.asarray(grid.occ),np.asarray(grid.clearance),
            np.asarray(fresh_cum_cost)*10.0,
            position=tuple(current_pose),
            last_targets=tuple(self.last_view_goals[-8:]),
            max_route_cost_px=max_route_cost_px)
        return v

    def note_view_execution(self, goal, *, reached, new_observed_cells):
        # Failed arrival or no actual observation must not mark the target done.
        if reached and new_observed_cells>0:
            self.last_view_goals.append((float(goal[0]),float(goal[1])))
            self.last_view_goals=self.last_view_goals[-32:]

    def propose_reconnected_loaded_path(self, *, grid, current_pose,
                                        center_witness_xy, segment_safe,
                                        min_clearance_px=18.0):
        """Offline/diagnostic first. Must verify actual payload sweep before actuation."""
        if grid.clearance is None:
            return None
        route=KnownFreeAStar(np.asarray(grid.occ), np.asarray(grid.clearance),
                             origin_xy=(-float(grid.width)/2.,-float(grid.height)/2.),
                             cell_px=10.,min_clearance_px=min_clearance_px)
        return route.plan(current_pose,center_witness_xy,local_segment_safe=segment_safe)

    def verified_waypoint(self, *, pose, path,index,segment_safe):
        return verified_lookahead(pose,path,index=index,segment_safe=segment_safe)

    def monitor_carrier(self, **kwargs):
        return self.monitor.observe(**kwargs)
