import sys
from pathlib import Path
import math
import unittest
import numpy as np

from swarm_rescue.solutions.blue_pid_explore.v8_macro_frontier import MacroFrontierPlanner
from swarm_rescue.solutions.blue_pid_explore.v8_mission_guard import MissionPreservationGuard, point_segment_distance
from swarm_rescue.solutions.blue_pid_explore.v8_ego_route import EgoRouteContinuity
from swarm_rescue.solutions.blue_pid_explore.v8_motion_guard import lidar_short_segment_safe

class TestV8(unittest.TestCase):
    def make_grid(self):
        occ=np.full((50,50),-2.0); occ[:,40:]=0; occ[12:22,2:15]=0
        clearance=np.full_like(occ,40.0)
        cum=np.arange(50)[:,None]*np.ones((1,50))
        cum = np.hypot(np.indices((50,50))[0]-25,np.indices((50,50))[1]-25)
        return occ,clearance,cum
    def test_frontier_returns_known_free(self):
        a,b,c=self.make_grid();o=MacroFrontierPlanner();g=o.select(occ=a,clearance_px=b,current_xy=(250,250),map_origin_xy=(0,0),local_start_safe=lambda a,b:True,now=300)
        self.assertIsNotNone(g);self.assertLess(a[g.cell],-.6)
    def test_frontier_region_cooldown(self):
        a,b,c=self.make_grid();o=MacroFrontierPlanner();g=o.select(occ=a,clearance_px=b,current_xy=(250,250),map_origin_xy=(0,0),local_start_safe=lambda a,b:True,now=300)
        o.mark_result(g.region,300,reached=True,new_semantic_cells=0)
        g2=o.select(occ=a,clearance_px=b,current_xy=(250,250),map_origin_xy=(0,0),local_start_safe=lambda a,b:True,now=301)
        self.assertNotEqual(g.region,g2.region)
    def test_frontier_none_when_all_free(self):
        o=MacroFrontierPlanner();a=np.full((10,10),-3.);b=np.full((10,10),25.);c=np.ones((10,10))
        self.assertIsNone(o.select(occ=a,clearance_px=b,current_xy=(250,250),map_origin_xy=(0,0),local_start_safe=lambda a,b:True,now=1))
    def test_frontier_no_unreachable(self):
        a,b,c=self.make_grid();a[25,:]=2.0  # split the free space; unreachable pockets stay excluded
        o=MacroFrontierPlanner();g=o.select(occ=a,clearance_px=b,current_xy=(250,310),map_origin_xy=(0,0),local_start_safe=lambda a,b:True,now=1)
        self.assertTrue(g is None or g.cell[0] >= 25)
    def test_point_segment(self):
        self.assertAlmostEqual(point_segment_distance((5,5),(0,0),(10,0)),5.)
    def test_guard_fetch_priority(self):
        d=MissionPreservationGuard().check(now=100,deadline=5400,state='EXPLORE',carrying=False,has_claim=True,visible_ungrasped_bomb=False,own_pose=(0,0),proposed_xy=(100,0))
        self.assertFalse(d.allowed)
    def test_guard_carrier_priority(self):
        d=MissionPreservationGuard().check(now=100,deadline=5400,state='EXPLORE',carrying=False,has_claim=False,visible_ungrasped_bomb=False,own_pose=(0,0),proposed_xy=(100,0),known_peers=[{'pos':(50,0),'carry':True,'last_seen':95}]);self.assertFalse(d.allowed)
    def test_guard_stale_peer_ok(self):
        d=MissionPreservationGuard().check(now=100,deadline=5400,state='EXPLORE',carrying=False,has_claim=False,visible_ungrasped_bomb=False,own_pose=(0,0),proposed_xy=(100,0),known_peers=[{'pos':(50,0),'carry':True,'last_seen':1}]);self.assertTrue(d.allowed)
    def test_guard_deadline(self):
        d=MissionPreservationGuard().check(now=5300,deadline=5400,state='EXPLORE',carrying=False,has_claim=False,visible_ungrasped_bomb=False,own_pose=(0,0),proposed_xy=(100,0));self.assertFalse(d.allowed)
    def test_ego_capture_and_forward(self):
        o=EgoRouteContinuity();self.assertTrue(o.capture(step=100,legal_pose_xy=(10,10),legal_heading=0,world_path=[(10,10),(25,10),(40,10)],gps_valid=True));o.odometry_update(101,(2,0,0));r=o.propose(step=101,gps_valid=False,carrying=True,stalled_or_missing=True,local_segment_safe=lambda a,d:True);self.assertTrue(r.ok);self.assertAlmostEqual(r.bearing_body,0.)
    def test_ego_capture_invalid_gps(self):
        o=EgoRouteContinuity();self.assertFalse(o.capture(step=100,legal_pose_xy=(0,0),legal_heading=0,world_path=[(1,0)],gps_valid=False))
    def test_ego_no_route(self):
        o=EgoRouteContinuity();r=o.propose(step=5,gps_valid=False,carrying=True,stalled_or_missing=True,local_segment_safe=lambda a,d:True);self.assertFalse(r.ok)
    def test_ego_stale(self):
        o=EgoRouteContinuity(max_age_frames=30);o.capture(step=1,legal_pose_xy=(0,0),legal_heading=0,world_path=[(0,0),(20,0)]);r=o.propose(step=90,gps_valid=False,carrying=True,stalled_or_missing=True,local_segment_safe=lambda a,d:True);self.assertFalse(r.ok)
    def test_ego_blocked(self):
        o=EgoRouteContinuity();o.capture(step=1,legal_pose_xy=(0,0),legal_heading=0,world_path=[(0,0),(20,0)]);r=o.propose(step=2,gps_valid=False,carrying=True,stalled_or_missing=True,local_segment_safe=lambda a,d:False);self.assertFalse(r.ok)
    def test_ego_not_required_when_gps_good(self):
        o=EgoRouteContinuity();o.capture(step=1,legal_pose_xy=(0,0),legal_heading=0,world_path=[(0,0),(20,0)]);r=o.propose(step=2,gps_valid=True,carrying=True,stalled_or_missing=True,local_segment_safe=lambda a,d:True);self.assertFalse(r.ok)
    def test_ego_odom_duplicates(self):
        o=EgoRouteContinuity();self.assertTrue(o.odometry_update(1,(1,0,0)));self.assertFalse(o.odometry_update(1,(1,0,0)))
    def test_lidar_open(self):
        a=np.linspace(-math.pi, math.pi,181);d=np.full_like(a,230.)
        self.assertTrue(lidar_short_segment_safe(bearing_body=0,distance_px=20,lidar_ranges=d,lidar_angles=a,hull_radius_px=24))
    def test_lidar_wall(self):
        a=np.linspace(-math.pi, math.pi,181);d=np.full_like(a,30.)
        self.assertFalse(lidar_short_segment_safe(bearing_body=0,distance_px=20,lidar_ranges=d,lidar_angles=a,hull_radius_px=24))
    def test_lidar_cargo_narrow(self):
        a=np.linspace(-math.pi, math.pi,181);d=np.full_like(a,56.)
        self.assertFalse(lidar_short_segment_safe(bearing_body=0,distance_px=15,lidar_ranges=d,lidar_angles=a,hull_radius_px=24,carrying=True))
    def test_lidar_peer(self):
        a=np.linspace(-math.pi,math.pi,181);d=np.full_like(a,230.)
        self.assertFalse(lidar_short_segment_safe(bearing_body=0,distance_px=20,lidar_ranges=d,lidar_angles=a,hull_radius_px=24,peer_nearby=True))

if __name__=='__main__':unittest.main()
