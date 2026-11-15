import unittest
import math
import numpy as np
from swarm_rescue.solutions.blue_pid_explore.v7_gateway import GatewayPlanner, short_unknown_probe
from swarm_rescue.solutions.blue_pid_explore.v7_probe_guard import ProbePersistenceGate
from swarm_rescue.solutions.blue_pid_explore.v7_yaw_sweep import YawPhaseSweep
from swarm_rescue.solutions.blue_pid_explore.v7_egotrail import LocalOdomTrail, retreat_desired_world_velocity, RetreatRequest
from swarm_rescue.solutions.blue_pid_explore.v7_brake_shadow import EMPTY, discrete_stopping_distance, cap_speed_for_clearance


class GatewayTests(unittest.TestCase):
    def make_map(self):
        occ=np.zeros((15,15),dtype=float)
        occ[2:13,2:8]=-1.0
        occ[2:13,8:10]=2.0
        clear=np.full_like(occ,100.)
        return occ,clear

    def test_not_cross_wall(self):
        occ,clear=self.make_map()
        occ[:,10:]=0
        g=GatewayPlanner(origin_xy=(0.,0.),min_clearance_px=10.)
        r=g.choose(occ=occ,clearance_px=clear,current_xy=(4.5*10,6.5*10),now=10)
        self.assertIsNotNone(r)
        self.assertTrue(all(x<80 for x,y in r.route))

    def test_no_unverified_start_snap(self):
        occ,clear=self.make_map()
        g=GatewayPlanner((0.,0.))
        self.assertIsNone(g.choose(occ=occ,clearance_px=clear,current_xy=(9.5*10,6.5*10),now=1))

    def test_reconnect_with_checked_start(self):
        occ,clear=self.make_map()
        g=GatewayPlanner((0.,0.))
        r=g.choose(occ=occ,clearance_px=clear,current_xy=(1.5*10,6.5*10),now=1,start_link_safe=lambda a,b: True)
        self.assertIsNotNone(r)

    def test_tabu_only_temporary(self):
        occ,clear=self.make_map();g=GatewayPlanner((0.,0.))
        r=g.choose(occ=occ,clearance_px=clear,current_xy=(4.5*10,6.5*10),now=20)
        g.mark_failed(r.cell,20,cooldown=50)
        self.assertGreater(g.failed_until[r.cell],20)
        s=g.choose(occ=occ,clearance_px=clear,current_xy=(4.5*10,6.5*10),now=21)
        self.assertTrue(s is None or s.cell!=r.cell)

    def test_live_probe_safeguard(self):
        ang=np.linspace(-math.pi,math.pi,181)
        d=np.full(181,100.)
        p=short_unknown_probe(current_xy=(0,0),heading=0,lidar_ranges=d,lidar_angles=ang,target_heading=0,hull_radius_px=25.)
        self.assertIsNotNone(p)
        self.assertLessEqual(p.length_px,10.)
        d[np.argmin(abs(ang))]=3.
        self.assertIsNone(short_unknown_probe(current_xy=(0,0),heading=0,lidar_ranges=d,lidar_angles=ang,target_heading=0,hull_radius_px=25.))

    def test_probe_short_validator(self):
        ang=np.linspace(-math.pi,math.pi,181);d=np.full(181,100.)
        self.assertIsNone(short_unknown_probe(current_xy=(0,0),heading=0,lidar_ranges=d,lidar_angles=ang,target_heading=0,hull_radius_px=25.,current_segment_safe=lambda a,b:False))

    def test_empty_exploration(self):
        occ=np.full((8,8),2.0); clr=np.full((8,8),30.0)
        self.assertIsNone(GatewayPlanner((0,0)).choose(occ=occ,clearance_px=clr,current_xy=(4,4),now=1))

    def test_conflict_doesnt_make_wall_traversable(self):
        occ,clr=self.make_map();own=np.full_like(occ,0.0);own[5,8]=-3.
        g=GatewayPlanner((0,0))
        v=g.choose(occ=occ,clearance_px=clr,current_xy=(45.,65.),own_evidence=own,now=1)
        self.assertIsNotNone(v)
        self.assertTrue(all(g.cell(point,occ.shape)[1]<8 for point in v.route))


class ProbeGateTests(unittest.TestCase):
    def test_three_samples_needed(self):
        g=ProbePersistenceGate(required_clear_frames=3)
        a=np.linspace(-math.pi,math.pi,181)
        d=np.full(181,100.)
        args=dict(gateway_cell=(3,4),position=(0.,0.),heading=0.,target_heading=0.,
                  lidar_ranges=d,lidar_angles=a,body_radius_px=21.)
        self.assertFalse(g.observe(**args).confirmed)
        self.assertFalse(g.observe(**args).confirmed)
        self.assertTrue(g.observe(**args).confirmed)
        self.assertFalse(g.observe(**args).confirmed)

    def test_dynamic_resets(self):
        g=ProbePersistenceGate()
        a=np.linspace(-math.pi,math.pi,181);d=np.full(181,100.)
        args=dict(gateway_cell=(3,4),position=(0.,0.),heading=0.,target_heading=0.,
                  lidar_ranges=d,lidar_angles=a,body_radius_px=21.)
        g.observe(**args);g.observe(**args)
        self.assertFalse(g.observe(**args,dynamic_obstruction=True).confirmed)
        self.assertFalse(g.observe(**args).confirmed)


class YawSweepTests(unittest.TestCase):
    def test_interleaved_spacing(self):
        y=YawPhaseSweep()
        self.assertAlmostEqual(y.ray_spacing,2*math.pi/34)
    def test_abort_on_cargo(self):
        y=YawPhaseSweep();d=y.update(current_heading=0,carrying=True,visible_target=False,stable_standpoint=True)
        self.assertTrue(d.done)
    def test_final_phase(self):
        y=YawPhaseSweep();h=0.
        for k in range(12):
            d=y.update(current_heading=h,carrying=False,visible_target=False,stable_standpoint=True)
            if d.done:break
            h=d.target_heading
        self.assertTrue(d.done)


class TrailTests(unittest.TestCase):
    def test_odom_integration(self):
        t=LocalOdomTrail(min_record_px=2.,min_age_frames=1)
        for k in range(1,10):
            self.assertTrue(t.update(step=k,odometry=(3.,0.,0.),gps_valid=k<3))
        self.assertAlmostEqual(t.pose.x,27.)
        self.assertAlmostEqual(t.pose.y,0.)

    def test_no_duplicate_updates(self):
        t=LocalOdomTrail()
        self.assertTrue(t.update(step=1,odometry=(1,0,0),gps_valid=False))
        self.assertFalse(t.update(step=1,odometry=(1,0,0),gps_valid=False))

    def test_invalid_odom_rejected(self):
        t=LocalOdomTrail()
        self.assertFalse(t.update(step=1,odometry=(float('nan'),0,0),gps_valid=False))

    def test_retreat_only_with_lost_gps_and_cargo(self):
        t=LocalOdomTrail(min_record_px=2.,min_age_frames=1)
        for k in range(1,14):t.update(step=k,odometry=(3.,0.,0.),gps_valid=k<3)
        self.assertIsNone(t.retreat_target(now=13,path_missing=True,carrying=False,local_segment_safe=lambda a,d:True))
        self.assertIsNone(t.retreat_target(now=13,path_missing=False,carrying=True,local_segment_safe=lambda a,d:True))
        req=t.retreat_target(now=13,path_missing=True,carrying=True,local_segment_safe=lambda a,d:True)
        self.assertIsNotNone(req)
        self.assertAlmostEqual(abs(req.bearing_body),math.pi,places=5)

    def test_unsafe_segment_refused(self):
        t=LocalOdomTrail(min_record_px=2.,min_age_frames=1)
        for k in range(1,14):t.update(step=k,odometry=(3.,0.,0.),gps_valid=False)
        self.assertIsNone(t.retreat_target(now=13,path_missing=True,carrying=True,local_segment_safe=lambda a,d:False))

    def test_suppression(self):
        t=LocalOdomTrail(min_record_px=2.,min_age_frames=1)
        for k in range(1,14):t.update(step=k,odometry=(3.,0.,0.),gps_valid=False)
        t.suppress(13,30)
        self.assertIsNone(t.retreat_target(now=14,path_missing=True,carrying=True,local_segment_safe=lambda a,d:True))

    def test_world_velocity_transform(self):
        r=RetreatRequest(0.,20.,10,(0.,0.),1.,'test')
        x,y=retreat_desired_world_velocity(r,math.pi/2,speed_px_per_frame=1.25)
        self.assertAlmostEqual(x,0.,places=6);self.assertAlmostEqual(y,1.25)


class BrakingTests(unittest.TestCase):
    def test_zero(self):self.assertEqual(discrete_stopping_distance(0.),0.)
    def test_monotone(self):self.assertGreater(discrete_stopping_distance(5.),discrete_stopping_distance(3.))
    def test_larger_clearance_allows_faster(self):
        self.assertGreater(cap_speed_for_clearance(80),cap_speed_for_clearance(20))
    def test_zero_gap(self):self.assertEqual(cap_speed_for_clearance(7,margin_px=8),0.)
    def test_bounded(self):self.assertLessEqual(cap_speed_for_clearance(400),11.4)
    def test_invalid(self):
        with self.assertRaises(ValueError):discrete_stopping_distance(-1.)

if __name__=='__main__':unittest.main()
