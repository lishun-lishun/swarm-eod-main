import math
import unittest
import numpy as np
from swarm_rescue.solutions.blue_pid_explore.v6_visibility import SemanticRayEvidence, official_semantic_angles, choose_next_view
from swarm_rescue.solutions.blue_pid_explore.v6_connected_route import KnownFreeAStar, verified_lookahead,reversed_legal_breadcrumbs
from swarm_rescue.solutions.blue_pid_explore.v6_route_monitor import RouteIntegrityMonitor
from swarm_rescue.solutions.blue_pid_explore.v6_task_policy import KnownTask,PeerCommitment,EventTaskPolicy


class V6Tests(unittest.TestCase):
    def test_angles_match_official(self):
        a=official_semantic_angles()
        self.assertEqual(len(a),35)
        self.assertAlmostEqual(a[0],-math.pi)
        self.assertAlmostEqual(a[-1],math.pi)
        self.assertAlmostEqual(a[17],0)

    def test_real_35_count_not_181(self):
        e=SemanticRayEvidence((50,50),origin_xy=(-250,-250))
        n=e.update(pose_xy=(0.,0.),heading=0.,lidar_ranges=np.full(181,230.),
          lidar_angles=np.linspace(-math.pi,math.pi,181),semantic_angles=official_semantic_angles(),step=1)
        self.assertLess(n,850)
        self.assertGreater(n,100)

    def test_occlusion_never_marks_behind_hit(self):
        e=SemanticRayEvidence((50,50),origin_xy=(-250,-250))
        lidar=np.full(181,230.)
        lidar[90]=30.0  # zero-angle semantic hit
        e.update(pose_xy=(0,0),heading=0,lidar_ranges=lidar,
          lidar_angles=np.linspace(-math.pi,math.pi,181),
          semantic_angles=official_semantic_angles(),step=1)
        self.assertEqual(e.seen_at[e.cell((75.,0.))],0)
        self.assertGreater(e.seen_at[e.cell((15.,0.))],0)

    def test_unknown_blocks_predicted_gain_behind_it(self):
        e=SemanticRayEvidence((30,30),origin_xy=(-150,-150))
        occ=np.full((30,30),-2.0)
        occ[15,16]=0.0
        occ[15,17]=0.0
        # This is a local-utility estimate, never a physical absence proof.
        self.assertGreater(e.predicted_information((0,0),0,occ),0)

    def test_viewer_returns_only_known_free(self):
        e=SemanticRayEvidence((30,30),origin_xy=(-150,-150))
        occ=np.full((30,30),-2.0)
        occ[:,20:]=0
        route=np.full((30,30),math.inf);route[:, :19]=50
        clr=np.full((30,30),30.0)
        v=choose_next_view(e,occ,clr,route,position=(0,0))
        self.assertIsNotNone(v)
        self.assertLess(v.cell[1],19)

    def test_astar_rejects_wall_even_if_nearest_free_other_side(self):
        occ=np.full((12,12),-2.)
        occ[:,6]=3.
        clr=np.full((12,12),25.)
        p=KnownFreeAStar(occ,clr,origin_xy=(0,0),cell_px=10)
        ok=lambda a,b:True
        r=p.plan((15,15),(95,95),local_segment_safe=ok)
        self.assertEqual(r.status,'DISCONNECTED')

    def test_diagonal_no_corner_cut(self):
        occ=np.full((3,3),3.0);occ[0,0]=-3.;occ[1,1]=-3.
        p=KnownFreeAStar(occ,np.full((3,3),40.),origin_xy=(0,0),cell_px=10)
        r=p.plan((5,5),(15,15),local_segment_safe=lambda a,b:True,goal_radius_cells=0)
        self.assertEqual(r.status,'DISCONNECTED')

    def test_astar_map_path(self):
        occ=np.full((10,10),-2.);cl=np.full((10,10),30.)
        p=KnownFreeAStar(occ,cl,origin_xy=(0,0),cell_px=10)
        r=p.plan((5,5),(85,85),local_segment_safe=lambda a,b:True,goal_radius_cells=0)
        self.assertEqual(r.status,'MAP_CONNECTED');self.assertGreater(len(r.points),2)

    def test_astar_requires_start_connector(self):
        occ=np.full((10,10),-2.);cl=np.full((10,10),30.)
        p=KnownFreeAStar(occ,cl,origin_xy=(0,0),cell_px=10)
        r=p.plan((5,5),(85,85),local_segment_safe=lambda a,b:False)
        self.assertEqual(r.status,'INVALID_START')

    def test_monotone_waypoint_selection(self):
        path=[(0,0),(10,0),(20,0),(30,0)]
        r=verified_lookahead((11,0),path,index=1,
           segment_safe=lambda a,b: b[0]<25.,lookahead_px=30.)
        self.assertEqual(r.waypoint,(20,0));self.assertGreaterEqual(r.index,1)

    def test_blocked_route_not_executed(self):
        path=[(0,0),(10,0)]
        r=verified_lookahead((0,0),path,index=0,segment_safe=lambda a,b:False)
        self.assertEqual(r.status,'NO_VERIFIED_SEGMENT')

    def test_reverse_breadcrumb_only_safe_prefix(self):
        r=reversed_legal_breadcrumbs((30,0),[(0,0),(10,0),(20,0)],
          segment_safe=lambda a,b:b[0]>=10)
        self.assertEqual(r.status,'VERIFIED_RETREAT');self.assertEqual(r.points[-1],(10,0))

    def test_monitor_no_path_is_not_arrival(self):
        m=RouteIntegrityMonitor()
        r=m.observe(step=5,route_tag='carry',pose=(0,0),waypoint=None,route_count=0,commanded_speed=2.)
        self.assertEqual(r.kind,'MISSING_ROUTE')

    def test_monitor_false_center(self):
        m=RouteIntegrityMonitor()
        r=m.observe(step=5,route_tag='carry',pose=(0,0),waypoint=(1,1),route_count=1,commanded_speed=1.,
              carrying=True,center_cost_px=0,center_surface_distance_px=120)
        self.assertEqual(r.kind,'WITNESS_NOT_CONTACT')

    def test_task_respects_v3_peer_lease_without_unit_mix(self):
        t=KnownTask('bomb',(100.,100.),60,100,20)
        p=PeerCommitment(2,(101,99),None,10)
        r=EventTaskPolicy(4).choose([t],now=12,peers=[p])
        self.assertIsNone(r.task)

    def test_task_lock_good_incumbent(self):
        tasks=[KnownTask('old',(10,10),100,150,20),KnownTask('new',(20,20),1,1,10)]
        r=EventTaskPolicy(4).choose(tasks,now=100,current_key='old')
        self.assertEqual(r.task.key,'old')

    def test_task_switch_after_failure(self):
        tasks=[KnownTask('old',(10,10),100,150,20,feasible=False),KnownTask('new',(20,20),1,1,10)]
        r=EventTaskPolicy(4).choose(tasks,now=100,current_key='old',allow_switch=True,current_feasible=False)
        self.assertEqual(r.task.key,'new')


if __name__=='__main__':unittest.main()

class AdapterTests(unittest.TestCase):
    def test_adapter_legal_geometry_data(self):
        from swarm_rescue.solutions.blue_pid_explore.v6_adapter import V6Adapter
        class Grid:
            width=200.;height=200.;occ=np.full((20,20),-2.)
            clearance=np.full((20,20),35.)
        a=V6Adapter(3,Grid())
        n=a.observe_semantic_rays(step=5,pose=(0,0),heading=0,
            lidar_distances=np.full(181,230.),lidar_angles=np.linspace(-math.pi,math.pi,181))
        self.assertGreater(n,0)
        r=a.propose_reconnected_loaded_path(grid=Grid(),current_pose=(0,0),
          center_witness_xy=(50,30),segment_safe=lambda a,b:True)
        self.assertEqual(r.status,'MAP_CONNECTED')

class PhysicsTests(unittest.TestCase):
    def test_stopping_distance_monotone(self):
        from swarm_rescue.solutions.blue_pid_explore.v6_braking_audit import stopping_distance_upper
        self.assertEqual(stopping_distance_upper(0),0)
        self.assertGreater(stopping_distance_upper(8),stopping_distance_upper(4))

    def test_speed_limit_not_exceed_available_gap(self):
        from swarm_rescue.solutions.blue_pid_explore.v6_braking_audit import max_certified_speed,stopping_distance_upper
        for gap in (5,10,30,70,100):
            v=max_certified_speed(gap,margin_px=3)
            self.assertLessEqual(stopping_distance_upper(v),max(0,gap-3)+1e-6)

class ValidatorTests(unittest.TestCase):
    def test_validator_cannot_cross_unknown_wall(self):
        from swarm_rescue.solutions.blue_pid_explore.v6_local_validator import make_short_segment_validator
        class G:
            width=300.;height=300.;occ=np.full((30,30),-3.0)
        G.occ[15,17]=2.
        verts=[(-4,-4),(4,-4),(4,4),(-4,4)]
        fn=make_short_segment_validator(grid=G(),position=(0,0),heading=0,
             lidar_ranges=np.full(181,230.0),
             lidar_angles=np.linspace(-math.pi,math.pi,181),footprint_vertices=verts)
        self.assertFalse(fn((0,0),(30,0)))

    def test_validator_short_free(self):
        from swarm_rescue.solutions.blue_pid_explore.v6_local_validator import make_short_segment_validator
        class G:
            width=300.;height=300.;occ=np.full((30,30),-3.0)
        verts=[(-4,-4),(4,-4),(4,4),(-4,4)]
        fn=make_short_segment_validator(grid=G(),position=(0,0),heading=0,
             lidar_ranges=np.full(181,230.0),
             lidar_angles=np.linspace(-math.pi,math.pi,181),footprint_vertices=verts)
        self.assertTrue(fn((0,0),(10,0)))
