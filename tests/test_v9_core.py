import math
import numpy as np
import pytest
from swarm_rescue.solutions.v9_rescue.config import V9Config
from swarm_rescue.solutions.v9_rescue.geometry import Pose, RelativeDetection, clip_disk, to_body, rotate,wrap
from swarm_rescue.solutions.v9_rescue.localization import EgoLocalizer, correlative_scan_match
from swarm_rescue.solutions.v9_rescue.mapping import BeliefGrid
from swarm_rescue.solutions.v9_rescue.tasks import TaskMemory,TaskScheduler
from swarm_rescue.solutions.v9_rescue.planning import KnownFreePlanner, RouteFollower, EgoBreadcrumbs, lidar_unknown_step
from swarm_rescue.solutions.v9_rescue.control import ModelAwareMotion,make_reference
from swarm_rescue.solutions.v9_rescue.mission import MissionSupervisor

R=np.full(181,230.)
A=np.linspace(-math.pi,math.pi,181)

def grid(w=200):
    return BeliefGrid((float(w),float(w)))

def free_grid(w=200):
    g=grid(w)
    g.log_odds[:,:]=-2
    g.revision+=1
    return g

def test_wrap():
    assert math.isclose(wrap(2*math.pi),0,abs_tol=1e-8)

def test_rotation_inverse():
    x=np.array([3.,-2.]); t=.73
    assert np.allclose(to_body(rotate(x,t),t),x)

def test_unit_disk():
    assert np.linalg.norm(clip_disk((1,1)))<=1+1e-12

def test_nan_force_safe():
    assert np.allclose(clip_disk((float('nan'),1.)),(0,0))

def test_localizer_without_gps_continues():
    l=EgoLocalizer();l.predict(step=0,odom=(2.,0.,0.),gps=None,compass=None)
    l.predict(step=1,odom=(3.,0.,0.),gps=None,compass=None)
    assert l.pose.x==5 and not l.anchored

def test_localizer_anchor_does_not_jump_map():
    l=EgoLocalizer()
    l.predict(step=0,odom=(1.,0.,0.),gps=None,compass=None)
    before=l.pose.x
    l.predict(step=1,odom=(0.,0.,0.),gps=(800.,-400.),compass=1.0)
    assert l.anchored and abs(l.pose.x-before)<1
    assert np.allclose(l.world(),(800,-400),atol=1)

def test_localizer_gps_outlier_rejected():
    l=EgoLocalizer();l.predict(step=0,odom=(0.,0.,0.),gps=(0,0),compass=0.)
    l.predict(step=1,odom=(1,0,0),gps=(2000,2000),compass=0.)
    assert l.pose.x<8

def test_scan_match_empty_map_rejected():
    g=grid()
    assert correlative_scan_match(g,Pose(0,0,0),R,A) is None

def test_scan_match_large_correction_rejected():
    l=EgoLocalizer();l.predict(step=0,odom=(1,0,0),gps=None,compass=None)
    assert not l.accept_scan_match(dx=20,dy=0,dyaw=0,improvement=.5,static_correspondences=30)

def test_mapping_lidar_builds_free():
    g=grid(300)
    for k in range(8):g.integrate_lidar(Pose(0,0,0),R,A,k)
    assert int(np.sum(g.log_odds<0))>100

def test_mapping_dynamic_not_wall():
    g=grid(300)
    r=R.copy();r[90]=24.0
    g.integrate_lidar(Pose(0,0,0),r,A,0,dynamic_detections=[RelativeDetection('bomb',24.,0.)])
    c=g.cell((21.,0.))
    assert g.log_odds[c]<=.5

def test_mapping_semantic_not_occupancy():
    g=grid(300);g.integrate_semantic_geometry(Pose(0,0,0),R,A,0)
    assert g.sem_rays.max()>0
    assert np.count_nonzero(g.log_odds)==0

def test_mapping_trace_clips_outside():
    g=grid(80)
    assert len(g.trace((0,0),(1000.,0)))>3

def test_mapping_unknown_not_verified():
    g=grid(100)
    assert not g.known_free_segment((0,0),(30,0))

def test_occupied_wall_blocks_route():
    g=free_grid(200)
    g.log_odds[:,10]=3.
    p=KnownFreePlanner(g)
    result=p.path((-75,0),(75,0))
    assert result.status!='CONNECTED'

def test_known_free_connected_path():
    g=free_grid(200)
    result=KnownFreePlanner(g).path((-30,-10),(30,10))
    assert result.status=='CONNECTED' and len(result.points)>2

def test_frontier_from_known_component():
    g=grid(200)
    g.log_odds[6:14,6:14]=-2.
    g.revision+=1
    front=KnownFreePlanner(g).select_frontier((-10,0),now=10,drone_id=0)
    assert front is not None and front.route.status=='CONNECTED'

def test_route_follower_no_skip_unsafe():
    f=RouteFollower();f.set_route(type('P',(),{'status':'CONNECTED','points':((0.,0.),(15.,0.),(30.,0.))})(),0)
    assert f.waypoint((0.,0.),now=2,local_segment_ok=lambda a,b:math.dist(a,b)<20)==(15.,0.)

def test_breadcrumb_recent_backtrack():
    b=EgoBreadcrumbs()
    for k in range(12):b.add((k*5.,0.),k*10)
    point=b.local_backtrack((55.,0.),current_safe=lambda a,b:True)
    assert point is not None and point[0]<55

def test_unknown_step_wall_rejected():
    r=R.copy();r[:]=10
    assert lidar_unknown_step(Pose(0,0,0),r,A,body_radius_px=18,yaw_relative=0.) is None

def test_unknown_step_clear_bounded():
    p=lidar_unknown_step(Pose(0,0,0),R,A,body_radius_px=18,yaw_relative=0.)
    assert 3 <= p[0] <= 8

def test_task_memory_cluster():
    m=TaskMemory(0)
    a=m.observe_bomb((20,20),1); b=m.observe_bomb((22,20),2)
    assert a is b and len(m.targets)==1

def test_task_pick_locked():
    m=TaskMemory(0);a=m.observe_bomb((20,20),1)
    sched=TaskScheduler(0)
    pick=sched.decide(m,now=2,current_xy=(0,0),carrying=False)
    assert pick.key==a.key
    assert sched.decide(m,now=3,current_xy=(0,0),carrying=True) is None

def test_task_radio_only_when_anchored():
    m=TaskMemory(0);m.observe_bomb((20,20),0)
    data=m.advertised(2,EgoLocalizer(),'EXPLORE',False)
    assert data['world_xy'] is None and not data['targets']

def test_task_disposal_not_inferred_from_drop():
    m=TaskMemory(0);t=m.observe_bomb((20,20),0)
    m.my_target=t.key
    m.observe_grasp_transition(3,True,reward=0.)
    m.observe_grasp_transition(4,False,reward=0.)
    assert t.state=='SUSPECT_DROP'

def test_task_reward_confirm():
    m=TaskMemory(0);t=m.observe_bomb((20,20),0);m.my_target=t.key
    m.observe_grasp_transition(3,True,reward=0.)
    m.observe_grasp_transition(4,False,reward=1.)
    assert t.state=='DISPOSED'

def test_control_output_finite_unit_disk():
    m=ModelAwareMotion()
    out=m.output(Pose(0,0,0),(0,0),(3,0),lidar_ranges=R,lidar_angles=A)
    assert abs(out.forward)**2+abs(out.lateral)**2<=1.00001

def test_control_close_wall_brakes():
    m=ModelAwareMotion();rr=np.full(181,30.)
    out=m.output(Pose(0,0,0),(0,0),(6,0),lidar_ranges=rr,lidar_angles=A)
    assert out.safe_speed<3

def test_loaded_not_empty_model():
    m=ModelAwareMotion();e=m.output(Pose(0,0,0),(0,0),(2,0),lidar_ranges=R,lidar_angles=A,carrying=False)
    c=m.output(Pose(0,0,0),(0,0),(2,0),lidar_ranges=R,lidar_angles=A,carrying=True)
    assert abs(e.forward-c.forward)>.05

def test_grasper_hold_carry():
    m=ModelAwareMotion()
    out=m.output(Pose(0,0,0),(0,0),(2,0),lidar_ranges=R,lidar_angles=A,carrying=True,grasper=True)
    assert out.grasper==1

def test_terminal_relative_semantic():
    m=MissionSupervisor()
    d=RelativeDetection('center',29.,0.1)
    out=m.step(now=10,pose=Pose(0,0,0),carrying=True,visible_centers=(d,))
    assert out.phase=='DOCK' and out.grasp and out.terminal_target==d

def test_bomb_direct():
    m=MissionSupervisor()
    d=RelativeDetection('bomb',28.,0.)
    out=m.step(now=10,pose=Pose(0,0,0),carrying=False,visible_bombs=(d,))
    assert out.phase=='FETCH' and out.grasp

def test_loaded_unknown_center_explores_without_drop():
    m=MissionSupervisor()
    out=m.step(now=0,pose=Pose(0,0,0),carrying=True,explore_goal_xy=(100,0))
    assert out.phase=='CARRY_EXPLORE' and out.grasp

def test_reference_slows_near_end():
    v1=make_reference((0,0),(100,0),speed_cap=6,brake_decel=.4)
    v2=make_reference((0,0),(11,0),speed_cap=6,brake_decel=.4)
    assert np.linalg.norm(v1)>np.linalg.norm(v2)

from swarm_rescue.solutions.v9_rescue.cargo import direction_dependent_extent,CargoSwingTracker

def test_cargo_extent_directional():
    behind=direction_dependent_extent(movement_relative_yaw=0.,cargo_bearing_body=math.pi,uncertainty_rad=.2)
    side=direction_dependent_extent(movement_relative_yaw=0.,cargo_bearing_body=math.pi/2,uncertainty_rad=.2)
    assert behind.along_px<side.along_px+10
    assert side.lateral_px>behind.lateral_px

def test_cargo_extent_unknown_conservative():
    u=direction_dependent_extent(movement_relative_yaw=0.,cargo_bearing_body=None)
    assert u.uncertainty_rad==math.pi and u.lateral_px>=39

def test_cargo_tracker_no_fabricated_bearing():
    t=CargoSwingTracker();b,u=t.observe(carrying=True,body_yaw=0,pickup_bearing=None)
    assert b is None and u==math.pi

def test_cargo_tracker_resets():
    t=CargoSwingTracker();t.observe(carrying=True,body_yaw=0,pickup_bearing=math.pi)
    t.observe(carrying=False,body_yaw=0)
    assert t.bearing is None

def test_route_cost_geodesic_vs_euclid():
    g=free_grid(300)
    # Vertical wall with passage only at upper edge.
    g.log_odds[0:21,15]=3.;g.revision+=1
    p=KnownFreePlanner(g)
    route=p.path((-100.,0),(100.,0))
    assert route.status=='CONNECTED'
    assert route.distance>200.0

def test_peers_claim_without_global_consensus():
    m=TaskMemory(4);t=m.observe_bomb((30,0),0)
    loc=EgoLocalizer();loc.predict(step=0,odom=(0,0,0),gps=(0,0),compass=0.)
    m.read_messages([(2,{'schema':'v9.1','id':2,'world_xy':[0,0],
        'task_world':[30,0],'pose_anchored':True,'phase':'FETCH',
        'carrying':False,'targets':[]})],now=5,localizer=loc)
    result=TaskScheduler(4).decide(m,now=6,current_xy=(0,0),carrying=False,localizer=loc)
    assert result is None


def test_stale_peer_claim_expires():
    m=TaskMemory(4);m.observe_bomb((30,0),0)
    loc=EgoLocalizer();loc.predict(step=0,odom=(0,0,0),gps=(0,0),compass=0.)
    m.read_messages([(2,{'schema':'v9.1','id':2,'task_world':[30,0],
        'pose_anchored':True,'phase':'FETCH','carrying':False})],now=0,localizer=loc)
    m.read_messages([],now=60,localizer=loc)
    result=TaskScheduler(4).decide(m,now=60,current_xy=(0,0),carrying=False,localizer=loc)
    assert result is not None


def test_peer_confirmed_disposal_clears_stale_remote_fetch():
    memory=TaskMemory(4)
    target=memory.observe_bomb((30.,0.),0)
    memory.my_target=target.key
    loc=EgoLocalizer()
    loc.predict(step=0,odom=(0,0,0),gps=(0,0),compass=0.)
    memory.read_messages([(2,{'schema':'v9.1','id':2,'pose_anchored':True,
        'targets':[{'world':[30.,0.],'state':'DISPOSED'}]})],now=5,localizer=loc)
    assert target.state=='DISPOSED'
    assert memory.my_target is None
    assert TaskScheduler(4).decide(memory,now=6,current_xy=(0,0),carrying=False) is None


def test_nearby_bombs_remain_separate_after_one_disposal():
    memory=TaskMemory(4,match_px=14.)
    first=memory.observe_bomb((-131.,382.),0)
    second=memory.observe_bomb((-105.,385.),0)
    assert first.key != second.key
    loc=EgoLocalizer()
    loc.predict(step=0,odom=(0,0,0),gps=(0,0),compass=0.)
    memory.read_messages([(2,{'schema':'v9.1','id':2,'pose_anchored':True,
        'targets':[{'world':[-131.,382.],'state':'DISPOSED'}]})],now=5,localizer=loc)
    assert first.state=='DISPOSED' and second.state=='KNOWN'
