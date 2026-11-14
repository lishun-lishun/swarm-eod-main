import importlib.util
from pathlib import Path
import numpy as np

BASE = (Path(__file__).resolve().parents[1] / 'src' / 'swarm_rescue'
        / 'solutions' / 'blue_pid_explore')

def load(name):
    import sys
    p = BASE/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m

S=load('v5_deadline_allocator'); P=load('v5_path_supervisor'); C=load('v5_control_gate')

def test_eta_monotonic_and_unknown():
    d=S.DeadlineTaskScheduler(0)
    a=S.TaskCandidate('a',(0.,0.),200.,400.)
    b=S.TaskCandidate('b',(0.,0.),300.,400.)
    unknown=S.TaskCandidate('u',(0.,0.),200.,None)
    assert d.eta(b)>d.eta(a)>0
    assert d.eta(unknown)>0

def test_switch_hysteresis():
    d=S.DeadlineTaskScheduler(0)
    a=S.TaskCandidate('a',(0.,0.),240.,400.)
    b=S.TaskCandidate('b',(100.,0.),220.,400.)
    z=d.choose([a,b],now=50,horizon=5400,incumbent_key='a')
    assert z.key=='a'

def test_unknown_only_reachable():
    d=S.DeadlineTaskScheduler(0)
    x=S.TaskCandidate('bad',(0.,0.),float('inf'),None)
    y=S.TaskCandidate('known',(20.,0.),200.,None)
    assert d.choose([x,y],now=300,horizon=5400,incumbent_key=None).key=='known'

def test_peer_bid_v5_unit_and_ttl():
    d=S.DeadlineTaskScheduler(2)
    a=S.TaskCandidate('a',(100.,100.),100.,300.)
    bid=S.TaskBidV5(1,(101.,100.),1.0,100)
    assert d.choose([a],now=105,horizon=5400,incumbent_key=None,peer_bids=[bid]) is None
    assert d.choose([a],now=130,horizon=5400,incumbent_key=None,peer_bids=[bid]) is not None

def test_workload_penalty():
    d=S.DeadlineTaskScheduler(0)
    a=S.TaskCandidate('a',(0.,0.),150.,200.)
    no,eta=d.evaluate(a,now=100,horizon=5400,own_recent_pickups=0)
    heavy,_=d.evaluate(a,now=100,horizon=5400,own_recent_pickups=3)
    assert heavy>no and eta>0

def test_terminal_conflict():
    p=P.PathProgressSupervisor(cooldown_frames=0)
    for step in range(6):
        v=p.observe(step=step,tag='CARRY',position=(0,0),path_cost_px=0,
                    waypoint_distance_px=0,path_points=1,is_carrying=True,live_center_distance_px=120)
    assert v.action=='TERMINAL_CONFLICT'

def test_empty_path_recovery():
    p=P.PathProgressSupervisor(cooldown_frames=0)
    v=p.observe(step=1,tag='CARRY',position=(0,0),path_cost_px=200,
                waypoint_distance_px=None,path_points=0,is_carrying=True,live_center_distance_px=None)
    assert v.action=='REPLAN'

def test_lookahead_requires_clear_segment():
    q=P.reachable_lookahead((0,0),[(10,0),(30,0),(60,0)],
                            lambda a,b: b[0]<50)
    assert q==(30.,0.)

def test_control_off_exact():
    g=C.BoundedModelBlend()
    a,enabled=g.apply(baseline_body=(.4,.1),model_world=(1.,0.),heading=0,
                      state='EXPLORE',carrying=False,lidar_min_px=150,
                      gps_valid=True,peer_nearby=False)
    assert np.allclose(a,(.4,.1)) and enabled is False

def test_control_on_bounded():
    g=C.BoundedModelBlend(alpha=.2,delta_max=.15)
    a,enabled=g.apply(baseline_body=(.3,.1),model_world=(.5,.6),heading=0,
                      state='EXPLORE',carrying=False,lidar_min_px=150,
                      gps_valid=True,peer_nearby=False)
    assert enabled and np.linalg.norm(a)<=1. and np.linalg.norm(a-np.array([.3,.1]))<=.031

def test_no_cargo_control():
    g=C.BoundedModelBlend(alpha=.5)
    a,enabled=g.apply(baseline_body=(.3,.1),model_world=(1.,0.),heading=0,
                      state='CARRY',carrying=True,lidar_min_px=200,
                      gps_valid=True,peer_nearby=False)
    assert not enabled and np.allclose(a,(.3,.1))

def test_v5_bid_transport():
    import sys
    from pathlib import Path
    # Inject a private package namespace to allow the module's relative import.
    import types
    package=types.ModuleType('v5testpkg')
    package.__path__=[str(BASE)]
    sys.modules['v5testpkg']=package
    spec=importlib.util.spec_from_file_location('v5testpkg.v5_deadline_allocator',BASE/'v5_deadline_allocator.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    spec2=importlib.util.spec_from_file_location('v5testpkg.v5_bid_wire',BASE/'v5_bid_wire.py')
    wire=importlib.util.module_from_spec(spec2);sys.modules[spec2.name]=wire;spec2.loader.exec_module(wire)
    bid=mod.TaskBidV5(3,(2.,4.),71.,120)
    encoded=wire.encode_bid(bid)
    assert wire.decode_bid(encoded,sender_id=3,now=130)==bid
    assert wire.decode_bid(encoded,sender_id=8,now=130) is None
    assert wire.decode_bid(encoded,sender_id=3,now=151) is None
    encoded['cost_frames']=float('nan')
    assert wire.decode_bid(encoded,sender_id=3,now=130) is None
