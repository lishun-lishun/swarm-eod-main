"""Import/contract smoke tests against a narrow MOCK of the official API.

Does NOT replace the required actual Arcade/Pymunk integration run.
"""
import importlib
import sys
import types
import math
import numpy as np
import pytest


def load_adapter(monkeypatch):
    import swarm_rescue.solutions.v9_rescue
    for name in ('swarm_rescue.simulation',
                 'swarm_rescue.simulation.drone',
                 'swarm_rescue.simulation.ray_sensors'):
        monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    abstract=types.ModuleType('swarm_rescue.simulation.drone.drone_abstract')
    semantic=types.ModuleType('swarm_rescue.simulation.ray_sensors.drone_semantic_sensor')

    class Base:
        def __init__(self,identifier=None,misc_data=None,**kwargs):
            self.identifier=identifier or 0
            self.size_area=(2000.,2000.)
            self.elapsed_timestep=0
            self.communicator=types.SimpleNamespace(received_messages=[])
            self.reward=0.
            self._carry=False
            self._gps=(0.,0.)
            self._sem=[]
        def lidar_values(self):return np.full(181,230.)
        def lidar_rays_angles(self):return np.linspace(-math.pi,math.pi,181)
        def odometer_values(self):return (0.,0.,0.)
        def measured_gps_position(self):return self._gps
        def measured_compass_angle(self):return 0.
        def semantic_values(self):return self._sem
        def communicator_is_disabled(self):return False
        def grasped_bombs(self):return [object()] if self._carry else []
    class En:
        BOMB=object()
        DRONE=object()
        DISPOSAL_CENTER=object()
    class Sensor:
        TypeEntity=En
    abstract.DroneAbstract=Base
    semantic.DroneSemanticSensor=Sensor
    monkeypatch.setitem(sys.modules,abstract.__name__,abstract)
    monkeypatch.setitem(sys.modules,semantic.__name__,semantic)
    # Core module tests do not import this fake environment.
    sys.modules.pop('swarm_rescue.solutions.v9_rescue.adapter',None)
    return importlib.import_module('swarm_rescue.solutions.v9_rescue.adapter'),En


def test_adapter_empty_motion_smoke(monkeypatch):
    module,_=load_adapter(monkeypatch)
    d=module.V9RescueDrone(identifier=9)
    for k in range(10):
        d.elapsed_timestep=k
        result=d.control()
        assert set(result)=={'forward','lateral','rotation','grasper'}
        assert all(math.isfinite(float(result[x])) for x in result)
        assert result['forward']**2+result['lateral']**2<=1.0001
    assert not any(x['kind']=='V9_EXCEPTION' for x in d._latest_events)


def test_adapter_missing_gps_no_crash(monkeypatch):
    module,_=load_adapter(monkeypatch)
    d=module.V9RescueDrone(identifier=6)
    d._gps=None
    for k in range(12):
        d.elapsed_timestep=k
        out=d.control()
        assert out['grasper']==0
    assert not d.loc.anchored
    assert not any(x['kind']=='V9_EXCEPTION' for x in d._latest_events)


def test_adapter_carry_holds_grasper(monkeypatch):
    module,_=load_adapter(monkeypatch)
    d=module.V9RescueDrone(identifier=6)
    d._carry=True
    d.elapsed_timestep=1
    out=d.control()
    assert out['grasper']==1


def test_adapter_bomb_legal_semantic(monkeypatch):
    module,kind=load_adapter(monkeypatch)
    d=module.V9RescueDrone(identifier=2)
    d._sem=[types.SimpleNamespace(entity_type=kind.BOMB,distance=36.0,angle=0.,grasped=False)]
    d.elapsed_timestep=0
    out=d.control()
    assert len(d.mem.targets)>=1
    assert out['grasper']==1


def test_adapter_communicator_contract(monkeypatch):
    module,_=load_adapter(monkeypatch)
    d=module.V9RescueDrone(identifier=4)
    d.elapsed_timestep=0
    d.control()
    msg=d.define_message_for_all()
    assert msg is not None and msg['schema']=='v9.1' and msg['id']==4
