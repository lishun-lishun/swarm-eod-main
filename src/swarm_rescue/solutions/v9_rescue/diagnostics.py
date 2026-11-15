"""Legal-only per-robot diagnostics. Use in tools; not gameplay truth."""
from __future__ import annotations
import math
import weakref
import numpy as np

_REGISTERED=weakref.WeakSet()

def register(controller):
    _REGISTERED.add(controller)

def snapshot():
    """No hidden bomb positions, true state, simulator maps or scores."""
    out=[]
    for drone in list(_REGISTERED):
        try:
            st=drone.loc.status()
            out.append({
                'id':int(drone.identifier or 0),
                'step':int(drone.elapsed_timestep),
                'phase':str(drone.mission.phase),
                'local_pose':[float(st.pose.x),float(st.pose.y),float(st.pose.yaw)],
                'localization_sigma_px':float(st.sigma_xy),
                'pose_anchored':bool(st.world_anchored),
                'gps_active':bool(st.gps_active),
                'known_free_cells':int(np.count_nonzero(drone.grid.free_mask())),
                'semantic_ray_cells':int(np.count_nonzero(drone.grid.sem_rays)),
                'known_targets':int(sum(t.state=='KNOWN' for t in drone.mem.targets.values())),
                'current_task':drone.mem.my_target,
                'grasped_count':int(bool(drone.grasped_bombs())),
                'route_points':len(drone.follow.route.points),
                'route_index':int(drone.follow.index),
                'last_safety_reason':str(drone.motion.last_obstacle_reason),
                'recent_events':list(drone.mem.events[-12:]),
                'task_events':list(drone.mem.events),
                'task_targets':[{'key':t.key,'state':t.state,
                    'last_seen':int(t.last_seen),'local_observed':bool(t.local_observed),
                    'world_xy':None if drone.loc.world(t.xy) is None else
                    [float(v) for v in drone.loc.world(t.xy)]}
                    for t in drone.mem.targets.values()],
                'exceptions':list(drone._latest_events[-8:]),
                'flight_trace':list(drone._trace),
            })
        except (ValueError,TypeError,AttributeError):
            continue
    return sorted(out,key=lambda x:x['id'])
