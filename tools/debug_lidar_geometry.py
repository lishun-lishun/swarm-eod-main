"""Print official LiDAR boundary fits at selected poses."""
from test_lidar_safety import SafetyPlayground, SafetyDrone, add_wall, reset_agent
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.solutions.blue_pid_explore.lidar_geometry import LidarGeometryEstimator, footprint_vertices_from_body
import numpy as np

world = SafetyPlayground()
drone = SafetyDrone(identifier=0, misc_data=MiscData(size_area=(700, 700), number_drones=1))
world.add(drone, ((0, 0), 0))
add_wall(world, (100, -260), (100, 260))
add_wall(world, (-260, 100), (260, 100))
footprint = footprint_vertices_from_body(drone.base)
for pose in [(0, -120), (80, -120), (80, 80)]:
    reset_agent(world, drone, pose, (0, 0), 0.0, 0)
    values = np.asarray(drone.lidar_values(), dtype=float)
    fits = LidarGeometryEstimator().estimate(values, drone.lidar_rays_angles(), pose, 0.0, footprint)
    print("POSE", pose, "finite", np.sum(np.isfinite(values)), "range", np.nanmin(values), np.nanmax(values))
    for fit in fits:
        print(" ", np.round(fit.normal, 3), "surface", round(fit.surface_distance, 3),
              "clear", round(fit.clearance, 3), "rms", round(fit.geometry_error - 1, 3),
              "points", fit.point_count)
world.cleanup()
