"""V6 short-horizon sensor + polygon sweep validator for optional path repair.

NEVER bypasses the baseline control safety layer. For carried loads it uses a
full 360-degree reach envelope because current joint swing is not legally
observable as an exact pose. It may reject a real narrow corridor: that is a
safe failure of the *new optional recovery*, not grounds to alter the baseline.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import ConvexHull
from .oriented_footprint import OrientedFootprint, Pose2


def make_short_segment_validator(*, grid, position, heading,
                                 lidar_ranges,lidar_angles,footprint_vertices,
                                 carrying=False,pose_sigma_px=0.0,
                                 max_segment_px=70.0,
                                 bomb_reach_px=33.0,bomb_radius_px=12.0,
                                 sensor_noise_margin_px=6.0):
    lidar=np.asarray(lidar_ranges,dtype=float)
    angles=np.asarray(lidar_angles,dtype=float)
    vertices=np.asarray(footprint_vertices,dtype=float)
    if lidar.ndim!=1 or lidar.shape!=angles.shape or vertices.ndim!=2 or vertices.shape[1]!=2:
        raise ValueError('bad official LiDAR or footprint arrays')
    if grid.occ is None:
        raise ValueError('need fused occupancy belief')
    margin=2.0+max(0.,float(pose_sigma_px))
    if carrying:
        # No unverified 'rigid cargo': official Grasper uses a PivotJoint.
        # Full swing envelope is deliberately conservative for *new routes*.
        radius=float(bomb_reach_px)+float(bomb_radius_px)+margin
        circle=np.array([(radius*math.cos(t),radius*math.sin(t))
            for t in np.linspace(0,2*math.pi,48,endpoint=False)])
        vertices=np.concatenate([vertices,circle],axis=0)
        vertices=vertices[ConvexHull(vertices).vertices]
    hull=OrientedFootprint(vertices,margin=margin)
    occupied=np.asarray(grid.occ,dtype=float)>=-0.6  # unknown IS NOT free
    origin=(-float(grid.width)/2.0,-float(grid.height)/2.0)
    p=np.asarray(position,dtype=float).reshape(2)

    def valid(start,end):
        a=np.asarray(start,dtype=float).reshape(2)
        b=np.asarray(end,dtype=float).reshape(2)
        if not np.isfinite(np.r_[a,b,p]).all():
            return False
        length=float(np.linalg.norm(b-a))
        if length>max_segment_px or np.linalg.norm(a-p)>8.0:
            # This validator certifies only a short segment from THIS pose.
            return False
        if length>1e-3:
            relative=math.atan2(float(b[1]-a[1]),float(b[0]-a[0]))-float(heading)
            err=np.abs((angles-relative+math.pi)%(2*math.pi)-math.pi)
            close=(err<0.07)&np.isfinite(lidar)
            # Require at least one trustworthy LiDAR ray for this direction.
            if not np.any(close):
                return False
            hit=float(np.min(lidar[close]))
            if hit < length+hull.radius+sensor_noise_margin_px:
                return False
        return not hull.swept_collides(
            Pose2(float(a[0]),float(a[1]),float(heading)),
            Pose2(float(b[0]),float(b[1]),float(heading)),
            occupied, origin, 10.0)

    return valid
