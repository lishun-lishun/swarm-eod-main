"""Only algorithm settings; never import game truth/map definitions here."""
from dataclasses import dataclass

@dataclass(frozen=True)
class V9Config:
    cell_px: float = 10.0
    lidar_range_px: float = 230.0
    semantic_range_px: float = 130.0
    comm_range_px: float = 180.0
    # Algorithm hypotheses, NOT authoritative sim constants; calibrate by Pymunk.
    empty_radius_px: float = 18.0
    loaded_radius_px: float = 26.0
    loaded_speed_px_frame: float = 2.8
    empty_speed_px_frame: float = 6.0
    loaded_decel_px_frame2: float = 0.17
    empty_decel_px_frame2: float = 0.42
    map_every: int = 4
    plan_every: int = 30
    map_free_threshold: float = -0.62
    map_occ_threshold: float = 0.95
    min_route_clear_px: float = 19.0
    current_collision_margin_px: float = 6.0
    max_route_expansions: int = 24000
    max_unknown_step_px: float = 8.0
    task_lease_frames: int = 35
    track_match_px: float = 24.0
    max_deadline_frames: int = 5400
    # Observation uncertainty limits are safety gates, not proof of accuracy.
    max_localization_sigma_px: float = 35.0
    scan_match_period: int = 8
    scan_match_min_hits: int = 15
