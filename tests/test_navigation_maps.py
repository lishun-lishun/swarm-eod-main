import numpy as np

from swarm_rescue.solutions.blue_pid_explore.astar import astar
from swarm_rescue.solutions.blue_pid_explore.frontier_explorer import FrontierExplorer
from swarm_rescue.solutions.blue_pid_explore.occupancy_map import Cell, OccupancyMap
from swarm_rescue.solutions.blue_pid_explore.semantic_coverage import SemanticCoverageMap


def test_no_return_does_not_create_occupied_cell():
    grid = OccupancyMap((200, 200), resolution=10)
    grid.update_scan([0, 0], 0.0, [230.0], [0.0], max_range=230.0)
    cells = grid.cells()
    assert not np.any(cells == Cell.OCCUPIED)
    assert np.any(cells == Cell.FREE)


def test_lidar_and_semantic_coverage_are_separate():
    occupancy = OccupancyMap((200, 200), resolution=10)
    semantic = SemanticCoverageMap(occupancy)
    occupancy.update_scan([0, 0], 0, [50], [0], max_range=230)
    assert not np.any(semantic.covered)
    semantic.update([0, 0], 0, [0], max_range=130)
    assert np.any(semantic.covered)


def test_astar_does_not_cross_inflated_obstacle():
    grid = np.full((20, 20), Cell.FREE, dtype=np.int8)
    grid[:, 10] = Cell.INFLATED; grid[10, 10] = Cell.FREE
    result = astar(grid, (2, 2), (17, 17))
    assert result.success
    assert (10, 10) in result.path
    assert all(grid[y, x] not in (Cell.OCCUPIED, Cell.INFLATED) for x, y in result.path)


def test_frontier_score_penalizes_repeated_selection():
    grid = np.full((10, 12), Cell.UNKNOWN, dtype=np.int8)
    grid[4:6, 1:8] = Cell.FREE
    distances = np.tile(np.arange(12), (10, 1)).astype(float)
    explorer = FrontierExplorer(history_weight=100)
    first = explorer.frontiers(grid, (1, 4), distances)[0]
    explorer.mark_selected(first.representative)
    second = explorer.frontiers(grid, (1, 4), distances)[0]
    if second.representative == first.representative:
        assert second.score < first.score

