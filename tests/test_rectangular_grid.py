#!/usr/bin/env python3
'''
Tests for rectangular grid cells.
Run from the repository root with: python -m unittest discover -s tests
'''
import unittest
import numpy as np

from stastar.grid import Grid as SquareGrid
from stastar.planner import Planner as SquareSTPlanner

from cbs_mapf.agent import Agent
from cbs_mapf.grid import Grid, RectSTPlanner, cell_dimensions
from cbs_mapf.planner import Planner


def border(width, height, step=10, x0=0, y0=0):
    '''Static obstacles outlining a width x height rectangle'''
    obstacles = []
    for x in range(x0, x0 + width + 1, step):
        obstacles += [(x, y0), (x, y0 + height)]
    for y in range(y0, y0 + height + 1, step):
        obstacles += [(x0, y), (x0 + width, y)]
    return obstacles


def keep_order(starts, goals):
    '''Assign the i-th start to the i-th goal so agents are forced to interact'''
    return [Agent(start, goal) for start, goal in zip(starts, goals)]


class TestCellDimensions(unittest.TestCase):

    def test_int_is_square(self):
        self.assertEqual(cell_dimensions(30), (30, 30))
        self.assertEqual(cell_dimensions(np.int64(30)), (30, 30))

    def test_pair_is_width_height(self):
        self.assertEqual(cell_dimensions((20, 10)), (20, 10))
        self.assertEqual(cell_dimensions([20, 10]), (20, 10))  # e.g. loaded from YAML

    def test_non_positive_rejected(self):
        for bad in (0, -5, (0, 10), (10, -1)):
            with self.assertRaises(ValueError):
                cell_dimensions(bad)


class TestGrid(unittest.TestCase):

    def test_square_cells_match_stastar(self):
        # Same output as the original square-only grid, including odd sizes and offset maps
        cases = [(30, border(1920, 1080, 30)),
                 (25, border(1920, 1080, 30)),
                 (7, border(97, 53, 5, x0=-40, y0=13))]
        for size, obstacles in cases:
            obstacles = np.array(obstacles)
            np.testing.assert_array_equal(Grid(size, obstacles).grid,
                                          SquareGrid(size, obstacles).grid)

    def test_rectangular_shape_and_centres(self):
        grid = Grid((20, 10), np.array(border(300, 120))).grid
        # 300 / 20 = 15 columns, 120 / 10 = 12 rows
        self.assertEqual(grid.shape, (12, 15, 2))
        np.testing.assert_array_equal(grid[0][0], [10, 5])
        np.testing.assert_array_equal(grid[0][1], [30, 5])
        np.testing.assert_array_equal(grid[1][0], [10, 15])
        np.testing.assert_array_equal(grid[-1][-1], [290, 115])

    def test_tall_cells(self):
        grid = Grid((10, 40), np.array(border(100, 200))).grid
        self.assertEqual(grid.shape, (5, 10, 2))
        np.testing.assert_array_equal(grid[1][2], [25, 60])

    def test_snap_to_grid(self):
        grid = Grid((20, 10), np.array(border(300, 120)))
        np.testing.assert_array_equal(grid.snap_to_grid(np.array([39, 7])), [30, 5])
        np.testing.assert_array_equal(grid.snap_to_grid(np.array([40, 10])), [50, 15])
        # Positions on or past the boundary clamp to the nearest edge cell
        np.testing.assert_array_equal(grid.snap_to_grid(np.array([300, 120])), [290, 115])
        np.testing.assert_array_equal(grid.snap_to_grid(np.array([-50, 500])), [10, 115])

    def test_cells_larger_than_map_rejected(self):
        with self.assertRaises(ValueError):
            Grid((400, 10), np.array(border(300, 120)))


class TestRectSTPlanner(unittest.TestCase):

    def test_neighbours_step_by_cell_dimensions(self):
        planner = RectSTPlanner((20, 10), 5, border(300, 120))
        neighbours = planner.neighbour_table.lookup(np.array([150, 55]))
        offsets = {tuple(n - [150, 55]) for n in neighbours}
        expected = {(dx, dy) for dx in (-20, 0, 20) for dy in (-10, 0, 10)}
        self.assertEqual(offsets, expected)

    def test_square_cells_plan_same_paths_as_stastar(self):
        obstacles = border(1920, 1080, 30) + border(165, 112, 10, x0=576, y0=277)
        square = SquareSTPlanner(30, 110, obstacles)
        rect = RectSTPlanner(30, 110, obstacles)
        for start, goal in [((331, 165), (1370, 565)), ((1530, 951), (540, 565))]:
            np.testing.assert_array_equal(rect.plan(start, goal, dict(), max_iter=500),
                                          square.plan(start, goal, dict(), max_iter=500))

    def test_single_agent_path_on_rectangular_cells(self):
        planner = RectSTPlanner((20, 10), 5, border(300, 120))
        path = planner.plan((25, 25), (275, 95), dict(), max_iter=1000)
        self.assertGreater(len(path), 0)
        np.testing.assert_array_equal(path[0], [30, 25])
        np.testing.assert_array_equal(path[-1], [270, 95])
        steps = np.abs(np.diff(path, axis=0))
        self.assertTrue(np.all(np.isin(steps[:, 0], [0, 20])))
        self.assertTrue(np.all(np.isin(steps[:, 1], [0, 10])))


class TestCBSRectangular(unittest.TestCase):

    def assert_valid_solution(self, planner, paths, starts, goals, cell):
        width, height = cell
        grid = planner.st_planner.grid
        self.assertEqual(len(paths), len(starts))
        for path, start, goal in zip(paths, starts, goals):
            np.testing.assert_array_equal(path[0], grid.snap_to_grid(np.array(start)))
            np.testing.assert_array_equal(path[-1], grid.snap_to_grid(np.array(goal)))
            # Every position is a cell centre and every move is at most one cell per axis
            centres = {tuple(c) for c in grid.grid.reshape(-1, 2)}
            self.assertTrue(all(tuple(p) in centres for p in path))
            steps = np.abs(np.diff(path, axis=0))
            self.assertTrue(np.all(np.isin(steps[:, 0], [0, width])))
            self.assertTrue(np.all(np.isin(steps[:, 1], [0, height])))
        # No two agents ever come within twice the robot radius. Paths are not padded:
        # an agent whose path has ended waits on its goal.
        for t in range(max(len(path) for path in paths)):
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    self.assertGreater(Planner.dist(Planner.position_at(paths[i], t),
                                                    Planner.position_at(paths[j], t)),
                                       2 * planner.robot_radius,
                                       'agents {0} and {1} collide at t={2}'.format(i, j, t))

    def test_agents_swap_sides_with_wide_cells(self):
        cell = (20, 10)
        planner = Planner(cell, 10, border(400, 100))
        starts = [(30, 25), (370, 75)]
        goals = [(370, 75), (30, 25)]
        paths = planner.plan(starts, goals, assign=keep_order,
                             low_level_max_iter=2000, max_process=2)
        self.assertEqual(len(paths), len(starts), 'no solution found')
        self.assert_valid_solution(planner, paths, starts, goals, cell)

    def test_four_agents_with_tall_cells(self):
        cell = (10, 30)
        planner = Planner(cell, 10, border(200, 300))
        starts = [(25, 45), (175, 45), (25, 255), (175, 255)]
        goals = [(175, 255), (25, 255), (175, 45), (25, 45)]
        paths = planner.plan(starts, goals, assign=keep_order,
                             low_level_max_iter=2000, max_process=4)
        self.assertEqual(len(paths), len(starts), 'no solution found')
        self.assert_valid_solution(planner, paths, starts, goals, cell)

    def test_agent_steps_off_its_goal_to_let_another_pass(self):
        # Agent 0 starts on its goal (3, 0) in a corridor along y=0 that agent 1 must cross;
        # (3, 1) is the only free cell off the corridor. Agent 0 has to leave its goal and
        # come back, which needs the goal constraints in Planner.calculate_constraints and
        # RectSTPlanner.plan's can_stay: without them CBS finds no plan. A radius below 0.5,
        # as in the warehouse, uses RectSTPlanner's own search, the one with can_stay.
        cell = (1, 1)
        walls = [(x, 1) for x in range(7) if x != 3]
        planner = Planner(1, 0.45, walls, bounds=(0, 7, 0, 2), allow_diagonal=False)
        starts = [(3, 0), (0, 0)]
        goals = [(3, 0), (6, 0)]
        paths = planner.plan(starts, goals, assign=keep_order,
                             low_level_max_iter=2000, max_process=1)
        self.assertEqual(len(paths), len(starts), 'no solution found')
        self.assert_valid_solution(planner, paths, starts, goals, cell)
        self.assertTrue(any(tuple(p) != goals[0] for p in paths[0]), 'agent 0 never left its goal')


if __name__ == '__main__':
    unittest.main()
