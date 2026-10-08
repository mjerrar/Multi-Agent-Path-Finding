#!/usr/bin/env python3
'''
Tests for agents that cover several cells (Agent.offsets), e.g. a robot carrying pallets.
Run from the repository root with: python -m unittest discover -s tests
'''
import unittest
from itertools import combinations
import numpy as np

from cbs_mapf.agent import Agent
from cbs_mapf.constraint_tree import CTNode
from cbs_mapf.constraints import Constraints
from cbs_mapf.planner import Planner

ROBOT = (0, 0)
LEFT, RIGHT, UP = (-1, 0), (1, 0), (0, 1)


def grid_planner(walls, bounds):
    '''1 x 1 cells, robots smaller than a cell and 4-connected moves, as in a warehouse'''
    return Planner(1, 0.45, walls, bounds=bounds, allow_diagonal=False)


def covered(agent, position):
    '''The cells an agent at position covers'''
    return {(int(position[0]) + dx, int(position[1]) + dy) for dx, dy in agent.offsets}


def plan(planner, agents):
    '''CBS paths for agents as given, in their order'''
    return planner.plan([agent.start for agent in agents], [agent.goal for agent in agents],
                        assign=lambda starts, goals: agents,
                        low_level_max_iter=2000, max_process=1)


class TestFootprintConflicts(unittest.TestCase):

    def setUp(self):
        self.planner = grid_planner([(9, 2)], bounds=(0, 10, 0, 3))

    def test_carried_pallet_is_a_conflict(self):
        # Agent 0 waits at (1, 1) with a pallet on (2, 1); agent 1 walks onto (2, 1) at t=2
        carrier = Agent((1, 1), (1, 1), [ROBOT, RIGHT])
        walker = Agent((4, 1), (2, 0))
        paths = {carrier: np.array([(1, 1)]), walker: np.array([(4, 1), (3, 1), (2, 1), (2, 0)])}
        node = CTNode(Constraints(), paths)
        agent_i, agent_j, time = self.planner.validate_paths([carrier, walker], node)
        self.assertIs(agent_i, carrier)
        self.assertIs(agent_j, walker)
        self.assertEqual(time, 2)
        # Without the pallet nothing collides
        point = Agent((1, 1), (1, 1))
        node = CTNode(Constraints(), {point: paths[carrier], walker: paths[walker]})
        self.assertIsNone(self.planner.validate_paths([point, walker], node)[0])

    def test_constraints_are_on_the_shared_cell(self):
        # Both robots carry a pallet; at t=1 their pallets meet on (2, 1), while the robots
        # stay apart. Each agent must keep its pallet off (2, 1) at t=1, not its robot
        left = Agent((1, 1), (1, 1), [ROBOT, RIGHT])
        right = Agent((4, 1), (3, 1), [ROBOT, LEFT])
        node = CTNode(Constraints(), {left: np.array([(1, 1)]), right: np.array([(4, 1), (3, 1)])})
        self.assertEqual(self.planner.calculate_constraints(node, left, right, 1)[left], {1: {(2, 1)}})
        self.assertEqual(self.planner.calculate_constraints(node, right, left, 1)[right], {1: {(2, 1)}})

    def test_goal_times_cover_the_pallet(self):
        carrier = Agent((1, 1), (3, 1), [ROBOT, UP])
        walker = Agent((8, 0), (8, 1))
        paths = {carrier: np.array([(1, 1), (2, 1), (3, 1)]), walker: np.array([(8, 0), (8, 1)])}
        goal_times = self.planner.calculate_goal_times(CTNode(Constraints(), paths), walker, [carrier, walker])
        self.assertEqual(goal_times, {2: {(3, 1), (3, 2)}})

    def test_offsets_need_cell_conflicts(self):
        planner = Planner(10, 10, [(0, 0), (100, 0), (0, 100), (100, 100)])
        with self.assertRaises(ValueError):
            plan(planner, [Agent((15, 15), (85, 85), [ROBOT, (10, 0)])])


class TestCBSFootprints(unittest.TestCase):

    def assert_valid_solution(self, paths, agents, walls, bounds):
        minx, maxx, miny, maxy = bounds
        self.assertEqual(len(paths), len(agents), 'no solution found')
        for path, agent in zip(paths, agents):
            np.testing.assert_array_equal(path[0], agent.start)
            np.testing.assert_array_equal(path[-1], agent.goal)
            # Each step waits or moves one cell up, down, left or right
            self.assertTrue(np.all(np.abs(np.diff(path, axis=0)).sum(axis=1) <= 1))
        # Every covered cell is on the map and off the walls, and no two agents ever cover the
        # same cell. Paths are not padded: an agent whose path has ended waits on its goal
        for t in range(max(len(path) for path in paths)):
            cells_at_t = [covered(agent, path[min(t, len(path) - 1)]) for agent, path in zip(agents, paths)]
            for cells in cells_at_t:
                for x, y in cells:
                    self.assertTrue(minx <= x < maxx and miny <= y < maxy, '({0}, {1}) is off the map'.format(x, y))
                    self.assertNotIn((x, y), walls)
            for i, j in combinations(range(len(agents)), 2):
                self.assertFalse(cells_at_t[i] & cells_at_t[j], 'agents {0} and {1} overlap at t={2}'.format(i, j, t))

    def test_carrier_and_walker_pass_head_on(self):
        # A 10 x 3 aisle: agent 0 carries a pallet on its right along the middle row, agent 1
        # walks the other way along the same row
        walls, bounds = [(5, 2)], (0, 10, 0, 3)
        agents = [Agent((1, 1), (7, 1), [ROBOT, RIGHT]), Agent((9, 1), (0, 1))]
        paths = plan(grid_planner(walls, bounds), agents)
        self.assert_valid_solution(paths, agents, walls, bounds)

    def test_two_carriers_and_a_walker(self):
        walls, bounds = [(5, 2)], (0, 10, 0, 3)
        agents = [Agent((1, 0), (8, 0), [ROBOT, UP]),
                  Agent((8, 1), (1, 1), [ROBOT, LEFT]),
                  Agent((4, 2), (4, 0))]
        paths = plan(grid_planner(walls, bounds), agents)
        self.assert_valid_solution(paths, agents, walls, bounds)

    def test_carrier_steps_off_its_goal_to_let_another_pass(self):
        # A corridor along y=0 with a two-cell pocket above (4, 0) and (5, 0). Agent 0 starts
        # on its goal (4, 0) with a pallet on (5, 0), and agent 1 must cross both cells, so
        # agent 0 has to back into the pocket with its pallet and come out again
        walls = [(x, 1) for x in range(10) if x not in (4, 5)]
        bounds = (0, 10, 0, 2)
        agents = [Agent((4, 0), (4, 0), [ROBOT, RIGHT]), Agent((0, 0), (9, 0))]
        paths = plan(grid_planner(walls, bounds), agents)
        self.assert_valid_solution(paths, agents, walls, bounds)
        self.assertIn([4, 1], paths[0].tolist(), 'agent 0 never backed into the pocket')


if __name__ == '__main__':
    unittest.main()
