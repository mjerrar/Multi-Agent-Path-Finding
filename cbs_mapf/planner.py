#!/usr/bin/env python3
'''
Author: Haoran Peng
Email: gavinsweden@gmail.com

An implementation of multi-agent path finding using conflict-based search
[Sharon et al., 2015]
'''
from typing import List, Optional, Tuple, Dict, Callable, Set
import multiprocessing as mp
from heapq import heappush, heappop
from itertools import combinations
from copy import deepcopy
import numpy as np

from .constraint_tree import CTNode
from .constraints import Constraints
from .agent import Agent
# The low level planner for CBS is the Space-Time A* planner
# https://github.com/GavinPHR/Space-Time-AStar, extended to rectangular grid cells
from .grid import Bounds, GridSize, RectSTPlanner
from .assigner import *
class Planner:

    '''
    grid_size is either a side length for square cells or (cell_width, cell_height).
    bounds is (minx, maxx, miny, maxy) with exclusive maxima; by default the map edges
    are the outermost static obstacles. allow_diagonal=False permits only right, left,
    down and up moves.
    With 2 * robot_radius < 1, agents collide only when they cover the same cell at the same
    time, and an agent can cover several cells (Agent.offsets), e.g. a robot carrying pallets.
    Otherwise agents are points that collide within 2 * robot_radius of each other.
    '''
    def __init__(self, grid_size: GridSize,
                       robot_radius: int,
                       static_obstacles: List[Tuple[int, int]],
                       bounds: Optional[Bounds] = None,
                       allow_diagonal: bool = True):

        self.robot_radius = robot_radius
        self.cell_conflicts = 2 * robot_radius < 1
        self.st_planner = RectSTPlanner(grid_size, robot_radius, static_obstacles,
                                        bounds, allow_diagonal)

    '''
    You can use your own assignment function, the default algorithm greedily assigns
    the closest goal to each start.
    '''
    def plan(self, starts: List[Tuple[int, int]],
                   goals: List[Tuple[int, int]],
                   assign:Callable = min_cost,
                   max_iter:int = 200,
                   low_level_max_iter:int = 100,
                   max_process:int = 10,
                   debug:bool = False) -> np.ndarray:

        self.low_level_max_iter = low_level_max_iter
        self.debug = debug

        # Do goal assignment
        self.agents = assign(starts, goals)
        if not self.cell_conflicts and any(agent.offsets != ((0, 0),) for agent in self.agents):
            raise ValueError('Agents covering several cells (Agent.offsets) need 2 * robot_radius < 1')

        constraints = Constraints()

        # Compute path for each agent using low level planner
        solution = dict((agent, self.calculate_path(agent, constraints, None)) for agent in self.agents)

        open = []
        if all(len(path) != 0 for path in solution.values()):
            # Make root node
            node = CTNode(constraints, solution)
            # Min heap for quick extraction
            open.append(node)

        iter_ = 0
        if max_process <= 1:
            # Search in this process: no worker or manager processes, so Ctrl+C
            # stops it like any other Python code.
            while open and iter_ < max_iter:
                iter_ += 1
                results = []
                self.search_node(heappop(open), results)
                paths = self.collect_results(results, open)
                if paths is not None:
                    if debug:
                        print('CBS_MAPF: Paths found after {0} iterations'.format(iter_))
                    return paths
        else:
            manager = mp.Manager()
            processes = []
            try:
                while open and iter_ < max_iter:
                    iter_ += 1

                    results = manager.list([])

                    processes = []

                    # Default to 10 processes maximum
                    for _ in range(max_process if len(open) > max_process else len(open)):
                        # daemon: exiting the main process never waits for a worker
                        p = mp.Process(target=self.search_node, args=[heappop(open), results],
                                       daemon=True)
                        processes.append(p)
                        p.start()

                    for p in processes:
                        p.join()

                    paths = self.collect_results(results, open)
                    if paths is not None:
                        if debug:
                            print('CBS_MAPF: Paths found after about {0} iterations'.format(4 * iter_))
                        return paths
            finally:
                # Also runs on Ctrl+C: stop any workers left and the manager process
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                manager.shutdown()

        if debug:
            print('CBS-MAPF: Open set is empty, no paths found.')
        return np.array([])

    '''
    Return the paths if a search result is a solution; otherwise push the
    result's child nodes onto the open set and return None.
    '''
    @staticmethod
    def collect_results(results, open) -> Optional[np.ndarray]:
        for result in results:
            if len(result) == 1:
                return result[0]
            if result[0]:
                heappush(open, result[0])
            if result[1]:
                heappush(open, result[1])
        return None

    '''
    Abstracted away the cbs search for multiprocessing.
    The parameters open and results MUST BE of type ListProxy to ensure synchronization.
    '''
    def search_node(self, best: CTNode, results):
        agent_i, agent_j, time_of_conflict = self.validate_paths(self.agents, best)

        # If there is not conflict, validate_paths returns (None, None, -1)
        if agent_i is None:
            results.append((self.reformat(self.agents, best.solution),))
            return
        # Calculate new constraints
        agent_i_constraint = self.calculate_constraints(best, agent_i, agent_j, time_of_conflict)
        agent_j_constraint = self.calculate_constraints(best, agent_j, agent_i, time_of_conflict)

        # Calculate new paths
        agent_i_path = self.calculate_path(agent_i,
                                           agent_i_constraint,
                                           self.calculate_goal_times(best, agent_i, self.agents))
        agent_j_path = self.calculate_path(agent_j,
                                           agent_j_constraint,
                                           self.calculate_goal_times(best, agent_j, self.agents))

        # Replace old paths with new ones in solution
        solution_i = best.solution
        solution_j = deepcopy(best.solution)
        solution_i[agent_i] = agent_i_path
        solution_j[agent_j] = agent_j_path

        node_i = None
        if all(len(path) != 0 for path in solution_i.values()):
            node_i = CTNode(agent_i_constraint, solution_i)

        node_j = None
        if all(len(path) != 0 for path in solution_j.values()):
            node_j = CTNode(agent_j_constraint, solution_j)

        results.append((node_i, node_j))


    '''
    Pair of agent, point of conflict
    '''
    def validate_paths(self, agents, node: CTNode):
        # Check collision pair-wise
        for agent_i, agent_j in combinations(agents, 2):
            time_of_conflict = self.safe_distance(node.solution, agent_i, agent_j)
            # time_of_conflict=1 if there is not conflict
            if time_of_conflict == -1:
                continue
            return agent_i, agent_j, time_of_conflict
        return None, None, -1


    def safe_distance(self, solution: Dict[Agent, np.ndarray], agent_i: Agent, agent_j: Agent) -> int:
        path_i, path_j = solution[agent_i], solution[agent_j]
        # Check until the longer path ends: an agent that has arrived stays on its
        # goal, so it can still be hit by the other agent.
        for idx in range(max(len(path_i), len(path_j))):
            if self.conflict_cell(agent_i, path_i, agent_j, path_j, idx) is None:
                continue
            return idx
        return -1

    '''
    Where agent_i collides with agent_j at a time step, or None. With cell conflicts, a cell
    both cover; otherwise the position of agent_j, if within 2 * robot_radius of agent_i.
    '''
    def conflict_cell(self, agent_i: Agent, path_i: np.ndarray,
                            agent_j: Agent, path_j: np.ndarray, time: int) -> Optional[Tuple[int, int]]:
        position_i, position_j = self.position_at(path_i, time), self.position_at(path_j, time)
        if self.cell_conflicts:
            overlap = self.footprint(agent_i, position_i) & self.footprint(agent_j, position_j)
            return min(overlap) if overlap else None
        if self.dist(position_i, position_j) > 2*self.robot_radius:
            return None
        return tuple(position_j.tolist())

    '''
    Whether an agent at position covers cell (with point agents, comes within 2 * robot_radius of it)
    '''
    def covers(self, agent: Agent, position: np.ndarray, cell: Tuple[int, int]) -> bool:
        if self.cell_conflicts:
            return cell in self.footprint(agent, position)
        return self.dist(position, np.array(cell)) < 2*self.robot_radius

    @staticmethod
    def footprint(agent: Agent, position: np.ndarray) -> Set[Tuple[int, int]]:
        '''The cells an agent at position covers: its own and those at its offsets.'''
        x, y = int(position[0]), int(position[1])
        return {(x + dx, y + dy) for dx, dy in agent.offsets}

    @staticmethod
    def position_at(path: np.ndarray, time: int) -> np.ndarray:
        '''Where an agent is at a time step; after its path ends it waits on its goal.'''
        return path[min(time, len(path) - 1)]

    @staticmethod
    def dist(point1: np.ndarray, point2: np.ndarray) -> int:
        return int(np.linalg.norm(point1-point2, 2))  # L2 norm

    def calculate_constraints(self, node: CTNode,
                                    constrained_agent: Agent,
                                    unchanged_agent: Agent,
                                    time_of_conflict: int) -> Constraints:
        contrained_path = node.solution[constrained_agent]
        unchanged_path = node.solution[unchanged_agent]

        # The cell the constrained agent must keep off. The low level planner checks every
        # cell an agent covers against its constraints, so this holds for carried pallets too
        pivot = self.conflict_cell(constrained_agent, contrained_path,
                                   unchanged_agent, unchanged_path, time_of_conflict)
        conflict_end_time = time_of_conflict
        while conflict_end_time < len(contrained_path) and \
                self.covers(constrained_agent, contrained_path[conflict_end_time], pivot):
            conflict_end_time += 1
        # An agent already waiting on its goal is past the end of its path: still constrain
        # the time of conflict, or the child node would repeat its parent
        conflict_end_time = max(conflict_end_time, time_of_conflict + 1)
        return node.constraints.fork(constrained_agent, pivot, time_of_conflict, conflict_end_time)

    def calculate_goal_times(self, node: CTNode, agent: Agent, agents: List[Agent]):
        solution = node.solution
        goal_times = dict()
        for other_agent in agents:
            if other_agent == agent:
                continue
            time = len(solution[other_agent]) - 1
            # Every cell the other agent covers once it waits on its goal
            goal_times.setdefault(time, set()).update(self.footprint(other_agent, solution[other_agent][time]))
        return goal_times

    '''
    Calculate the paths for all agents with space-time constraints
    '''
    def calculate_path(self, agent: Agent, 
                       constraints: Constraints, 
                       goal_times: Dict[int, Set[Tuple[int, int]]]) -> np.ndarray:
        return self.st_planner.plan(agent.start,
                                    agent.goal,
                                    constraints.setdefault(agent, dict()),
                                    semi_dynamic_obstacles=goal_times,
                                    max_iter=self.low_level_max_iter,
                                    debug=self.debug,
                                    offsets=agent.offsets)

    '''
    Reformat the solution to a list of paths in agent order, each at its own (minimum)
    length, not padded. An agent whose path ends first waits on its goal: the conflict
    checks above already assume that (position_at), so the paths stay collision-free.
    '''
    @staticmethod
    def reformat(agents: List[Agent], solution: Dict[Agent, np.ndarray]):
        return [solution[agent] for agent in agents]

    '''
    Pad paths to equal length, inefficient but well..
    '''
    @staticmethod
    def pad(solution: Dict[Agent, np.ndarray]):
        max_ = max(len(path) for path in solution.values())
        for agent, path in solution.items():
            if len(path) == max_:
                continue
            padded = np.concatenate([path, np.array(list([path[-1]])*(max_-len(path)))])
            solution[agent] = padded
        return solution

