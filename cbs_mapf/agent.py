#!/usr/bin/env python3
'''
Author: Haoran Peng
Email: gavinsweden@gmail.com
'''
from typing import Sequence, Tuple
import numpy as np


class Agent:

    '''
    offsets are the cells the agent covers relative to its own position, in the same units:
    (0, 0) for the agent itself, plus e.g. the cells of pallets docked to it. They move with
    the agent and need a planner with 2 * robot_radius < 1 (see Planner).
    '''
    def __init__(self, start: Tuple[int, int], goal: Tuple[int, int],
                 offsets: Sequence[Tuple[int, int]] = ((0, 0),)):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.offsets = tuple((int(dx), int(dy)) for dx, dy in offsets)

    # Uniquely identify an agent with its start position
    def __hash__(self):
        return int(str(self.start[0]) + str(self.start[1]))

    def __eq__(self, other: 'Agent'):
        return np.array_equal(self.start, other.start) and \
               np.array_equal(self.goal, other.goal)

    def __str__(self):
        return str(self.start.tolist())

    def __repr__(self):
        return self.__str__()
