#!/usr/bin/env python3
'''
Grid with rectangular cells for the Space-Time A* low level planner.
stastar's own Grid takes a single side length, so its cells are always square.
'''
from typing import Dict, List, Optional, Set, Tuple, Union
from heapq import heappush, heappop
import numpy as np
from scipy.spatial import KDTree

from stastar.planner import Planner as STPlanner
from stastar.neighbour_table import NeighbourTable

# Either a side length for square cells, or (cell_width, cell_height)
GridSize = Union[int, Tuple[int, int]]

# Map edges as (minx, maxx, miny, maxy); maxx and maxy are exclusive
Bounds = Tuple[int, int, int, int]


class FourConnectedNeighbourTable(NeighbourTable):
    '''Neighbours without diagonal moves: wait, right, left, down, up.'''
    directions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def cell_dimensions(grid_size: GridSize) -> Tuple[int, int]:
    if np.isscalar(grid_size):
        width = height = grid_size
    else:
        width, height = grid_size
    if width <= 0 or height <= 0:
        raise ValueError('Grid cell dimensions must be positive, got {0}'.format(grid_size))
    return width, height


class Grid:

    '''
    bounds sets the map edges explicitly. Without it, the edges are the outermost
    static obstacles, so any free space beyond them is not part of the map.
    '''
    def __init__(self, grid_size: GridSize, static_obstacles: np.ndarray,
                 bounds: Optional[Bounds] = None):
        self.cell_width, self.cell_height = cell_dimensions(grid_size)
        if bounds is None:
            bounds = self.calculate_boundaries(static_obstacles)
        self.minx, self.maxx, self.miny, self.maxy = bounds
        self.grid = self.make_grid(self.cell_width, self.cell_height,
                                   self.minx, self.maxx, self.miny, self.maxy)
        if self.grid.size == 0:
            raise ValueError('Grid cells of {0}x{1} do not fit inside the map bounded by the static obstacles'
                             .format(self.cell_width, self.cell_height))

    @staticmethod
    def calculate_boundaries(static_obstacles: np.ndarray) -> Tuple[int, int, int, int]:
        min_ = np.min(static_obstacles, axis=0)
        max_ = np.max(static_obstacles, axis=0)
        return min_[0], max_[0], min_[1], max_[1]

    '''
    Grid of cell centres indexed as grid[row][column] = [x, y]
    '''
    @staticmethod
    def make_grid(cell_width: int, cell_height: int,
                  minx: int, maxx: int, miny: int, maxy: int) -> np.ndarray:
        x_size = (maxx - minx) // cell_width
        y_size = (maxy - miny) // cell_height
        xs = minx + cell_width * (np.arange(x_size) + 0.5)
        ys = miny + cell_height * (np.arange(y_size) + 0.5)
        # Assigning into int32 truncates the half-cell offsets the same way stastar does
        grid = np.zeros([y_size, x_size, 2], dtype=np.int32)
        grid[:, :, 0] = xs[np.newaxis, :]
        grid[:, :, 1] = ys[:, np.newaxis]
        return grid

    '''
    Snap an arbitrary position to the centre of the nearest cell
    '''
    def snap_to_grid(self, position: np.ndarray) -> np.ndarray:
        i = int((position[1] - self.miny) // self.cell_height)
        j = int((position[0] - self.minx) // self.cell_width)
        i = min(max(i, 0), len(self.grid) - 1)
        j = min(max(j, 0), len(self.grid[0]) - 1)
        return self.grid[i][j]


class RectSTPlanner(STPlanner):
    '''
    Space-Time A* planner on a grid of rectangular cells.
    Mirrors STPlanner.__init__ with the Grid above; the search itself is inherited.
    bounds sets the map edges (see Grid); allow_diagonal=False restricts moves to
    right, left, down and up.
    '''
    def __init__(self, grid_size: GridSize,
                       robot_radius: int,
                       static_obstacles: List[Tuple[int, int]],
                       bounds: Optional[Bounds] = None,
                       allow_diagonal: bool = True):

        self.grid_size = cell_dimensions(grid_size)
        self.robot_radius = robot_radius
        np_static_obstacles = np.array(static_obstacles)
        self.static_obstacles = KDTree(np_static_obstacles)

        self.grid = Grid(grid_size, np_static_obstacles, bounds)
        table = NeighbourTable if allow_diagonal else FourConnectedNeighbourTable
        self.neighbour_table = table(self.grid.grid)

        # Plain-tuple copies of the map for the fast search in plan()
        self._blocked = {(int(x), int(y)) for x, y in np_static_obstacles}
        self._neighbours = {(int(key[0]), int(key[1])): [(int(x), int(y)) for x, y in cells]
                            for key, cells in self.neighbour_table.table.items()}
        self.set_reserved({})

    '''
    Cells taken by something outside the search (e.g. a robot carrying a pallet on a fixed
    path), as {time: {(x, y), ...}} with time 0 = the start of every search. No path may
    enter them at that time, and no path may end on a cell that is reserved later.
    '''
    def set_reserved(self, reserved: Dict[int, Set[Tuple[int, int]]]):
        self.reserved = reserved
        self._reserved_until = {}
        for t, cells in reserved.items():
            for cell in cells:
                self._reserved_until[cell] = max(t, self._reserved_until.get(cell, -1))

    '''
    Space-Time A*, same inputs, outputs and limits as STPlanner.plan, but with sets and
    tuples instead of a list scan, string hashing and KDTree queries per neighbour.
    Grid cells are at least 1 apart, so when 2 * robot_radius < 1 the inherited distance
    checks (int(l2) > radius) only reject the obstacle's own cell: a set lookup is exact.
    Otherwise it falls back to the inherited search.

    Extras for a robot carrying pallets: offsets is its footprint relative to its own
    cell (every cell must stay in bounds and clear), ignore lists static obstacles it
    may cover (the home cell of the pallet it carries), and start_time is the time of
    the first state, so dynamic obstacles and reserved cells line up with a later leg.
    '''
    def plan(self, start: Tuple[int, int],
                   goal: Tuple[int, int],
                   dynamic_obstacles: Dict[int, Set[Tuple[int, int]]],
                   semi_dynamic_obstacles: Dict[int, Set[Tuple[int, int]]] = None,
                   max_iter: int = 500,
                   debug: bool = False,
                   offsets: List[Tuple[int, int]] = ((0, 0),),
                   ignore: Set[Tuple[int, int]] = frozenset(),
                   start_time: int = 0) -> np.ndarray:
        if 2 * self.robot_radius >= 1:
            return super().plan(start, goal, dynamic_obstacles, semi_dynamic_obstacles, max_iter, debug)

        dynamic = {t: {(int(x), int(y)) for x, y in cells} for t, cells in dynamic_obstacles.items()}
        dynamic_until = {}  # last time each cell is constrained for this agent
        for t, cells in dynamic.items():
            for cell in cells:
                dynamic_until[cell] = max(t, dynamic_until.get(cell, -1))
        semi_dynamic = [(t, {(int(x), int(y)) for x, y in cells})
                        for t, cells in (semi_dynamic_obstacles or {}).items()]
        start = tuple(int(v) for v in self.grid.snap_to_grid(np.array(start)))
        goal = tuple(int(v) for v in self.grid.snap_to_grid(np.array(goal)))
        gx, gy = goal
        minx, maxx, miny, maxy = self.grid.minx, self.grid.maxx, self.grid.miny, self.grid.maxy
        reserved, reserved_until = self.reserved, self._reserved_until

        def blocked(pos, time):
            for dx, dy in offsets:
                cell = (pos[0] + dx, pos[1] + dy)
                if not (minx <= cell[0] < maxx and miny <= cell[1] < maxy):
                    return True
                if (cell in self._blocked and cell not in ignore) or cell in dynamic.get(time, ()) \
                        or cell in reserved.get(time, ()):
                    return True
                if any(time >= t and cell in cells for t, cells in semi_dynamic):
                    return True
            return False

        def can_stay(pos, time):
            # Arriving at the goal means staying there: nothing may be reserved there later,
            # and no constraint may forbid this agent from being there later
            cells = [(pos[0] + dx, pos[1] + dy) for dx, dy in offsets]
            return all(reserved_until.get(cell, -1) <= time and dynamic_until.get(cell, -1) <= time
                       for cell in cells)

        tie = 0  # FIFO among equal f scores
        open_heap = [(abs(start[0] - gx) + abs(start[1] - gy), tie, 0, start, start_time)]
        open_keys = {(start, start_time)}
        closed = set()
        came_from = {}
        iter_ = 0
        while open_heap and iter_ < max_iter:
            iter_ += 1
            _, _, g, pos, time = heappop(open_heap)
            if pos == goal and can_stay(pos, time):
                if debug:
                    print('STA*: Path found after {0} iterations'.format(iter_))
                path = [pos]
                key = (pos, time)
                while key in came_from:
                    key = came_from[key]
                    path.append(key[0])
                return np.array(path[::-1], dtype=np.int32)
            key = (pos, time)
            open_keys.discard(key)
            closed.add(key)
            epoch = time + 1
            for nb in self._neighbours[pos]:
                nb_key = (nb, epoch)
                if nb_key in closed or nb_key in open_keys or blocked(nb, epoch):
                    continue
                came_from[nb_key] = key
                open_keys.add(nb_key)
                tie += 1
                heappush(open_heap, (g + 1 + abs(nb[0] - gx) + abs(nb[1] - gy), tie, g + 1, nb, epoch))

        if debug:
            print('STA*: Open set is empty, no path found.')
        return np.array([])
