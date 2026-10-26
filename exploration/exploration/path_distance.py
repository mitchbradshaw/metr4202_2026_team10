"""
path_distance.py

How far the robot would actually have to drive to each candidate, instead of
the straight-line distance through walls. A frontier on the other side of a
wall looks close in a straight line but can be a long way round, and one the
robot can't reach at all (e.g. seen through a gap in the lidar scan) gets no
distance, so it is never sent.

One Dijkstra search over the free cells of /map, out from the robot, gives the
driving distance to every free cell at once. Steps go to the 8 neighbours: a
straight step costs 1 cell, a diagonal step sqrt(2), and a diagonal step is
not allowed to squeeze between two cells that aren't free (no corner cutting).

Pure logic with no ROS imports, like the other modules, so it is easy to test.
"""

import heapq
import math

# (d_row, d_col, step length in cells)
NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
]


def path_distances(data, width, height, start, occupied_threshold=65):
    """
    data      : 1D occupancy list (msg.data)
    width     : msg.info.width
    height    : msg.info.height
    start     : (row, col) of the robot's cell
    returns   : list with the driving distance [cells] to every cell, indexed
                row * width + col: math.inf for a free cell the robot can't
                get to, None for a cell that isn't free (unknown or wall);
                None overall if start is outside the map
    """
    s_row, s_col = start
    if s_row < 0 or s_row >= height or s_col < 0 or s_col >= width:
        return None

    # Work out once which cells the robot can drive on: not unknown (-1), not a wall.
    free = [0 <= value < occupied_threshold for value in data]

    dist = [math.inf if is_free else None for is_free in free]
    dist[s_row * width + s_col] = 0.0       # start here even if the robot's own cell isn't free
    heap = [(0.0, s_row, s_col)]

    while heap:
        d, row, col = heapq.heappop(heap)
        if d > dist[row * width + col]:
            continue                        # stale entry: already found a shorter way here

        for d_row, d_col, step in NEIGHBOURS:
            n_row = row + d_row
            n_col = col + d_col
            if n_row < 0 or n_row >= height or n_col < 0 or n_col >= width:
                continue
            idx = n_row * width + n_col
            if not free[idx]:
                continue
            # a diagonal step needs both cells beside it free, or it cuts a corner
            if d_row != 0 and d_col != 0:
                if not (free[row * width + n_col] and free[n_row * width + col]):
                    continue

            new_d = d + step
            if new_d < dist[idx]:
                dist[idx] = new_d
                heapq.heappush(heap, (new_d, n_row, n_col))

    return dist


def distance_to(dist, width, height, cell, search=2):
    """
    Driving distance [cells] to cell, or None if the robot can't get there.

    A goal nudged off a wall in the costmap can land on a cell that /map still
    has as unknown. If cell itself isn't free, the reachable cell nearest to it
    (within a square of +-search cells) is used instead, plus the step from
    there. A free cell the search never reached stays unreachable, however
    close reachable cells are (they may be on the other side of a wall).
    """
    row, col = cell
    d = None
    if 0 <= row < height and 0 <= col < width:
        d = dist[row * width + col]

    if d is None:
        d = math.inf
        best_step = None
        for n_row in range(max(0, row - search), min(height, row + search + 1)):
            for n_col in range(max(0, col - search), min(width, col + search + 1)):
                n_d = dist[n_row * width + n_col]
                if n_d is None or n_d == math.inf:
                    continue
                step = math.hypot(n_row - row, n_col - col)
                if best_step is None or step < best_step:
                    best_step = step
                    d = n_d + step

    return None if d == math.inf else d


def world_to_cell(x, y, info):
    """Map-frame (x, y) [m] -> (row, col) of the /map cell it falls in."""
    col = int(math.floor((x - info.origin.position.x) / info.resolution))
    row = int(math.floor((y - info.origin.position.y) / info.resolution))
    return row, col
