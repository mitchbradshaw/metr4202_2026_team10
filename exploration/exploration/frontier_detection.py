"""
frontier_detection.py

Pure frontier-detection logic with no ROS dependencies, so the frontier_explorer
node can use it and the unit tests can check it on hand-made grids.

A frontier cell is a FREE cell with at least one UNKNOWN 4-neighbour: it sits on
the boundary between mapped and unmapped space.
"""

FREE = 0
UNKNOWN = -1


def find_frontiers(data, width, height):
    """
    data   : 1D occupancy list (msg.data)
    width  : msg.info.width
    height : msg.info.height
    returns: list of (row, col) frontier cells
    """
    frontiers = []

    for row in range(height):
        for col in range(width):
            idx = row * width + col   # index of this cell in the 1D list

            if data[idx] != FREE:
                continue

            # 4-connected neighbours: up, down, left, right.
            neighbours = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ]

            for n_row, n_col in neighbours:
                # Skip neighbours outside the map bounds.
                if n_row < 0 or n_row >= height:
                    continue
                if n_col < 0 or n_col >= width:
                    continue

                neighbour_idx = n_row * width + n_col
                if data[neighbour_idx] == UNKNOWN:
                    frontiers.append((row, col))
                    break   # one unknown neighbour is enough

    return frontiers
