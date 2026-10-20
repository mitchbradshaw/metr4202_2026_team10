"""
waypoint_finder.py

Turns raw frontier cells into a small list of candidate waypoints.

Pipeline inside find_waypoints():
    frontier cells
        -> size-capped BFS clustering          (group connected frontier into bites)
        -> drop clusters that are too small     (noise rejection)
        -> centroid of each cluster, snapped to a real frontier cell (the waypoint)
        -> estimate info gain at each waypoint  (unknown area its rays reach before a wall)
        -> convert the waypoint cell to world (map-frame) coordinates
    -> list[Waypoint]

The class holds no ROS state; it is pure logic so it is easy to unit test.
The owning node passes in the latest /map data every cycle.
"""

from collections import deque
import math


class Waypoint:
    """A single candidate goal the robot could drive to."""

    def __init__(self, row, col, x, y, gain, cluster_size,
                 marker_id=None, marker_x=None, marker_y=None):
        self.row = row                    # grid row of the waypoint cell
        self.col = col                    # grid column of the waypoint cell
        self.x = x                        # world x in the map frame [m]
        self.y = y                        # world y in the map frame [m]
        self.gain = gain                  # estimated info gain (visible unknown area, m^2)
        self.cluster_size = cluster_size  # number of frontier cells that formed it
        self.marker_id = marker_id        # set when this is a view pose for an ArUco marker
        self.marker_x = marker_x          # that marker's position, to face it on arrival
        self.marker_y = marker_y


class WaypointFinder:

    def __init__(self,
                 max_cluster_size,
                 min_cluster_size,
                 sensor_radius_cells,
                 unknown_value=-1,
                 occupied_threshold=65,
                 n_rays=120):
        # Largest a cluster may grow before we cut it and start a new one.
        # Roughly one sensor footprint, so each waypoint is one "bite".
        self.max_cluster_size = max_cluster_size

        # Clusters smaller than this are treated as map noise and skipped.
        self.min_cluster_size = min_cluster_size

        # How far the gain rays reach, in cells.
        self.sensor_radius_cells = sensor_radius_cells

        # Value that marks an unknown cell in the OccupancyGrid (-1 in ROS).
        self.unknown_value = unknown_value

        # Cells at or above this value are walls and stop a ray (slam_toolbox uses 100).
        self.occupied_threshold = occupied_threshold

        # Directions of the gain rays as (d_row, d_col), spread evenly round a circle.
        self.ray_dirs = []
        for i in range(n_rays):
            angle = 2.0 * math.pi * i / n_rays
            self.ray_dirs.append((math.sin(angle), math.cos(angle)))

    # ---- public entry point -------------------------------------------------

    def find_waypoints(self, data, info, frontier_cells):
        """
        data           : msg.data   (1D list of occupancy values)
        info           : msg.info   (has width, height, resolution, origin)
        frontier_cells : list of (row, col) from the frontier detector
        returns        : list[Waypoint]
        """
        width = info.width
        height = info.height

        clusters = self._cluster_frontiers_capped(frontier_cells)

        waypoints = []
        for cluster in clusters:
            # Noise rejection: ignore tiny specks of frontier.
            if len(cluster) < self.min_cluster_size:
                continue

            # One representative cell for the whole cluster.
            cell = self._cluster_waypoint(cluster)
            w_row, w_col = cell

            # How much unknown space would we likely reveal from here?
            gain = self._estimate_gain(data, width, height, cell, info.resolution)

            # Grid cell -> world coordinates so the decider can measure distance
            # against the robot pose (which lives in the map frame, in metres).
            x, y = self._grid_to_world(w_row, w_col, info)

            waypoints.append(
                Waypoint(w_row, w_col, x, y, gain, len(cluster))
            )

        return waypoints

    # ---- step 1: size-capped BFS clustering ---------------------------------

    def _cluster_frontiers_capped(self, frontier_cells):
        frontier_set = set(frontier_cells)   # fast membership test
        assigned = set()                     # cells already placed in a finished cluster
        clusters = []

        for seed in frontier_cells:
            if seed in assigned:
                continue

            # Start a new cluster with a BFS out from this seed cell.
            cluster = []
            queue = deque()
            queue.append(seed)
            enqueued = set()                 # prevents queuing the same cell twice
            enqueued.add(seed)

            # Grow until we hit the size cap OR run out of connected frontier.
            while queue and len(cluster) < self.max_cluster_size:
                current = queue.popleft()    # popleft = BFS = compact blobs
                cluster.append(current)
                assigned.add(current)
                c_row, c_col = current

                # 8-connectivity keeps thin diagonal curves in one piece.
                for d_row in (-1, 0, 1):
                    for d_col in (-1, 0, 1):
                        if d_row == 0 and d_col == 0:
                            continue
                        neighbour = (c_row + d_row, c_col + d_col)
                        if neighbour not in frontier_set:
                            continue
                        if neighbour in assigned:
                            continue
                        if neighbour in enqueued:
                            continue
                        enqueued.add(neighbour)
                        queue.append(neighbour)

            # Any cell that was enqueued but never popped (because we hit the cap)
            # was never added to 'assigned', so the outer loop will find it later
            # and start the next cluster from it. That is what tiles a long curve.
            clusters.append(cluster)

        return clusters

    # ---- step 2: waypoint = centroid snapped to a real frontier cell ---------

    def _cluster_waypoint(self, cluster):
        # Geometric centroid of the cluster.
        sum_row = 0
        sum_col = 0
        for (row, col) in cluster:
            sum_row += row
            sum_col += col
        n = len(cluster)
        centroid_row = sum_row / n
        centroid_col = sum_col / n

        # Snap to the actual frontier cell nearest the centroid, so the waypoint
        # always lands on a free boundary cell (never in unknown/occupied space).
        best_cell = None
        best_dist_sq = None
        for (row, col) in cluster:
            d_row = row - centroid_row
            d_col = col - centroid_col
            dist_sq = d_row * d_row + d_col * d_col
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_cell = (row, col)

        return best_cell

    # ---- step 3: info gain = unknown area the lidar could see from here ------

    def _estimate_gain(self, data, width, height, waypoint, resolution):
        """
        Cast rays out from the waypoint like a lidar scan and add up the
        unknown cells they cross before they hit a wall. Unknown space behind
        a wall (e.g. outside the arena) can't be seen, so it no longer counts.

        Rays spread apart as they go out, so a sample k cells out stands for a
        wedge of floor 2*pi*k / n_rays cells wide. Weighting each sample by
        that keeps the result an area in m^2, whatever n_rays is.
        """
        w_row, w_col = waypoint
        n_rays = len(self.ray_dirs)
        gain_cells = 0.0

        for d_row, d_col in self.ray_dirs:
            row, col = w_row, w_col                 # last cell this ray was in
            for k in range(1, self.sensor_radius_cells + 1):
                # k cell lengths out from the centre of the waypoint cell
                n_row = int(math.floor(w_row + 0.5 + k * d_row))
                n_col = int(math.floor(w_col + 0.5 + k * d_col))

                if n_row < 0 or n_row >= height or n_col < 0 or n_col >= width:
                    break                           # off the map
                if self._is_wall(data, width, n_row, n_col):
                    break                           # the lidar stops at walls

                # A diagonal step slips between two cells that touch at a corner.
                # SLAM draws slanted walls corner to corner like that, so if both
                # of those cells are walls the ray is blocked too.
                if (n_row != row and n_col != col
                        and self._is_wall(data, width, row, n_col)
                        and self._is_wall(data, width, n_row, col)):
                    break

                if data[n_row * width + n_col] == self.unknown_value:
                    gain_cells += 2.0 * math.pi * k / n_rays
                row, col = n_row, n_col

        return gain_cells * resolution * resolution

    def _is_wall(self, data, width, row, col):
        return data[row * width + col] >= self.occupied_threshold

    # ---- step 4: grid -> world ----------------------------------------------

    def _grid_to_world(self, row, col, info):
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        # +0.5 puts the point at the centre of the cell, not its corner.
        x = origin_x + (col + 0.5) * resolution
        y = origin_y + (row + 0.5) * resolution
        return x, y
