"""
test_exploration_logic.py

ROS-free unit tests for the three logic modules. Because none of them import
rclpy, we can build small hand-made occupancy grids and check that the right
values come out — no simulator, no Nav2, no tf2 required.

Maps are written as lists of strings for readability:
    '.'  free      -> 0
    '?'  unknown   -> -1
    '#'  occupied  -> 100

Run with:
    colcon test --packages-select exploration
    colcon test-result --verbose
or, from the package directory with the workspace sourced:
    python -m pytest test/test_exploration_logic.py -v
"""

from exploration.frontier_detection import find_frontiers
from exploration.path_distance import distance_to, path_distances
from exploration.waypoint_finder import WaypointFinder, Waypoint
from exploration.waypoint_decider import WaypointDecider, ranked


# ---- helpers ------------------------------------------------------------

def build_grid(rows):
    """Turn a list of strings into (data, width, height) like a ROS map."""
    height = len(rows)
    width = len(rows[0])

    data = []
    for r in range(height):
        for c in range(width):
            ch = rows[r][c]
            if ch == '.':
                data.append(0)      # free
            elif ch == '?':
                data.append(-1)     # unknown
            else:
                data.append(100)    # occupied
    return data, width, height


class FakePosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakeOrigin:
    def __init__(self, x, y):
        self.position = FakePosition(x, y)


class FakeMapInfo:
    """Stand-in for msg.info — only the fields the finder actually reads."""

    def __init__(self, width, height, resolution,
                 origin_x=0.0, origin_y=0.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = FakeOrigin(origin_x, origin_y)


# ---- frontier detection -------------------------------------------------

def test_frontiers_lie_on_the_free_unknown_boundary():
    # Left half free, right half unknown. The only free cells that touch
    # unknown are in column 4, so the frontier is that vertical line.
    data, width, height = build_grid([
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
    ])

    frontiers = find_frontiers(data, width, height)

    assert len(frontiers) == 10
    for (row, col) in frontiers:
        assert col == 4


def test_no_frontiers_in_fully_known_map():
    # No unknown cells anywhere -> no frontier.
    data, width, height = build_grid([
        '.....',
        '.....',
        '.....',
    ])

    frontiers = find_frontiers(data, width, height)

    assert frontiers == []


# ---- clustering (via the finder) ----------------------------------------

def test_large_frontier_stays_one_cluster_when_cap_is_high():
    # The 10-cell vertical frontier fits under a high cap, so it is not split.
    data, width, height = build_grid([
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
    ])
    info = FakeMapInfo(width, height, resolution=0.05)
    frontiers = find_frontiers(data, width, height)

    finder = WaypointFinder(
        max_cluster_size=100,     # far bigger than the frontier
        min_cluster_size=1,
        sensor_radius_cells=3,
    )
    waypoints = finder.find_waypoints(data, info, frontiers)

    assert len(waypoints) == 1


def test_long_frontier_is_split_when_cap_is_low():
    # Same 10-cell frontier, but a cap of 5 forces it into two bites.
    data, width, height = build_grid([
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
        '.....?????',
    ])
    info = FakeMapInfo(width, height, resolution=0.05)
    frontiers = find_frontiers(data, width, height)

    finder = WaypointFinder(
        max_cluster_size=5,       # forces a split of the 10-cell line
        min_cluster_size=1,
        sensor_radius_cells=3,
    )
    waypoints = finder.find_waypoints(data, info, frontiers)

    assert len(waypoints) == 2


def test_tiny_frontier_is_rejected_as_noise():
    # A single unknown cell in the corner creates only a 2-cell frontier.
    # With min_cluster_size = 5 it is discarded as noise.
    data, width, height = build_grid([
        '?....',
        '.....',
        '.....',
        '.....',
        '.....',
    ])
    info = FakeMapInfo(width, height, resolution=0.05)
    frontiers = find_frontiers(data, width, height)

    finder = WaypointFinder(
        max_cluster_size=100,
        min_cluster_size=5,       # bigger than the 2-cell frontier
        sensor_radius_cells=3,
    )
    waypoints = finder.find_waypoints(data, info, frontiers)

    assert len(waypoints) == 0


# ---- gain estimate ------------------------------------------------------

def test_gain_counts_unknown_and_is_zero_with_none_nearby():
    finder = WaypointFinder(
        max_cluster_size=100,
        min_cluster_size=1,
        sensor_radius_cells=4,
    )

    # Map A: a big unknown region sits right next to the waypoint.
    data_a, width_a, height_a = build_grid([
        '......?????',
        '......?????',
        '......?????',
        '......?????',
        '......?????',
        '......?????',
        '......?????',
        '......?????',
        '......?????',
        '......?????',
        '......?????',
    ])
    gain_a = finder._estimate_gain(data_a, width_a, height_a, (5, 5), 1.0)

    # Map B: fully known, so nothing to reveal.
    data_b, width_b, height_b = build_grid([
        '...........',
        '...........',
        '...........',
        '...........',
        '...........',
        '...........',
        '...........',
        '...........',
        '...........',
        '...........',
        '...........',
    ])
    gain_b = finder._estimate_gain(data_b, width_b, height_b, (5, 5), 1.0)

    assert gain_a > 0
    assert gain_b == 0

def test_gain_ignores_unknown_behind_a_wall():
    finder = WaypointFinder(
        max_cluster_size=100,
        min_cluster_size=1,
        sensor_radius_cells=8,
    )

    # Unknown space a few cells right of the waypoint (5, 2)...
    open_map = ['......?????'] * 11
    # ...and the same map with a solid wall in front of it.
    walled_map = ['.....#?????'] * 11

    gain_open = finder._estimate_gain(*build_grid(open_map), (5, 2), 1.0)
    gain_walled = finder._estimate_gain(*build_grid(walled_map), (5, 2), 1.0)

    assert gain_open > 0
    assert gain_walled == 0


def test_gain_does_not_leak_through_a_slanted_wall():
    # SLAM draws a 45 degree wall as cells touching corner to corner. A
    # diagonal ray could squeeze between two of them, so check it can't.
    finder = WaypointFinder(
        max_cluster_size=100,
        min_cluster_size=1,
        sensor_radius_cells=10,
    )
    data, width, height = build_grid([
        '.......#',
        '......#?',
        '.....#??',
        '....#???',
        '...#????',
        '..#?????',
        '.#??????',
        '#???????',
    ])

    gain = finder._estimate_gain(data, width, height, (2, 2), 1.0)

    assert gain == 0


# ---- grid -> world conversion -------------------------------------------

def test_grid_to_world_uses_cell_centre_and_origin():
    finder = WaypointFinder(
        max_cluster_size=100,
        min_cluster_size=1,
        sensor_radius_cells=3,
    )
    info = FakeMapInfo(width=10, height=10, resolution=0.5,
                       origin_x=1.0, origin_y=-2.0)

    # Cell (row=2, col=3):
    #   x = 1.0 + (3 + 0.5) * 0.5 =  2.75
    #   y = -2.0 + (2 + 0.5) * 0.5 = -0.75
    x, y = finder._grid_to_world(2, 3, info)

    assert abs(x - 2.75) < 1e-9
    assert abs(y - (-0.75)) < 1e-9


# ---- the decider --------------------------------------------------------

def make_waypoint(x, y, gain):
    # row/col/cluster_size are irrelevant to the decider; fill with dummies.
    return Waypoint(row=0, col=0, x=x, y=y, gain=gain, cluster_size=1)


def test_decider_prefers_higher_utility():
    robot_x, robot_y = 0.0, 0.0
    # A: lots of gain a bit far. B: little gain, close.
    a_wp = make_waypoint(x=10.0, y=0.0, gain=100)
    b_wp = make_waypoint(x=1.0, y=0.0, gain=10)

    decider = WaypointDecider(a=1.0, b=1.0)
    best, utility = decider.choose([a_wp, b_wp], robot_x, robot_y, blacklist=[])

    # U_A = 100 - 10 = 90 ; U_B = 10 - 1 = 9  ->  A wins.
    assert best is a_wp


def test_decider_skips_blacklisted_goals():
    robot_x, robot_y = 0.0, 0.0
    bad = make_waypoint(x=5.0, y=0.0, gain=100)   # high gain but blacklisted
    ok = make_waypoint(x=2.0, y=0.0, gain=10)

    decider = WaypointDecider(a=1.0, b=1.0, blacklist_radius=0.5)
    best, utility = decider.choose(
        [bad, ok], robot_x, robot_y, blacklist=[(5.0, 0.0)])

    assert best is ok


def test_decider_penalises_goals_on_top_of_the_robot():
    robot_x, robot_y = 0.0, 0.0
    # Close waypoint has more raw gain, but sits inside min_travel_dist,
    # so it is skipped in favour of the far one.
    close_wp = make_waypoint(x=0.1, y=0.0, gain=50)
    far_wp = make_waypoint(x=2.0, y=0.0, gain=10)

    decider = WaypointDecider(a=1.0, b=1.0, min_travel_dist=0.3)
    best, utility = decider.choose(
        [close_wp, far_wp], robot_x, robot_y, blacklist=[])

    assert best is far_wp


def test_decider_adds_bonus_to_marker_waypoints():
    # A marker view pose has no gain, but the marker bonus should still let
    # it beat a frontier with some gain: U_marker = 0 - 2 + 50 = 48 vs 10 - 2 = 8.
    frontier = make_waypoint(x=2.0, y=0.0, gain=10)
    marker = Waypoint(row=None, col=None, x=0.0, y=2.0, gain=0.0, cluster_size=0,
                      marker_id=42, marker_x=0.0, marker_y=2.5)

    decider = WaypointDecider(a=1.0, b=1.0, marker_bonus=50.0)
    best, utility = decider.choose([frontier, marker], 0.0, 0.0, blacklist=[])

    assert best is marker
    assert abs(utility - 48.0) < 1e-9


def test_decider_returns_none_when_no_candidates():
    decider = WaypointDecider(a=1.0, b=1.0)
    best, utility = decider.choose([], 0.0, 0.0, blacklist=[])

    assert best is None
    assert utility is None


def test_decider_returns_none_when_all_blacklisted():
    only = make_waypoint(x=3.0, y=0.0, gain=100)

    decider = WaypointDecider(a=1.0, b=1.0, blacklist_radius=0.5)
    best, utility = decider.choose(
        [only], 0.0, 0.0, blacklist=[(3.0, 0.0)])

    assert best is None
    assert utility is None


def test_score_keeps_skipped_candidates_with_no_utility():
    bad = make_waypoint(x=5.0, y=0.0, gain=100)     # blacklisted
    close = make_waypoint(x=0.1, y=0.0, gain=50)    # inside min_travel_dist
    ok = make_waypoint(x=2.0, y=0.0, gain=10)

    decider = WaypointDecider(
        a=1.0, b=1.0, blacklist_radius=0.5, min_travel_dist=0.3)
    scored = decider.score(
        [bad, close, ok], 0.0, 0.0, blacklist=[(5.0, 0.0)])

    # One entry per waypoint, same order; only ok is scored: U = 10 - 2 = 8.
    assert [s.waypoint for s in scored] == [bad, close, ok]
    assert scored[0].utility is None
    assert scored[1].utility is None
    assert abs(scored[2].utility - 8.0) < 1e-9
    assert abs(scored[2].dist - 2.0) < 1e-9


def test_ranked_drops_skipped_and_sorts_best_first():
    far = make_waypoint(x=10.0, y=0.0, gain=100)    # U = 100 - 10 = 90
    near = make_waypoint(x=1.0, y=0.0, gain=10)     # U = 10 - 1 = 9
    mid = make_waypoint(x=4.0, y=0.0, gain=30)      # U = 30 - 4 = 26
    bad = make_waypoint(x=6.0, y=0.0, gain=500)     # blacklisted

    decider = WaypointDecider(a=1.0, b=1.0, blacklist_radius=0.5)
    scored = decider.score(
        [near, bad, far, mid], 0.0, 0.0, blacklist=[(6.0, 0.0)])

    assert [s.waypoint for s in ranked(scored)] == [far, mid, near]


# ---- driving distance ---------------------------------------------------

def test_path_distance_goes_round_a_wall():
    # The goal is 4 cells straight below the robot, but a wall is in the way
    # and the only gap is at the far right, so the drive is much longer.
    data, width, height = build_grid([
        '.......',
        '.......',
        '######.',
        '.......',
        '.......',
    ])
    dist = path_distances(data, width, height, (0, 0))

    assert distance_to(dist, width, height, (4, 0), search=0) > 2 * 4.0


def test_path_distance_is_none_for_a_walled_off_goal():
    data, width, height = build_grid([
        '.....',
        '.....',
        '#####',
        '.....',
    ])
    dist = path_distances(data, width, height, (0, 0))

    assert distance_to(dist, width, height, (3, 2), search=0) is None
    assert distance_to(dist, width, height, (1, 4), search=0) is not None


def test_path_distance_cannot_cut_through_a_slanted_wall():
    # SLAM draws a slanted wall as cells touching at the corners; a diagonal
    # step must not squeeze between two of them.
    data, width, height = build_grid([
        '...#',
        '..#.',
        '.#..',
        '#...',
    ])
    dist = path_distances(data, width, height, (0, 0))

    assert distance_to(dist, width, height, (3, 3), search=0) is None


def test_decider_skips_candidates_it_cannot_drive_to():
    near = make_waypoint(x=1.0, y=0.0, gain=10)
    walled_off = make_waypoint(x=2.0, y=0.0, gain=100)

    decider = WaypointDecider(a=1.0, b=1.0)
    best, utility = decider.choose(
        [near, walled_off], 0.0, 0.0, blacklist=[], path_dists=[1.0, None])

    assert best is near


def test_decider_uses_driving_distance_not_straight_line():
    # Same gain, both 2 m away in a straight line, but the first is a 10 m
    # drive round a wall.
    behind_wall = make_waypoint(x=2.0, y=0.0, gain=10)
    open_side = make_waypoint(x=0.0, y=2.0, gain=10)

    decider = WaypointDecider(a=1.0, b=1.0)
    best, utility = decider.choose(
        [behind_wall, open_side], 0.0, 0.0, blacklist=[], path_dists=[10.0, 2.0])

    assert best is open_side


# ---- local first --------------------------------------------------------

def test_local_first_keeps_a_nearby_candidate_over_a_slightly_better_far_one():
    near = make_waypoint(x=2.0, y=0.0, gain=10)     # U = 10 - 2 = 8
    far = make_waypoint(x=4.0, y=0.0, gain=14)      # U = 14 - 4 = 10, but only 1.4x the gain

    decider = WaypointDecider(a=1.0, b=1.0, local_radius=3.0, far_ratio=1.5)
    best, utility = decider.choose([near, far], 0.0, 0.0, blacklist=[])

    assert best is near


def test_local_first_still_leaves_for_a_much_better_far_candidate():
    near = make_waypoint(x=2.0, y=0.0, gain=10)     # U = 10 - 2 = 8
    far = make_waypoint(x=8.0, y=0.0, gain=20)      # U = 20 - 8 = 12, and 2x the gain

    decider = WaypointDecider(a=1.0, b=1.0, local_radius=3.0, far_ratio=1.5)
    best, utility = decider.choose([near, far], 0.0, 0.0, blacklist=[])

    assert best is far


# ---- marker bonus -------------------------------------------------------

def test_marker_bonus_scales_with_a():
    # The bonus is in m^2 like gain, so at a = 8 a marker still beats a frontier
    # with 10 m^2 of gain: 8 * (0 + 50) - 2 = 398 vs 8 * 10 - 2 = 78.
    frontier = make_waypoint(x=2.0, y=0.0, gain=10)
    marker = Waypoint(row=None, col=None, x=0.0, y=2.0, gain=0.0, cluster_size=0,
                      marker_id=7, marker_x=0.0, marker_y=2.5)

    decider = WaypointDecider(a=8.0, b=1.0, marker_bonus=50.0)
    best, utility = decider.choose([frontier, marker], 0.0, 0.0, blacklist=[])

    assert best is marker
    assert abs(utility - 398.0) < 1e-9
