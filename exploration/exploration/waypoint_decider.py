"""
waypoint_decider.py

Scores the candidate waypoints from WaypointFinder and returns the best one.

Utility per candidate:
    U = a * gain  -  b * cost  +  bonus

    gain      : from the finder (unknown cells reachable from the waypoint)
    cost      : straight-line distance from the robot to the waypoint [m]
    bonus     : bonus gain for markers

The class is pure logic. The owning node supplies the robot pose (from tf2)
and the current blacklist every cycle.

score() keeps every candidate's numbers, not just the winner's, so the node
can log them and draw them in RViz; ranked() sorts them best first.
"""

import math


class Scored:
    """One candidate's score from this planning cycle."""

    def __init__(self, waypoint, dist, utility):
        self.waypoint = waypoint      # the Waypoint from WaypointFinder
        self.dist = dist              # robot -> waypoint [m]
        self.utility = utility        # None if the candidate was skipped


def ranked(scored):
    """The candidates that were not skipped, highest utility first."""
    usable = [s for s in scored if s.utility is not None]
    # sorted() is stable, so equal utilities keep their original order
    return sorted(usable, key=lambda s: s.utility, reverse=True)


class WaypointDecider:

    def __init__(self,
                 a,
                 b,
                 blacklist_radius=0.5,
                 min_travel_dist=0.5,
                 marker_bonus=50.0):
        self.a = a                                # weight on information gain
        self.b = b                                # weight on travel cost
        self.blacklist_radius = blacklist_radius  # [m] how near a failed goal counts as "same"
        self.min_travel_dist = min_travel_dist    # [m] candidates closer than this are skipped
        self.marker_bonus = marker_bonus          # bonus for markers

    # ---- public entry point -------------------------------------------------

    def choose(self, waypoints, robot_x, robot_y, blacklist):
        """
        waypoints : list[Waypoint] from WaypointFinder
        robot_x/y : robot position in the map frame [m] (from tf2)
        blacklist : list of (x, y) world points that previously failed
        returns   : (best_waypoint, best_utility), or (None, None) if none are valid
        """
        top = ranked(self.score(waypoints, robot_x, robot_y, blacklist))
        if not top:
            return None, None
        return top[0].waypoint, top[0].utility

    def score(self, waypoints, robot_x, robot_y, blacklist):
        """
        Same arguments as choose().
        returns : list[Scored], one per waypoint in the same order
        """
        scored = []

        for wp in waypoints:
            cost = self._distance(robot_x, robot_y, wp.x, wp.y)

            # Hard skips: a goal we already failed to reach, or one too close.
            if self._is_blacklisted(wp, blacklist) or cost < self.min_travel_dist:
                scored.append(Scored(wp, cost, None))
                continue

            bonus = 0.0
            if wp.marker_id is not None:
                bonus = self.marker_bonus

            utility = self.a * wp.gain - self.b * cost + bonus
            scored.append(Scored(wp, cost, utility))

        return scored

    # ---- helpers ------------------------------------------------------------

    def _distance(self, x1, y1, x2, y2):
        d_x = x2 - x1
        d_y = y2 - y1
        return math.sqrt(d_x * d_x + d_y * d_y)

    def _is_blacklisted(self, wp, blacklist):
        for (b_x, b_y) in blacklist:
            if self._distance(wp.x, wp.y, b_x, b_y) < self.blacklist_radius:
                return True
        return False
