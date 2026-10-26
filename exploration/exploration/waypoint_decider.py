"""
waypoint_decider.py

Scores the candidate waypoints from WaypointFinder and returns the best one.

Utility per candidate:
    U = a * (gain + bonus)  -  b * cost

    gain      : from the finder (unknown area the lidar could see from the waypoint)
    bonus     : marker_bonus for a marker view pose, 0 otherwise. It is in the
                same m^2 units as gain, so changing a doesn't change how
                markers rank against frontiers.
    cost      : driving distance from the robot to the waypoint [m] (straight
                line if the node doesn't pass path distances)

Local first: the best candidate within local_radius of the robot wins, unless
a further one offers far_ratio times as much (a * (gain + bonus)). This keeps
the robot finishing the area it is in instead of crossing the map for a
slightly bigger frontier.

The class is pure logic. The owning node supplies the robot pose (from tf2),
the current blacklist and the path distances every cycle.

score() keeps every candidate's numbers, not just the winner's, so the node
can log them and draw them in RViz; ranked() sorts them best first.
"""

import math


class Scored:
    """One candidate's score from this planning cycle."""

    def __init__(self, waypoint, dist, utility):
        self.waypoint = waypoint      # the Waypoint from WaypointFinder
        self.dist = dist              # robot -> waypoint [m], None if it can't be reached
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
                 marker_bonus=50.0,
                 local_radius=0.0,
                 far_ratio=1.5):
        self.a = a                                # weight on information gain
        self.b = b                                # weight on travel cost
        self.blacklist_radius = blacklist_radius  # [m] how near a failed goal counts as "same"
        self.min_travel_dist = min_travel_dist    # [m] candidates closer than this are skipped
        self.marker_bonus = marker_bonus          # [m^2] gain a marker view pose counts as
        self.local_radius = local_radius          # [m] "local" candidates; 0 turns local-first off
        self.far_ratio = far_ratio                # a far candidate must offer this many times more

    # ---- public entry point -------------------------------------------------

    def choose(self, waypoints, robot_x, robot_y, blacklist, path_dists=None):
        """
        waypoints  : list[Waypoint] from WaypointFinder
        robot_x/y  : robot position in the map frame [m] (from tf2)
        blacklist  : list of (x, y) world points that previously failed
        path_dists : optional list of driving distances [m], one per waypoint,
                     None for a waypoint the robot can't reach
        returns    : (best_waypoint, best_utility), or (None, None) if none are valid
        """
        top = self.prefer_local(
            ranked(self.score(waypoints, robot_x, robot_y, blacklist, path_dists)))
        if not top:
            return None, None
        return top[0].waypoint, top[0].utility

    def score(self, waypoints, robot_x, robot_y, blacklist, path_dists=None):
        """
        Same arguments as choose().
        returns : list[Scored], one per waypoint in the same order
        """
        scored = []

        for i, wp in enumerate(waypoints):
            if path_dists is None:
                cost = self._distance(robot_x, robot_y, wp.x, wp.y)
            else:
                cost = path_dists[i]

            # Hard skips: can't drive there, already failed there, or too close.
            if cost is None:
                scored.append(Scored(wp, None, None))
                continue
            if self._is_blacklisted(wp, blacklist) or cost < self.min_travel_dist:
                scored.append(Scored(wp, cost, None))
                continue

            bonus = 0.0
            if wp.marker_id is not None:
                bonus = self.marker_bonus

            utility = self.a * (wp.gain + bonus) - self.b * cost
            scored.append(Scored(wp, cost, utility))

        return scored

    def prefer_local(self, top):
        """
        top     : ranked() list, highest utility first
        returns : the same candidates with the chosen one moved to the front:
                  the best local one, unless the best overall is further away
                  and offers at least far_ratio times as much
        """
        if not top or self.local_radius <= 0.0:
            return top

        local = [s for s in top if s.dist <= self.local_radius]
        if not local or local[0] is top[0]:
            return top                  # nothing nearby, or the best is nearby anyway

        best_far = top[0]
        best_local = local[0]
        if self._offer(best_far) >= self.far_ratio * self._offer(best_local):
            return top                  # worth the trip

        return [best_local] + [s for s in top if s is not best_local]

    # ---- helpers ------------------------------------------------------------

    def _offer(self, s):
        """What a candidate offers before the distance cost: a * (gain + bonus)."""
        return s.utility + self.b * s.dist

    def _distance(self, x1, y1, x2, y2):
        d_x = x2 - x1
        d_y = y2 - y1
        return math.sqrt(d_x * d_x + d_y * d_y)

    def _is_blacklisted(self, wp, blacklist):
        for (b_x, b_y) in blacklist:
            if self._distance(wp.x, wp.y, b_x, b_y) < self.blacklist_radius:
                return True
        return False
