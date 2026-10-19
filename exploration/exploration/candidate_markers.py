"""
candidate_markers.py

Builds the RViz MarkerArray that shows why the explorer picked its goal.
Add a MarkerArray display on /exploration/candidates to see it.

    frontiers  : every frontier cell, small orange squares
    candidates : one sphere per waypoint, blue (lowest utility) to red (highest),
                 grey if it was skipped (blacklisted or too close to the robot)
    labels     : rank, utility, gain and distance above the best few
    chosen     : a larger see-through green sphere on the goal that was sent

It is redrawn from scratch every planning cycle, so it always shows the
decision behind the goal the robot is currently driving to.
"""

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def candidate_markers(scored, top, frontier_cells, map_info, n_labels=5, frame_id='map'):
    """
    scored         : list[Scored] from WaypointDecider.score()
    top            : ranked(scored), best first
    frontier_cells : list of (row, col) from find_frontiers()
    map_info       : msg.info of the /map the frontiers came from
    returns        : MarkerArray
    """
    out = MarkerArray()

    # Wipe last cycle's markers first, so candidates that are gone disappear.
    clear = Marker()
    clear.header.frame_id = frame_id
    clear.action = Marker.DELETEALL
    out.markers.append(clear)

    # ---- frontier cells ----
    res = map_info.resolution
    origin_x = map_info.origin.position.x
    origin_y = map_info.origin.position.y
    frontiers = _marker(frame_id, 'frontiers', 0, Marker.POINTS, res, (1.0, 0.5, 0.0))
    for row, col in frontier_cells:
        frontiers.points.append(Point(
            x=origin_x + (col + 0.5) * res,
            y=origin_y + (row + 0.5) * res))
    out.markers.append(frontiers)

    # ---- one sphere per candidate, coloured by utility ----
    if top:
        u_best = top[0].utility
        u_worst = top[-1].utility
    for i, s in enumerate(scored):
        if s.utility is None:
            rgb = (0.5, 0.5, 0.5)                   # skipped
        else:
            if u_best == u_worst:
                t = 1.0
            else:
                t = (s.utility - u_worst) / (u_best - u_worst)
            rgb = (t, 0.0, 1.0 - t)                 # blue = worst, red = best
        out.markers.append(_marker_at(
            frame_id, 'candidates', i, Marker.SPHERE, 0.15, rgb, s.waypoint))

    # ---- labels on the best few ----
    for rank, s in enumerate(top[:n_labels], start=1):
        label = _marker_at(
            frame_id, 'labels', rank, Marker.TEXT_VIEW_FACING, 0.15, (0.0, 0.0, 0.0),
            s.waypoint)
        label.pose.position.z = 0.3                 # float above the sphere
        label.text = (f'#{rank} U={s.utility:.1f}\n'
                      f'gain={s.waypoint.gain:.1f} dist={s.dist:.1f}')
        out.markers.append(label)

    # ---- the goal that was sent ----
    if top:
        chosen = _marker_at(
            frame_id, 'chosen', 0, Marker.SPHERE, 0.35, (0.0, 1.0, 0.0), top[0].waypoint)
        chosen.color.a = 0.5                        # see the red sphere inside it
        out.markers.append(chosen)

    return out


# ---- helpers ------------------------------------------------------------

def _marker(frame_id, ns, marker_id, marker_type, size, rgb):
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = marker_id
    m.type = marker_type
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    # spheres use all three; points use x, y as the square size; text uses z as height
    m.scale.x = size
    m.scale.y = size
    m.scale.z = size
    m.color.r, m.color.g, m.color.b = rgb
    m.color.a = 1.0
    return m


def _marker_at(frame_id, ns, marker_id, marker_type, size, rgb, waypoint):
    m = _marker(frame_id, ns, marker_id, marker_type, size, rgb)
    m.pose.position.x = waypoint.x
    m.pose.position.y = waypoint.y
    return m
