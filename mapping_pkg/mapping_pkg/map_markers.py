"""
map_markers.py

Places the ArUco markers seen by camera_node on the map and publishes them
for RViz.

For each detection it looks up where the camera was when the image was taken
(tf2, using the image's timestamp), transforms the marker's position from the
camera frame into the map frame, and averages it with the earlier sightings
of the same marker to reduce noise. The averaged positions are published as
spheres on /aruco/map_markers and written to a CSV file every 5 s.

Based on the ROS 2 Humble tf2 tutorials, using a PointStamped for the marker:
https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Adding-A-Frame-Py.html
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

# detections from camera_node
from mapping_interfaces.msg import MarkerDetectionArray

# tf2
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

# RViz display
from visualization_msgs.msg import Marker, MarkerArray


class MapMarkersNode(Node):

    def __init__(self):
        super().__init__('map_markers_node')

        # ---- ROS parameters (values come from team10_bringup/config/params.yaml) ----
        # needs SLAM running; to test without SLAM set map_frame to odom there
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        # sighting filters, read each time they are used so `ros2 param set` works live
        self.declare_parameter('max_range_m', 2.0)        # sightings further away are too noisy
        self.declare_parameter('outlier_min_sightings', 5)
        self.declare_parameter('outlier_dist_m', 1.0)
        # write the marker table to a CSV every few seconds so we have numbers to check
        self.declare_parameter('output_file', '/tmp/aruco_markers.csv')

        # marker id -> averaged position in the map
        self.detected_markers = {}
        # marker id -> how many detections have been averaged
        self.detection_counts = {}

        # the buffer stores transforms and the listener fills it
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # camera_node's output
        self.create_subscription(
            MarkerDetectionArray, '/aruco/detections', self.get_detected_markers, 10)

        # for RViz: add a MarkerArray display on this topic
        self.rviz_pub = self.create_publisher(MarkerArray, '/aruco/map_markers', 10)

        self.create_timer(5.0, self.write_report)

    # ---- detections -> map positions ----------------------------------------

    def get_detected_markers(self, msg):
        from_frame = msg.header.frame_id
        to_frame = self.map_frame
        frame_time = Time.from_msg(msg.header.stamp)

        # The error grows with range, so keep the close sightings and skip the far ones.
        max_range = self.get_parameter('max_range_m').value
        in_range = [m for m in msg.markers if m.range_m <= max_range]
        if not in_range:
            return

        try:
            # where the camera was when the image was taken
            transform = self.tf_buffer.lookup_transform_full(
                target_frame=to_frame,           # map ...
                target_time=Time(),              # ... using the latest SLAM correction
                source_frame=from_frame,         # camera ...
                source_time=frame_time,          # ... where it was when the image was taken
                fixed_frame='odom')
        except TransformException as ex:
            # at most once every 2 s, so it doesn't flood the terminal
            self.get_logger().warn(
                f'Could not transform {from_frame} to {to_frame}: {ex}',
                throttle_duration_sec=2.0)
            return

        for mkr in in_range:
            camera_point = PointStamped()
            camera_point.header = msg.header
            camera_point.point = mkr.position

            # camera frame -> map frame
            p = do_transform_point(camera_point, transform).point

            self.update_markers(mkr.id, np.array([p.x, p.y, p.z]))

        self.place_marker()

    def update_markers(self, marker_id, position):
        """
        Add a new sighting to a marker's position, as a running mean of all its
        sightings so far.
        """
        n = self.detection_counts.get(marker_id, 0)

        # once a marker has been seen enough times, sightings far from its
        # current estimate are treated as outliers and ignored
        if n >= self.get_parameter('outlier_min_sightings').value:
            dist = float(np.linalg.norm(position[:2] - self.detected_markers[marker_id][:2]))
            if dist > self.get_parameter('outlier_dist_m').value:
                self.get_logger().warn(
                    f'marker {marker_id}: ignoring outlier {dist:.1f} m from estimate',
                    throttle_duration_sec=2.0)
                return

        if n == 0:
            self.detected_markers[marker_id] = position
        else:
            # running mean
            old = self.detected_markers[marker_id]
            self.detected_markers[marker_id] = old + (position - old) / (n + 1)
        self.detection_counts[marker_id] = n + 1

        p = self.detected_markers[marker_id]
        # log marker changes to the terminal; the throttle prevents flooding
        self.get_logger().info(
            f'marker {marker_id}: x={p[0]:.2f} y={p[1]:.2f} (map) '
            f'n={self.detection_counts[marker_id]}',
            throttle_duration_sec=2.0)

    def write_report(self):
        """Write the marker positions to a CSV file, to check the results against."""
        path = self.get_parameter('output_file').value
        with open(path, 'w') as f:
            f.write('id,x,y\n')
            for marker_id, p in sorted(self.detected_markers.items()):
                f.write(f'{marker_id},{p[0]:.3f},{p[1]:.3f}\n')

    # ---- RViz ---------------------------------------------------------------

    def place_marker(self):
        """Publish a green sphere for every marker so RViz can show them on the map."""
        out = MarkerArray()

        for marker_id, position in self.detected_markers.items():
            mkr = Marker()
            mkr.header.frame_id = self.map_frame
            mkr.id = marker_id
            mkr.type = Marker.SPHERE
            mkr.pose.position.x = float(position[0])
            mkr.pose.position.y = float(position[1])
            mkr.pose.position.z = 0.0
            mkr.scale.x = 0.1
            mkr.scale.y = 0.1
            mkr.scale.z = 0.1
            mkr.color.r = 0.0
            mkr.color.g = 1.0
            mkr.color.b = 0.0
            mkr.color.a = 1.0
            out.markers.append(mkr)

        self.rviz_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MapMarkersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
