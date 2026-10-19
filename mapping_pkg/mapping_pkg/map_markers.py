"""
A node object that takes the detections from the camera node and publishes them as markers for RViz.

It should be run after the camera node, and it will subscribe to the camera node's output topic. 
It will then publish the markers to a topic that RViz can subscribe to.

It will update the markers in real time as the camera node detects new markers, and average frames over time to reduce noise in the marker positions.

This node should find the pose at time that the detection frame was taken, the gets the markers position relative to that. 
Then add them to get markers position on map

I've been going along with these
https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Adding-A-Frame-Py.html (i'm using PointStamp tho)
"""

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
 
#detection from camera node
from mapping_interfaces.msg import MarkerDetectionArray
 
#tf2 imports
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
 
#rviz display
from visualization_msgs.msg import Marker, MarkerArray


class MapMarkersNode(Node):
 
    def __init__(self):
        super().__init__('map_markers_node')
 
        # ---- ROS Parameter ----------------------------------------------
        # values come from team10_bringup/config/params.yaml
        #needs SLAM running; to test without SLAM set map_frame to odom there
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        # sighting filters, read each time they are used so `ros2 param set` works live
        self.declare_parameter('max_range_m', 2.0)        # sightings further away are too noisy
        self.declare_parameter('outlier_min_sightings', 5)
        self.declare_parameter('outlier_dist_m', 1.0)
 
        # Initialize marker storage: marker id -> position in the map
        self.detected_markers = {}
        # marker id -> how many detections have been averaged
        self.detection_counts = {}
 
        #Transform stuff, initialises buffer for transform storage and the listener fills it
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
 
        #subscriber for the camera node output, queue of 10
        self.create_subscription(MarkerDetectionArray, '/aruco/detections', self.get_detected_markers, 10)
 
        #publisher for RViz, add a MarkerArray display on this topic
        self.rviz_pub = self.create_publisher(MarkerArray, '/aruco/map_markers', 10)
        # write the marker table to a csv every few seconds so we have numbers to check
        self.declare_parameter('output_file','/tmp/aruco_markers.csv')
        self.create_timer(5.0, self.write_report)

 
    #assigns map positions to aruco marker
    def get_detected_markers(self, msg):
        from_frame = msg.header.frame_id
        to_frame = self.map_frame
        frame_time = Time.from_msg(msg.header.stamp)
 
        #Checks range of marker detection, bigger will have larger error.
        #Keeps the close markers and skips the far ones.
        max_range = self.get_parameter('max_range_m').value
        in_range = [m for m in msg.markers if m.range_m <= max_range]
        if not in_range:
            return
 
        try:
            # this gets the position of the camera when detection was made
                        transform = self.tf_buffer.lookup_transform_full(
                        target_frame=to_frame,           # map ...
                        target_time=Time(),              # ... using the latest SLAM correction
                        source_frame=from_frame,         # camera ...
                        source_time=frame_time,          # ... where it was when the image was taken
                        fixed_frame='odom')
        # just error message, at most once every 2 s so it doesn't flood the terminal
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform {from_frame} to {to_frame} : {ex}')
            return
 
        for mkr in in_range:
 
            cameraPoint = PointStamped()
 
            cameraPoint.header = msg.header
            cameraPoint.point = mkr.position
 
            # camera to map transform
            mapPoint_stamped = do_transform_point(cameraPoint, transform)
 
            # x, y and z
            p = mapPoint_stamped.point
 
            self.update_markers(mkr.id, np.array([p.x, p.y, p.z]))
 
        self.place_marker()
 
    def update_markers(self, marker_id, position):
        """
        Update the detected markers with new positions.
        Average all detections with a running mean.
        """
        n = self.detection_counts.get(marker_id, 0)

        # markers seen enough times have worse estimates removed
        if n >= self.get_parameter('outlier_min_sightings').value:
             dist = float(np.linalg.norm(position[:2] - self.detected_markers[marker_id][:2]))
             # ignore a sighting too far from the current estimate
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
             self.detected_markers[marker_id] = old + (position - old) / (n+1)
        self.detection_counts[marker_id] = n + 1

        p = self.detected_markers[marker_id]
        # log marker changes to terminal, throttle prevents flooding
        self.get_logger().info(
             f'marker {marker_id}: x={p[0]:.2f} y={p[1]:.2f} (map) n={self.detection_counts[marker_id]}', 
             throttle_duration_sec=2.0)

    def write_report(self):
         """
         Write marker locations to csv file for verification of results
         """
         path = self.get_parameter('output_file').value
         with open(path, 'w') as f:
              f.write('id,x,y\n')
              for marker_id, p in sorted(self.detected_markers.items()):
                   f.write(f'{marker_id},{p[0]:.3f},{p[1]:.3f}\n')
 
    #publishes a green sphere for every marker so RViz can show them on the map
    def place_marker(self):
 
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