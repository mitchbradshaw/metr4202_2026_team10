"""Thin ROS 2 wrapper around aruco_core.

Jobs: subscribe, convert messages <-> NumPy with cv_bridge, call aruco_core,
publish the debug image, log. No detection maths belongs in this file.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from mapping_interfaces.msg import MarkerDetection, MarkerDetectionArray 
from mapping_pkg import aruco_core


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # ---- Parameters ------------------------------------------------- 
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('debug_image_topic', '/aruco/debug_image')
        self.declare_parameter('detections_topic', '/aruco/detections')
        self.declare_parameter('marker_size', aruco_core.DEFAULT_MARKER_SIZE_M)
        self.declare_parameter('aruco_dictionary', aruco_core.DEFAULT_DICTIONARY)
        self.declare_parameter('image_qos_depth', 5)

        # Frame the detections are published in - empty = the image's own frame_id.
        # Set it when the camera driver stamps images with a body frame (x forward,
        # z up) instead of the optical frame the pose maths is in.
        self.declare_parameter('camera_frame', '')

        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        debug_image_topic = self.get_parameter('debug_image_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        self.marker_size = float(self.get_parameter('marker_size').value)
        self.dictionary_name = self.get_parameter('aruco_dictionary').value
        image_qos_depth = self.get_parameter('image_qos_depth').value
        self.camera_frame = self.get_parameter('camera_frame').value

        self.bridge = CvBridge()

        # Filled in by the first valid CameraInfo message.
        self.K = None
        self.dist = None

        # ---- Subscribers --------------------------------------------------------
        # qos_profile_sensor_data = BEST_EFFORT reliability, KEEP_LAST depth 5.
        self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)

        # Images: same BEST_EFFORT as above, but the queue depth comes from params.yaml.
        image_qos = QoSProfile(depth=image_qos_depth, reliability=ReliabilityPolicy.BEST_EFFORT)
        if self.use_compressed:
            self.create_subscription(
                CompressedImage, image_topic, self.image_callback, image_qos)
        else:
            self.create_subscription(
                Image, image_topic, self.image_callback, image_qos)

        # ---- Publishers --------------------------------------------------------
        self.detections_pub = self.create_publisher(MarkerDetectionArray, detections_topic, 10)

        # Publisher is RELIABLE (default), depth 1
        self.debug_pub = self.create_publisher(Image, debug_image_topic, 1)

        self.get_logger().info(
            f'Listening: image={image_topic} (compressed={self.use_compressed}, '
            f'queue depth {image_qos_depth}), '
            f'camera_info={camera_info_topic}; marker_size={self.marker_size} m, '
            f'dictionary={self.dictionary_name}; publishing {debug_image_topic}; '
            f'detections in frame {self.camera_frame or "<from image header>"}')

    def camera_info_callback(self, msg):
        if self.K is not None:
            return  # Intrinsics don't change - cache the first valid message only.

        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            # An uncalibrated CameraInfo would make solvePnP garbage.
            self.get_logger().warn(
                'CameraInfo has non-positive focal length; ignoring it.',
                throttle_duration_sec=5.0)
            return

        self.K = K
        self.dist = np.array(msg.d, dtype=np.float64)  # may be empty, aruco_core handles that
        self.get_logger().info(
            f'Cached intrinsics from CameraInfo ({msg.width}x{msg.height}, '
            f'frame {msg.header.frame_id}): fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} '
            f'cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}, dist={list(self.dist)}')

    def image_callback(self, msg):
        if self.K is None:
            # Throttling uses wall-clock time, this rate-limits
            self.get_logger().warn(
                'No CameraInfo received yet; skipping image.', throttle_duration_sec=2.0)
            return

        # ---- message -> array
        try:
            if self.use_compressed:
                image_bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc: 
            self.get_logger().error(f'cv_bridge conversion failed: {exc}',
                                    throttle_duration_sec=2.0)
            return

        # ---- algorithm (maths in aruco_core)
        detections = aruco_core.detect_markers(
            image_bgr, self.K, self.dist, self.marker_size, self.dictionary_name)

        # ---- which frame the detections are in
        # tvec is always in OpenCV's optical convention (z forward, x right, y down).
        # Gazebo stamps images with camera_rgb_optical_frame, which matches; a real
        # camera driver may stamp a body frame, which would put the range on the
        # vertical axis and drop every marker next to the robot. This also relabels
        # the debug image below, which is right: an image belongs in the optical frame.
        if self.camera_frame:
            msg.header.frame_id = self.camera_frame
        elif 'optical' not in msg.header.frame_id:
            self.get_logger().warn(
                f"Image frame '{msg.header.frame_id}' is not an optical frame, so markers "
                f'will be misplaced; set camera_frame in the config.',
                throttle_duration_sec=30.0)

        # ---- detections -> published for each frame
        self.detections_pub.publish(self.to_detection_array(msg.header, detections))

        annotated = aruco_core.draw_detections(
            image_bgr, detections, self.K, self.dist, self.marker_size)

        # ---- array -> message
        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

        # ---- logging
        # tvec is in the camera OPTICAL frame: z forward, x right, y down.
        if detections:
            parts = [
                f'id {d.marker_id}: {d.range_m:.2f} m '
                f'(optical x={d.tvec[0]:+.2f} y={d.tvec[1]:+.2f} z={d.tvec[2]:+.2f})'
                for d in detections
            ]
            self.get_logger().info('Detected ' + '; '.join(parts), throttle_duration_sec=1.0)

    @staticmethod
    def to_detection_array(header, detections):
        """list[aruco_core.Detection] -> MarkerDetectionArray.

        header is the source image's header: its stamp is the capture time and
        its frame_id is the optical frame the poses are expressed in. Map node can use this for positioning
        """
        out = MarkerDetectionArray()
        out.header = header
        for d in detections:
            m = MarkerDetection()
            m.id = d.marker_id
            m.position.x = float(d.tvec[0])
            m.position.y = float(d.tvec[1])
            m.position.z = float(d.tvec[2])
            m.range_m = d.range_m
            m.pixel_area = d.pixel_area
            out.markers.append(m)
        return out



def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
