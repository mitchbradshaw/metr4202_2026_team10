"""ArUco detection and single-marker pose estimation.

Pure Python: only numpy and cv2. No ROS imports, so this file can be
imported and tested on any machine with `pip install numpy opencv-contrib-python`.

Verified facts about the METR4202 world (metr4202_aruco_explore @ 4e5e9b0):
  * Marker textures are named 6x6_1000-{0,2,42}.png -> DICT_6X6_1000, IDs 0, 2, 42.
  * The marker face is a 0.1 x 0.1 m box and the black square fills the whole
    texture (no white margin), so the black-square side length is 0.1 m.
"""

from dataclasses import dataclass

import cv2
import numpy as np


DEFAULT_DICTIONARY = 'DICT_6X6_1000'
DEFAULT_MARKER_SIZE_M = 0.1


@dataclass
class Detection:
    marker_id: int
    corners: np.ndarray   # shape (4, 2), pixel coords, order TL, TR, BR, BL
    rvec: np.ndarray      # shape (3,), Rodrigues rotation, marker -> camera optical frame
    tvec: np.ndarray      # shape (3,), marker centre in camera optical frame, metres
    range_m: float        # Euclidean distance camera centre -> marker centre, metres
    pixel_area: float     # marker outline area in the image, px^2 (hints at viewing angle/range)


# --------------------------------------------------------------------------
# OpenCV version handling.
#
# The cv2.aruco API changed in OpenCV 4.7:
#   < 4.7 : free function cv2.aruco.detectMarkers(img, dict, parameters=...)
#           and cv2.aruco.DetectorParameters_create()
#   >= 4.7: cv2.aruco.ArucoDetector(dict, params).detectMarkers(img)
#           and cv2.aruco.DetectorParameters()
#
# Ubuntu 22.04's apt python3-opencv (what cv_bridge uses) is 4.5.x, while a
# fresh `pip install opencv-contrib-python` on Windows is >= 4.7. This is the
# ONLY place in the file that cares which one is installed.
# --------------------------------------------------------------------------
_HAS_NEW_ARUCO_API = hasattr(cv2.aruco, 'ArucoDetector')


class _ArucoDetectorCompat:
    """Wraps whichever detection API the installed OpenCV provides."""

    def __init__(self, dictionary_name):
        dictionary_id = getattr(cv2.aruco, dictionary_name)  # e.g. cv2.aruco.DICT_6X6_1000
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

        if _HAS_NEW_ARUCO_API:
            self.params = cv2.aruco.DetectorParameters()
        else:
            self.params = cv2.aruco.DetectorParameters_create()

        # Sub-pixel corner refinement. Corner accuracy feeds straight into
        # solvePnP, and at 3 m a 0.1 m marker is only ~18 px wide in the
        # 640x480 Gazebo camera, so every fraction of a pixel matters.
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        if _HAS_NEW_ARUCO_API:
            self._detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, grey):
        """Returns (corners, ids) exactly as OpenCV does: ids is None if nothing found."""
        if _HAS_NEW_ARUCO_API:
            corners, ids, _rejected = self._detector.detectMarkers(grey)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                grey, self.dictionary, parameters=self.params)
        return corners, ids


# Building a detector is cheap but not free; cache one per dictionary name.
_detector_cache = {}


def _get_detector(dictionary_name):
    if dictionary_name not in _detector_cache:
        _detector_cache[dictionary_name] = _ArucoDetectorCompat(dictionary_name)
    return _detector_cache[dictionary_name]


def marker_object_points(marker_size_m):
    """3D corner coordinates of a square marker in its own frame, metres.

    ORDER MATTERS. cv2.aruco returns each marker's image corners clockwise
    starting from the marker's own top-left: TL, TR, BR, BL. The object points
    must be listed in the same order so that point i in the image corresponds
    to point i here.

    Marker frame: origin at the marker centre, x to the right, y up, z out of
    the marker face towards the viewer (z = 0 on the marker plane). This exact
    layout and ordering is also what SOLVEPNP_IPPE_SQUARE requires.
    """
    half = marker_size_m / 2.0
    return np.array([
        [-half,  half, 0.0],   # top-left
        [ half,  half, 0.0],   # top-right
        [ half, -half, 0.0],   # bottom-right
        [-half, -half, 0.0],   # bottom-left
    ], dtype=np.float64)


def detect_markers(image_bgr, K, dist_coeffs, marker_size_m=DEFAULT_MARKER_SIZE_M,
                   dictionary_name=DEFAULT_DICTIONARY):
    """Detect ArUco markers and estimate each one's pose relative to the camera.

    Args:
        image_bgr: HxWx3 uint8 BGR image (a HxW greyscale image is also accepted).
        K: 3x3 camera intrinsic matrix.
        dist_coeffs: distortion coefficients (any length OpenCV accepts; may be empty).
        marker_size_m: side length of the black square, metres.
        dictionary_name: name of a cv2.aruco predefined dictionary.

    Returns:
        list[Detection]; empty list if the image is empty or no markers are found.

    The returned rvec/tvec are in the CAMERA OPTICAL FRAME (OpenCV convention):
        z forward (out of the lens), x right, y down in the image.
    This is NOT the ROS body convention (x forward, y left, z up). In ROS terms
    these numbers are expressed in the camera's *_optical_frame
    (camera_rgb_optical_frame on the Waffle Pi), not camera_link.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    if image_bgr.ndim == 3:
        grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        grey = image_bgr

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = _as_dist_array(dist_coeffs)

    detector = _get_detector(dictionary_name)
    corners_list, ids = detector.detect(grey)
    if ids is None or len(ids) == 0:
        return []

    object_points = marker_object_points(marker_size_m)
    detections = []

    for marker_corners, marker_id in zip(corners_list, ids.flatten()):
        image_points = marker_corners.reshape(4, 2).astype(np.float64)

        # IPPE_SQUARE is the solver designed for exactly this case: 4 coplanar
        # points forming a square, in the order above. Like any planar PnP it
        # can have a two-fold orientation ambiguity for small/distant markers;
        # that mostly affects rvec, the translation (and so range) is stable.
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue

        rvec = rvec.reshape(3)
        tvec = tvec.reshape(3)
        detections.append(Detection(
            marker_id=int(marker_id),
            corners=image_points,
            rvec=rvec,
            tvec=tvec,
            range_m=float(np.linalg.norm(tvec)),
            pixel_area=float(cv2.contourArea(image_points.astype(np.float32))),
        ))

    return detections


def draw_detections(image_bgr, detections, K, dist_coeffs, marker_size_m=DEFAULT_MARKER_SIZE_M):
    """Return a copy of the image with marker outlines, axes, ID and range drawn on.

    Axes colours from drawFrameAxes: x red, y green, z blue (z points out of the marker).
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    annotated = image_bgr.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    if not detections:
        return annotated

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = _as_dist_array(dist_coeffs)

    # drawDetectedMarkers expects the same shapes detectMarkers produced:
    # a list of (1, 4, 2) float32 arrays and an (N, 1) int array of ids.
    corners_for_draw = [d.corners.reshape(1, 4, 2).astype(np.float32) for d in detections]
    ids_for_draw = np.array([[d.marker_id] for d in detections], dtype=np.int32)
    cv2.aruco.drawDetectedMarkers(annotated, corners_for_draw, ids_for_draw)

    for d in detections:
        cv2.drawFrameAxes(annotated, K, dist, d.rvec, d.tvec, marker_size_m * 0.5)

        # Put the label just above the marker's top-most corner.
        x = int(d.corners[:, 0].min())
        y = int(d.corners[:, 1].min()) - 8
        y = max(y, 15)
        label = f'id {d.marker_id}  {d.range_m:.2f} m'
        cv2.putText(annotated, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)       # dark outline
        cv2.putText(annotated, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)   # yellow text

    return annotated


def _as_dist_array(dist_coeffs):
    """CameraInfo.d can legitimately be empty; OpenCV wants an array either way."""
    if dist_coeffs is None:
        return np.zeros(5, dtype=np.float64)
    dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if dist.size == 0:
        return np.zeros(5, dtype=np.float64)
    return dist
