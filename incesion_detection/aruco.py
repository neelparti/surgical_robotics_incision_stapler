import cv2
import numpy as np
from picamera2 import Picamera2

def detect_aruco_marker(frame, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    """
    Detects an ArUco marker in the given frame.

    Parameters:
        frame (numpy.ndarray): The input image from the camera.
        aruco_dict_type (int): The type of ArUco dictionary to use. Default is DICT_4X4_50.

    Returns:
        tuple: (ids, corners, frame_with_markers)
        - ids (numpy.ndarray or None): Detected marker IDs.
        - corners (list or None): Detected marker corner points.
        - frame_with_markers (numpy.ndarray): The frame with detected markers drawn.
    """
    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()

    # Convert to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Draw detected markers on the frame
    frame_with_markers = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_with_markers, corners, ids)

    return ids, corners, frame_with_markers


# Initialize the Raspberry Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  # Adjust resolution if needed
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    # Capture a frame from the Pi camera
    frame = picam2.capture_array()

    # Detect ArUco markers
    ids, corners, frame_with_markers = detect_aruco_marker(frame)

    if ids is not None:
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            print(f"Marker {marker_id} detected at corners: {corner}")

    # Display the frame with detected markers
    cv2.imshow("ArUco Detection", frame_with_markers)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()