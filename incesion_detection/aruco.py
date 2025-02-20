import cv2
import numpy as np
from picamera2 import Picamera2

REAL_WIDTH = 5.0        # cm
FOCAL_LENGTH = 2957     # pixels

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
    aruco_params = cv2.aruco.DetectorParameters_create()

    # Convert to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Draw detected markers on the frame
    frame_with_markers = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_with_markers, corners, ids)

    return ids, corners, frame_with_markers

def calculate_depth(pixel_width, real_width=REAL_WIDTH, focal_length=FOCAL_LENGTH):
    """
    Calculates the depth (distance) of an object using the pinhole camera model.

    Parameters:
        pixel_width (float): The width of the detected marker in pixels.
        real_width (float): The real-world width of the marker (in cm).
        focal_length (float): The precomputed focal length of the camera (in pixels).

    Returns:
        float: Estimated depth in cm.
    """
    if pixel_width > 0:
        depth = (focal_length * real_width) / pixel_width
        return depth
    return None

# Initialize the Raspberry Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)  # Adjust resolution if needed
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    # Capture a frame from the Pi camera
    frame = picam2.capture_array("main")
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

    # Detect ArUco markers
    ids, corners, frame_with_markers = detect_aruco_marker(frame)

    if ids is not None:
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]

            # Extract the top-left and top-right corners of the marker
            top_left, top_right = corner[0][0], corner[0][1]
            
            # Compute pixel width of the marker
            pixel_width = np.linalg.norm(top_right - top_left)

            # Compute depth
            depth = calculate_depth(pixel_width)

            if depth:
                print(f"Marker {marker_id} detected - Depth: {depth:.2f} cm")

    # Display the frame with detected markers
    cv2.imshow("ArUco Detection", frame_with_markers)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()