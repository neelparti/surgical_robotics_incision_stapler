import cv2
import numpy as np
from picamera2 import Picamera2

REAL_WIDTH = 2.0  # Real-world width of the ArUco marker (in cm)

def load_camera_calibration(calib_file, frame_width, frame_height):
    """
    Loads camera calibration from an .npz file and computes the
    new camera matrix for the specified frame resolution.
    
    Parameters:
    -----------
    calib_file : str
        Path to the .npz file containing mtx and dist.
    frame_width : int
        Width of the frame you intend to use for capture.
    frame_height : int
        Height of the frame you intend to use for capture.

    Returns:
    --------
    dict
        Dictionary with keys:
        'mtx': Original camera matrix (3x3).
        'dist': Distortion coefficients.
        'new_mtx': Optimal new camera matrix (3x3) for the specified frame size.
        'roi': Region of interest (x, y, width, height) for cropping.
        'focal_length': Approximate focal length in pixels (the average of fx and fy).
    """
    # Load calibration
    with np.load(calib_file) as data:
        mtx = data['mtx']  # 3x3 camera matrix
        dist = data['dist']  # distortion coefficients

    # Compute new camera matrix
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (frame_width, frame_height), 1, (frame_width, frame_height)
    )

    # Approximate focal length by averaging fx and fy
    fx = new_mtx[0, 0]
    fy = new_mtx[1, 1]
    focal_length = (fx + fy) / 2.0

    return {
        'mtx': mtx,
        'dist': dist,
        'new_mtx': new_mtx,
        'roi': roi,
        'focal_length': focal_length
    }

def detect_aruco_marker(frame, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    """
    Detects an ArUco marker in the given frame.

    Parameters:
        frame (numpy.ndarray): The input image.
        aruco_dict_type (int): The type of ArUco dictionary to use. Default is DICT_4X4_50.

    Returns:
        tuple: (ids, corners, frame_with_markers)
        - ids (numpy.ndarray or None): Detected marker IDs.
        - corners (list or None): Detected marker corner points.
        - frame_with_markers (numpy.ndarray): Frame with markers drawn for visualization.
    """
    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters_create()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Draw detected markers on the frame
    frame_with_markers = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_with_markers, corners, ids)

    return ids, corners, frame_with_markers

def calculate_depth(pixel_width, real_width, focal_length):
    """
    Calculates the depth (distance) of an object using the pinhole camera model.

    Parameters:
        pixel_width (float): The width of the detected marker in pixels.
        real_width (float): The real-world width of the marker (in cm).
        focal_length (float): Focal length derived from calibration (in pixels).

    Returns:
        float or None: Estimated depth in cm if pixel_width > 0, else None.
    """
    if pixel_width > 0:
        depth = (focal_length * real_width) / pixel_width
        return depth
    return None

def main():
    # Camera resolution
    frame_width, frame_height = 320, 240

    # 1. Load the calibration data (camera matrix, etc.) from .npz
    calib_info = load_camera_calibration(
        calib_file="camera_calib.npz",
        frame_width=frame_width,
        frame_height=frame_height
    )
    # If desired, you can print out some of the calibration parameters:
    print(f"Using focal length: {calib_info['focal_length']:.2f} px")

    # 2. Initialize the Raspberry Pi Camera
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (frame_width, frame_height)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    # 3. Capture and process frames in a loop
    while True:
        # Capture a frame
        frame = picam2.capture_array("main")

        # Undistort using calibration
        undistorted_frame = cv2.undistort(
            frame, 
            calib_info['mtx'], 
            calib_info['dist'], 
            None, 
            calib_info['new_mtx']
        )

        # Detect ArUco markers in the undistorted image
        ids, corners, frame_with_markers = detect_aruco_marker(undistorted_frame)

        # Calculate depth for any detected markers
        if ids is not None:
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]

                # Each corner set is in the format:
                # [top_left, top_right, bottom_right, bottom_left]
                top_left, top_right = corner[0][0], corner[0][1]

                # Compute pixel width of the marker
                pixel_width = np.linalg.norm(top_right - top_left)

                # Compute depth using the new focal length
                depth = calculate_depth(
                    pixel_width=pixel_width,
                    real_width=REAL_WIDTH,
                    focal_length=calib_info['focal_length']
                )
                if depth:
                    print(f"Marker {marker_id} detected - Depth: {depth:.2f} cm")

        # Display the frame with detected markers
        cv2.imshow("ArUco Detection", frame_with_markers)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    picam2.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
