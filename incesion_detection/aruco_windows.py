import cv2
import numpy as np

REAL_WIDTH_CM = 2.0  # Real-world width of the ArUco marker (in cm)
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
ROBOT_CENTER = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)

def detect_aruco_marker(frame, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    frame_with_markers = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_with_markers, corners)

    # Draw the robot center clearly on the frame
    robot_center_coords = (int(ROBOT_CENTER[0]), int(ROBOT_CENTER[1]))
    cv2.circle(frame_with_markers, robot_center_coords, radius=5, color=(255, 0, 0), thickness=-1)
    cv2.putText(frame_with_markers, "RC", 
                (robot_center_coords[0] + 10, robot_center_coords[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return ids, corners, frame_with_markers

def calculate_depth(pixel_width, real_width, focal_length):
    if pixel_width > 0:
        return (focal_length * real_width) / pixel_width
    return None

def get_range(FOCAL_LENGTH, WEBCAM_INDEX):
    
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return None, None

    detected_depth = None
    detected_frame = None

    print("Press 's' to capture depth and frame, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        ids, corners, frame_with_markers = detect_aruco_marker(frame)

        if ids is not None:
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                
                # Compute marker width in pixels
                detected_top_left, detected_top_right, detected_bottom_right, detected_bottom_left = corner[0]
                width = np.linalg.norm(detected_top_right - detected_top_left)
                height = np.linalg.norm(detected_bottom_right - detected_bottom_left)
                pixel_width = (width + height) / 2  # Take average

                # Calculate center coordinates in pixels
                center_x = int((detected_top_left[0] + detected_bottom_right[0]) / 2)
                center_y = int((detected_top_left[1] + detected_bottom_right[1]) / 2)
                center_coords = (center_x, center_y)
                
                detected_depth = calculate_depth(pixel_width, REAL_WIDTH_CM, FOCAL_LENGTH)
                detected_depth /= 2  # Adjust depth if needed
                detected_frame = frame.copy()

                print(f"Marker {marker_id} detected - Depth: {detected_depth:.2f} cm")
                cv2.putText(frame_with_markers, f"Depth: {detected_depth:.2f} cm", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("ArUco Detection", frame_with_markers)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and detected_depth is not None:  # Capture depth and return on 's' press
            break
        elif key == ord('q'):  # Quit on 'q' press
            detected_depth, detected_frame = None, None
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected_depth, detected_frame, pixel_width, center_coords # Returns on 's' press
    
def pixels_to_cm(pixel_width, real_width_cm, FOCAL_LENGTH):
    """
    Converts a pixel measurement to centimeters using the focal length.
    """
    if pixel_width <= 0:
        return None  # Avoid division by zero
    return (pixel_width * real_width_cm) / FOCAL_LENGTH

def calculate_distance_robot_to_point(pixel_point, FOCAL_LENGTH, reference_pixel_width):
    """
    Calculates the real-world distance (in cm) between two points in pixel coordinates.
    
    Args:
        pixel_point (tuple): (x, y) coordinates of the destination point in pixels.
        real_width_cm (float): Known real-world width of the ArUco marker.
        FOCAL_LENGTH (float): Camera focal length in pixels.
        reference_pixel_width (float): Width of the ArUco marker in pixels.

    Returns:
        float: The real-world distance between the two points in cm.
    """
    # Convert pixel distances to real-world cm
    scale_factor = pixels_to_cm(1, REAL_WIDTH_CM, FOCAL_LENGTH) / reference_pixel_width  # Convert pixels to cm
    real_x1, real_y1 = ROBOT_CENTER[0] * scale_factor, ROBOT_CENTER[1] * scale_factor
    real_x2, real_y2 = pixel_point[0] * scale_factor, pixel_point[1] * scale_factor

    # Compute Euclidean distance in cm
    real_distance_cm = np.sqrt((real_x2 - real_x1) ** 2 + (real_y2 - real_y1) ** 2)
    return real_distance_cm

def main():
    FOCAL_LENGTH = 615  # Assumed focal length for depth calculation
    WEBCAM_INDEX = 0 

    depth, frame, top_left, top_right, bottom_left, bottom_right = get_range(FOCAL_LENGTH, WEBCAM_INDEX)
    cv2.imshow("Aruco tag", frame)
    print("Aruco tag depth: " + str(depth))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
