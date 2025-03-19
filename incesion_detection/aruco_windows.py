import cv2
import numpy as np

REAL_WIDTH = 2.0  # Real-world width of the ArUco marker (in cm)

def detect_aruco_marker(frame, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    frame_with_markers = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_with_markers, corners, ids)
    return ids, corners, frame_with_markers

def calculate_depth(pixel_width, real_width, focal_length):
    if pixel_width > 0:
        return (focal_length * real_width) / pixel_width
    return None

def get_range(FOCAL_LENGTH, WEBCAM_INDEX):
    frame_width, frame_height = 320, 240
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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
                top_left, top_right, bottom_right, bottom_left = corner[0]
                width = np.linalg.norm(top_right - top_left)
                height = np.linalg.norm(bottom_right - bottom_left)
                pixel_width = (width + height) / 2  # Take average
                
                detected_depth = calculate_depth(pixel_width, REAL_WIDTH, FOCAL_LENGTH)
                detected_depth /= 2  # Adjust depth as needed
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
    return detected_depth, detected_frame  # Returns on 's' press
    

def main():
    FOCAL_LENGTH = 615  # Assumed focal length for depth calculation
    WEBCAM_INDEX = 0 

    depth, frame = get_range(FOCAL_LENGTH, WEBCAM_INDEX)
    cv2.imshow("Aruco tag", frame)
    print("Aruco tag depth: " + str(depth))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
