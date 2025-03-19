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

def main():
    frame_width, frame_height = 320, 240
    FOCAL_LENGTH = 615  # Assumed focal length for depth calculation
    WEBCAM_INDEX = 0 

    # On Windows, use the default webcam (usually index 0)
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        ids, corners, frame_with_markers = detect_aruco_marker(frame)

        if ids is not None:
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                top_left, top_right = corner[0][0], corner[0][1]
                pixel_width = np.linalg.norm(top_right - top_left)
                depth = calculate_depth(pixel_width, REAL_WIDTH, FOCAL_LENGTH)
                depth = depth / 2
                if depth:
                    print(f"Marker {marker_id} detected - Depth: {depth:.2f} cm")

        cv2.imshow("ArUco Detection", frame_with_markers)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
