import cv2
import numpy as np

# Constants
REAL_WIDTH = 2.0  # Real-world width of the ArUco marker (in cm)
KNOWN_DISTANCE = 20.0  # Known distance from the camera to the marker (in cm)

def detect_aruco_marker(frame, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    return ids, corners

def calibrate_focal_length(known_distance, real_width, pixel_width):
    return (pixel_width * known_distance) / real_width

def main():
    cap = cv2.VideoCapture(0)  # Change to 0 if using a built-in webcam

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("Place the ArUco marker at", KNOWN_DISTANCE, "cm from the camera.")
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        ids, corners = detect_aruco_marker(frame)

        if ids is not None:
            for i, corner in enumerate(corners):
                top_left, top_right, bottom_right, bottom_left = corner[0]

                # Calculate average width in pixels
                width = np.linalg.norm(top_right - top_left)
                height = np.linalg.norm(bottom_right - bottom_left)
                pixel_width = (width + height) / 2  # Take the average

                # Compute the focal length
                focal_length = calibrate_focal_length(KNOWN_DISTANCE, REAL_WIDTH, pixel_width)
                print(f"Calibrated Focal Length: {focal_length:.2f} pixels")

                # Draw marker for visualization
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("ArUco Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
