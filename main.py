import sys 
import os
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'incesion_detection'))

import aruco_windows

def main():
    FOCAL_LENGTH = 615  # focal length for depth calculation
    WEBCAM_INDEX = 0 

    depth, frame, pixel_width, aruco_center_coords = aruco_windows.get_range(FOCAL_LENGTH, WEBCAM_INDEX)
    cv2.imshow("Aruco tag", frame)
    print("Aruco tag depth: " + str(depth))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()