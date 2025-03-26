import numpy as np
import sys
import os

# Append the directory for the incesion_detection module
sys.path.append(os.path.join(os.path.dirname(__file__), 'incesion_detection'))
import aruco_windows

def copy_params(real_width_cm=2.0, frame_width=320, frame_height=240, focal_length=615, webcam_index=0):
    """
    Sets global parameters for camera calibration and image dimensions.
    
    Global Variables:
        REAL_WIDTH_CM: The known real-world width of the reference object (cm)
        FRAME_WIDTH: The width of the camera frame in pixels
        FRAME_HEIGHT: The height of the camera frame in pixels
        ROBOT_CENTER: The (x, y) center of the frame
        FOCAL_LENGTH: The camera's focal length in pixels
        WEBCAM_INDEX: The webcam index used for capture
    """
    global REAL_WIDTH_CM, FRAME_WIDTH, FRAME_HEIGHT, ROBOT_CENTER, FOCAL_LENGTH, WEBCAM_INDEX
    REAL_WIDTH_CM = real_width_cm
    FRAME_WIDTH = frame_width
    FRAME_HEIGHT = frame_height
    ROBOT_CENTER = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)
    FOCAL_LENGTH = focal_length
    WEBCAM_INDEX = webcam_index

def calculate_distance_robot_to_point(pixel_point, aruco_pixel_width):
    # 1) Convert from pixel units to cm in the plane of the marker
    pixel_scale = REAL_WIDTH_CM / aruco_pixel_width
    
    # 2) Convert the robot center and the target point from pixel to cm
    real_robot_center = (ROBOT_CENTER[0] * pixel_scale, ROBOT_CENTER[1] * pixel_scale)
    real_pixel_point  = (pixel_point[0]  * pixel_scale, pixel_point[1]  * pixel_scale)
    
    # 3) Compute Euclidean distance in cm
    real_distance_cm = np.sqrt(
        (real_pixel_point[0]  - real_robot_center[0]) ** 2 +
        (real_pixel_point[1]  - real_robot_center[1]) ** 2
    )

    # 4) Distance in x and y cm
    dx_cm = real_pixel_point[0] - real_robot_center[0]
    dy_cm = real_pixel_point[1] - real_robot_center[1]

    return real_distance_cm, dx_cm, dy_cm

