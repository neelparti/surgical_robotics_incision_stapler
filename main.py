import sys 
import os
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'incesion_detection'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'robot_control'))

import aruco_windows
import pixel_translation
import line_detection
import robot_control

def main():
    REAL_WIDTH_CM = 2.0  # Real-world width of the ArUco marker (in cm)
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240
    ROBOT_CENTER = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)
    FOCAL_LENGTH = 615  # focal length for depth calculation
    WEBCAM_INDEX = 0 

    #robot_control.robot.create_robot()
    robot_control.activate_robot()

    aruco_windows.copy_params(REAL_WIDTH_CM, FRAME_WIDTH, FRAME_HEIGHT, FOCAL_LENGTH, WEBCAM_INDEX)
    pixel_translation.copy_params(REAL_WIDTH_CM, FRAME_WIDTH, FRAME_HEIGHT, FOCAL_LENGTH, WEBCAM_INDEX)

    sample_destination_point = (100, 100)       # for testing

    depth, frame, aruco_pixel_width, aruco_center_coords = aruco_windows.get_range()
    cv2.imshow("Aruco tag", frame)
    print("Aruco tag depth: " + str(depth))
    cv2.waitKey(0)

    path = line_detection.detect_line_dotted_2(frame,selected_lines=[0])
    path = path[0]      # hacky hack
    print(path)     # debugging 
    robot_control.move_to_lin_trf(0, 0, depth)
    for point in path:
        euclidean_dist, x, y = pixel_translation.calculate_distance_robot_to_point(point, aruco_pixel_width)
        
        #robot_control.move_to_lin(x,y,depth)
        # here send x,y to robot, once robot reaches, loop again

    robot_control.close_robot()
        
if __name__ == "__main__":
    main()