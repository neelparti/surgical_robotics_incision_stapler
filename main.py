import sys 
import os
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'incesion_detection'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'robot_control'))

import aruco_windows
import pixel_translation
import line_detection
import robot_control
import time

def main():
    REAL_WIDTH_CM = 2.0  # Real-world width of the ArUco marker (in cm)
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240
    ROBOT_CENTER = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)
    FOCAL_LENGTH = 615  # focal length for depth calculation
    WEBCAM_INDEX = 0 

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
    # go to the first point in the path list realive to TRF  
    current_point = path[0]
    #robot_control.move_to_lin_trf(0,0,(depth*10)-35)
    robot_control.move_to_defined()
    #robot_control.move_to_lin_trf(12.385955,0, 0.0)
    #robot_control.move_to_lin_trf(24.77,0, 0.0)
    #robot_control.move_to_lin_trf(37.15,0, 0.0)
    #robot_control.move_to_lin_trf(49.54,0, 0.0)
    #robot_control.move_to_lin_trf(61.92,0, 0.0)
    #robot_control.move_to_lin_trf(74.315,0, 0.0)
    #robot_control.move_to_lin_trf(86.7,0, 0.0)
    current_point = (0, 0)
    prev_x = 0
    prev_y = 0
    for dest_point in path:
        #x,y is the delta between current position and destination position
      
        euclidean_dist, curr_x, curr_y = pixel_translation.calculate_distance_robot_to_point(current_point, dest_point, aruco_pixel_width)
        delta_x = curr_x - prev_x
        delta_y =  curr_y - prev_y
        prev_x = curr_x
        prev_y =  curr_y

        print(delta_x,delta_y)
        
        #robot_control.move_to_lin_trf(0,0,-15)
        robot_control.move_to_lin_trf(delta_x,0,0)
        #robot_control.move_to_lin_trf(0,0,15)
        time.sleep(1)



        # move realative to current TRF and pass in x, y
        
    # here send x,y to robot, once robot reaches, loop again
    robot_control.close_robot()
        
if __name__ == "__main__":
    main()