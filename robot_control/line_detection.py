#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt



def save_image():
    # Image path
    image_path = r'/home/lucasc/lucasc/Winter_2025/SYSC_4206/final_project/test_img1.png'

    # Image directory
    directory = r'/home/lucasc/lucasc/Winter_2025/SYSC_4206/final_project/'

    # Open the default webcam (device index 0)
    cap = cv2.VideoCapture(2)

    # Check if the webcam is opened properly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        # Read one frame from the webcam
        result, image = cap.read()
        if result:
            # Display the captured frame in an OpenCV window
            cv2.imshow("Captured Image", image)

            # Save the captured image to disk
            cv2.imwrite(f"{directory}captured_image.png", image)
            print("Image captured and saved as 'captured_image.png'.")

            # Wait for a key press and close the window
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print("Error: No image detected. Please try again.")

    # Release the webcam resource
    cap.release()
    

import cv2
import numpy as np
def detect_line():
    # Read the image
    img = cv2.imread('captured_image.png')

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for detecting red color in HSV space
    # Lower red: 0-10 degrees (Hue), 100-255 (Saturation), 100-255 (Value)
    # Upper red: 170-180 degrees (Hue), 100-255 (Saturation), 100-255 (Value)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the image to get only red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks to capture both red hues
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Use the mask to extract the red regions from the image
    red_img = cv2.bitwise_and(img, img, mask=red_mask)

    # Convert the red regions to grayscale
    gray = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Edge detection using Canny
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Hough Line Transform parameters
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 12  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25  # minimum number of pixels making up a line
    max_line_gap = 45  # maximum gap in pixels between connectable line segments

    # Create a blank image to draw the lines on
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on the edge-detected image to find the lines
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # If lines are detected, draw them on the blank image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Draw the red lines (in blue for visibility on BGR image)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Combine the original image with the detected lines
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    # Show the result
    cv2.imshow('Red Lines Detection', line_image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2
import numpy as np

def detect_line_dotted():
 # Read the image
    img = cv2.imread('captured_image.png')

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for detecting red color in HSV space
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the image to get only red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Use the mask to extract the red regions from the image
    red_img = cv2.bitwise_and(img, img, mask=red_mask)

    # Convert the red regions to grayscale
    gray = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blur_gray, 50, 150)

    # Hough Line Transform parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 12, np.array([]), minLineLength=25, maxLineGap=45)
    
    line_image = np.zeros_like(img)  # Create a blank image to draw on
    dot_coordinates = []  # Store the dot coordinates

    if lines is not None:
        # Sort lines by their middle point to eliminate overlapping lines
        unique_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                unique_lines.append((mid_x, mid_y, x1, y1, x2, y2))

        unique_lines = sorted(unique_lines, key=lambda x: (x[0], x[1]))  # Sort by X and Y
        filtered_lines = []
        min_distance = 23  # Minimum distance to avoid overlapping dots

        for line in unique_lines:
            if not filtered_lines or all(np.hypot(line[0] - l[0], line[1] - l[1]) > min_distance for l in filtered_lines):
                filtered_lines.append(line)

        for mid_x, mid_y, x1, y1, x2, y2 in filtered_lines:
            # Compute the number of dots based on spacing
            line_length = int(np.hypot(x2 - x1, y2 - y1))
            num_dots = line_length // min_distance
            
            # Get direction vector
            dx = (x2 - x1) / num_dots
            dy = (y2 - y1) / num_dots
            
            for i in range(num_dots + 1):
                dot_x = int(x1 + i * dx)
                dot_y = int(y1 + i * dy)
                if not any(np.hypot(dot_x - cx, dot_y - cy) < min_distance for cx, cy in dot_coordinates):
                    dot_coordinates.append((dot_x, dot_y))
                    cv2.circle(line_image, (dot_x, dot_y), 3, (255, 0, 0), -1)  # Draw dots

    # Combine the original image with the dotted lines
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    # Show the result
    cv2.imshow('Dotted Line Detection', line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return dot_coordinates

# Run detection
#coordinates = detect_line_dotted()
#print("Dot Coordinates:", coordinates)


import cv2
import numpy as np



def detect_line_dotted_2(frame, selected_lines=None):
   # Read the image
    #img = cv2.imread('captured_image.png')
    img = frame

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for detecting red color in HSV space
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the image to get only red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Use the mask to extract the red regions from the image
    red_img = cv2.bitwise_and(img, img, mask=red_mask)

    # Convert the red regions to grayscale
    gray = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blur_gray, 50, 150)

    # Hough Line Transform parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 12, np.array([]), minLineLength=25, maxLineGap=45)
    
    line_image = np.zeros_like(img)  # Create a blank image to draw on
    dot_coordinates = []  # Store the dot coordinates

    if lines is not None:
        clusters = []
        threshold_distance = 15  # Distance threshold for clustering

        for line in lines:
            x1, y1, x2, y2 = line[0]
            added = False
            
            for cluster in clusters:
                for cx1, cy1, cx2, cy2 in cluster:
                    if np.hypot(cx1 - x1, cy1 - y1) < threshold_distance and np.hypot(cx2 - x2, cy2 - y2) < threshold_distance:
                        cluster.append((x1, y1, x2, y2))
                        added = True
                        break
                if added:
                    break
            
            if not added:
                clusters.append([(x1, y1, x2, y2)])

        # Find representative lines for each cluster
        merged_lines = []
        for cluster in clusters:
            x1_vals, y1_vals, x2_vals, y2_vals = zip(*cluster)
            merged_lines.append((min(x1_vals), min(y1_vals), max(x2_vals), max(y2_vals)))

        # Store all line dots separately
        all_dot_coordinates = []
        for x1, y1, x2, y2 in merged_lines:
            line_length = int(np.hypot(x2 - x1, y2 - y1))
            num_dots = line_length // 15
            dx = (x2 - x1) / num_dots
            dy = (y2 - y1) / num_dots
            line_dots = []
            
            for i in range(num_dots + 1):
                dot_x = int(x1 + i * dx)
                dot_y = int(y1 + i * dy)
                line_dots.append((dot_x, dot_y))
            all_dot_coordinates.append(line_dots)
        
        # If specific lines are selected, only draw those
        if selected_lines is not None:
            for idx in selected_lines:
                if idx < len(all_dot_coordinates):
                    for dot_x, dot_y in all_dot_coordinates[idx]:
                        cv2.circle(line_image, (dot_x, dot_y), 3, (255, 0, 0), -1)
            dot_coordinates = [all_dot_coordinates[idx] for idx in selected_lines if idx < len(all_dot_coordinates)]
        else:
            dot_coordinates = all_dot_coordinates

    # Combine the original image with the dotted lines
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    # Show the result
    cv2.imshow('Dotted Line Detection', line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return dot_coordinates

# Run detection with a specific line index
#coordinates = detect_line_dotted_2(selected_lines=[0])  # Example: Draw only the first detected line
#print("Dot Coordinates:", coordinates)








#save_image()
#detect_line()
#detect_line_dotted()

  

'''
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the resulting frame
    cv2.imshow("Webcam Feed", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()'
'''