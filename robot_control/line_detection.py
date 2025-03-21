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


#save_image()
detect_line()
  

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