import cv2
import numpy as np
import glob

# -----------------------------------------------------------
# Parameters you need to adjust for your setup:
# -----------------------------------------------------------
CHECKERBOARD = (9, 6)  # (columns of internal corners, rows of internal corners)
SQUARE_SIZE = 0.025    # real-world size of a checkerboard square (in meters, or any unit you prefer)

# Path to calibration images
images_path = "calib_images/*.jpg"

# -----------------------------------------------------------
# Prepare object points (3D points in real-world space)
# -----------------------------------------------------------
# For a 9x6 checkerboard, we have 54 corners.
# We'll assume the top-left corner is (0,0,0) in 3D, next corner along x, etc.
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE  # Optionally multiply by the real square size (e.g., 25 mm => 0.025 m)

# Arrays to store object points and image points for all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in the image plane

# -----------------------------------------------------------
# Load images and find chessboard corners
# -----------------------------------------------------------
image_files = glob.glob(images_path)
if not image_files:
    raise ValueError("No images found in the specified path. Check the path and file extension.")

for filename in image_files:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Attempt to find the corners on the chessboard
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # Refine corner positions to subpixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners_refined)

        # Optionally draw corners for visualization
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)  # pause 500 ms to see each detection
    else:
        print(f"Checkerboard not detected in {filename}")

cv2.destroyAllWindows()

# -----------------------------------------------------------
# Calibrate the camera
# -----------------------------------------------------------
# Returns the camera matrix, distortion coefficients, rotation and translation vectors
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

if ret:
    print("Camera calibrated successfully!")
    print("\nCamera matrix:\n", camera_matrix)
    print("\nDistortion coefficients:\n", dist_coeffs.ravel())
else:
    print("Calibration failed. Not enough valid checkerboard detections.")

# -----------------------------------------------------------
# (Optional) Compute the reprojection error for a sanity check
# -----------------------------------------------------------
total_error = 0
for i in range(len(objpoints)):
    # Project the 3D points to 2D using the found parameters
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
    )
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print(f"\nMean reprojection error: {mean_error:.4f}")
