import numpy as np
import cv2
import glob

def main():
    # Set the dimensions of the chessboard pattern you are using.
    # For a 9x6 board, you'll have 9 corners in one dimension and 6 in the other.
    # Adjust these to match your actual calibration board.
    chessboard_size = (9, 6)

    # Criteria for refining corner detection
    # We stop either after max iterations or when the corner refinement has moved
    # less than 0.001 of a pixel.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare the 3D object points for each corner in the chessboard.
    # (0,0,0), (1,0,0), (2,0,0) ... (8,5,0) for a 9x6 board.
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store 3D points (objpoints) and 2D points in image plane (imgpoints).
    objpoints = []  # 3D world points
    imgpoints = []  # 2D image points

    # Load all images that match the pattern (adjust the path and file extension as needed).
    images = glob.glob('*.jpg')  # e.g., 'calibration_*.jpg' if your files are named that way.

    # Iterate over each calibration image
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Refine corner positions
            corners_subpix = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            # Add points to our arrays
            objpoints.append(objp)
            imgpoints.append(corners_subpix)

            # Draw and display the corners for checking
            cv2.drawChessboardCorners(img, chessboard_size, corners_subpix, ret)
            cv2.imshow('Chessboard corners', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrate the camera using the collected points
    # ret is the RMS (root mean square) re-projection error.
    # mtx is the camera matrix,
    # dist is the distortion coefficients,
    # rvecs and tvecs are arrays of rotation and translation vectors for each image.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print(f"Calibration successful.\nRMS re-projection error: {ret}")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # Save the calibration result to a .npz file
    np.savez('camera_calib.npz', 
             mtx=mtx, 
             dist=dist, 
             rvecs=rvecs, 
             tvecs=tvecs)

    print("Calibration parameters saved to camera_calib.npz")

if __name__ == "__main__":
    main()
