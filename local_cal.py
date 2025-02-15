#!/usr/bin/env python
import cv2
import numpy as np
import glob
import os

def main():
    # Folder containing calibration images
    calib_folder = "calibration_images"
    image_files = glob.glob(os.path.join(calib_folder, "*.png"))

    # Chessboard dimensions (number of inner corners per a chessboard row and column)
    chessboard_size = (8,4)  # adjust if necessary
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points for the chessboard pattern (e.g., (0,0,0), (1,0,0), ..., (8,5,0))
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all valid images.
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane.
    image_shape = None

    valid_images = 0

    print("Starting to process calibration images...")

    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"Could not load image: {image_file}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]  # (width, height)

        # Try to find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            print(f"Chessboard detected in: {image_file}")
            objpoints.append(objp)
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_images += 1
        else:
            print(f"Chessboard not detected in: {image_file}. Skipping.")

    if valid_images < 1:
        print("No valid calibration images found. Exiting calibration.")
        return

    print(f"Running calibration using {valid_images} valid images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    if ret:
        np.savez("camera_calibration.npz", camera_matrix=mtx, dist_coeffs=dist)
        print("Calibration successful!")
        print("Camera matrix and distortion coefficients saved to 'camera_calibration.npz'.")
    else:
        print("Calibration failed.")

if __name__ == "__main__":
    main()
