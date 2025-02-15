from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import os

S = 60
FPS = 120

class FrontEnd(object):

    def __init__(self):
        pygame.init()

        self.S = 60

        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        self.tello = Tello()

        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

        # Ensure calibration image folder exists
        self.calib_folder = "calibration_images"
        if not os.path.exists(self.calib_folder):
            os.makedirs(self.calib_folder)

    def run(self):

        self.tello.connect()
        self.tello.set_speed(self.speed)
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame
            text = "Battery: {}% Speed: {}".format(self.tello.get_battery(), self.S)
            cv2.putText(frame, text, (5, 720 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Rotate and flip frame for correct orientation
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Always end the Tello connection properly.
        self.tello.end()

    def keydown(self, key):
        if key == pygame.K_w:  # set forward velocity
            self.for_back_velocity = self.S
        elif key == pygame.K_s:  # set backward velocity
            self.for_back_velocity = -self.S
        elif key == pygame.K_a:  # set left velocity
            self.left_right_velocity = -self.S
        elif key == pygame.K_d:  # set right velocity
            self.left_right_velocity = self.S
        elif key == pygame.K_UP:  # set up velocity
            self.up_down_velocity = self.S
        elif key == pygame.K_DOWN:  # set down velocity
            self.up_down_velocity = -self.S
        elif key == pygame.K_LEFT:  # set yaw counter clockwise velocity
            self.yaw_velocity = -self.S
        elif key == pygame.K_RIGHT:  # set yaw clockwise velocity
            self.yaw_velocity = self.S
        elif key == pygame.K_1:
            self.S = 10
        elif key == pygame.K_2:
            self.S = 20
        elif key == pygame.K_3:
            self.S = 30
        elif key == pygame.K_4:
            self.S = 40
        elif key == pygame.K_5:
            self.S = 50
        elif key == pygame.K_6:
            self.S = 60
        elif key == pygame.K_7:
            self.S = 70
        elif key == pygame.K_8:
            self.S = 80
        elif key == pygame.K_9:
            self.S = 90
        elif key == pygame.K_0:
            self.S = 100
        elif key == pygame.K_BACKSPACE:
            self.tello.emergency()
        elif key == pygame.K_c:
            # Start the camera calibration mode
            self.calibrate_camera()

    def keyup(self, key):
        if key in (pygame.K_w, pygame.K_s):
            self.for_back_velocity = 0
        elif key in (pygame.K_a, pygame.K_d):
            self.left_right_velocity = 0
        elif key in (pygame.K_UP, pygame.K_DOWN):
            self.up_down_velocity = 0
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                                         self.up_down_velocity, self.yaw_velocity)

    def calibrate_camera(self):
        """
        Enters camera calibration mode.
        Instructions are overlaid on the video:
          - Press SPACE to capture an image (of the chessboard pattern).
          - Press ENTER to perform calibration and save the camera matrix.
          - Press ESC to cancel calibration.
        Captured images are saved locally to the 'calibration_images' folder.
        """
        calibration_images = []
        instructions = [
            "Camera Calibration Mode",
            "Place the chessboard pattern in view.",
            "Press SPACE to capture image.",
            "Press ENTER to calibrate and save.",
            "Press ESC to cancel."
        ]
        capturing = True
        while capturing:
            # Get a copy of the current frame
            frame = self.tello.get_frame_read().frame.copy()

            # Overlay instructions on the frame
            for i, text in enumerate(instructions):
                cv2.putText(frame, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Captured Images: {len(calibration_images)}", (10, 30 + len(instructions)*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Convert frame for pygame display
            frame_disp = np.rot90(frame)
            frame_disp = np.flipud(frame_disp)
            surf = pygame.surfarray.make_surface(frame_disp)
            self.screen.blit(surf, (0, 0))
            pygame.display.update()

            # Process events for calibration mode
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Capture the current frame for calibration
                        img_copy = frame.copy()
                        calibration_images.append(img_copy)
                        # Save the captured image locally with a unique name
                        img_filename = os.path.join(self.calib_folder, f"calib_{len(calibration_images)}.png")
                        cv2.imwrite(img_filename, img_copy)
                        print(f"Captured and saved: {img_filename}")
                    elif event.key == pygame.K_RETURN:
                        # Proceed to calibrate using the captured images
                        self.run_calibration(calibration_images)
                        capturing = False
                    elif event.key == pygame.K_ESCAPE:
                        # Cancel calibration mode
                        capturing = False
            time.sleep(1 / FPS)

    def run_calibration(self, images):
        """
        Performs camera calibration on the provided images using a 9x6 chessboard.
        The computed camera matrix and distortion coefficients are saved to 'camera_calibration.npz'.
        """
        if len(images) == 0:
            print("No calibration images captured.")
            self.display_message("No calibration images captured.", 3)
            return

        # Define chessboard dimensions (inner corners)
        CHECKERBOARD = (9, 6)
        # Termination criteria for refining corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points (e.g. (0,0,0), (1,0,0), ..., (8,5,0))
        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        objpoints = []  # 3D points in real-world space
        imgpoints = []  # 2D points in image plane

        valid_images = 0
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret:
                valid_images += 1
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        if valid_images > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            # Save the calibration results to a file
            np.savez("camera_calibration.npz", camera_matrix=mtx, dist_coeffs=dist)
            print("Calibration successful. Camera matrix saved to 'camera_calibration.npz'.")
            self.display_message("Calibration successful. Saved as camera_calibration.npz", 3)
        else:
            print("Calibration failed. No valid chessboard patterns detected.")
            self.display_message("Calibration failed. No valid chessboard patterns detected.", 3)

    def display_message(self, message, duration):
        """
        Displays a message on the pygame screen for the specified duration (in seconds).
        """
        end_time = time.time() + duration
        font = pygame.font.SysFont('Arial', 24)
        while time.time() < end_time:
            self.screen.fill((0, 0, 0))
            text_surface = font.render(message, True, (255, 255, 255))
            self.screen.blit(text_surface, (50, 350))
            pygame.display.update()
            time.sleep(0.1)

def main():
    frontend = FrontEnd()
    frontend.run()

if __name__ == '__main__':
    main()
from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import os

S = 60
FPS = 120

class FrontEnd(object):

    def __init__(self):
        pygame.init()

        self.S = 60

        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        self.tello = Tello()

        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

        # Ensure calibration image folder exists
        self.calib_folder = "calibration_images"
        if not os.path.exists(self.calib_folder):
            os.makedirs(self.calib_folder)

    def run(self):

        self.tello.connect()
        self.tello.set_speed(self.speed)
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame
            text = "Battery: {}% Speed: {}".format(self.tello.get_battery(), self.S)
            cv2.putText(frame, text, (5, 720 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Rotate and flip frame for correct orientation
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Always end the Tello connection properly.
        self.tello.end()

    def keydown(self, key):
        if key == pygame.K_w:  # set forward velocity
            self.for_back_velocity = self.S
        elif key == pygame.K_s:  # set backward velocity
            self.for_back_velocity = -self.S
        elif key == pygame.K_a:  # set left velocity
            self.left_right_velocity = -self.S
        elif key == pygame.K_d:  # set right velocity
            self.left_right_velocity = self.S
        elif key == pygame.K_UP:  # set up velocity
            self.up_down_velocity = self.S
        elif key == pygame.K_DOWN:  # set down velocity
            self.up_down_velocity = -self.S
        elif key == pygame.K_LEFT:  # set yaw counter clockwise velocity
            self.yaw_velocity = -self.S
        elif key == pygame.K_RIGHT:  # set yaw clockwise velocity
            self.yaw_velocity = self.S
        elif key == pygame.K_1:
            self.S = 10
        elif key == pygame.K_2:
            self.S = 20
        elif key == pygame.K_3:
            self.S = 30
        elif key == pygame.K_4:
            self.S = 40
        elif key == pygame.K_5:
            self.S = 50
        elif key == pygame.K_6:
            self.S = 60
        elif key == pygame.K_7:
            self.S = 70
        elif key == pygame.K_8:
            self.S = 80
        elif key == pygame.K_9:
            self.S = 90
        elif key == pygame.K_0:
            self.S = 100
        elif key == pygame.K_BACKSPACE:
            self.tello.emergency()
        elif key == pygame.K_c:
            # Start the camera calibration mode
            self.calibrate_camera()

    def keyup(self, key):
        if key in (pygame.K_w, pygame.K_s):
            self.for_back_velocity = 0
        elif key in (pygame.K_a, pygame.K_d):
            self.left_right_velocity = 0
        elif key in (pygame.K_UP, pygame.K_DOWN):
            self.up_down_velocity = 0
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                                         self.up_down_velocity, self.yaw_velocity)

    def calibrate_camera(self):
        """
        Enters camera calibration mode.
        Instructions are overlaid on the video:
          - Press SPACE to capture an image (of the chessboard pattern).
          - Press ENTER to perform calibration and save the camera matrix.
          - Press ESC to cancel calibration.
        Captured images are saved locally to the 'calibration_images' folder.
        """
        calibration_images = []
        instructions = [
            "Camera Calibration Mode",
            "Place the chessboard pattern in view.",
            "Press SPACE to capture image.",
            "Press ENTER to calibrate and save.",
            "Press ESC to cancel."
        ]
        capturing = True
        while capturing:
            # Get a copy of the current frame
            frame = self.tello.get_frame_read().frame.copy()

            # Overlay instructions on the frame
            for i, text in enumerate(instructions):
                cv2.putText(frame, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Captured Images: {len(calibration_images)}", (10, 30 + len(instructions)*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Convert frame for pygame display
            frame_disp = np.rot90(frame)
            frame_disp = np.flipud(frame_disp)
            surf = pygame.surfarray.make_surface(frame_disp)
            self.screen.blit(surf, (0, 0))
            pygame.display.update()

            # Process events for calibration mode
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Capture the current frame for calibration
                        img_copy = frame.copy()
                        calibration_images.append(img_copy)
                        # Save the captured image locally with a unique name
                        img_filename = os.path.join(self.calib_folder, f"calib_{len(calibration_images)}.png")
                        cv2.imwrite(img_filename, img_copy)
                        print(f"Captured and saved: {img_filename}")
                    elif event.key == pygame.K_RETURN:
                        # Proceed to calibrate using the captured images
                        self.run_calibration(calibration_images)
                        capturing = False
                    elif event.key == pygame.K_ESCAPE:
                        # Cancel calibration mode
                        capturing = False
            time.sleep(1 / FPS)

    def run_calibration(self, images):
        """
        Performs camera calibration on the provided images using a 9x6 chessboard.
        The computed camera matrix and distortion coefficients are saved to 'camera_calibration.npz'.
        """
        if len(images) == 0:
            print("No calibration images captured.")
            self.display_message("No calibration images captured.", 3)
            return

        # Define chessboard dimensions (inner corners)
        CHECKERBOARD = (9, 6)
        # Termination criteria for refining corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points (e.g. (0,0,0), (1,0,0), ..., (8,5,0))
        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        objpoints = []  # 3D points in real-world space
        imgpoints = []  # 2D points in image plane

        valid_images = 0
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret:
                valid_images += 1
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        if valid_images > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            # Save the calibration results to a file
            np.savez("camera_calibration.npz", camera_matrix=mtx, dist_coeffs=dist)
            print("Calibration successful. Camera matrix saved to 'camera_calibration.npz'.")
            self.display_message("Calibration successful. Saved as camera_calibration.npz", 3)
        else:
            print("Calibration failed. No valid chessboard patterns detected.")
            self.display_message("Calibration failed. No valid chessboard patterns detected.", 3)

    def display_message(self, message, duration):
        """
        Displays a message on the pygame screen for the specified duration (in seconds).
        """
        end_time = time.time() + duration
        font = pygame.font.SysFont('Arial', 24)
        while time.time() < end_time:
            self.screen.fill((0, 0, 0))
            text_surface = font.render(message, True, (255, 255, 255))
            self.screen.blit(text_surface, (50, 350))
            pygame.display.update()
            time.sleep(0.1)

def main():
    frontend = FrontEnd()
    frontend.run()

if __name__ == '__main__':
    main()
