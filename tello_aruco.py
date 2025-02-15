import cv2
import numpy as np
import pygame
import time
from djitellopy import Tello

S = 60
FPS = 120

class FrontEnd(object):
    def __init__(self):
        pygame.init()
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
        
        # Load camera calibration data
        try:
            calib_data = np.load("camera_calibration.npz")
            self.camera_matrix = calib_data["camera_matrix"]
            self.dist_coeffs = calib_data["dist_coeffs"]
            print("Camera calibration loaded successfully.")
        except Exception as e:
            print("Could not load camera calibration. Pose estimation will not be accurate.")
            self.camera_matrix = None
            self.dist_coeffs = None

        # Load ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
    
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
            frame = self.detect_aruco_marker(frame)
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()
            
            time.sleep(1 / FPS)
        
        self.tello.end()
    
    def detect_aruco_marker(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print(f"Detected ArUco markers: {ids.flatten()}")
            
            for i in range(len(ids)):
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.1, self.camera_matrix, self.dist_coeffs)
                    distance = np.linalg.norm(tvec)
                    print(f"Marker {ids[i][0]} Distance: {distance:.2f}m")
                    cv2.putText(frame, f"ID: {ids[i][0]} Dist: {distance:.2f}m", (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
        
        return frame
    
    def keydown(self, key):
        if key == pygame.K_w:
            self.for_back_velocity = S
        elif key == pygame.K_s:
            self.for_back_velocity = -S
        elif key == pygame.K_a:
            self.left_right_velocity = -S
        elif key == pygame.K_d:
            self.left_right_velocity = S
        elif key == pygame.K_UP:
            self.up_down_velocity = S
        elif key == pygame.K_DOWN:
            self.up_down_velocity = -S
        elif key == pygame.K_LEFT:
            self.yaw_velocity = -S
        elif key == pygame.K_RIGHT:
            self.yaw_velocity = S
        elif key == pygame.K_BACKSPACE:
            self.tello.emergency()
    
    def keyup(self, key):
        if key in [pygame.K_w, pygame.K_s]:
            self.for_back_velocity = 0
        elif key in [pygame.K_a, pygame.K_d]:
            self.left_right_velocity = 0
        elif key in [pygame.K_UP, pygame.K_DOWN]:
            self.up_down_velocity = 0
        elif key in [pygame.K_LEFT, pygame.K_RIGHT]:
            self.yaw_velocity = 0
        elif key == pygame.K_t:
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:
            self.tello.land()
            self.send_rc_control = False
    
    def update(self):
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity)

def main():
    frontend = FrontEnd()
    frontend.run()

if __name__ == '__main__':
    main()
