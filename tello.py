import cv2
import numpy as np
import pygame
import time
import threading
from djitellopy import Tello

# ---------------- Adjustable Parameters ----------------
S_DEFAULT = 60                # Default movement speed (velocity value)
FPS = 120                     # Frames per second for display/update
ARUCO_MARKER_SIZE = 0.158     # Size of the ArUco marker in meters (for pose estimation)
ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250  # Type of ArUco dictionary to use
# --------------------------------------------------------

class FrontEnd:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tello Video Stream")
        self.screen = pygame.display.set_mode([960, 720])
        
        # Initialize Tello
        self.tello = Tello()
        
        # Movement velocities (in cm/s)
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        
        # Adjustable speed for movement commands
        self.speed_val = S_DEFAULT  
        # Drone’s set speed (for commands like takeoff, etc.)
        self.tello_speed = 10
        
        self.send_rc_control = False
        
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)
        
        # Load camera calibration data if available
        try:
            calib_data = np.load("camera_calibration.npz")
            self.camera_matrix = calib_data["camera_matrix"]
            self.dist_coeffs = calib_data["dist_coeffs"]
            print("Camera calibration loaded successfully.")
        except Exception as e:
            print("Could not load camera calibration. Pose estimation will not be accurate.")
            self.camera_matrix = None
            self.dist_coeffs = None
        
        # Set up the ArUco dictionary and detector parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()  # For newer OpenCV, you might use DetectorParameters_create()
        
        # For user-defined autonomous sequence (list of (command, parameter))
        self.user_sequence = []  
        self.executing_sequence = False

    def run(self):
        self.tello.connect()
        self.tello.set_speed(self.tello_speed)
        self.tello.streamoff()
        self.tello.streamon()
        
        # Store the frame reader so that other functions can access the current frame
        self.frame_read = self.tello.get_frame_read()
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
                    elif event.key == pygame.K_f:
                        # When 'f' is pressed, run the user sequence in a new thread.
                        if not self.executing_sequence:
                            threading.Thread(target=self.execute_sequence, args=(self.user_sequence,)).start()
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)
            
            if self.frame_read.stopped:
                break

            # Get the current frame and process it
            frame = self.frame_read.frame.copy()
            
            # Detect and annotate ArUco markers (for display purposes)
            frame = self.detect_aruco_marker(frame)
            
            # Display battery and speed information on the frame
            battery = self.tello.get_battery()
            text = f"Battery: {battery}%  Speed: {self.speed_val}"
            cv2.putText(frame, text, (5, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Rotate and flip the frame for the correct orientation
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
                    # Estimate the pose of each marker
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], ARUCO_MARKER_SIZE, self.camera_matrix, self.dist_coeffs)
                    distance = np.linalg.norm(tvec)
                    print(f"Marker {ids[i][0]} Distance: {distance:.2f}m")
                    # Annotate the frame with marker ID and distance
                    pos = (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10)
                    cv2.putText(frame, f"ID: {ids[i][0]} Dist: {distance:.2f}m", pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # Draw the coordinate axes on the marker
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, ARUCO_MARKER_SIZE)
        return frame

    def keydown(self, key):
        # Movement controls
        if key == pygame.K_w:
            self.for_back_velocity = self.speed_val
        elif key == pygame.K_s:
            self.for_back_velocity = -self.speed_val
        elif key == pygame.K_a:
            self.left_right_velocity = -self.speed_val
        elif key == pygame.K_d:
            self.left_right_velocity = self.speed_val
        elif key == pygame.K_UP:
            self.up_down_velocity = self.speed_val
        elif key == pygame.K_DOWN:
            self.up_down_velocity = -self.speed_val
        elif key == pygame.K_LEFT:
            self.yaw_velocity = -self.speed_val
        elif key == pygame.K_RIGHT:
            self.yaw_velocity = self.speed_val
        elif key == pygame.K_BACKSPACE:
            self.tello.emergency()
        # Speed adjustment keys
        elif key == pygame.K_1:
            self.speed_val = 10
        elif key == pygame.K_2:
            self.speed_val = 20
        elif key == pygame.K_3:
            self.speed_val = 30
        elif key == pygame.K_4:
            self.speed_val = 40
        elif key == pygame.K_5:
            self.speed_val = 50
        elif key == pygame.K_6:
            self.speed_val = 60
        elif key == pygame.K_7:
            self.speed_val = 70
        elif key == pygame.K_8:
            self.speed_val = 80
        elif key == pygame.K_9:
            self.speed_val = 90
        elif key == pygame.K_0:
            self.speed_val = 100

    def keyup(self, key):
        # Stop movement when keys are released
        if key in [pygame.K_w, pygame.K_s]:
            self.for_back_velocity = 0
        elif key in [pygame.K_a, pygame.K_d]:
            self.left_right_velocity = 0
        elif key in [pygame.K_UP, pygame.K_DOWN]:
            self.up_down_velocity = 0
        elif key in [pygame.K_LEFT, pygame.K_RIGHT]:
            self.yaw_velocity = 0
        # Takeoff and landing controls
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        if self.send_rc_control:
            self.tello.send_rc_control(
                self.left_right_velocity,
                self.for_back_velocity,
                self.up_down_velocity,
                self.yaw_velocity
            )

    # ---------------- New High-Level Drone Functions ----------------
    def fly_forward(self, distance_m):
        """Fly the Tello forward by a specified distance (in meters) at maximum speed."""
        distance_cm = int(distance_m * 100)
        max_speed = 100  # Maximum speed in cm/s

        # Set Tello speed to maximum
        self.tello.set_speed(max_speed)
        
        print(f"Setting speed to {max_speed} cm/s and flying forward {distance_m} meter(s) ({distance_cm} cm).")
        self.tello.move_forward(distance_cm)

    def fly_backward(self, distance_m):
        """Fly the Tello backward by a specified distance (in meters) at maximum speed."""
        distance_cm = int(distance_m * 100)
        max_speed = 100  # Maximum speed in cm/s

        # Set Tello speed to maximum
        self.tello.set_speed(max_speed)
        
        print(f"Setting speed to {max_speed} cm/s and flying backward {distance_m} meter(s) ({distance_cm} cm).")
        self.tello.move_back(distance_cm)

    def fly_up(self, distance_m):
        """Fly the Tello up by a specified distance (in meters) at maximum speed."""
        distance_cm = int(distance_m * 100)
        max_speed = 100  # Maximum speed in cm/s

        # Set Tello speed to maximum
        self.tello.set_speed(max_speed)
        
        print(f"Setting speed to {max_speed} cm/s and flying up {distance_m} meter(s) ({distance_cm} cm).")
        self.tello.move_up(distance_cm)
    
    def fly_down(self, distance_m):
        """Fly the Tello up by a specified distance (in meters) at maximum speed."""
        distance_cm = int(distance_m * 100)
        max_speed = 100  # Maximum speed in cm/s

        # Set Tello speed to maximum
        self.tello.set_speed(max_speed)
        
        print(f"Setting speed to {max_speed} cm/s and flying up {distance_m} meter(s) ({distance_cm} cm).")
        self.tello.move_down(distance_cm)




    def turn_clock_wise(self, degree):
        """Turn the Tello clockwise by a specified degree."""
        print(f"Turning clockwise {degree} degrees.")
        self.tello.rotate_clockwise(degree)

    def turn_counter_clock_wise(self, degree):
        """Turn the Tello counter-clockwise by a specified degree."""
        print(f"Turning counter-clockwise {degree} degrees.")
        self.tello.rotate_counter_clockwise(degree)

    def land(self):
        """Land the Tello."""
        print("Landing.")
        self.tello.land()

    def detect_aruco(self, marker_id):
        """
        Return True if the specified ArUco marker is detected
        in the current frame.
        """
        frame = self.frame_read.frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None and marker_id in ids.flatten():
            print(f"Marker {marker_id} detected.")
            return True
        print(f"Marker {marker_id} not detected.")
        return False

    def fly_toward_aruco(self, marker_id):
        """
        Fly the Tello toward the specified ArUco marker until the drone is approximately 1 meter away.
        Returns True when done.
        """
        print(f"Flying toward ArUco marker {marker_id} until ~1 meter away.")
        # Temporarily disable RC control to avoid interference with high-level commands
        original_rc_control = self.send_rc_control
        self.send_rc_control = False
        max_speed = 100  # Maximum speed in cm/s

        # Set Tello speed to maximum
        self.tello.set_speed(max_speed)
        try:
            while True:
                frame = self.frame_read.frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                if ids is not None:
                    for i in range(len(ids)):
                        if ids[i][0] == marker_id:
                            if self.camera_matrix is not None and self.dist_coeffs is not None:
                                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                                    corners[i], ARUCO_MARKER_SIZE, self.camera_matrix, self.dist_coeffs)
                                distance = np.linalg.norm(tvec)
                                print(f"Marker {marker_id} distance: {distance:.2f} m")
                                if distance > 1.0:
                                    # Move forward a small increment (20 cm) at a time
                                    self.tello.move_forward(20)
                                    time.sleep(1)
                                else:
                                    print(f"Reached ~1 meter from marker {marker_id}.")
                                    return True
                else:
                    print("Marker not found. Hovering...")
                    time.sleep(0.5)
        finally:
            self.send_rc_control = original_rc_control

    def execute_sequence(self, sequence):
        """
        Execute a user-defined sequence of actions.
        Each action is a tuple: (command, parameter)
        For example: ("fly_forward", 2) to fly forward 2 meters.
        """
        print("Starting execution of user sequence.")
        self.executing_sequence = True
        # Optionally, disable manual RC control during sequence execution
        original_rc_control = self.send_rc_control
        self.send_rc_control = False
        try:
            for action in sequence:
                command = action[0]
                param = action[1] if len(action) > 1 else None
                print(f"Executing command: {command} with parameter: {param}")
                if command == "fly_forward":
                    self.fly_forward(param)
                elif command == "turn_clock_wise":
                    self.turn_clock_wise(param)
                elif command == "turn_counter_clock_wise":
                    self.turn_counter_clock_wise(param)
                elif command == "land":
                    self.land()
                elif command == "detect_aruco":
                    result = self.detect_aruco(param)
                    print(f"detect_aruco({param}) returned {result}")
                elif command == "fly_toward_aruco":
                    result = self.fly_toward_aruco(param)
                    print(f"fly_toward_aruco({param}) returned {result}")
                else:
                    print(f"Unknown command: {command}")
                time.sleep(1)  # Brief pause between commands
        except Exception as e:
            print(f"Error during sequence execution: {e}")
        finally:
            self.executing_sequence = False
            self.send_rc_control = original_rc_control
            print("Finished execution of user sequence.")

def main():
    frontend = FrontEnd()
    
    # ================= User-Defined Autonomous Sequence =================
    # Define your sequence of actions here.
    # Each tuple is (command, parameter). For commands that do not require a parameter (like "land"),
    # you can use None as the parameter.
    #
    # For example:
    #   ("fly_forward", 2)         -> Fly forward 2 meters
    #   ("turn_clock_wise", 90)      -> Turn clockwise 90 degrees
    #   ("detect_aruco", 23)         -> Check for ArUco marker with ID 23
    #   ("fly_toward_aruco", 23)     -> Fly toward marker 23 until ~1 meter away
    #   ("land", None)             -> Land the Tello
    #
    frontend.user_sequence = [
        # ("fly_forward", 2),
        # ("turn_clock_wise", 90),
        ("detect_aruco", 42),
        # ("fly_toward_aruco", 42),
        # ("land", None)
    ]
    # =======================================================================
    
    frontend.run()

if __name__ == '__main__':
    main()
