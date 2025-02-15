import cv2
import numpy as np
import pygame
import time
import threading
from djitellopy import Tello
import csv

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
                    elif event.key == pygame.K_i:
                        # NEW: When 'i' is pressed, perform a 360° scan and plan the flight path.
                        if not self.executing_sequence:
                            threading.Thread(target=self.scan_and_plan).start()
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

    # ---------------- High-Level Drone Functions ----------------
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
        max_speed = 100

        self.tello.set_speed(max_speed)
        print(f"Setting speed to {max_speed} cm/s and flying backward {distance_m} meter(s) ({distance_cm} cm).")
        self.tello.move_back(distance_cm)

    def fly_up(self, distance_m):
        """Fly the Tello up by a specified distance (in meters) at maximum speed."""
        distance_cm = int(distance_m * 100)
        max_speed = 100

        self.tello.set_speed(max_speed)
        print(f"Setting speed to {max_speed} cm/s and flying up {distance_m} meter(s) ({distance_cm} cm).")
        self.tello.move_up(distance_cm)
    
    def fly_down(self, distance_m):
        """Fly the Tello down by a specified distance (in meters) at maximum speed."""
        distance_cm = int(distance_m * 100)
        max_speed = 100

        self.tello.set_speed(max_speed)
        print(f"Setting speed to {max_speed} cm/s and flying down {distance_m} meter(s) ({distance_cm} cm).")
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
        While approaching, adjust yaw so that the marker remains centered.
        Returns True when done.
        """
        detection_count = 0
        print(f"Flying toward ArUco marker {marker_id} until ~1 meter away.")
        original_rc_control = self.send_rc_control
        self.send_rc_control = False
        max_speed = 100
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
                                
                                marker_center = np.mean(corners[i][0], axis=0)
                                frame_center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
                                error_x = marker_center[0] - frame_center[0]
                                
                                threshold = 20  
                                
                                if abs(error_x) > threshold:
                                    if error_x > 0:
                                        print("Marker is to the right, rotating clockwise.")
                                        self.tello.rotate_clockwise(10)
                                    else:
                                        print("Marker is to the left, rotating counter-clockwise.")
                                        self.tello.rotate_counter_clockwise(10)
                                    time.sleep(0.5)
                                else:
                                    print("Marker is centered.")

                                if distance > 1.0:
                                    self.tello.move_forward(100)
                                    time.sleep(1)
                                else:
                                    print(f"Reached ~1 meter from marker {marker_id}.")
                                    return True
                else:
                    print("Marker not found. Hovering...")
                    time.sleep(0.5)
                    detection_count += 1

                    if detection_count > 10:
                        print('Could not detect Aruco Code')
                        return True

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
                time.sleep(1)
        except Exception as e:
            print(f"Error during sequence execution: {e}")
        finally:
            self.executing_sequence = False
            self.send_rc_control = original_rc_control
            print("Finished execution of user sequence.")

    # ---------------- New: Scan and Plan Function ----------------
    def scan_and_plan(self):
        """
        When key 'i' is pressed, rotate 360° in small increments while capturing all target
        ArUco markers (IDs: 12,27,42,69,88). Then, choose from possibly multiple detections
        per marker the combination that gives the shortest overall travel distance (starting
        from (0,0)) and plan a sequence of moves to fly to each marker in order.
        The resulting flight sequence and marker detections are saved to CSV files.
        """
        print("Initiating 360° scan for target markers.")
        # Temporarily disable manual RC control.
        original_rc_control = self.send_rc_control
        self.send_rc_control = False

        target_ids = [12, 27, 42, 69, 88]
        # Dictionary to store detections: for each marker id, store a list of dicts
        # containing 'distance', 'global_angle', and computed Cartesian coordinates ('x','y').
        detections = {marker: [] for marker in target_ids}

        accumulated_angle = 0
        step_angle = 10  # degrees per step
        num_steps = 360 // step_angle  # e.g., 36 steps for a full rotation
        # We assume the drone’s starting orientation is 0°.
        current_heading = 0

        for i in range(num_steps):
            print(f"Rotating by {step_angle}° (Accumulated: {accumulated_angle}°)")
            self.tello.rotate_clockwise(step_angle)
            time.sleep(0.5)  # Allow time for rotation and frame update

            accumulated_angle = (accumulated_angle + step_angle) % 360
            # For a clockwise rotation, assume the heading decreases.
            current_heading = (current_heading - step_angle) % 360

            # Capture a frame and detect markers.
            frame = self.frame_read.frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            if ids is not None:
                for j, marker_id in enumerate(ids.flatten()):
                    if marker_id in target_ids:
                        # Use pose estimation to get distance.
                        if self.camera_matrix is not None and self.dist_coeffs is not None:
                            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                                corners[j], ARUCO_MARKER_SIZE, self.camera_matrix, self.dist_coeffs)
                            distance = np.linalg.norm(tvec)
                        else:
                            distance = None
                        # Compute the center of the marker in the image.
                        marker_center = np.mean(corners[j][0], axis=0)
                        frame_center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
                        error_pixels = marker_center[0] - frame_center[0]
                        # Convert pixel error to an angle (in degrees) using fx from the camera matrix.
                        if self.camera_matrix is not None:
                            fx = self.camera_matrix[0, 0]
                            angle_error_rad = np.arctan(error_pixels / fx)
                            angle_error_deg = np.degrees(angle_error_rad)
                        else:
                            angle_error_deg = 0
                        # Global angle is the current heading plus the error.
                        global_angle = (current_heading + angle_error_deg) % 360
                        # Compute Cartesian coordinates (relative to starting position).
                        if distance is not None:
                            x = distance * np.cos(np.radians(global_angle))
                            y = distance * np.sin(np.radians(global_angle))
                        else:
                            x, y = None, None
                        print(f"Detected marker {marker_id}: distance = {distance:.2f}m, global angle = {global_angle:.2f}°")
                        detections[marker_id].append({
                            'distance': distance,
                            'global_angle': global_angle,
                            'x': x,
                            'y': y
                        })
            else:
                print("No target markers detected in this step.")

        # Merge detections that are close together (within 0.5 m) for each marker.
        def merge_detections(detection_list, threshold=0.5):
            merged = []
            for det in detection_list:
                if det['x'] is None or det['y'] is None:
                    continue
                found = False
                for i, m in enumerate(merged):
                    if np.hypot(det['x'] - m['x'], det['y'] - m['y']) <= threshold:
                        # Replace the old detection with the new value.
                        merged[i] = det
                        found = True
                        break
                if not found:
                    merged.append(det)
            return merged

        for marker in target_ids:
            detections[marker] = merge_detections(detections[marker], threshold=0.5)

        # Check that we have at least one detection for each target marker.
        missing = [m for m in target_ids if len(detections[m]) == 0]
        if missing:
            print(f"Missing detections for markers: {missing}. Cannot plan path.")
            self.send_rc_control = original_rc_control
            return

        # Evaluate every combination: one candidate per marker in the fixed order.
        import itertools
        best_path = None
        best_total_distance = float('inf')
        # Note: if there are multiple sets (groups) of markers, each detection list for a given marker
        # should have the distinct candidates after merging.
        for cand in itertools.product(detections[12], detections[27], detections[42], detections[69], detections[88]):
            # cand is a tuple: (detection for 12, for 27, for 42, for 69, for 88)
            total_dist = 0
            # From start (0,0) to marker 12.
            total_dist += np.hypot(cand[0]['x'], cand[0]['y'])
            # Then from marker to marker.
            total_dist += np.hypot(cand[1]['x'] - cand[0]['x'], cand[1]['y'] - cand[0]['y'])
            total_dist += np.hypot(cand[2]['x'] - cand[1]['x'], cand[2]['y'] - cand[1]['y'])
            total_dist += np.hypot(cand[3]['x'] - cand[2]['x'], cand[3]['y'] - cand[2]['y'])
            total_dist += np.hypot(cand[4]['x'] - cand[3]['x'], cand[4]['y'] - cand[3]['y'])
            if total_dist < best_total_distance:
                best_total_distance = total_dist
                best_path = cand

        if best_path is None:
            print("Could not compute a valid flight path.")
            self.send_rc_control = original_rc_control
            return

        print(f"Optimal path total distance: {best_total_distance:.2f} m")
        for marker_id, det in zip(target_ids, best_path):
            print(f"Marker {marker_id}: x = {det['x']:.2f}, y = {det['y']:.2f}, distance = {det['distance']:.2f}m, global angle = {det['global_angle']:.2f}°")

        # Now plan a sequence of flight commands.
        # Assume the drone starts at (0,0) with heading 0°.
        current_x, current_y = 0, 0
        current_heading = 0
        sequence = []
        for marker_id, det in zip(target_ids, best_path):
            # Compute the vector from current position to the target marker.
            target_x, target_y = det['x'], det['y']
            dx = target_x - current_x
            dy = target_y - current_y
            # Desired angle (in degrees) to the target.
            desired_angle = np.degrees(np.arctan2(dy, dx))
            # Calculate the turn needed (normalize to [-180, 180]).
            turn_angle = desired_angle - current_heading
            turn_angle = (turn_angle + 180) % 360 - 180
            if turn_angle > 0:
                sequence.append(("turn_counter_clock_wise", abs(turn_angle)))
            elif turn_angle < 0:
                sequence.append(("turn_clock_wise", abs(turn_angle)))
            # Distance to fly.
            dist = np.hypot(dx, dy)
            sequence.append(("fly_forward", dist))
            # Update current position and heading.
            current_x, current_y = target_x, target_y
            current_heading = desired_angle

        print("Planned flight sequence:")
        for cmd in sequence:
            print(cmd)

        # Save the flight sequence to a CSV file.
        try:
            with open("flight_sequence.csv", "w", newline="") as f_seq:
                writer = csv.writer(f_seq)
                writer.writerow(["Command", "Parameter"])
                for cmd in sequence:
                    writer.writerow(cmd)
            print("Flight sequence saved to flight_sequence.csv")
        except Exception as e:
            print(f"Error saving flight sequence CSV: {e}")

        # Save the marker sequence to a CSV file.
        try:
            with open("marker_sequence.csv", "w", newline="") as f_marker:
                writer = csv.writer(f_marker)
                writer.writerow(["Marker_ID", "x", "y", "distance", "global_angle"])
                for marker_id, det in zip(target_ids, best_path):
                    writer.writerow([marker_id, det['x'], det['y'], det['distance'], det['global_angle']])
            print("Marker sequence saved to marker_sequence.csv")
        except Exception as e:
            print(f"Error saving marker sequence CSV: {e}")

        
        # Optionally, automatically execute the planned sequence:
        self.execute_sequence(sequence)
        self.send_rc_control = original_rc_control

def main():
    frontend = FrontEnd()
    
    # ================= User-Defined Autonomous Sequence =================
    # You can still define your manual autonomous sequence here.
    frontend.user_sequence = [
        # ("fly_toward_aruco", 27),
    ]
    # =======================================================================
    
    frontend.run()

if __name__ == '__main__':
    main()
