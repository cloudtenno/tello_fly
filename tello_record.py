from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time

S = 60
FPS = 120

class FrontEnd(object):

    def __init__(self):
        pygame.init()
        self.S = 60

        # Set the pygame window to 960 (width) x 720 (height)
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        self.tello = Tello()

        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # Recording state variables
        self.recording = False
        self.video_writer = None

        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def toggle_recording(self):
        """Toggle video recording on/off when 'r' is pressed."""
        if not self.recording:
            # Create a VideoWriter that will record at 960x720 (width x height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter("recorded_video.mp4", fourcc, FPS, (960, 720))
            self.recording = True
            print("Recording started.")
        else:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            print("Recording stopped.")

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

            # Get the latest frame from the drone.
            frame = frame_read.frame.copy()

            # --- Process the frame for correct display orientation ---
            # (The Tello feed is rotated/flipped so that the pygame display looks correct.)
            processed_frame = np.rot90(frame)
            processed_frame = np.flipud(processed_frame)
            # Note: The resulting array 'processed_frame' has shape (960, 720, 3),
            # which is suitable for pygame.surfarray.make_surface (it expects [width, height, channels]).

            if self.recording:
                # While recording, show the live feed without any text overlay.
                frame_surface = pygame.surfarray.make_surface(processed_frame)
                self.screen.blit(frame_surface, (0, 0))
            else:
                # When not recording, add an overlay (battery, speed) on the original frame.
                text = "Battery: {}% Speed: {}".format(self.tello.get_battery(), self.S)
                cv2.putText(frame, text, (5, 720 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Then process for display.
                frame_with_text = np.rot90(frame)
                frame_with_text = np.flipud(frame_with_text)
                frame_surface = pygame.surfarray.make_surface(frame_with_text)
                self.screen.blit(frame_surface, (0, 0))

            pygame.display.update()

            # --- Record the displayed frame if recording is active ---
            if self.recording and self.video_writer is not None:
                # Grab the current display pixels.
                recorded_frame = pygame.surfarray.array3d(self.screen)
                # Pygame's array shape is (width, height, channels); transpose it to (height, width, channels)
                recorded_frame = np.transpose(recorded_frame, (1, 0, 2))
                # Convert from RGB (pygame) to BGR (OpenCV)
                recorded_frame = cv2.cvtColor(recorded_frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(recorded_frame)

            time.sleep(1 / FPS)

        # On exit, release resources.
        if self.video_writer is not None:
            self.video_writer.release()
        self.tello.end()

    def keydown(self, key):
        if key == pygame.K_w:      # forward
            self.for_back_velocity = self.S
        elif key == pygame.K_s:    # backward
            self.for_back_velocity = -self.S
        elif key == pygame.K_a:    # left
            self.left_right_velocity = -self.S
        elif key == pygame.K_d:    # right
            self.left_right_velocity = self.S
        elif key == pygame.K_UP:   # up
            self.up_down_velocity = self.S
        elif key == pygame.K_DOWN: # down
            self.up_down_velocity = -self.S
        elif key == pygame.K_LEFT: # yaw counter-clockwise
            self.yaw_velocity = -self.S
        elif key == pygame.K_RIGHT:# yaw clockwise
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
        elif key == pygame.K_r:
            self.toggle_recording()

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
            self.tello.send_rc_control(self.left_right_velocity,
                                       self.for_back_velocity,
                                       self.up_down_velocity,
                                       self.yaw_velocity)

def main():
    frontend = FrontEnd()
    frontend.run()

if __name__ == '__main__':
    main()
