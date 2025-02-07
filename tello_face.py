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

        # Face detection is initially disabled.
        self.face_detection_active = False
        # Load the Haar cascade for frontal face detection.
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Sensitivity parameters for face detection:
        self.face_scale_factor = 1.3  # Default value; lower values increase sensitivity.
        self.face_min_neighbors = 5   # Default value; lower values increase sensitivity.

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

            frame = frame_read.frame.copy()

            # If face detection is active, detect faces and draw green rectangles.
            if self.face_detection_active:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.face_scale_factor,
                    minNeighbors=self.face_min_neighbors
                )
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Display current sensitivity parameters on the frame.
                sens_text = "Scale: {:.1f}  Neighbors: {}".format(
                    self.face_scale_factor, self.face_min_neighbors
                )
                cv2.putText(frame, sens_text, (5, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display battery and speed information.
            text = "Battery: {}%  Speed: {}".format(self.tello.get_battery(), self.S)
            cv2.putText(frame, text, (5, 720 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Rotate and flip the frame to match the pygame window.
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Always deallocate resources before finishing.
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
        elif key == pygame.K_LEFT:  # set yaw counter-clockwise velocity
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
            # Toggle face detection mode on/off.
            self.face_detection_active = not self.face_detection_active
        # Adjust face_scale_factor using [ and ] keys.
        elif key == pygame.K_LEFTBRACKET:
            # Decrease scale factor down to a minimum of 1.0 (increasing sensitivity).
            self.face_scale_factor = max(1.0, self.face_scale_factor - 0.1)
        elif key == pygame.K_RIGHTBRACKET:
            # Increase scale factor (decreasing sensitivity).
            self.face_scale_factor += 0.1
        # Adjust face_min_neighbors using , and . keys.
        elif key == pygame.K_COMMA:
            # Decrease min_neighbors down to a minimum of 1 (increasing sensitivity).
            self.face_min_neighbors = max(1, self.face_min_neighbors - 1)
        elif key == pygame.K_PERIOD:
            # Increase min_neighbors (decreasing sensitivity).
            self.face_min_neighbors += 1

    def keyup(self, key):
        if key in (pygame.K_w, pygame.K_s):  # stop forward/backward motion
            self.for_back_velocity = 0
        elif key in (pygame.K_a, pygame.K_d):  # stop left/right motion
            self.left_right_velocity = 0
        elif key in (pygame.K_UP, pygame.K_DOWN):  # stop up/down motion
            self.up_down_velocity = 0
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):  # stop yaw motion
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
