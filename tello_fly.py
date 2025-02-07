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
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Call it always before finishing. To deallocate resources.
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
        elif key == pygame.K_1:     # set  velocity
            self.S = 10
        elif key == pygame.K_2:     # set  velocity
            self.S = 20
        elif key == pygame.K_3:     # set  velocity
            self.S = 30
        elif key == pygame.K_4:     # set  velocity
            self.S = 40
        elif key == pygame.K_5:     # set  velocity
            self.S = 50
        elif key == pygame.K_6:     # set  velocity
            self.S = 60
        elif key == pygame.K_7:     # set  velocity
            self.S = 70
        elif key == pygame.K_8:     # set  velocity
            self.S = 80
        elif key == pygame.K_9:     # set  velocity
            self.S = 90
        elif key == pygame.K_0:     # set  velocity
            self.S = 100
        elif key == pygame.K_BACKSPACE:
            self.tello.emergency()

    def keyup(self, key):
        if key == pygame.K_w or key == pygame.K_s:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_UP or key == pygame.K_DOWN:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False

    def update(self):
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)


def main():
    frontend = FrontEnd()

    # run frontend

    frontend.run()


if __name__ == '__main__':
    main()