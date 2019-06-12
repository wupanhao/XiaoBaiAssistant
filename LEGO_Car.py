#!/usr/bin/env python

from __future__ import print_function # use python 3 syntax but make it compatible with python 2
from __future__ import division       #                           ''

import time     # import the time library for the sleep function
import brickpi3 # import the BrickPi3 drivers


class Car:
    def __init__(self):
        self.BP = brickpi3.BrickPi3()
        self.speed = 0
        self.start = (self.BP.get_motor_encoder(self.BP.PORT_A), self.BP.get_motor_encoder(self.BP.PORT_B), self.BP.get_motor_encoder(self.BP.PORT_C), self.BP.get_motor_encoder(self.BP.PORT_D))
        self.dist = list(self.start)
    def get_encoder(self):
        try:
            # Each of the following BP.get_motor_encoder functions returns the encoder value (what we want to display).
            encoders = (self.BP.get_motor_encoder(self.BP.PORT_A), self.BP.get_motor_encoder(self.BP.PORT_B), self.BP.get_motor_encoder(self.BP.PORT_C), self.BP.get_motor_encoder(self.BP.PORT_D))
            print("Encoder A: %6d  B: %6d  C: %6d  D: %6d" % encoders)
            return encoders
        except IOError as error:
            print(error)           
            return None
    def set_speed(self,speed = 10):
        self.speed = speed
    def go_forward(self,dist):
        # 1000 ~= 28cm
        dist = dist*250/7.0
        self.dist[0] = self.dist[0]+dist
        self.dist[1] = self.dist[1]+dist
        self.BP.set_motor_power(self.BP.PORT_A + self.BP.PORT_B , self.speed)
        self.to_dist()
    def go_backward(self,dist):
        dist = dist*250/7.0
        self.dist[0] = self.dist[0]-dist
        self.dist[1] = self.dist[1]-dist
        self.BP.set_motor_power(self.BP.PORT_A + self.BP.PORT_B , -self.speed)
        self.to_dist()
    def turn_left(self,angle = 90):
        # 1000 ~= 190
        dist = angle*100/19.0
        self.dist[0] = self.dist[0]-dist
        self.dist[1] = self.dist[1]+dist
        self.BP.set_motor_power(self.BP.PORT_A, -self.speed)
        self.BP.set_motor_power(self.BP.PORT_B , self.speed)
        self.to_dist()
    def turn_right(self,angle = 90):
        dist = angle*100/19.0
        self.dist[0] = self.dist[0]+dist
        self.dist[1] = self.dist[1]-dist
        self.BP.set_motor_power(self.BP.PORT_A, self.speed)
        self.BP.set_motor_power(self.BP.PORT_B , -self.speed)
        self.to_dist()
    def to_dist(self):
        while True:
            encoders = self.get_encoder()
            if encoders is not None:
                if abs( encoders[0] - self.dist[0]) < 10:
                    self.BP.set_motor_power(self.BP.PORT_A,0)
                elif encoders[0] < self.dist[0]:
                    self.BP.set_motor_power(self.BP.PORT_A,abs(self.speed))
                elif encoders[0] > self.dist[0]:
                    self.BP.set_motor_power(self.BP.PORT_A,-abs(self.speed))
                if abs( encoders[1] - self.dist[1]) < 10:
                    self.BP.set_motor_power(self.BP.PORT_B,0)
                elif encoders[1] < self.dist[1]:
                    self.BP.set_motor_power(self.BP.PORT_B, abs(self.speed))
                elif encoders[1] > self.dist[1]:
                    self.BP.set_motor_power(self.BP.PORT_B, -abs(self.speed))
                if abs( encoders[0] - self.dist[0]) < 10 and abs( encoders[1] - self.dist[1]) < 10:
                    return
                time.sleep(0.02)  # delay for 0.02 seconds (20ms) to reduce the Raspberry Pi CPU load.
            else:
                return
if __name__ == '__main__':
    car = Car()
    car.set_speed(50)
    car.turn_right(45)
    car.BP.reset_all()
