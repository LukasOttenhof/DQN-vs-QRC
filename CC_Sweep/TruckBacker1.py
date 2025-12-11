# Retrieved 2025-01-01 and modified from https://github.com/Matthew-Vandergrift/TruckBackerUpper-ENV

import numpy as np
from math import degrees, radians

class TruckBackerUpper:    
    def __init__(self, np_random, trailer_length=14, cab_length=6, x_bounds=[0,200], y_bounds=[-100, 100]):
        self.np_random = np_random
        self.l_t = trailer_length
        self.l_c = cab_length
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def reset_truck(self, x_rand_val, y_rand_val, theta_t_rand_val, theta_c_rand_val):
        # Position Variables
        self.x = x_rand_val
        self.y = y_rand_val
        # Angle Variables
        self.theta_t = theta_t_rand_val
        self.theta_c = theta_c_rand_val

    def step(self, u):
        # Intermediate Variables for computation
        a = 3 * np.cos(u)
        b = a * np.cos(self.theta_c)
        # Updating State Variables
        self.x += -1 * b * np.cos(self.theta_t)
        self.y += -1 * b * np.sin(self.theta_t)
        self.theta_t += -1 * np.arcsin(a * np.sin(self.theta_c) / self.l_t)
        self.theta_c += np.arcsin(3 * np.sin(u) / (self.l_c + self.l_t))
        # Returning flags for termination (success, or fail)
        terminated_goal = self.at_goal()
        terminated_fail = not(self.valid())
        return terminated_goal, terminated_fail

    # Checking if truck is in a valid position
    def valid(self):
        return ((not self.is_jackknifed()) and self.valid_location() and self.valid_angles())

    # Checking if truck has reached the goal (using relaxed goal from the paper)
    def at_goal(self):
        return (np.sqrt(self.x**2 + self.y**2) <= 5.0 and np.abs(self.theta_t) <= 0.5)

    # Some utility functions
    def is_jackknifed(self):
        return(self.theta_c > np.pi/2)
    # This is the one function, I need to check  
    def valid_location(self):
        return (self.x <= self.x_bounds[1] and self.x >= self.x_bounds[0]) and \
              (self.y >= self.y_bounds[0] and self.y <= self.y_bounds[1])
    # Checking for valid angles
    def valid_angles(self):
        return self.theta_t <= 4*np.pi