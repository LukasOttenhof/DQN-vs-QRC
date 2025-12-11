# Retrieved 2025-01-01 and modified from https://github.com/Matthew-Vandergrift/TruckBackerUpper-ENV
#
# Custom Truck Backer Upper Gymnasium Environment 
# Description of Truck Backer Upper from : An application of the temporal difference
# algorithm to the truck backer-upper problem
# Inspiration of PyGame Code from : https://github.com/johnnycode8/gym_custom_env

import numpy as np
import gym
import sys
# import pygame

# For Declaring Observation and Action Space
from gym import spaces
from gym.utils import seeding

# My problem code
import sys
sys.path.append("../")
import TruckBacker1 as tbu_prob


# Environment Class
class TruckBackerEnv_D(gym.Env):
    # Required (idk why)
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    # INIT FUNCTION
    def __init__(self, trailer_length=14, cab_length=6, x_bounds=[0,200], y_bounds=[-100, 100], render_mode = None, fps = 1, seed=None):
        self.seed(seed)
        self.render_mode = render_mode
        # Initalizing the Problem Object
        self.truck = tbu_prob.TruckBackerUpper(self.np_random, trailer_length, cab_length, x_bounds, y_bounds)
        # Defining Action Space
        self.action_space = spaces.Discrete(5)
        # Defining Obseration Space
        self.observation_space = spaces.Box(
            low=np.array([-3.00, -3.00, -2*np.pi, -2*np.pi]),
            high=np.array([3.00, 3.00, 2*np.pi, 2*np.pi]),
            shape=(4,),
            dtype=np.float64
        )
        # Defining Max Number of Steps 
        self.step_counter = 0
        # Pygame stuff
        if render_mode == "human":
            self.fps = fps
            self.last_action=''
        # Stuff for Testing 
        self.had_recent_stochastic = False

    # GYM FUNCTION
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        #self.truck.reset_truck(self.np_random.uniform(-10 , 10), self.np_random.uniform(-1.5, 1.5))
        self.truck.reset_truck(x_rand_val = self.np_random.uniform(100, 150), y_rand_val=self.np_random.uniform(-20 , 20), theta_t_rand_val=self.np_random.uniform(-1, 1), theta_c_rand_val=self.np_random.uniform(-0.5,0.5))
        self.step_counter = 0
        # Observation is simply the four state variables 
        normed_x = (6*self.truck.x / self.truck.x_bounds[1] - 3)
        normed_y = (3*self.truck.y / self.truck.y_bounds[1])
        obs = np.array([normed_x, normed_y, self.truck.theta_c, self.truck.theta_t])
        # Checking for rendering 
        if self.render_mode == "human":
            self.render()
        # Return observation
        return obs

    def step(self, action):
        # Adding a step to the counter
        action = action/2 - 1
        self.step_counter += 1
        self.had_recent_stochastic = False

        # Taking the action
        terminated_goal, terminated_fail = self.truck.step(u = action) 
        terminated = terminated_goal #| terminated_fail 
        truncated = (self.step_counter == 300)

        if terminated_fail:
            self.had_recent_stochastic = True
            #self.truck.reset_truck(self.np_random.uniform(-10,10), self.np_random.uniform(-1.5, 1.5))
            self.truck.reset_truck(x_rand_val = self.np_random.uniform(100, 150), y_rand_val=self.np_random.uniform(-20 , 20), theta_t_rand_val=self.np_random.uniform(-1, 1), theta_c_rand_val=self.np_random.uniform(-0.5,0.5))
            
        
        # Reward Function
        if terminated_goal == True:
            reward = 100
        else:
            reward = -1
            
        # State Observation
        normed_x = (6*(self.truck.x / self.truck.x_bounds[1]) - 3)
        normed_y = (3* self.truck.y / self.truck.y_bounds[1])
        obs = np.array([normed_x, normed_y, self.truck.theta_c, self.truck.theta_t])

        # Checking for Rendering
        if(self.render_mode=='human'):
            self.render()

        if terminated_fail:
            info = {}
            # info = {
            #     "terminated_failure": True,
            #     "should_set_steps_to_max": False
            # }
        else:
            info = {}

        # Return observation, reward, terminated, info
        return obs, reward, terminated or truncated, info
    
# For unit testing
if __name__=="__main__":
    env = gym.make('TBU_v0', render_mode=None)

    # Reset environment
    obs, info = env.reset()
    print("First Observation is : ", obs)
    rand_action = env.action_space.sample()
    print("Random Action is :", rand_action)

    # Reset environment
    obs,info = env.reset()

    # Take some random 
    num_actions_per_episode = 0
    while(True):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)
        num_actions_per_episode += 1
        if reward > 0:
            print("reward is ", reward)
        if(terminated):
            print("Reset after %s actions" %num_actions_per_episode)
            obs, info  = env.reset()