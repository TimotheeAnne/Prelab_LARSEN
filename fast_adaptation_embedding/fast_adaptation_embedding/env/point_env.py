import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import time

class Point(object):
    def __init__(self, goal):
        self.state = np.array([0.,0.])
        self.goal = np.array(goal)
        self.action_space = spaces.Box(low= -np.ones(2), high=np.ones(2))
        self.observation_space = spaces.Box(low= np.array([-100,-100]), high=np.array([100,100])) #(x, y, sin_theta, cos_theta)

    def step(self, action):
        self.state = self.state + action
        cost = np.linalg.norm(self.state-self.goal)
        return self.state.copy(), cost, False, {}

    def render(self, t_wait=0.3):
        plt.clf()
        plt.plot(self.state[0], self.state[1], 'o')
        plt.plot(self.goal[0], self.goal[1], '*')
        plt.xlim([-30, 30])
        plt.ylim([-30, 30])
        plt.pause(t_wait)

    def reset(self, state = np.array([0.,0.])):
        self.state =  state.copy() 
        return self.state.copy()

    def state(self):
        return self.state.copy()

if __name__=='__main__':
    env = Point([10,-20])
    print(env.action_space.sample())
    print(env.action_space.shape[0])


