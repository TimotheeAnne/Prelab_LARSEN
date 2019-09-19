import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pygame as game
import numpy as np
import os

black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
window_width = 800
window_height = 800


class arm:
    def __init__(self, img, pivot_adjust=-0.0, scale=1.0):
        self.base_arm = game.image.load(img)
        self.base_center = (int(window_width / 2), int(window_height / 2))
        # self.base_center = (1000, 1000)
        self.angle = 0
        self.length = self.base_arm.get_rect().w
        self.scale = scale
        self.offset = self.length * scale / 2.0 + pivot_adjust

    def rotate(self, angle, gui, pivot=None):
        if pivot is None:
            pivot = self.base_center
        if gui:
            rotated_arm = game.transform.rotozoom(self.base_arm, angle, self.scale)
            rect = rotated_arm.get_rect()
            rect.center = (self.offset * np.cos(np.deg2rad(angle)) + pivot[0], \
                           -self.offset * np.sin(np.deg2rad(angle)) + pivot[1])
            return rotated_arm, rect
        else:
            return (int(self.offset * np.cos(np.deg2rad(angle)) + pivot[0]), \
                    int(-self.offset * np.sin(np.deg2rad(angle)) + pivot[1]))


class Arm_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, goal=(300, 300)):
        self.sim_step = 1 / 30.
        self.T = 50
        self.dt = 0.1
        self.scale = 0.15
        self.pivot_adjust = 0
        dirname = os.path.dirname(__file__)
        arm_img_path = os.path.join(dirname, 'assets/arm.png')
        self.arm1 = arm(arm_img_path, scale=self.scale, pivot_adjust=self.pivot_adjust)
        self.angle1 = self.arm1.angle
        self.joint1 = self.angle1

        self.arm2 = arm(arm_img_path, scale=self.scale, pivot_adjust=self.pivot_adjust)
        self.angle2 = self.arm2.angle
        self.joint2 = self.angle2 - self.joint1

        self.arm3 = arm(arm_img_path, scale=self.scale, pivot_adjust=self.pivot_adjust)
        self.angle3 = self.arm3.angle
        self.joint3 = self.angle3 - self.joint2

        self.arm4 = arm(arm_img_path, scale=self.scale, pivot_adjust=self.pivot_adjust)
        self.angle4 = self.arm4.angle
        self.joint4 = self.angle4 - self.joint3

        self.arm5 = arm(arm_img_path, scale=self.scale, pivot_adjust=-self.pivot_adjust)
        self.angle5 = self.arm5.angle
        self.joint5 = self.angle5 - self.joint4

        self.states = []
        self.goal = goal
        self.ac_goal_pos = np.array([goal[0]/window_width,  goal[1]/window_height])
        self.actions = []
        self.obs = []

        self.n_act = 5
        self.n_obs = 7

        self.action_space = spaces.Box(low=-np.ones(self.n_act),
                                       high=np.ones(self.n_act),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.ones(self.n_obs),
                                       high=np.ones(self.n_obs),
                                       dtype='float32')
        self.reset()

    def get_obs(self):
        center1 = self.arm1.rotate(self.angle1, gui=False)
        pivot = (int(self.arm2.offset * np.cos(np.deg2rad(self.angle1)) + center1[0]), \
                 int(-self.arm2.offset * np.sin(np.deg2rad(self.angle1)) + center1[1]))

        center2 = self.arm2.rotate(angle=self.angle2, pivot=pivot, gui=False)
        pivot = (int(self.arm3.offset * np.cos(np.deg2rad(self.angle2)) + center2[0]), \
                 int(-self.arm3.offset * np.sin(np.deg2rad(self.angle2)) + center2[1]))

        center3 = self.arm3.rotate(angle=self.angle3, pivot=pivot, gui=False)
        pivot = (int(self.arm4.offset * np.cos(np.deg2rad(self.angle3)) + center3[0]), \
                 int(-self.arm4.offset * np.sin(np.deg2rad(self.angle3)) + center3[1]))

        center4 = self.arm4.rotate(angle=self.angle4, pivot=pivot, gui=False)
        pivot = (int(self.arm5.offset * np.cos(np.deg2rad(self.angle4)) + center4[0]), \
                 int(-self.arm5.offset * np.sin(np.deg2rad(self.angle4)) + center4[1]))

        center5 = self.arm5.rotate(angle=self.angle5, pivot=pivot, gui=False)
        end_effector = (int(self.arm5.offset * np.cos(np.deg2rad(self.angle5)) + center5[0]), \
                        int(-self.arm5.offset * np.sin(np.deg2rad(self.angle5)) + center5[1]))

        return np.array([self.joint1, self.joint2, self.joint3, self.joint4, self.joint5, end_effector[0], end_effector[1]])

    def step(self, action):
        action = np.clip(action, -1, 1) * 180/np.pi

        self.joint1 += self.sim_step * action[0]
        self.joint1 = np.clip(self.joint1, -100.0, 100.0)
        self.angle1 = self.joint1

        self.joint2 += self.sim_step * action[1]
        self.joint2 = np.clip(self.joint2, -100.0, 100.0)
        self.angle2 = self.joint1 + self.joint2

        self.joint3 += self.sim_step * action[2]
        self.joint3 = np.clip(self.joint3, -100.0, 100.0)
        self.angle3 = self.joint1 + self.joint2 + self.joint3

        self.joint4 += self.sim_step * action[3]
        self.joint4 = np.clip(self.joint4, -100.0, 100.0)
        self.angle4 = self.joint1 + self.joint2 + self.joint3 + self.joint4

        self.joint5 += self.sim_step * action[4]
        self.joint5 = np.clip(self.joint5, -100.0, 100.0)
        self.angle5 = self.joint1 + self.joint2 + self.joint3 + self.joint4 + self.joint5

        for event in game.event.get():
            if event.type == game.QUIT:
                game.quit()

        self.timer += self.sim_step

        obs = self.get_obs()
        self.states.append(obs)
        self.done = self.timer > self.T
        info = {}
        normalized_obs = self.get_normalize_obs()
        return np.copy(normalized_obs), self.get_reward(), self.done, info

    def get_normalize_obs(self):
        obs = self.get_obs()
        normalized_obs = np.concatenate((obs[:5]/100, obs[5:6]/window_width, obs[6:7]/window_height))
        return normalized_obs

    def render(self, mode='human', close=False):
        display = game.display.set_mode((window_width, window_height))
        fpsClock = game.time.Clock()
        arm_rotated1, rect1 = self.arm1.rotate(self.angle1, gui=True)
        pivot = (int(self.arm2.offset * np.cos(np.deg2rad(self.angle1)) + rect1.center[0]), \
                 int(-self.arm2.offset * np.sin(np.deg2rad(self.angle1)) + rect1.center[1]))

        arm_rotated2, rect2 = self.arm2.rotate(angle=self.angle2, pivot=pivot, gui=True)
        pivot = (int(self.arm3.offset * np.cos(np.deg2rad(self.angle2)) + rect2.center[0]), \
                 int(-self.arm3.offset * np.sin(np.deg2rad(self.angle2)) + rect2.center[1]))

        arm_rotated3, rect3 = self.arm3.rotate(angle=self.angle3, pivot=pivot, gui=True)
        pivot = (int(self.arm4.offset * np.cos(np.deg2rad(self.angle3)) + rect3.center[0]), \
                 int(-self.arm4.offset * np.sin(np.deg2rad(self.angle3)) + rect3.center[1]))

        arm_rotated4, rect4 = self.arm4.rotate(angle=self.angle4, pivot=pivot, gui=True)
        pivot = (int(self.arm5.offset * np.cos(np.deg2rad(self.angle4)) + rect4.center[0]), \
                 int(-self.arm5.offset * np.sin(np.deg2rad(self.angle4)) + rect4.center[1]))

        arm_rotated5, rect5 = self.arm5.rotate(angle=self.angle5, pivot=pivot, gui=True)
        end_effector = (int(self.arm5.offset * np.cos(np.deg2rad(self.angle5)) + rect5.center[0]), \
                        int(-self.arm5.offset * np.sin(np.deg2rad(self.angle5)) + rect5.center[1]))
        display.fill(white)
        display.blit(arm_rotated1, rect1)
        display.blit(arm_rotated2, rect2)
        display.blit(arm_rotated3, rect3)
        display.blit(arm_rotated4, rect4)
        display.blit(arm_rotated5, rect5)
        game.draw.circle(display, black, end_effector, 5)
        game.draw.circle(display, blue, self.goal, 4)
        game.display.update()
        fpsClock.tick(1 / self.sim_step)

    def reset(self):
        self.angle1 = self.arm1.angle
        self.angle2 = self.arm2.angle
        self.angle3 = self.arm3.angle
        self.angle4 = self.arm4.angle
        self.angle5 = self.arm5.angle
        self.joint1 = self.angle1
        self.joint2 = self.angle2 - self.joint1
        self.joint3 = self.angle3 - self.joint2
        self.joint4 = self.angle4 - self.joint3
        self.joint5 = self.angle5 - self.joint4
        self.states = []
        game.init()
        self.timer = 0.0
        return self.get_obs()

    def get_states(self):
        return self.states

    def hard_reset(self):
        self.reset()
        game.quit()

    def get_reward(self):
        self.state = self.get_obs()
        eff_state = self.state[5::]
        dist = np.linalg.norm(np.array(eff_state) - np.array(self.goal)) / float(window_width)
        return dist

    def get_commands(self):
        return self.actions