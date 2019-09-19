import pygame as game
import numpy as np
import gym
from gym import spaces
import os

black = (0, 0, 0)
white = (255, 255, 255) 
blue = (0, 0, 255)
window_width = 2000
window_height = 1200
class arm:
    def __init__(self, img, pivot_adjust=-33.0, scale = 1.0):
        self.base_arm = game.image.load(img)
        self.base_center = (int(window_width/2), int(window_height/2))
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
                -self.offset*np.sin(np.deg2rad(angle)) + pivot[1])
            return rotated_arm, rect
        else:
            return (int(self.offset * np.cos(np.deg2rad(angle)) + pivot[0]), \
                int(-self.offset*np.sin(np.deg2rad(angle)) + pivot[1]))

class Arm_env(gym.Env):
    def __init__(self, goal=(0.5, 0.5), joint_mismatch=np.ones(5)):
        dirname = os.path.dirname(__file__)
        arm_img_path = os.path.join(dirname, 'assets/arm.png')
        self.arm1 = arm(arm_img_path, scale=0.5, pivot_adjust=-10)
        self.angle1 = self.arm1.angle
        self.joint1 = self.angle1

        self.arm2 = arm(arm_img_path, scale=0.5, pivot_adjust=-10)
        self.angle2 = self.arm2.angle
        self.joint2 = self.angle2 - self.joint1

        self.arm3 = arm(arm_img_path, scale=0.5, pivot_adjust=-10)
        self.angle3 = self.arm3.angle
        self.joint3 = self.angle3 - self.joint2

        self.arm4 = arm(arm_img_path, scale=0.5, pivot_adjust=-10)
        self.angle4 = self.arm4.angle
        self.joint4 = self.angle4 - self.joint3

        self.arm5 = arm(arm_img_path, scale=0.5, pivot_adjust=-10)
        self.angle5 = self.arm5.angle
        self.joint5 = self.angle5 - self.joint4

        self.states = [] 
        self.real_goal = (np.array(goal) + 1.0) * 0.5 * np.array([float(window_width), float(window_height)])
        self.normalized_goal = goal

        self.gui = False
        self.display = None

        self.sim_step = 3.0

        self.joint_mismatch=joint_mismatch
        self.action_space = spaces.Box(low= -np.ones(5), high=np.ones(5))
        # self.observation_space = spaces.Box(low= np.array([-100.,-100., -100.,-100., -100., 0.,  0.]), high=np.array([100.,100., 100.,100., 100.,float(window_width), float(window_height)])) #(x, y, sin_theta, cos_theta)1
        self.observation_space = spaces.Box(low= np.array([-1.,-1., -1.,-1., -1., -1.,  -1.]), high=np.array([1., 1., 1.,1., 1., 1., 1.])) #(x, y, sin_theta, cos_theta)1

    def normalize(self, state):
        space_range = np.array([200., 200., 200., 200., 200., float(window_width), float(window_height)])
        return (state/space_range) * np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) - np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]) 

    def set_goal(self, goal):
        self.real_goal = (np.array(goal) + 1.0) * 0.5 * np.array([float(window_width), float(window_height)])
        self.normalized_goal = goal

    def render(self,mode=None):
        if mode is not None:
            self.gui = True
            self.display = game.display.set_mode((window_width, window_height))

        if self.gui == True:
            arm_rotated1, rect1 = self.arm1.rotate(self.angle1, gui=True)
            pivot = (int(self.arm2.offset * np.cos(np.deg2rad(self.angle1)) + rect1.center[0]),  \
                int(-self.arm2.offset*np.sin(np.deg2rad(self.angle1)) + rect1.center[1]))
            
            arm_rotated2, rect2 = self.arm2.rotate(angle=self.angle2, pivot=pivot, gui=True)
            pivot = (int(self.arm3.offset * np.cos(np.deg2rad(self.angle2)) + rect2.center[0]),  \
                int(-self.arm3.offset*np.sin(np.deg2rad(self.angle2)) + rect2.center[1]))
            
            arm_rotated3, rect3 = self.arm3.rotate(angle=self.angle3, pivot=pivot, gui=True)
            pivot = (int(self.arm4.offset * np.cos(np.deg2rad(self.angle3)) + rect3.center[0]),  \
                int(-self.arm4.offset*np.sin(np.deg2rad(self.angle3)) + rect3.center[1]))

            arm_rotated4, rect4 = self.arm4.rotate(angle=self.angle4, pivot=pivot, gui=True)
            pivot = (int(self.arm5.offset * np.cos(np.deg2rad(self.angle4)) + rect4.center[0]),  \
                int(-self.arm5.offset*np.sin(np.deg2rad(self.angle4)) + rect4.center[1]))

            arm_rotated5, rect5 = self.arm5.rotate(angle=self.angle5, pivot=pivot, gui=True)
            end_effector = (int(self.arm5.offset * np.cos(np.deg2rad(self.angle5)) + rect5.center[0]),  \
                int(-self.arm5.offset*np.sin(np.deg2rad(self.angle5)) + rect5.center[1]))
            
            self.display.fill(white)
            self.display.blit(arm_rotated1, rect1)
            self.display.blit(arm_rotated2, rect2)
            self.display.blit(arm_rotated3, rect3)
            self.display.blit(arm_rotated4, rect4)
            self.display.blit(arm_rotated5, rect5)
            game.draw.circle(self.display, black, end_effector, 30)
            game.draw.circle(self.display, blue, self.real_goal.astype(int), 20)
            
            game.display.update()
        
    def step(self, act):
        action = self.joint_mismatch * act
        self.joint1 += self.sim_step * action[0]
        self.joint1 = np.clip(self.joint1, -100.0, 100.0)
        self.angle1 = self.joint1

        self.joint2 += self.sim_step * action[1]
        self.joint2 = np.clip(self.joint2, -100.0, 100.0)
        self.angle2 = self.joint1 + self.joint2
        
        self.joint3 += self.sim_step * action[2]
        self.joint3 = np.clip(self.joint3, -100.0, 100.0)
        self.angle3 =self.joint1 + self.joint2 + self.joint3
        
        self.joint4 += self.sim_step * action[3]
        self.joint4 = np.clip(self.joint4, -100.0, 100.0)
        self.angle4 = self.joint1 + self.joint2 + self.joint3 + self.joint4
        
        self.joint5 += self.sim_step * action[4]
        self.joint5 = np.clip(self.joint5, -100.0, 100.0)
        self.angle5 = self.joint1 + self.joint2 + self.joint3 + self.joint4 + self.joint5
        

        center1 = self.arm1.rotate(self.angle1, gui=False)
        pivot = (int(self.arm2.offset * np.cos(np.deg2rad(self.angle1)) + center1[0]),  \
            int(-self.arm2.offset*np.sin(np.deg2rad(self.angle1)) + center1[1]))

        center2 = self.arm2.rotate(angle=self.angle2, pivot=pivot, gui=False)
        pivot = (int(self.arm3.offset * np.cos(np.deg2rad(self.angle2)) + center2[0]),  \
            int(-self.arm3.offset*np.sin(np.deg2rad(self.angle2)) + center2[1]))
        
        center3 = self.arm3.rotate(angle=self.angle3, pivot=pivot, gui=False)
        pivot = (int(self.arm4.offset * np.cos(np.deg2rad(self.angle3)) + center3[0]),  \
            int(-self.arm4.offset*np.sin(np.deg2rad(self.angle3)) + center3[1]))

        center4 = self.arm4.rotate(angle=self.angle4, pivot=pivot, gui=False)
        pivot = (int(self.arm5.offset * np.cos(np.deg2rad(self.angle4)) + center4[0]),  \
            int(-self.arm5.offset*np.sin(np.deg2rad(self.angle4)) + center4[1]))

        center5 = self.arm5.rotate(angle=self.angle5, pivot=pivot, gui=False)
        end_effector = (int(self.arm5.offset * np.cos(np.deg2rad(self.angle5)) + center5[0]),  \
            int(-self.arm5.offset*np.sin(np.deg2rad(self.angle5)) + center5[1]))
    
        self.state = np.array([self.joint1, self.joint2, self.joint3, self.joint4, self.joint5, end_effector[0], end_effector[1]])
        
        # for event in game.event.get():
        #     if event.type == game.QUIT:
        #         game.quit()

        eff_state = self.normalize(self.state)[5::]
        dist = np.linalg.norm(eff_state - self.normalized_goal)
        done= False
        if self.gui == True:
            self.render()
        return self.normalize(self.state), -dist, done, {}

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

        center1 = self.arm1.rotate(self.angle1, gui=False)
        pivot = (int(self.arm2.offset * np.cos(np.deg2rad(self.angle1)) + center1[0]),  \
            int(-self.arm2.offset*np.sin(np.deg2rad(self.angle1)) + center1[1]))

        center2 = self.arm2.rotate(angle=self.angle2, pivot=pivot, gui=False)
        pivot = (int(self.arm3.offset * np.cos(np.deg2rad(self.angle2)) + center2[0]),  \
            int(-self.arm3.offset*np.sin(np.deg2rad(self.angle2)) + center2[1]))
        
        center3 = self.arm3.rotate(angle=self.angle3, pivot=pivot, gui=False)
        pivot = (int(self.arm4.offset * np.cos(np.deg2rad(self.angle3)) + center3[0]),  \
            int(-self.arm4.offset*np.sin(np.deg2rad(self.angle3)) + center3[1]))

        center4 = self.arm4.rotate(angle=self.angle4, pivot=pivot, gui=False)
        pivot = (int(self.arm5.offset * np.cos(np.deg2rad(self.angle4)) + center4[0]),  \
            int(-self.arm5.offset*np.sin(np.deg2rad(self.angle4)) + center4[1]))

        center5 = self.arm5.rotate(angle=self.angle5, pivot=pivot, gui=False)
        end_effector = (int(self.arm5.offset * np.cos(np.deg2rad(self.angle5)) + center5[0]),  \
            int(-self.arm5.offset*np.sin(np.deg2rad(self.angle5)) + center5[1]))

        self.state = self.state = np.array([self.joint1, self.joint2, self.joint3, self.joint4, self.joint5, end_effector[0], end_effector[1]])
        return self.normalize(self.state)

    def hard_reset(self):
        self.reset()
        game.quit()

if __name__ == "__main__":
    system = Arm_env(goal=(0, 0.5))
    system.render(mode="human")
    import time
    # actions = np.load("best_action_seq.npy")
    # print(actions)
    system.reset()
    for i in range(50):
        obs = None
        for k in range(100):
            if i%2 == 0:
                obs, r ,_ ,_ = system.step(np.ones(5))
            else:
               obs, r ,_ ,_ =  system.step(-np.ones(5))
            time.sleep(0.01)
        print(obs[5::], r)
            # system.step(action=actions[i*5 : i*5 + 5])
    