from pybulletgym.envs.mujoco.robots.locomotors.walker_base import WalkerBase
from pybulletgym.envs.mujoco.robots.robot_bases import MJCFBasedRobot
from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.robots.locomotors.half_cheetah import HalfCheetah
from pybulletgym.envs.mujoco.robots.robot_bases import XmlBasedRobot
import numpy as np

import os
import pybullet

DEFAULT_SIZE = 500


class HalfCheetah(WalkerBase, MJCFBasedRobot):
    """
    Half Cheetah implementation based on MuJoCo.
    """
    foot_list = ["ffoot", "fshin", "fthigh", "bfoot", "bshin", "bthigh"]  # track these contacts with ground

    def __init__(self):
        WalkerBase.__init__(self, power=1)
        MJCFBasedRobot.__init__(self, "half_cheetah.xml", "torso", action_dim=6, obs_dim=18, add_ignored_joints=True)
        self.pos_after = 0
        self.prev_qpos = None

    def calc_state(self):
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (9,)
        qvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (9,)
        state = np.concatenate([
            (qpos.flat[:1] - self.prev_qpos[:1]) / self.scene.dt,
            qpos.flat[1:],
            qvel.flat,
        ])
        self.body_xyz = [qpos.flat[0], qpos.flat[2], qpos.flat[1]]
        self.prev_qpos = np.copy(qpos)
        return state

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        pos_before = self.pos_after
        self.pos_after = self.robot_body.get_pose()[0]
        debugmode = 0
        if debugmode:
            print("calc_potential: self.walk_target_dist")
            print(self.walk_target_dist)
            print("self.scene.dt")
            print(self.scene.dt)
            print("self.scene.frame_skip")
            print(self.scene.frame_skip)
            print("self.scene.timestep")
            print(self.scene.timestep)
        return (self.pos_after - pos_before) / self.scene.dt

    def calc_survival(self):
        survival_rew = 0 if self.body_xyz[2] > -0.2 else -1
        return survival_rew

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        for part_id, part in self.parts.items():
            self._p.changeDynamics(part.bodyIndex, part.bodyPartIndex, lateralFriction=0.8, spinningFriction=0.1,
                                   rollingFriction=0.1, restitution=0.5)

        # self.jdict["bthigh"].power_coef = 120.0
        # self.jdict["bshin"].power_coef = 90.0
        # self.jdict["bfoot"].power_coef = 60.0
        # self.jdict["fthigh"].power_coef = 160.0
        # self.jdict["fshin"].power_coef = 110.0
        # self.jdict["ffoot"].power_coef = 80.0
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        self.prev_qpos = np.copy(qpos)

class PybulletHalfCheetahMuJoCoEnv(WalkerBaseMuJoCoEnv):
    def __init__(self):
        self.robot = HalfCheetah()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot)


    def step(self, a):
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        potential = self.robot.calc_potential()
        survival = self.robot.calc_survival()
        power_cost = -0. * np.square(a).sum()
        state = self.robot.calc_state()

        done = False

        debugmode = 0
        if debugmode:
            print("potential=")
            print(potential)
            print("power_cost=")
            print(power_cost)

        self.rewards = [
            potential,
            power_cost,
            survival
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}


if __name__ == "__main__":
    import gym
    import time
    import sys
    sys.path.insert(0, '../..')
    import dmbrl.env
    # system = gym.make("PybulletHalfCheetahMuJoCoEnv-v0")
    # system = gym.make("HalfCheetahBulletEnv-v0")
    system.render(mode="human")
    import time
    actions = np.random.random(6)*10-5
    system.reset()
    rew = 0
    for i in range(200):
        _, r, _, _ = system.step(actions)
        rew += r
        time.sleep(0.05)
    print(rew)
