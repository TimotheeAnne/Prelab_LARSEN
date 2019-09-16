from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes import StadiumScene
from pybulletgym.envs.mujoco.robots.locomotors.walker_base import WalkerBase
from pybulletgym.envs.mujoco.robots.robot_bases import XmlBasedRobot
import pybullet
import numpy as np
import os


class MJCFBasedRobot(XmlBasedRobot):
    """
    Base class for mujoco .xml based agents.
    """

    def __init__(self, model_xml, robot_name, action_dim, obs_dim, self_collision=True, add_ignored_joints=False):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision, add_ignored_joints)
        self.model_xml = model_xml
        self.doneLoading = 0

    def reset(self, bullet_client):

        # full_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "mjcf", self.model_xml)
        full_path = os.path.join(os.path.dirname(__file__), "assets",  self.model_xml)

        self._p = bullet_client
        # print("Created bullet_client with id=", self._p._client)
        if self.doneLoading == 0:
            self.ordered_joints = []
            self.doneLoading = 1
            if self.self_collision:
                self.objects = self._p.loadMJCF(full_path,
                                                flags=pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(full_path)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)
        self.robot_specific_reset(self._p)

        s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s

    def calc_potential(self):
        return 0


class Ant(WalkerBase, MJCFBasedRobot):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self):
        WalkerBase.__init__(self, power=2.5)
        MJCFBasedRobot.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=27)

    def calc_state(self):
        WalkerBase.calc_state(self)
        pose = self.parts['torso'].get_pose()
        qpos = np.hstack((pose, [j.get_position() for j in self.ordered_joints])).flatten()  # shape (15,)

        velocity = self.parts['torso'].get_velocity()
        qvel = np.hstack(
            (velocity[0], velocity[1], [j.get_velocity() for j in self.ordered_joints])).flatten()  # shape (14,)

        return np.concatenate([
            qpos.flat[2:],  # self.sim.data.qpos.flat[2:],
            qvel.flat  # self.sim.data.qvel.flat,
        ])

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        for key in self.jdict.keys():
            self.jdict[key].power_coef = 400.0  # joint effort


class AntMuJoCoEnv(WalkerBaseMuJoCoEnv):
    def __init__(self):
        self.robot = Ant()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot)
        self.mismatch = np.ones(8)

    def step(self, a):
        act = self.mismatch * a
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(act)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        # electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        # electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if debugmode:
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            # print("electricity_cost")
            # print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        # self.rewards = [
        #     alive,
        #     progress,
        #     #electricity_cost,
        #     joints_at_limit_cost,
        #     feet_collision_cost
        # ]
        self.rewards = [
            # alive,
            # state[0]*0.01, #height
            state[13]  # x lelocity
            # electricity_cost,
            # joints_at_limit_cost,
            # feet_collision_cost
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def __str__(self):
        s = "\n---------Ant environment object---------------\n"
        s += "Custom Ant Environment similar to Mujoco's env, but with pybullet.\n"
        s += "Action dim 8 (for 8 joints)\n"
        s += "Observation dim 27 (Reduced from 111 dim since remaining values are zero)\n"
        s += "\ndim 0           : Height\n"
        s += "dim 1 to dim 4  : Body pose (Quaternion)\n"
        s += "dim 5 to dim 12 : Joint positions\n"
        s += "dim 13 to dim 15: body x,y,z linear velocity (verify!!)\n"
        s += "dim 16 to dim 18: body x,y,z angular velocity (verify!!)\n"
        s += "dim 19 to dim 26: joint velocities\n"
        s += "---------------------------------------------\n"
        return s

    def get_reward(self, prev_obs, action, next_obs):
        return next_obs[13]

    def set_mismatch(self, mismatch=np.ones(8)):
        self.mismatch = mismatch


if __name__ == "__main__":
    import gym
    import time
    import sys
    sys.path.insert(0, '../..')
    import dmbrl.env
    system = gym.make("AntMuJoCoEnv_fastAdapt-v0")
    system.render(mode="human")
    import time


    system.reset()
    rew = 0
    for i in range(200):
        actions = np.random.random(8)*2-1
        _, r, _, _ = system.step(actions)
        rew += r
        time.sleep(0.05)
    print(rew)
# /home/rkaushik/projects/fast_adaptation_embedding/arm_data/best_action_seq_ant.npy
