from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes import StadiumScene
from pybulletgym.envs.mujoco.robots.locomotors.walker_base import WalkerBase
from pybulletgym.envs.mujoco.robots.robot_bases import XmlBasedRobot
import pybullet
import numpy as np
import os

import sys

sys.path.insert(0, '/home/timothee/Documents/Prelab_LARSEN/fast_adaptation_embedding')
sys.path.insert(0, '/home/tanne-local/Documents/Prelab_LARSEN/fast_adaptation_embedding')


if __name__ == "__main__":
    import gym
    import time
    import fast_adaptation_embedding.env
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
    render = True
    # render = False
    system = gym.make("MinitaurBulletEnv-v0", render=render)
    recorder = None
    # recorder = VideoRecorder(system, "test.mp4")
    system.reset()
    rew = 0
    for i in range(200):
        if recorder is not None:
            recorder.capture_frame()
        a = np.random.random(8) * 2 - 1
        _, r, _, _ = system.step(a)
        rew += r
        time.sleep(0.01)
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    print(rew)

