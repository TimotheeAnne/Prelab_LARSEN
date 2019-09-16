import gym
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pickle
import roboschool
import time
import pybullet
from tqdm import tqdm
import sys
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tqdm import tqdm

sys.path.insert(0, '/home/tanne-local/Documents/handful-of-trials/')
sys.path.insert(0, '..')
import dmbrl.env

HC_PETS_1 = "working_basic_HC"
HC_PETS_2 = "working_basic_HC_video_recorder"

PBHC_PETS_1 = "PBHC"

acs = "2019-09-16--14:12:16"

folder = acs

""" PETS saving files """
data = scipy.io.loadmat("/home/timothee/Documents/handful-of-trials/scripts/log/Saved/"+folder+"/logs.mat")
actions = data['actions']
rewards = data['returns'][0]


# file = "/home/timothee/Documents/gym-kinematicArm/HC_actions.pk"
# with open(file, 'rb') as f:
#     [actions, observations] = pickle.load(f)


def plot_arm():
    actions = data['actions']
    goals = data['goals']
    env = gym.make('kinematicArm-v0')


    for j in range(0,len(actions),5):
        goal = (np.array(goals[j])+1)*400
        goal = (int(goal[0]), int(goal[1]))
        env.reset(goal=goal)
        env.render(mode='human')
        for i in range(50):
            a = actions[j][i]
            obs, rew, _, _ = env.step(a)
            env.render()


def print_HC_obs(obs):
    print("x: ", obs[0], "z:", obs[1], "y: ", obs[2])
    print("vx: ", obs[9], "vz:", obs[10], "vy: ", obs[11])
    print("bthight: ", obs[3], "bshin: ", obs[4], "bfoot: ", obs[5])
    print("vbthight: ", obs[12], "vbshin: ", obs[13], "vbfoot: ", obs[14])
    print("fthight: ", obs[6], "fshin: ", obs[7], "ffoot: ", obs[8])
    print("vfthight: ", obs[15], "vfshin: ", obs[16], "vffoot: ", obs[17])
    print("________________________________")


def plot_bipedal(j=None):
    env = gym.make('BipedalWalker-v2')
    env.reset()
    env.render(mode='human')
    N = 100 if j is None else len(actions[j])
    print("Reward:", rewards[j])
    for i in range(N):
        if j is None:
            a = np.random.random(4)
        else:
            a = actions[j][i]
        obs, rew, done, _ = env.step(a)
        time.sleep(0.01)
        if done:
            break
        env.render()


def plot_hopper(j=None):
    env = gym.make('RoboschoolHopper-v1')
    print(env.reset())
    env.render(mode='human')
    N = 100 if j is None else len(actions[j])
    # print("Reward:", rewards[j])
    for i in range(N):
        if j is None:
            a = np.random.random(3)
        else:
            a = actions[j][i]
        obs, rew, done, _ = env.step(a)
        time.sleep(0.01)
        if done:
            break
        env.render()


def plot_cheetah(j=None):
    """
    PETS: MBRLHalfCheetah-v0
    Roboschool: RoboschoolHalfCheetah-v1
    Mine: RoboschoolHalfCheetah-v1

    :param j:
    :return:
    """
    # env = gym.make('MBRLHalfCheetah-v0')
    env = gym.make('PybulletHalfCheetahMuJoCoEnv-v0')
    render = True
    # render = False
    if render:
        env.render(mode='human')
    Obs = []
    obs = env.reset()
    Obs.append(obs)
    print_HC_obs(obs)
    video_recorder = None
    # video_recorder = VideoRecorder(env, "test.mp4")
    N = 100 if j is None else len(actions[j])
    tqdm = lambda x: x
    reward = 0
    for i in tqdm(range(N)):
        if video_recorder is not None:
            env.camera_adjust()
            video_recorder.capture_frame()
        if j is None:
            a = np.random.random(6)*2-1
            # a = [0, 0, 1, 0, 0, 0]
        else:
            a = actions[j][i]
        obs, rew, done, info = env.step(a)
        Obs.append(obs)
        # print(a)
        # print_HC_obs(obs)
        # time.sleep(0.05)
        reward += rew
        if done:
            break
        if render:
            env.render(mode='human')
    if video_recorder is not None:
        video_recorder.capture_frame()
        video_recorder.close()
    env.close()
    with open("./log/test/HC2.pk", 'wb') as f:
        pickle.dump(Obs, f)
    print("Final reward", reward)

def plot_Ant(j=None):
    env = gym.make('AntMuJoCoEnv_fastAdapt-v0')
    render = True
    # render = False
    if render:
        env.render(mode='human')
    Obs = []
    obs = env.reset()
    Obs.append(obs)
    # print(obs)
    video_recorder = None
    # video_recorder = VideoRecorder(env, "test.mp4")
    N = 1 if j is None else len(actions[j])
    tqdm = lambda x: x
    reward = 0
    for i in tqdm(range(N)):
        if video_recorder is not None:
            env.camera_adjust()
            video_recorder.capture_frame()
        if j is None:
            a = np.random.random(8)*2-1
            a = [0, 1, 0, 1, 0, 1, 0, 1]
        else:
            a = actions[j][i]
        obs, rew, done, info = env.step(a)
        # print(obs)
        time.sleep(0.01)
        reward += rew
        if done:
            break
        if render:
            env.render(mode='human')
    if video_recorder is not None:
        video_recorder.capture_frame()
        video_recorder.close()
    env.close()
    # with open("./log/test/HC2.pk", 'wb') as f:
    #     pickle.dump(Obs, f)
    print("Final reward", reward)

def plot_reward():
    plt.plot(rewards, lw=3, label="gym reward")
    plt.legend()
    # plt.yscale('log')
    plt.show()

# print(np.argmax(ep_length))
plot_Ant(4)
# plot_reward()
# print(np.min(observations[:-2], axis=0)[0])
#