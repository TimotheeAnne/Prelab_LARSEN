import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import fast_adaptation_embedding.env
import fast_adaptation_embedding.models.embedding_nn as nn_model
from fast_adaptation_embedding.models.ffnn import FFNN_Ensemble_Model
from fast_adaptation_embedding.controllers.cem import CEM_opt
from fast_adaptation_embedding.controllers.random_shooting import RS_opt
import torch
import numpy as np
import copy
import gym
import time
import pickle
from os import path
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tqdm import tqdm
from time import time, localtime, strftime
import pprint
from scipy.io import savemat
import cma


def execute_random(env, steps, samples, K, config):
    current_state = env.reset()
    obs = [current_state]
    acs = []
    reward = []
    traject_cost = 0
    params = np.random.uniform(config['lb'], config['ub'])
    for i in tqdm(range(steps)):
        if config["controller"] is not None:
            t = env.minitaur.GetTimeSinceReset()
            a = config["controller"](t, config['omega'], params)
        else:
            a = env.action_space.sample()
        next_state, r = 0, 0
        for k in range(K):
            next_state, r, done, _ = env.step(a)
        obs.append(next_state)
        acs.append(a)
        reward.append(r)
        traject_cost += -r
        if done:
            break
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['controller'].append(np.copy(params))
    samples['reward'].append(np.copy(reward))
    return traject_cost


def execute_2(env, steps, init_var, model, config, pred_high, pred_low, index_iter, samples):
    current_state = env.reset()
    f_rec = config['video_recording_frequency']
    recorder = None
    if f_rec and index_iter % f_rec == (f_rec - 1):
        recorder = VideoRecorder(env, os.path.join(config['logdir'], "iter_" + str(index_iter) + ".mp4"))
    obs = [current_state]
    acs = []
    trajectory = []
    reward = []
    traject_cost = 0
    model_error = 0
    sliding_mean = np.zeros(config["sol_dim"])
    mutation = np.random.rand(config["sol_dim"]) * 2. * 0.5 - 0.5
    rand = np.random.rand(config["sol_dim"])
    mutation *= np.array([1.0 if r > 0.25 else 0.0 for r in rand])
    goal = None
    for i in tqdm(range(steps)):
        cost_object = config['Cost_ensemble'](ensemble_model=model, init_state=current_state, horizon=config["horizon"],
                                              action_dim=env.action_space.shape[0], goal=goal, pred_high=pred_high,
                                              pred_low=pred_low, config=config)
        config["cost_fn"] = cost_object.cost_fn
        if config['opt'] == "RS":
            optimizer = RS_opt(config)
            sol = optimizer.obtain_solution()
        elif config['opt'] == "CEM":
            optimizer = CEM_opt(config)
            sol = optimizer.obtain_solution(sliding_mean, init_var)
        elif config['opt'] == "CMA-ES":
            xopt, es = cma.fmin2(None, np.zeros(config["sol_dim"]), 0.5,
                                 parallel_objective=lambda x: list(config["cost_fn"](x)),
                                 options={'maxfevals': config['max_iters'] * config['popsize'],
                                          'popsize': config['popsize']})
            sol = xopt
        a = sol[0:env.action_space.shape[0]]
        next_state, r = 0, 0
        if recorder is not None:
            recorder.capture_frame()
        for k in range(config["K"]):
            next_state, r, _, _ = env.step(a)
            obs.append(next_state)
            acs.append(a)
            reward.append(r)
        trajectory.append([current_state.copy(), a.copy(), next_state - current_state, -r])
        model_error += test_model(model, current_state.copy(), a.copy(), next_state - current_state)
        current_state = next_state
        traject_cost += -r
        sliding_mean[0:-len(a)] = sol[len(a)::]
    print("Model error: ", model_error / steps)
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['reward_sum'].append(-traject_cost)
    samples['model_error'].append(model_error / steps)
    return np.array(trajectory), traject_cost


def execute_3(env, steps, init_var, model, config, pred_high, pred_low, index_iter, samples):
    current_state = env.reset()
    controller = config["controller"]
    f_rec = config['video_recording_frequency']
    recorder = None
    if f_rec and (index_iter == 1 or index_iter % f_rec == (f_rec - 1)):
        recorder = VideoRecorder(env, os.path.join(config['logdir'], "iter_" + str(index_iter) + ".mp4"))
    obs = [current_state]
    acs, trajectory, reward, control_sol, motor_actions = [], [], [], [], []
    traject_cost = 0
    model_error = np.zeros(len(model.get_models()))
    sliding_mean = np.zeros(config["sol_dim"])
    goal = None
    omega = config['omega']
    for i in tqdm(range(steps)):
        t = env.minitaur.GetTimeSinceReset()
        if i % config["optimizer_frequency"] == 0:
            cost_object = config['Cost_ensemble'](ensemble_model=model, init_state=current_state,
                                                  horizon=config["horizon"],
                                                  action_dim=env.action_space.shape[0], goal=goal, pred_high=pred_high,
                                                  pred_low=pred_low, config=config)
            config["cost_fn"] = cost_object.cost_fn
            if config['opt'] == "RS":
                optimizer = RS_opt(config)
                if i == 0:
                    sol = optimizer.obtain_solution(t0=t)
                else:
                    sol = optimizer.obtain_solution(init_mean=sliding_mean, init_var=init_var, t0=t)
            elif config['opt'] == "CEM":
                optimizer = CEM_opt(config)
                sol = optimizer.obtain_solution(sliding_mean, init_var)
            elif config['opt'] == "CMA-ES":
                xopt, es = cma.fmin2(None, np.zeros(config["sol_dim"]), 0.5,
                                     parallel_objective=lambda x: list(config["cost_fn"](x)),
                                     options={'maxfevals': config['max_iters'] * config['popsize'],
                                              'popsize': config['popsize']})
                sol = xopt
        a = controller(t, omega, sol)
        next_state, r = 0, 0
        if recorder is not None:
            recorder.capture_frame()
        for k in range(config["K"]):
            next_state, r, done, motor_action = env.step(a)
        obs.append(next_state)
        acs.append(a)
        reward.append(r)
        control_sol.append(np.copy(sol))
        motor_actions.append(np.copy(motor_action['action']))
        trajectory.append([current_state.copy(), a.copy(), next_state - current_state, -r])
        model_error += test_model(model, current_state.copy(), a.copy(), next_state - current_state)
        current_state = next_state
        traject_cost += -r
        sliding_mean = sol
        if done:
            break
    print("Model error: ", model_error / (i + 1))
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['reward_sum'].append(-traject_cost)
    samples['model_error'].append(model_error / (i + 1))
    samples['controller_sol'].append(np.copy(control_sol))
    samples['motor_actions'].append(np.copy(motor_actions))
    return np.array(trajectory), traject_cost


def collect(config):
    logdir = os.path.join(config['logdir'],
                          strftime("%Y-%m-%d--%H:%M:%S", localtime()) + str(np.random.randint(10 ** 5)))
    config['logdir'] = logdir
    os.makedirs(logdir)
    with open(os.path.join(config['logdir'], "config.txt"), 'w') as f:
        f.write(pprint.pformat(config))
    n_task = 1
    envs = [gym.make(config['env'], **config['env_args']) for i in range(n_task)]
    samples = {'acs': [], 'obs': [], 'reward': [], "controller": []}

    for index_iter in range(config["iterations"]):
        '''Pick a random environment'''
        env_index = int(index_iter % n_task)
        env = envs[env_index]
        print("Episode: ", index_iter)
        c = execute_random(env=env, steps=config["episode_length"], samples=samples,
                           K=config["K"], config=config)
        print("cost :", c)
        print("-------------------------------\n")

    with open(config['save_data'] + '.pk', 'wb') as f:
        pickle.dump(samples, f)

