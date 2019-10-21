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



class Evaluation_ensemble(object):
    def __init__(self, config, ensemble_model=None, pred_high=None, pred_low=None):
        self.__ensemble_model = ensemble_model
        self.__horizon = config['horizon']
        self.__action_dim = config['action_dim']
        self.__pred_high = pred_high
        self.__pred_low = pred_low
        self.__obs_dim = config['ensemble_dim_out']
        self.__episode_length = config['episode_length']
        self.__inputs = np.zeros((0, config['ensemble_dim_in']))
        self.__outputs = np.zeros((0, config['ensemble_dim_out']))

    def preprocess_data_traj(self, traj_obs, traj_acs):
        N = len(traj_acs)
        actions, init_observations, observations = [], [], []
        for i in range(N):
            for t in range(0, len(traj_acs[i]) - self.__horizon, self.__horizon):
                actions.append(traj_acs[i][t:t + self.__horizon].flatten())
                init_observations.append(traj_obs[i][t])
                observations.append(traj_obs[i][t + 1:t + 1 + self.__horizon].flatten())
        return np.array(actions), np.array(init_observations), np.array(observations)

    def add_sample(self, obs, acs):
        assert (len(obs) == (len(acs) + 1))
        for t in range(len(acs)):
            self.__inputs = np.concatenate((self.__inputs, [np.concatenate((obs[t], acs[t]))]))
            self.__outputs = np.concatenate((self.__outputs, [obs[t + 1] - obs[t]]))

    def get_data(self):
        return self.__inputs, self.__outputs

    def set_data(self, inputs, outputs):
        self.__inputs = inputs
        self.__outputs = outputs

    def eval_traj(self, actions, init_states, observations):
        action_batch = torch.FloatTensor(actions).cuda() \
            if self.__ensemble_model.CUDA \
            else torch.FloatTensor(actions)
        traj_states = torch.FloatTensor(observations).cuda() \
            if self.__ensemble_model.CUDA \
            else torch.FloatTensor(observations)
        error = torch.FloatTensor(np.zeros((len(self.__models), len(actions)))).cuda() \
            if self.__ensemble_model.CUDA \
            else torch.FloatTensor(np.zeros((len(self.__models), len(actions))))
        one_step_error = torch.FloatTensor(np.zeros((len(self.__models), len(actions)))).cuda() \
            if self.__ensemble_model.CUDA \
            else torch.FloatTensor(np.zeros((len(self.__models), len(actions))))


        for model_index in range(len(self.__models)):
            dyn_model = self.__models[model_index]
            start_states = torch.FloatTensor(init_states).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(
                init_states)
            for h in range(self.__horizon):
                actions = action_batch[:, h * self.__action_dim: h * self.__action_dim + self.__action_dim]
                model_input = torch.cat((start_states, actions), dim=1)
                diff_state = dyn_model.predict_tensor(model_input)
                start_states += diff_state
                for dim in range(self.__obs_dim):
                    start_states[:, dim].clamp_(self.__pred_low[dim], self.__pred_high[dim])
                pred_error = torch.sqrt(
                    (start_states - traj_states[:, h * self.__obs_dim: h * self.__obs_dim + self.__obs_dim]).pow(2).sum(
                        1))
                state_norm = torch.sqrt(
                    (traj_states[:, h * self.__obs_dim: h * self.__obs_dim + self.__obs_dim]).pow(2).sum(1))
                error[model_index] += (pred_error / state_norm) / self.__horizon
                if h == 0:
                    one_step_error[model_index] = (pred_error / state_norm)
        return error.cpu().detach().numpy(), one_step_error.cpu().detach().numpy()

    def eval_model(self, ensemble):
        models = ensemble.get_models()
        inputs = torch.FloatTensor(self.__inputs).cuda()
        outputs = torch.FloatTensor(self.__outputs).cuda()
        error = torch.FloatTensor(np.zeros((len(models), len(self.__inputs)))).cuda()
        error0 = torch.FloatTensor(np.zeros((len(models), len(self.__inputs)))).cuda()

        for model_index in range(len(models)):
            dyn_model = models[model_index]
            error[model_index], error0[model_index] = dyn_model.compute_error(inputs, outputs)
        return error.cpu().detach().numpy(), error0.cpu().detach().numpy()

def process_data(data):
    # Assuming dada: an array containing [state, action, state_transition, cost]
    training_in = []
    training_out = []
    for d in data:
        s = d[0]
        a = d[1]
        training_in.append(np.concatenate((s, a)))
        training_out.append(d[2])
    return np.array(training_in), np.array(training_out), np.max(training_in, axis=0), np.min(training_in, axis=0)


def execute_random(env, steps, samples, K, config):
    current_state = env.reset()
    obs = [current_state]
    acs = []
    trajectory = []
    reward = []
    traject_cost = 0
    for i in tqdm(range(steps)):
        if config["controller"] is not None:
            t = env.minitaur.GetTimeSinceReset()
            a = config["controller"](t, config['omega'], np.random.uniform(config['lb'], config['ub']))
        else:
            a = env.action_space.sample()
        next_state, r = 0, 0
        for k in range(K):
            next_state, r, done, _ = env.step(a)
        obs.append(next_state)
        acs.append(a)
        reward.append(r)
        trajectory.append([current_state.copy(), a.copy(), next_state-current_state, -r])
        current_state = next_state
        traject_cost += -r
        if done:
            break
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['reward_sum'].append(-traject_cost)
    return np.array(trajectory), traject_cost


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
        trajectory.append([current_state.copy(), a.copy(), next_state-current_state, -r])
        model_error += test_model(model, current_state.copy(), a.copy(), next_state-current_state)
        current_state = next_state
        traject_cost += -r
        sliding_mean[0:-len(a)] = sol[len(a)::]
    print("Model error: ", model_error/steps)
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['reward_sum'].append(-traject_cost)
    samples['model_error'].append(model_error/steps)
    return np.array(trajectory), traject_cost


def execute_3(env, steps, init_var, model, config, pred_high, pred_low, index_iter, samples):
    current_state = env.reset()
    controller = config["controller"]
    f_rec = config['video_recording_frequency']
    recorder = None
    if f_rec and  (index_iter ==1 or index_iter % f_rec == (f_rec - 1)):
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
            cost_object = config['Cost_ensemble'](ensemble_model=model, init_state=current_state, horizon=config["horizon"],
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
        trajectory.append([current_state.copy(), a.copy(), next_state-current_state, -r])
        model_error += test_model(model, current_state.copy(), a.copy(), next_state-current_state)
        current_state = next_state
        traject_cost += -r
        sliding_mean = sol
        if done:
            break
    print("Model error: ", model_error/(i+1))
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['reward_sum'].append(-traject_cost)
    samples['model_error'].append(model_error/(i+1))
    samples['controller_sol'].append(np.copy(control_sol))
    samples['motor_actions'].append(np.copy(motor_actions))
    return np.array(trajectory), traject_cost


def test_model(ensemble_model, init_state, action, state_diff):
    x = np.concatenate(([init_state], [action]), axis=1)
    y = state_diff.reshape(1, -1)
    error = []
    for model_index in range(len(ensemble_model.get_models())):
        y_pred = ensemble_model.get_models()[model_index].predict(x)
        error.append(np.linalg.norm(y-y_pred)/np.linalg.norm(y))
    return np.array(error)


def extract_action_seq(data):
    actions = []
    for d in data:
        actions += d[1].tolist()
    return np.array(actions)


def main(config):
    logdir = os.path.join(config['logdir'], strftime("%Y-%m-%d--%H:%M:%S", localtime()) + str(np.random.randint(10**5)))
    config['logdir'] = logdir
    os.makedirs(logdir)
    with open(os.path.join(config['logdir'], "config.txt"), 'w') as f:
        f.write(pprint.pformat(config))
    n_task = 1
    envs = [gym.make(config['env'], **config['env_args']) for i in range(n_task)]
    envs[0].metadata["video.frames_per_second"] = config['video.frames_per_second']
    data = n_task * [None]

    traj_obs, traj_acs, traj_rets, traj_rews, traj_error, traj_eval, traj_sol, traj_motor = [], [], [], [], [], [], [], []

    evaluator = Evaluation_ensemble(config=config)

    for index_iter in range(config["iterations"]):
        '''Pick a random environment'''
        env_index = int(index_iter % n_task)
        env = envs[env_index]

        print("Episode: ", index_iter)
        c = None

        samples = {'acs': [], 'obs': [], 'reward': [], 'reward_sum': [], 'model_error': [],
                   "controller_sol": [], 'motor_actions': []}

        trajectory, c = execute_random(env=env, steps=config["episode_length"], samples=samples,
                                       K=config["K"], config=config)
        if data[env_index] is None:
            data[env_index] = trajectory
        else:
            data[env_index] = np.concatenate((data[env_index], trajectory), axis=0)

        traj_obs.extend(samples["obs"])
        traj_acs.extend(samples["acs"])
        evaluator.add_sample(traj_obs[-1], traj_acs[-1])
        print("-------------------------------\n")

    with open('train_data'+config['save_data']+'.pk', 'wb') as f:
        pickle.dump(data)
    with open('eval_data'+config['save_data']+'.pk', 'wb') as f:
        train_in, train_out = evaluator.get_data()
        pickle.dump((train_in, train_out))
