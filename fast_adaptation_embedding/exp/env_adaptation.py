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
import argparse
from fast_adaptation_embedding.env.kinematic_arm import Arm_env
import cma

# class Cost_meta(object):
#     def __init__(self, model, init_state, horizon, action_dim, goal, task_likelihoods):
#        self.__model = model
#        self.__init_state = init_state
#        self.__horizon = horizon
#        self.__action_dim = action_dim
#        self.__goal = goal
#        self.__task_likelihoods = task_likelihoods
#
#     def cost_given_task(self, samples, task_id):
#         all_costs = []
#         for s in samples:
#             state = self.__init_state.copy()
#             episode_cost = 0
#             for i in range(self.__horizon):
#                 action = s[self.__action_dim*i : self.__action_dim*i + self.__action_dim]
#                 x = np.concatenate((state, action), axis=1).reshape(1, self.__action_dim)
#                 diff_state = self.__model.predict(x, np.array([[task_id]])).data.cpu().numpy().flatten()
#                 state += diff_state
#                 episode_cost += np.linalg.norm(self.__goal - state)
#             all_costs.append(episode_cost * self.__task_likelihoods[task_id])
#         return np.array(all_costs)
#
#     def cost_fn(self, samples):
#         all_costs = np.zeros(len(samples))
#         for i in range(len(self.__task_likelihoods)):
#             all_costs += self.cost_given_task(samples, i)
#         return all_costs / np.sum(self.__task_likelihoods)


class Cost_ensemble(object):
    def __init__(self, ensemble_model, init_state, horizon, action_dim, goal):
        self.__ensemble_model = ensemble_model
        self.__init_state = init_state
        self.__horizon = horizon
        self.__action_dim = action_dim
        self.__goal = torch.FloatTensor(goal).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(goal)
        self.__models = self.__ensemble_model.get_models()

    def cost_fn(self, samples):
        action_samples = torch.FloatTensor(samples).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(samples)
        init_states = torch.FloatTensor(np.repeat([self.__init_state], len(samples),
                                                  axis=0)).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(
            np.repeat([self.__init_state], len(samples), axis=0))
        all_costs = torch.FloatTensor(
            np.zeros(len(samples))).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(np.zeros(len(samples)))

        n_model = len(self.__models)
        n_batch = min(n_model, len(samples))
        per_batch = len(samples) / n_batch

        for i in range(n_batch):
            start_index = int(i * per_batch)
            end_index = len(samples) if i == n_batch - 1 else int(i * per_batch + per_batch)
            action_batch = action_samples[start_index:end_index]
            start_states = init_states[start_index:end_index]
            __cost = 0
            dyn_model = self.__models[np.random.randint(0, len(self.__models))]
            for h in range(self.__horizon):
                actions = action_batch[:, h * self.__action_dim: h * self.__action_dim + self.__action_dim]
                model_input = torch.cat((start_states, actions), dim=1)
                diff_state = dyn_model.predict_tensor(model_input)
                start_states += diff_state
                all_costs[start_index: end_index] += torch.sum(torch.pow(start_states[:, 5::] - self.__goal, 2), dim=1)
                # * config["discount"]**h #+ torch.sum(actions * actions, dim=1) * 0.1 * config["discount"]**h

        return all_costs.cpu().detach().numpy()


def train_meta(tasks_in, tasks_out, config):
    model = nn_model.Embedding_NN(dim_in=config["dim_in"], hidden=config["hidden_layers"], dim_out=config["dim_out"], embedding_dim=config["embedding_size"], num_tasks=len(tasks_in), CUDA=config["cuda"], SEED=None, input_limit=1.0, dropout=0.0)
    nn_model.train_meta(model, tasks_in, tasks_out, meta_iter=config["meta_iter"], inner_iter=config["inner_iter"], inner_step=config["inner_step"], meta_step=config["meta_step"], minibatch=config["meta_batch_size"])
    return model


def train_model(model, train_in, train_out, task_id, config):
    cloned_model = copy.deepcopy(model)
    nn_model.train(cloned_model, train_in, train_out, task_id=task_id, inner_iter=config["epoch"],
                   inner_lr=config["learning_rate"], minibatch=config["training_batch_size"])
    return cloned_model


def train_ensemble_model(train_in, train_out, sampling_size, config, model=None):
    network = model
    if network is None:
        network = FFNN_Ensemble_Model(dim_in=config["ensemble_dim_in"],
                                      hidden=config["ensemble_hidden"],
                                      hidden_activation=config["hidden_activation"],
                                      dim_out=config["ensemble_dim_out"],
                                      CUDA=config["ensemble_cuda"],
                                      SEED=config["ensemble_seed"],
                                      output_limit=config["ensemble_output_limit"],
                                      dropout=config["ensemble_dropout"],
                                      n_ensembles=config["n_ensembles"])
    network.train(epochs=config["ensemble_epoch"], training_inputs=train_in, training_targets=train_out,
                  batch_size=config["ensemble_batch_size"], logInterval=config["ensemble_log_interval"],
                  sampling_size=sampling_size)
    return copy.deepcopy(network)


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


def execute_random(env, steps, init_state, samples, K):
    current_state = env.reset()
    obs = [current_state]
    acs = []
    trajectory = []
    reward = []
    traject_cost = 0
    for i in tqdm(range(steps)):
        a = env.action_space.sample()
        next_state, r = 0, 0
        for k in range(K):
            next_state, r, _, _ = env.step(a)
            obs.append(next_state)
            acs.append(a)
            reward.append(r)
        trajectory.append([current_state.copy(), a.copy(), next_state-current_state, -r])
        current_state = next_state
        traject_cost += -r
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['reward_sum'].append(-traject_cost)
    return np.array(trajectory), traject_cost


def execute_2(env, init_state, steps, init_mean, init_var, model, config, last_action_seq,
              pred_high, pred_low, index_iter, samples):
    current_state = env.reset()
    f_rec = config['video_recording_frequency']
    recorder = None
    if 1: #f_rec and index_iter % f_rec == (f_rec - 1):
        recorder = VideoRecorder(env, os.path.join(config['logdir'], "iter_" + str(index_iter) + ".mp4"))
        env.render(mode='human')
    obs = [current_state]
    acs = []
    trajectory = []
    reward = []
    traject_cost = 0
    model_error = 0
    sliding_mean = np.zeros(config["sol_dim"])
    # random = np.random.rand()
    mutation = np.random.rand(config["sol_dim"]) * 2. * 0.5 - 0.5
    rand = np.random.rand(config["sol_dim"])
    mutation *= np.array([1.0 if r > 0.25 else 0.0 for r in rand])
    goal = None
    for i in tqdm(range(steps)):
        cost_object = Cost_ensemble(ensemble_model=model, init_state=current_state, horizon=config["horizon"],
                                    action_dim=env.action_space.shape[0], goal=config["goal"])
        config["cost_fn"] = cost_object.cost_fn
        if config['opt'] == "RS":
            optimizer = RS_opt(config)
            sol =  optimizer.obtain_solution()
        elif config['opt'] == "CEM":
            optimizer = CEM_opt(config)
            sol = optimizer.obtain_solution(sliding_mean, init_var)
        elif config['opt'] == "CMA-ES":
            xopt, es = cma.fmin2(None, np.zeros(config["sol_dim"]), 0.5, parallel_objective=lambda x: list(config["cost_fn"](x)))
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


def test_model(ensemble_model, init_state, action, state_diff):
    x = np.concatenate(([init_state], [action]), axis=1)
    y = state_diff.reshape(1, -1)
    y_pred = ensemble_model.get_models()[0].predict(x)
    return np.linalg.norm(y-y_pred)/np.linalg.norm(y)


def extract_action_seq(data):
    actions = []
    for d in data:
        actions += d[1].tolist()
    return np.array(actions)


def main(args, logdir):
    config = {
        # exp parameters:
        "horizon": 20,  # NOTE: "sol_dim" must be adjusted
        "iterations": 200,
        "random_iter": 1,
        "episode_length": 1000,
        "init_state": None,  # Must be updated before passing config as param
        "action_dim": 5,
        "video_recording_frequency": 20,
        "logdir": logdir,
        "load_data": None,
        "goal": [0, 0.5],
        "motor_velocity_limit": np.inf,
        "angle_limit": 1,
        "K": 1,

        # # Model_parameters
        # "dim_in": 8+31,
        # "dim_out": 31,
        # "hidden_layers": [128, 128],
        # "embedding_size": 5,
        # "cuda": True,

        # # Meta learning parameters
        # "meta_iter": 5000,
        # "meta_step": 0.1,
        # "inner_iter": 10,
        # "inner_step": 0.02,

        # Model learning parameters
        "epoch": 1000,
        "learning_rate": 1e-3,
        "minibatch_size": 32,

        # Ensemble model params'log'
        "ensemble_epoch": 5,
        "ensemble_dim_in": 5+7,
        "ensemble_dim_out": 7,
        "ensemble_hidden": [100, 100],
        "hidden_activation": "relu",
        "ensemble_cuda": True,
        "ensemble_seed": None,
        "ensemble_output_limit": None,
        "ensemble_dropout": 0.0,
        "n_ensembles": 1,
        "ensemble_batch_size": 64,
        "ensemble_log_interval": 500,

        # Optimizer parameters
        "max_iters": 1,
        "epsilon": 0.0001,
        "opt": "CMA-ES",
        "lb": -1.,
        "ub": 1.,
        "popsize": 500,
        "pop_batch": 16384,
        "sol_dim": 8*20,  # NOTE: Depends on Horizon
        "num_elites": 50,
        "cost_fn": None,
        "alpha": 0.1,
        "discount": 1.
    }
    for (key, val) in args:
        if key in ['horizon', 'K', 'popsize', 'iterations']:
            config[key] = int(val)
        elif key in ['load_data', 'opt']:
            config[key] = val
        elif key in ['ensemble_hidden']:
            config[key] = [int(val), int(val), int(val)]
        else:
            config[key] = float(val)
    config['sol_dim'] = config['horizon'] * config['action_dim']
    #  ************************************************

    logdir = os.path.join(config['logdir'], strftime("%Y-%m-%d--%H:%M:%S", localtime()) + str(np.random.randint(10**5)))
    config['logdir'] = logdir
    os.makedirs(logdir)
    with open(os.path.join(config['logdir'], "config.txt"), 'w') as f:
        f.write(pprint.pformat(config))
    mismatches = np.array([[1., 1., 1., 1., 1.]])
    n_task = len(mismatches)
    goal = [1000, 0]
    render = False
    envs = [Arm_env(goal=tuple(config["goal"]), joint_mismatch=mismatches[0]) for i in range(n_task)]
    random_iter = config['random_iter']
    data = n_task * [None]
    models = n_task * [None]
    best_action_seq = np.random.rand(config["sol_dim"])*2.0 - 1.0
    best_cost = 10000
    last_action_seq = None
    all_action_seq = []
    all_costs = []

    for i in range(n_task):
        with open(os.path.join(config['logdir'], "ant_costs_task_"+ str(i)+".txt"), "w+") as f:
            f.write("")

    traj_obs, traj_acs, traj_rets, traj_rews, traj_error = [], [], [], [], []

    for index_iter in range(config["iterations"]):
        '''Pick a random environment'''
        env_index = int(index_iter % n_task)
        env = envs[env_index]

        print("Episode: ", index_iter)
        print("Env index: ", env_index)
        c = None

        samples = {'acs': [], 'obs': [], 'reward': [], 'reward_sum': [], 'model_error': []}
        if (not config['load_data'] is None) and (data[env_index] is None):
            with open(config['load_data'], 'rb') as f:
                data = pickle.load(f)
            random_iter = 0
        if data[env_index] is None or index_iter < random_iter * n_task:
            print("Execution (Random actions)...")
            trajectory, c = execute_random(env=env, steps=config["episode_length"],
                                           init_state=config["init_state"], samples=samples, K=config["K"])
            if data[env_index] is None:
                data[env_index] = trajectory
            else:
                data[env_index] = np.concatenate((data[env_index], trajectory), axis=0)
            print("Cost : ", c)

            if c < best_cost:
                best_cost = c
                best_action_seq = []
                for d in trajectory:
                    best_action_seq += d[1].tolist()
                best_action_seq = np.array(best_action_seq)
                last_action_seq = best_action_seq
            all_action_seq.append(extract_action_seq(trajectory))
            all_costs.append(c)
        else:
            '''------------Update models------------'''
            # with open("random_data.pk", 'wb') as f:
            #     pickle.dump(data, f)
            x, y, high, low = process_data(data[env_index])
            print("Learning model...")
            models[env_index] = train_ensemble_model(train_in=x, train_out=y, sampling_size=-1, config=config, model=models[env_index])
            print("Execution...")

            trajectory, c = execute_2(env=env,
                                    init_state=config["init_state"],
                                    model=models[env_index],
                                    steps=config["episode_length"],
                                    init_mean=best_action_seq[0:config["sol_dim"]] ,
                                    init_var=0.1 * np.ones(config["sol_dim"]),
                                    config=config,
                                    last_action_seq=best_action_seq,
                                    pred_high= high,
                                    pred_low=low,
                                    index_iter=index_iter,
                                    samples=samples)
            data[env_index] = np.concatenate((data[env_index], trajectory), axis=0)
            print("Cost : ", c)

            if c < best_cost:
                best_cost = c
                best_action_seq = []
                for d in trajectory:
                    best_action_seq += d[1].tolist()
                best_action_seq = np.array(best_action_seq)
                last_action_seq = extract_action_seq(trajectory)

            all_action_seq.append(extract_action_seq(trajectory))
            all_costs.append(c)

            print("Saving trajectories..")
            # if index_iter % 10 == 0:
            #     np.save(os.path.join(config['logdir'], "trajectories_ant.npy"), data)
            np.save(os.path.join(config['logdir'], "best_cost_ant.npy"), best_cost)
            np.save(os.path.join(config['logdir'], "best_action_seq_ant.npy"), best_action_seq)
        with open(os.path.join(config['logdir'], "ant_costs_task_" + str(env_index)+".txt"), "a+") as f:
            f.write(str(c)+"\n")

        traj_obs.extend(samples["obs"])
        traj_acs.extend(samples["acs"])
        traj_rets.extend(samples["reward_sum"])
        traj_rews.extend(samples["reward"])
        traj_error.extend(samples["model_error"])

        savemat(
            os.path.join(config['logdir'], "logs.mat"),
            {
                "observations": traj_obs,
                "actions": traj_acs,
                "reward_sum": traj_rets,
                "rewards": traj_rews,
                "model_error": traj_error
            }
        )
        print("-------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='append', nargs=2, default=[])
    parser.add_argument('-logdir', type=str, default='log')
    args = parser.parse_args()

    main(args.config, args.logdir)