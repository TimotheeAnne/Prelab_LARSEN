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


class Evaluation_ensemble(object):
    def __init__(self, config, ensemble_model=None, pred_high=None, pred_low=None):
        self.__ensemble_model = ensemble_model
        self.__horizon = config['horizon']
        self.__action_dim = config['action_dim']
        self.__pred_high = pred_high
        self.__pred_low = pred_low
        self.__obs_dim = config['ensemble_dim_out']
        self.__episode_length = config['episode_length']
        self.__inputs = []
        self.__outputs = []
        self.__type = config['model_type']
        self.H = config['horizon']
        self.indexes = []
        self.inv_indexes = []
        self.env = config['env']
        self.__pop_batch = config['pop_batch']

    def add_samples(self, samples):
        if self.__type == "D":
            acs = samples['acs']
            obs = samples['obs']
            indexes, inv_indexes = [], []
            for i in range(len(acs)):
                T = len(obs[i])
                for t in range(len(acs[i])):
                    if t < T - self.H:
                        my_index = len(self.__inputs)
                        self.indexes.append(my_index)
                        self.inv_indexes.append((i, t))
                    if self.env == "AntMuJoCoEnv_fastAdapt-v0":
                        self.__inputs.append(np.concatenate((obs[i][t], acs[i][t])))
                    elif self.env in ["PexodQuad-v0", "MinitaurControlledEnv_fastAdapt-v0"]:
                        self.__inputs.append(np.concatenate((obs[i][t][2:], acs[i][t])))
                    else:
                        self.__inputs.append(np.concatenate((obs[i][t][:28], obs[i][t][30:31], acs[i][t])))
                    self.__outputs.append(obs[i][t + 1] - obs[i][t])
        else:
            ctrl = samples['controller']
            obs = samples['obs']
            t0 = samples['t0']
            for i in range(len(obs)):
                T = len(obs[i])
                for t in range(0, T - self.H):
                    self.__inputs.append(
                        np.concatenate((obs[i][t][:28], obs[i][t][30:31], ctrl[i], [((t + t0[i]) % self.H) / self.H])))
                    self.__outputs.append(obs[i][t + self.H][28:31] - obs[i][t][28:31])

    def get_data(self):
        return np.array(self.__inputs), np.array(self.__outputs)

    def set_bounds(self, low, high):
        self.__pred_low = low
        self.__pred_high = high

    def get_bounds(self):
        return self.__pred_low, self.__pred_high

    def eval_traj(self, ensemble):
        models = ensemble.get_models()
        if len(self.indexes) == 0:
            print("Not enough data for evaluation.")
            return np.zeros((len(models), self.H, 1)), np.zeros((len(models), self.H, 1)), np.zeros(
                (len(models), self.H, 1))
        inputs = torch.FloatTensor(self.__inputs).cuda()
        outputs = torch.FloatTensor(self.__outputs).cuda()
        if self.__type == "D":
            error = torch.FloatTensor(np.zeros((len(models), self.H, len(self.indexes)))).cuda()
            error0 = torch.FloatTensor(np.zeros((len(models), self.H, len(self.indexes)))).cuda()
            traj_pred = torch.FloatTensor(np.zeros((len(models), self.H + 1, len(self.indexes), 3))).cuda()

            n_batch = max(1, int(len(self.indexes) / self.__pop_batch))
            per_batch = len(self.indexes) / n_batch

            for i in range(n_batch):
                start_index = int(i * per_batch)
                end_index = len(self.indexes) if i == n_batch - 1 else int(i * per_batch + per_batch)
                for model_index in range(len(models)):
                    dyn_model = models[model_index]
                    indexes = np.array(self.indexes[start_index:end_index])
                    if self.env == "AntMuJoCoEnv_fastAdapt-v0":
                        current_state = inputs[indexes, :27]
                        for t in range(self.H):
                            current_input = torch.cat((current_state, inputs[indexes + t, 27:]), dim=1)
                            current_output = outputs[indexes + t]
                            error[model_index, t, start_index:end_index], error0[model_index, t,
                                                                          start_index:end_index], diff_pred = dyn_model.compute_error(
                                current_input, current_output, True)
                            current_state[:, :27] += diff_pred
                    elif self.env in ["PexodQuad-v0", "MinitaurControlledEnv_fastAdapt-v0"]:
                        current_state = inputs[indexes, :2]
                        for t in range(self.H):
                            current_input = torch.cat((current_state, inputs[indexes + t, 2:]), dim=1)
                            current_output = outputs[indexes + t]
                            error[model_index, t, start_index:end_index], error0[model_index, t,
                                                                          start_index:end_index], diff_pred = dyn_model.compute_error(
                                current_input, current_output, True)
                            current_state += diff_pred[:, 2:]
                            traj_pred[model_index, t + 1, start_index:end_index, :2] = traj_pred[model_index, t,
                                                                                       start_index:end_index,
                                                                                       :2] + diff_pred[:, :2]
                    else:
                        current_state = inputs[indexes, :29]
                        for t in range(self.H):
                            current_input = torch.cat((current_state, inputs[indexes + t, 29:]), dim=1)
                            current_output = outputs[indexes + t]
                            error[model_index, t, start_index:end_index], error0[model_index, t,
                                                                          start_index:end_index], diff_pred = dyn_model.compute_error(
                                current_input, current_output, True)
                            current_state[:, :28] += diff_pred[:, :28]
                            current_state[:, 28:29] += diff_pred[:, 30:31]
                            traj_pred[model_index, t + 1, start_index:end_index] = traj_pred[model_index, t,
                                                                                   start_index:end_index] + diff_pred[:,
                                                                                                            28:31]
            return error.cpu().detach().numpy(), error0.cpu().detach().numpy(), traj_pred.cpu().detach().numpy()
        else:
            error = torch.FloatTensor(np.zeros((len(models), len(self.__outputs)))).cuda()
            error0 = torch.FloatTensor(np.zeros((len(models), len(self.__outputs)))).cuda()
            pred = torch.FloatTensor(np.zeros((len(models), len(self.__outputs), 3))).cuda()

            n_batch = max(1, int(len(inputs) / self.__pop_batch))
            per_batch = len(inputs) / n_batch

            for i in range(n_batch):
                start_index = int(i * per_batch)
                end_index = len(inputs) if i == n_batch - 1 else int(i * per_batch + per_batch)
                for model_index in range(len(models)):
                    dyn_model = models[model_index]
                    error[model_index][start_index:end_index], error0[model_index][start_index:end_index], pred[
                                                                                                               model_index][
                                                                                                           start_index:end_index] = dyn_model.compute_error(
                        inputs[start_index:end_index], outputs[start_index:end_index], True)

            # ~ pred_notfallen = (pred[:, :, 2] + inputs[:, 28]) > 0.13
            # ~ not_fallen = (outputs[:, 2] + inputs[:, 28]) > 0.13
            # ~ CM = [[[0, 0], [0, 0]] for _ in range(len(models))]
            # ~ for model_index in range(len(models)):
            # ~ for i in range(len(not_fallen)):
            # ~ CM[model_index][pred_notfallen[model_index][i]][not_fallen[i]] += 1
            return error.cpu().detach().numpy(), error0.cpu().detach().numpy(), pred.cpu().detach().numpy()  # , CM

    def eval_model(self, ensemble, return_pred=False):
        models = ensemble.get_models()
        inputs = torch.FloatTensor(self.__inputs).cuda()
        outputs = torch.FloatTensor(self.__outputs).cuda()
        error = torch.FloatTensor(np.zeros((len(models), len(self.__inputs)))).cuda()
        error0 = torch.FloatTensor(np.zeros((len(models), len(self.__inputs)))).cuda()

        if return_pred:
            for model_index in range(len(models)):
                dyn_model = models[model_index]
                error[model_index], error0[model_index], pred = dyn_model.compute_error(inputs, outputs, True)
            return error.cpu().detach().numpy(), error0.cpu().detach().numpy(), pred.cpu().detach().numpy()
        else:
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


def execute_random(env, steps, samples, K, config, index_iter):
    current_state = env.reset(friction=config['friction'])
    obs = [current_state]
    acs = []
    trajectory = []
    reward = []
    traject_cost = 0
    param = []
    motor_actions = []
    recorder = None
    # ~ recorder = VideoRecorder(env, os.path.join(config['logdir'], "iter_" + str(index_iter) + ".mp4"))
    for i in tqdm(range(steps)):
        a = env.action_space.sample()
        rew = 0
        for k in range(K):
            next_state, r, done, info = env.step(a)
            rew += r
            motor_actions.append(info['action'])
            if recorder is not None:
                recorder.capture_frame()
        if config['env'] == "PexodQuad-v0" and not config['controller'] is None:
            obs.append(info['obs'])
            acs.append(info['acs'])
        else:
            obs.append(next_state)
            acs.append(a)
        reward.append(rew)
        param.append(a)
        current_state = next_state
        traject_cost += -rew
        if done:
            break
    samples['t0'].append(0)
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['reward_sum'].append(-traject_cost)
    samples['controller'].append(np.copy(param))
    samples['motor_actions'].append(np.copy(motor_actions))
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    return traject_cost


def execute_2(env, steps, init_var, model, config, pred_high, pred_low, index_iter, samples):
    # for environment without controller
    current_state = env.reset(friction=config['friction'])
    f_rec = config['video_recording_frequency']
    recorder = None
    if f_rec and index_iter % f_rec == (f_rec - 1):
        recorder = VideoRecorder(env, os.path.join(config['logdir'], "iter_" + str(index_iter) + ".mp4"))
    obs = [current_state]
    acs = []
    reward = []
    param = []
    motor_actions = []
    traject_cost = 0
    model_error = 0
    sliding_mean = np.zeros(config["sol_dim"])
    mutation = np.random.rand(config["sol_dim"]) * 2. * 0.5 - 0.5
    rand = np.random.rand(config["sol_dim"])
    mutation *= np.array([1.0 if r > 0.25 else 0.0 for r in rand])
    goal = None
    for i in tqdm(range(steps)):
        cost_object = config['Cost_ensemble'](ensemble_model=model, init_state=current_state[2:],
                                              horizon=config["horizon"],
                                              action_dim=env.action_space.shape[0], goal=goal, pred_high=pred_high,
                                              pred_low=pred_low, config=config)
        config["cost_fn"] = cost_object.cost_fn
        if config['opt'] == "RS":
            optimizer = RS_opt(config)
            if True:
                sol = optimizer.obtain_solution()
            else:
                sol = optimizer.obtain_solution(init_mean=sliding_mean, init_var=init_var)
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
        rew = 0
        for k in range(config["K"]):
            next_state, r, done, info = env.step(a)
            rew += r
            motor_actions.append(info['action'])
            if not recorder is None:
                recorder.capture_frame()
        obs.append(next_state)
        acs.append(a)
        param.append(a)
        reward.append(rew)
        # ~ trajectory.append([current_state.copy(), a.copy(), next_state - current_state, -r])
        model_error += test_model(model, current_state[2:], a.copy(), next_state-current_state)
        current_state = next_state
        traject_cost += -rew
        sliding_mean[0:-len(a)] = sol[len(a)::]
        if done:
            break
    print("Model error: ", model_error / (i + 1))
    if recorder is not None:
        recorder.close()
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['reward_sum'].append(-traject_cost)
    samples['model_error'].append(model_error / steps)
    samples['controller'].append(np.copy(param))
    samples['motor_actions'].append(np.copy(motor_actions))
    return traject_cost


def execute_3(env, steps, init_var, model, config, pred_high, pred_low, index_iter, samples):
    # Environment with controller
    current_state = env.reset()
    controller = config["controller"]
    f_rec = config['video_recording_frequency']
    recorder = None
    # ~ if f_rec and  (index_iter ==1 or index_iter % f_rec == (f_rec - 1)):
    # ~ recorder = VideoRecorder(env, os.path.join(config['logdir'], "iter_" + str(index_iter) + ".mp4"))
    obs = [current_state]
    acs, reward, control_sol, motor_actions = [], [], [], []
    traject_cost = 0
    model_error = np.zeros(len(model.get_models()))
    sliding_mean = np.zeros(config["sol_dim"])
    goal = None
    omega = config['omega']
    with open('../data/good_controllers.pk', 'rb') as f:
        controllers = pickle.load(f)
    for i in tqdm(range(steps)):
        if config['env'] == "PexodQuad-v0":
            t = env.get_current_time()
        else:
            t = env.minitaur.GetTimeSinceReset()

        cost_object = config['Cost_ensemble'](ensemble_model=model,
                                              init_state=np.concatenate((current_state[:28], current_state[30:31])),
                                              horizon=config["horizon"],
                                              action_dim=env.action_space.shape[0], goal=goal, pred_high=pred_high,
                                              t0=t,
                                              pred_low=pred_low,
                                              config=config)
        config["cost_fn"] = cost_object.cost_fn
        if config['opt'] == "RS":
            optimizer = RS_opt(config)
            if i == 0 or config['only_random']:
                init_mean = controllers[np.random.randint(0, len(controllers))]
                sol = optimizer.obtain_solution(init_mean, init_var, t0=t)
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
        # ~ trajectory.append([current_state.copy(), a.copy(), next_state - current_state, -r])
        # ~ model_error += test_model(model, np.concatenate((current_state[:28], current_state[30:31])), a.copy(), next_state-current_state)
        current_state = next_state
        traject_cost += -r
        sliding_mean = sol
        if done:
            break
    print("Model error: ", model_error / (i + 1))
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    samples['t0'].append(0)
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['reward_sum'].append(-traject_cost)
    samples['model_error'].append(model_error / (i + 1))
    samples['controller_sol'].append(np.copy(control_sol))
    samples['motor_actions'].append(np.copy(motor_actions))
    return traject_cost


def execute_4(env, steps, init_var, model, config, pred_high, pred_low, index_iter, samples):
    # For Pexod Env with full action space and observation space: D
    current_state = env.reset()
    controller = config["controller"]
    f_rec = config['video_recording_frequency']
    recorder = None
    if f_rec and (index_iter == 1 or index_iter % f_rec == (f_rec - 1)):
        recorder = VideoRecorder(env, os.path.join(config['logdir'], "iter_" + str(index_iter) + ".mp4"))
    obs = [current_state]
    acs, reward, control_sol, motor_actions = [], [], [], []
    traject_cost = 0
    model_error = np.zeros(len(model.get_models()))
    sliding_mean = np.zeros(config["sol_dim"])
    goal = None
    omega = config['omega']
    for i in tqdm(range(steps)):
        t = env.get_current_time()
        cost_object = config['Cost_ensemble'](ensemble_model=model,
                                              init_state=current_state[2:],
                                              horizon=config["horizon"],
                                              action_dim=env.action_space.shape[0], goal=goal, pred_high=pred_high,
                                              t0=t,
                                              pred_low=pred_low,
                                              config=config)
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

        a = sol
        next_state, r = 0, 0
        if recorder is not None:
            recorder.capture_frame()
        for k in range(config["K"]):
            next_state, r, done, info = env.step(a)
        obs.append(info['obs'])
        acs.append(info['acs'])
        reward.append(r)
        control_sol.append(np.copy(sol))
        # ~ trajectory.append([current_state.copy(), a.copy(), next_state - current_state, -r])
        # ~ model_error += test_model(model, np.concatenate(current_state[2:]), a.copy(), next_state-current_state)
        current_state = info['obs']
        traject_cost += -r
        sliding_mean = sol
        if done:
            break
    print("Model error: ", model_error / (i + 1))
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    samples['t0'].append(0)
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['reward_sum'].append(-traject_cost)
    samples['model_error'].append(model_error / (i + 1))
    samples['controller_sol'].append(np.copy(control_sol))
    samples['motor_actions'].append(np.copy(motor_actions))
    return traject_cost


def test_model(ensemble_model, init_state, action, state_diff):
    x = np.concatenate(([init_state], [action]), axis=1)
    y = state_diff.reshape(1, -1)
    error = []
    for model_index in range(len(ensemble_model.get_models())):
        y_pred = ensemble_model.get_models()[model_index].predict(x)
        error.append(np.linalg.norm(y - y_pred) / np.linalg.norm(y))
    return np.array(error)


def extract_action_seq(data):
    actions = [], CMs
    for d in data:
        actions += d[1].tolist()
    return np.array(actions)


def analyse_CM(CM, toprint=False):
    [[TP, FP], [FN, TN]] = CM
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    TNR = TN / (TN + FP) if (FP + TN) != 0 else 0
    ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    FS = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
    if toprint:
        print("TPR: ", TPR)
        print("TNR: ", TNR)
        print("ACC: ", ACC)
        print("Precision: ", Precision)
        print("FS: ", FS)
    return [TPR, TNR, ACC, Precision, FS]


def main(config):
    logdir = os.path.join(config['logdir'],
                          strftime("%Y-%m-%d--%H:%M:%S", localtime()) + str(np.random.randint(10 ** 5)))
    config['logdir'] = logdir
    os.makedirs(logdir)
    with open(os.path.join(config['logdir'], "config.txt"), 'w') as f:
        f.write(pprint.pformat(config))
    n_task = 1
    render = False
    print(config)
    envs = [gym.make(config['env'], **config['env_args']) for i in range(n_task)]
    envs[0].metadata["video.frames_per_second"] = config['video.frames_per_second']
    random_iter = config['random_iter']
    data = n_task * [None]
    models = n_task * [None]
    evaluations = n_task * [None]
    trainer = Evaluation_ensemble(config=config)

    for i in range(n_task):
        with open(os.path.join(config['logdir'], "env_costs_task_" + str(i) + ".txt"), "w+") as f:
            f.write("")

    traj_obs, traj_acs, traj_rets, traj_rews, traj_error, traj_eval, traj_sol, traj_motor = [], [], [], [], [], [], [], []

    if not config['load_data'] is None:
        with open("../data/train_" + config['load_data'] + ".pk", 'rb') as f:
            training_samples = pickle.load(f)
        trainer = Evaluation_ensemble(config=config)
        trainer.add_samples(training_samples)
        random_iter = 0

    for index_iter in range(config["iterations"]):
        '''Pick a random environment'''
        env_index = int(index_iter % n_task)
        env = envs[env_index]

        # ~ if index_iter == 0:
        # ~ mismatches = [1] * 12 + [0]
        # ~ mismatches[0] = 0
        # ~ env.set_mismatch(mismatches)

        print("Episode: ", index_iter)
        print("Env index: ", env_index)
        c = None

        samples = {'acs': [], 'obs': [], 'reward': [], 'reward_sum': [], 'model_error': [],
                   "controller": [], 'motor_actions': [], 't0': []}
        if index_iter < random_iter * n_task:
            print("Execution (Random actions)...")
            c = execute_random(env=env, steps=config["episode_length"], samples=samples,
                               K=config["K"], config=config, index_iter=index_iter)
            print("Cost : ", c)
            trainer.add_samples(samples)
        else:
            '''------------Update models------------'''
            if (index_iter - random_iter) % 1 == 0 and index_iter < np.inf:
                low, high = trainer.get_bounds()
                x, y = trainer.get_data()
                print("Learning model...")
                sampling_size = -1  # if config['n_ensembles'] == 1 else len(x)
                models[env_index] = train_ensemble_model(train_in=x, train_out=y, sampling_size=sampling_size,
                                                         config=config, model=models[env_index])
                print("Evaluate model...")
                training_error, training_error0, train_pred = trainer.eval_traj(models[env_index])
                print("Training error:", np.mean(training_error, axis=2))
                traj_eval.append(np.mean(training_error, axis=2))
            print("Execution...")

            if config['env'] == "AntMuJoCoEnv_fastAdapt-v0" or (
                    config['env'] in ["PexodQuad-v0", "MinitaurControlledEnv_fastAdapt-v0"] and config[
                'controller'] is None):
                execute = execute_2
            elif config['env'] in ["PexodQuad-v0", "MinitaurControlledEnv_fastAdapt-v0"]:
                execute = execute_4
            else:
                execute = execute_3

            c = execute(env=env,
                        model=models[env_index],
                        steps=config["episode_length"],
                        init_var=config["init_var"] * np.ones(config["sol_dim"]),
                        config=config,
                        pred_high=high,
                        pred_low=low,
                        index_iter=index_iter,
                        samples=samples)
            print("Cost : ", c)
            if config['model_type'] == "D":
                trainer.add_samples(samples)
        with open(os.path.join(config['logdir'], "ant_costs_task_" + str(env_index) + ".txt"), "a+") as f:
            f.write(str(c) + "\n")

        traj_obs.extend(samples["obs"])
        traj_acs.extend(samples["acs"])
        traj_rets.extend(samples["reward_sum"])
        traj_rews.extend(samples["reward"])
        traj_error.extend(samples["model_error"])
        traj_sol.extend(samples["controller"])
        traj_motor.extend(samples['motor_actions'])

        # ~ savemat(
            # ~ os.path.join(config['logdir'], "logs.mat"),
            # ~ {
                # ~ "observations": traj_obs,
                # ~ "actions": traj_acs,
                # ~ "reward_sum": traj_rets,
                # ~ "rewards": traj_rews,
                # ~ "model_error": traj_error,
                # ~ "model_eval": traj_eval,
                # ~ "controller": traj_sol,
                # ~ "motor_actions": traj_motor
            # ~ }
        # ~ )
        with open(os.path.join(config['logdir'], "logs.mat"), 'wb') as f:
            pickle.dump(             {
                "observations": traj_obs,
                "actions": traj_acs,
                "reward_sum": traj_rets,
                "rewards": traj_rews,
                "model_error": traj_error,
                "model_eval": traj_eval,
                "controller": traj_sol,
                "motor_actions": traj_motor
            }, f)
        print("-------------------------------\n")
