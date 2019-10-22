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
                                      n_ensembles=config["n_ensembles"],
                                      contact=config['ensemble_contact'])
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
            error[model_index], error0[model_index], pred = dyn_model.compute_error(inputs, outputs)
        return error.cpu().detach().numpy(), error0.cpu().detach().numpy(), pred.cpu().detach().numpy()


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


def compare(config):
    logdir = os.path.join(config['logdir'],
                          strftime("%Y-%m-%d--%H:%M:%S", localtime()) + str(np.random.randint(10 ** 5)))
    config['logdir'] = logdir
    os.makedirs(logdir)
    with open(os.path.join(config['logdir'], "config.txt"), 'w') as f:
        f.write(pprint.pformat(config))
    n_task = 1
    render = False
    envs = [gym.make(config['env'], **config['env_args']) for i in range(n_task)]
    envs[0].metadata["video.frames_per_second"] = config['video.frames_per_second']
    random_iter = config['random_iter']
    data = n_task * [None]
    models = n_task * [None]
    evaluations = n_task * [None]

    traj_eval0, traj_error0, traj_rets, traj_rews, traj_error, traj_eval, traj_sol, traj_motor = [], [], [], [], [], [], [], []
    all_training_error, all_eval_error, traj_train_pred, traj_eval_pred = [], [], [], []
    with open("./data/train_data_"+config['load_data']+".pk", 'rb') as f:
        data = pickle.load(f)
    with open("./data/train_eval_"+config['load_data']+".pk", 'rb') as f:
        train_in, train_out = pickle.load(f)
    with open("./data/test_eval_"+config['load_data']+".pk", 'rb') as f:
        eval_in, eval_out = pickle.load(f)

    evaluator_train = Evaluation_ensemble(config=config)
    evaluator_train.set_data(train_in, train_out)
    evaluator_eval = Evaluation_ensemble(config=config)
    evaluator_eval.set_data(eval_in, eval_out)

    for index_iter in range(config["iterations"]):
        print("iteration :", index_iter)
        env_index = 0
        '''------------Update models------------'''
        x, y, high, low = process_data(data)
        print("Learning model...")
        sampling_size = -1 if config['n_ensembles'] == 1 else len(x)
        models[env_index] = train_ensemble_model(train_in=x, train_out=y, sampling_size=sampling_size, config=config,
                                                 model=models[env_index])
        print("Evaluate model...")
        training_error, training_error0, train_pred = evaluator_train.eval_model(models[env_index])
        eval_error, eval_error0, eval_pred = evaluator_eval.eval_model(models[env_index])
        print("Training error:", np.mean(training_error, axis=1))
        print("Test error:", np.mean(eval_error, axis=1))
        print("Training R²:", np.mean(1-training_error/training_error0, axis=1))
        print("Test R²:", np.mean(1-eval_error/eval_error0, axis=1))
        traj_eval.append(np.mean(eval_error, axis=1))
        traj_error.append(np.mean(training_error, axis=1))
        traj_eval0.append(np.mean(1-eval_error/eval_error0, axis=1))
        traj_error0.append(np.mean(1-training_error/training_error0, axis=1))
        all_training_error.append(training_error)
        all_eval_error.append(eval_error)
        traj_eval_pred.append(eval_pred)
        traj_train_pred.append(train_pred)

        savemat(
            os.path.join(config['logdir'], "logs.mat"),
            {
                "train_error": traj_error,
                "test_error": traj_eval,
                "train_R2": traj_error0,
                "test_R2": traj_eval0,
                "all_eval_error": all_eval_error,
                "all_training_error": all_training_error,
                "eval_pred": traj_eval_pred,
                "train_pred": traj_train_pred,
            }
        )
        print("-------------------------------\n")
