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
        self.__inputs = []
        self.__outputs = []
        self.__type = config['model_type']
        self.H = config['horizon']
        self.indexes = None
        self.inv_indexes = None

    def add_samples(self, samples):
        if self.__type == "D":
            acs = samples['acs']
            obs = samples['obs']
            indexes, inv_indexes = [], []
            for i in tqdm(range(len(acs))):
                T = len(obs[i])
                for t in range(len(acs[i])):
                    if t < T - self.H:
                        my_index = len(self.__inputs)
                        indexes.append(my_index)
                        inv_indexes.append((i, t))
                    self.__inputs.append(np.concatenate((obs[i][t][:28], obs[i][t][30:31], acs[i][t])))
                    self.__outputs.append(obs[i][t + 1] - obs[i][t])
            self.indexes = np.array(indexes)
            self.inv_indexes = np.array(inv_indexes)
            self.__pred_low = np.min(self.__inputs[:29], axis=0)
            self.__pred_high = np.max(self.__inputs[:29], axis=0)
        else:
            ctrl = samples['controller']
            obs = samples['obs']
            for i in range(len(obs)):
                T = len(obs[i])
                for t in range(0, T - self.H):
                    self.__inputs.append(np.concatenate((obs[i][t][:28], obs[i][t][30:31], ctrl[i], [(t % 25) / 25])))
                    self.__outputs.append(obs[i][t + self.H][28:31] - obs[i][t][28:31])

    def get_data(self):
        return self.__inputs, self.__outputs

    def set_bounds(self, low, high):
        self.__pred_low = low
        self.__pred_high = high

    def get_bounds(self):
        return self.__pred_low, self.__pred_high

    def eval_traj(self, ensemble):
        models = ensemble.get_models()
        inputs = torch.FloatTensor(self.__inputs).cuda()
        outputs = torch.FloatTensor(self.__outputs).cuda()

        if self.__type == "D":
            error = torch.FloatTensor(np.zeros((len(models), self.H, len(self.indexes)))).cuda()
            error0 = torch.FloatTensor(np.zeros((len(models), self.H, len(self.indexes)))).cuda()
            traj_pred = torch.FloatTensor(np.zeros((len(models), self.H + 1, len(self.indexes), 3))).cuda()
            for model_index in range(len(models)):
                dyn_model = models[model_index]
                current_state = inputs[self.indexes, :29]
                traj_pred[model_index][0] = torch.zeros((len(self.indexes), 3))
                for t in range(self.H):
                    current_input = torch.cat((current_state, inputs[self.indexes + t, 29:]), dim=1)
                    current_output = outputs[self.indexes + t]
                    error[model_index, t], error0[model_index, t], diff_pred = dyn_model.compute_error(current_input,
                                                                                                       current_output,
                                                                                                       True)
                    current_state[:, :28] += diff_pred[:, :28]
                    current_state[:, 28:29] += diff_pred[:, 30:31]

                    # ~ for dim in range(self.__obs_dim-2):
                    # ~ current_state[:, dim].clamp_(self.__pred_low[dim], self.__pred_high[dim])
                    traj_pred[model_index][t + 1] = traj_pred[model_index][t] + diff_pred[:, 28:31]
            return error.cpu().detach().numpy(), error0.cpu().detach().numpy(), traj_pred.cpu().detach().numpy()
        else:
            error = torch.FloatTensor(np.zeros((len(models), len(self.__inputs)))).cuda()
            error0 = torch.FloatTensor(np.zeros((len(models), len(self.__inputs)))).cuda()
            for model_index in range(len(models)):
                dyn_model = models[model_index]
                error[model_index], error0[model_index], pred = dyn_model.compute_error(inputs, outputs, True)
            return error.cpu().detach().numpy(), error0.cpu().detach().numpy(), pred.cpu().detach().numpy()

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

    with open("../data/train_" + config['load_data'] + ".pk", 'rb') as f:
        training_samples = pickle.load(f)
    with open("../data/eval_" + config['load_data'] + ".pk", 'rb') as f:
        eval_samples = pickle.load(f)

    trainer = Evaluation_ensemble(config=config)
    trainer.add_samples(training_samples)
    evaluator = Evaluation_ensemble(config=config)
    evaluator.add_samples(eval_samples)
    save_train_indexes = [trainer.indexes, trainer.inv_indexes]
    save_eval_indexes = [evaluator.indexes, evaluator.inv_indexes]
    with open(config['logdir'] + "/indexes.pk", 'wb') as f:
        pickle.dump([save_train_indexes, save_eval_indexes], f)

    l, h = trainer.get_bounds()
    evaluator.set_bounds(l, h)
    for index_iter in range(config["iterations"]):
        print("iteration :", index_iter)
        env_index = 0
        '''------------Update models------------'''
        x, y = trainer.get_data()
        print("Learning model...")
        sampling_size = -1 if config['n_ensembles'] == 1 else len(x)
        models[env_index] = train_ensemble_model(train_in=np.array(x), train_out=np.array(y),
                                                 sampling_size=sampling_size, config=config,
                                                 model=models[env_index])
        print("Evaluate model...")

        if (index_iter % 1 == 0) or (index_iter == (config["iterations"] - 1)):
            training_error, training_error0, train_pred = trainer.eval_traj(models[env_index])
            eval_error, eval_error0, eval_pred = evaluator.eval_traj(models[env_index])
            traj_eval_pred.append(eval_pred)
            traj_train_pred.append(train_pred)
            print("Training error:", np.mean(training_error, axis=1))
            print("Test error:", np.mean(eval_error, axis=1))
            # ~ print("Training R²:", np.mean(1 - training_error / training_error0, axis=2))
            # ~ print("Test R²:", np.mean(1 - eval_error / eval_error0, axis=2))

        # ~ else:
        # ~ training_error, training_error0 = trainer.eval_model(models[env_index])
        # ~ eval_error, eval_error0 = evaluator.eval_model(models[env_index])
        # ~ print("Training error:", np.mean(training_error, axis=1))
        # ~ print("Test error:", np.mean(eval_error, axis=1))
        # ~ print("Training R²:", np.mean(1 - training_error / training_error0, axis=1))
        # ~ print("Test R²:", np.mean(1 - eval_error / eval_error0, axis=1))

        traj_eval.append(np.mean(eval_error, axis=1))
        traj_error.append(np.mean(training_error, axis=1))
        # ~ traj_eval0.append(np.mean(1 - eval_error / eval_error0, axis=1))
        # ~ traj_error0.append(np.mean(1 - training_error / training_error0, axis=1))

        savemat(
            os.path.join(config['logdir'], "logs.mat"),
            {
                "train_error": traj_error,
                "test_error": traj_eval,
                # ~ "train_R2": traj_error0,
                # ~ "test_R2": traj_eval0,

                "eval_pred": traj_eval_pred,
                "train_pred": traj_train_pred,
            }
        )
        print("-------------------------------\n")
