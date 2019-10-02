import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import argparse
import torch
import numpy as np
from exp.env_adaptation import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='append', nargs=2, default=[])
    parser.add_argument('-logdir', type=str, default='log')
    args = parser.parse_args()
    logdir = args.logdir


    class Cost_ensemble(object):
        def __init__(self, ensemble_model, init_state, horizon, action_dim, goal, pred_high, pred_low, config):
            self.__ensemble_model = ensemble_model
            self.__init_state = init_state
            self.__horizon = horizon
            self.__action_dim = action_dim
            self.__goal = goal
            self.__models = self.__ensemble_model.get_models()
            self.__pred_high = pred_high
            self.__pred_low = pred_low
            self.__obs_dim = len(init_state)

        def cost_fn(self, samples):
            action_samples = torch.FloatTensor(samples).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(
                samples)
            init_states = torch.FloatTensor(np.repeat([self.__init_state], len(samples),
                                                      axis=0)).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(
                np.repeat([self.__init_state], len(samples), axis=0))
            all_costs = torch.FloatTensor(
                np.zeros(len(samples))).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(
                np.zeros(len(samples)))

            n_model = len(self.__models)
            # n_batch = min(n_model, int(len(samples)/1024))
            n_batch = max(1, int(len(samples) / 16384))
            per_batch = len(samples) / n_batch

            for i in range(n_batch):
                start_index = int(i * per_batch)
                end_index = len(samples) if i == n_batch - 1 else int(i * per_batch + per_batch)
                action_batch = action_samples[start_index:end_index]
                start_states = init_states[start_index:end_index]
                dyn_model = self.__models[np.random.randint(0, len(self.__models))]
                for h in range(self.__horizon):
                    actions = action_batch[:, h * self.__action_dim: h * self.__action_dim + self.__action_dim]
                    model_input = torch.cat((start_states, actions), dim=1)
                    diff_state = dyn_model.predict_tensor(model_input)
                    start_states += diff_state
                    for dim in range(self.__obs_dim):
                        start_states[:, dim].clamp_(self.__pred_low[dim], self.__pred_high[dim])

                    action_cost = torch.sum(actions * actions, dim=1) * 0.0
                    x_vel_cost = -start_states[:, 13]
                    survive_cost = (start_states[:, 0] < 0.26).type(start_states.dtype) * 2.0
                    all_costs[start_index: end_index] += x_vel_cost * config["discount"] ** h + action_cost * config[
                        "discount"] ** h + survive_cost * config["discount"] ** h
            return all_costs.cpu().detach().numpy()

    config = {
        # exp parameters:
        "horizon": 20,  # NOTE: "sol_dim" must be adjusted
        "iterations": 100,
        "random_iter": 1,
        "episode_length": 1000,
        "init_state": None,  # Must be updated before passing config as param
        "action_dim": 8,
        "video_recording_frequency": 20,
        "logdir": logdir,
        "load_data": None,
        "motor_velocity_limit": np.inf,
        "angle_limit": 1,
        "K": 1,

        # Model learning parameters
        "epoch": 1000,
        "learning_rate": 1e-3,
        "minibatch_size": 32,

        # Ensemble model params'log'
        "ensemble_epoch": 5,
        "ensemble_dim_in": 8+27,
        "ensemble_dim_out": 27,
        "ensemble_hidden": [200, 200, 100],
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
        "opt": "RS",
        "lb": -0.5,
        "ub": 0.5,
        "popsize": 500,
        "pop_batch": 16384,
        "sol_dim": 8*20,  # NOTE: Depends on Horizon
        "num_elites": 50,
        "cost_fn": None,
        "alpha": 0.1,
        "discount": 1.,
        "Cost_ensemble": Cost_ensemble
    }
    for (key, val) in args.config:
        if key in ['horizon', 'K', 'popsize', 'iterations']:
            config[key] = int(val)
        elif key in ['load_data']:
            config[key] = val
        elif key in ['ensemble_hidden']:
            config[key] = [int(val), int(val), int(val)]
        else:
            config[key] = float(val)
    config['sol_dim'] = config['horizon'] * config['action_dim']
    main(config)