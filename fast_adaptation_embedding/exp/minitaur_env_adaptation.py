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
            self.__energy_weight = config['energy_weight']
            self.__distance_weight = config['distance_weight']
            self.__survival_weight = config['survival_weight']
            self.__drift_weight = config['drift_weight']
            self.__shake_weight = config['shake_weight']
            self.__action_weight = config['action_weight']
            self.__discount = config['discount']
            self.__pop_batch = config['pop_batch']

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
            n_batch = max(1, int(len(samples) / self.__pop_batch))
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

                    action_cost = torch.sum(actions * actions, dim=1) * self.__action_weight
                    energy_cost = abs(
                        torch.sum(start_states[:, 16:24] * start_states[:, 8:16], dim=1)) * 0.02 * self.__energy_weight
                    x_vel_cost = -diff_state[:, 28] * self.__distance_weight
                    y_vel_cost = abs(diff_state[:, 29]) * self.__drift_weight
                    shake_cost = abs(diff_state[:, 30]) * self.__shake_weight
                    survival_cost = (start_states[:, 30] < 0.13).type(start_states.dtype) * self.__survival_weight
                    all_costs[start_index: end_index] += x_vel_cost * self.__discount ** h + \
                                                         action_cost * self.__discount ** h + \
                                                         survival_cost * self.__discount ** h + \
                                                         y_vel_cost * self.__discount ** h + \
                                                         shake_cost * self.__discount ** h + \
                                                         energy_cost * self.__discount ** h
            return all_costs.cpu().detach().numpy()


    def controller(t, w, params):
        a = np.zeros(8)
        for i in range(4):
            a[i] = params[i] * np.sin(w * t + params[8+i])
            a[4 + i] = params[4+i] * np.sin(w * t + params[12+i])
        return a

    config = {
        # exp parameters:
        # "env": 'MinitaurBulletEnv_fastAdapt-v0',
        "env": 'MinitaurGymEnv_fastAdapt-v0',
        "horizon": 25,  # NOTE: "sol_dim" must be adjusted
        "iterations": 200,
        "random_iter": 1,
        "episode_length": 1000,
        "init_state": None,
        "action_dim": 8,
        "video_recording_frequency": 20,
        "logdir": logdir,
        "load_data": None,
        "distance_weight": 1.0,
        "energy_weight": 0.,
        "shake_weight": 0.,
        "drift_weight": 0.,
        "survival_weight": 0.,
        "action_weight": 0.,
        "motor_velocity_limit": np.inf,
        "angle_limit": 1,
        "K": 1,
        "controller": controller,
        "omega": 4*np.pi,

        # Model learning parameters
        "epoch": 1000,
        "learning_rate": 1e-3,
        "minibatch_size": 32,

        # Ensemble model params'log'
        "ensemble_epoch": 5,
        "ensemble_dim_in": 8+31,
        "ensemble_dim_out": 31,
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
    env_args = {'render': False, 'control_time_step': 0.02,
                # 'distance_weight': config['distance_weight'],
                #      'energy_weight': config['energy_weight'], 'survival_weight':config['survival_weight'],
                #      'drift_weight': config['drift_weight'], 'shake_weight':config['shake_weight'],
                #      'action_weight': config['action_weight'], 'motor_velocity_limit': config['motor_velocity_limit'],
                #      'angle_limit': config['angle_limit']
    }
    config['env_args'] = env_args
    if config['controller'] is not None:
        lb = [-0.5] * 8 + [-np.pi] * 8
        ub = [0.5] * 8 + [np.pi] * 8
        config['lb'] = lb
        config['ub'] = ub
        config['sol_dim'] = 16
    main(config)