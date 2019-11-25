import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import argparse
import torch
import numpy as np
from exp.env_adaptation import main
from exp.model_comparison import compare
from exp.collect_data import collect
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='append', nargs=2, default=[])
    parser.add_argument('-logdir', type=str, default='log')
    args = parser.parse_args()
    logdir = args.logdir


    class Cost_ensemble(object):
        def __init__(self, ensemble_model, init_state, horizon, action_dim, goal, pred_high, pred_low, config, t0=0):
            self.__ensemble_model = ensemble_model
            self.__init_state = init_state
            self.__horizon = horizon
            self.__action_dim = action_dim
            self.__goal = goal
            self.__models = self.__ensemble_model.get_models()
            self.__pred_high = pred_high
            self.__pred_low = pred_low
            self.__obs_dim = len(init_state)
            self.__discount = config['discount']
            self.__t0 = t0

        def cost_fn(self, samples):
            action_samples = torch.FloatTensor(samples).cuda() \
                if self.__ensemble_model.CUDA \
                else torch.FloatTensor(samples)
            init_states = torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0)).cuda() \
                if self.__ensemble_model.CUDA \
                else torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0))
            all_costs = torch.FloatTensor(np.zeros(len(samples))).cuda() \
                if self.__ensemble_model.CUDA \
                else torch.FloatTensor(np.zeros(len(samples)))

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
                    start_states += diff_state[:, 2:]
                    x_vel_cost = -diff_state[:, 0] * config['xreward']
                    y_vel_cost = -diff_state[:, 1] * config['yreward']
                    all_costs[start_index: end_index] += x_vel_cost * self.__discount ** h + y_vel_cost * self.__discount ** h
            return all_costs.cpu().detach().numpy()


    config = {
        # exp parameters:
        "env": "MinitaurControlledEnv_fastAdapt-v0",
        "env_args": {},
        "horizon": 5,  # NOTE: "sol_dim" must be adjusted
        "T": 1,
        "iterations": 100,
        "random_iter": 50,
        "episode_length": 40,
        "init_state": None,  # Must be updated before passing config as param
        "action_dim": 5,
        "video_recording_frequency": 10,
        "logdir": logdir,
        "load_data": None,
        "motor_velocity_limit": np.inf,
        "angle_limit": 1,
        "K": 25,
        'video.frames_per_second': 50,
        'controller': None,
        'stop_training': np.inf,
        "control_time_step": 0.02,
        "script": 'main',
        "xreward": 1,
        "yreward": 0,
        "friction": 1,
        "slope": 0,
        "accurate_model": False,

        # Model learning parameters
        "epoch": 1000,
        "learning_rate": 1e-3,
        "minibatch_size": 32,

        # Ensemble model params'log'
        "ensemble_epoch": 5,
        "ensemble_dim_in": 5 + 2,
        "ensemble_dim_out": 4,
        "ensemble_hidden": [20, 20],
        "ensemble_contact": False,
        "hidden_activation": "relu",
        "ensemble_cuda": True,
        "ensemble_seed": None,
        "ensemble_output_limit": None,
        "ensemble_dropout": 0.0,
        "n_ensembles": 1,
        "ensemble_batch_size": 64,
        "ensemble_log_interval": 500,
        "model_type": "D",

        # Optimizer parameters
        "max_iters": 1,
        "epsilon": 0.0001,
        "opt": "RS",
        "lb": -1,
        "ub": 1,
        "popsize": 2000,
        "pop_batch": 16384,
        "sol_dim": 0,  # NOTE: Depends on Horizon
        "num_elites": 50,
        "cost_fn": None,
        "alpha": 0.1,
        "discount": 1.,
        "Cost_ensemble": Cost_ensemble,
        "init_var": 0.05,
        "initial_boost": 5,
        "omega": None,
        "only_random": True,

    }
    for (key, val) in args.config:
        if key in ['horizon', 'K', 'popsize', 'iterations', 'n_ensembles', 'episode_length']:
            config[key] = int(val)
        elif key in ['load_data', 'hidden_activation', 'data_size', 'save_data', 'script', 'model_type']:
            config[key] = val
        elif key in ['ensemble_hidden']:
            config[key] = [int(val), int(val), int(val)]
        elif key in ['only_random', 'accurate_model']:
            config[key] = True if val == 'True' else False
        else:
            config[key] = float(val)
    if config['model_type'] == 'C':
        config["ensemble_dim_in"] = 0
        config["ensemble_dim_out"] = 0

    config['video.frames_per_second'] = int(1 / config['control_time_step'])
    config['sol_dim'] = config['horizon'] * config['action_dim']
    config["action_repeat"] = int(250 * config['control_time_step'])
    env_args = {'control_time_step': config['control_time_step'], 'action_repeat': config["action_repeat"],
                'accurate_motor_model_enabled': config['accurate_model'], 'pd_control_enabled': True,
                }
    config['env_args'] = env_args
    if config['ensemble_contact']:
        config['ensemble_dim_out'] += 4
        config['ensemble_dim_in'] += 4
    if config['controller'] is not None:
        lb = -1
        ub = 1
        config['lb'] = lb
        config['ub'] = ub
        config['sol_dim'] = config['action_dim'] * int(config['horizon']/config['T'])
        config['init_var'] = np.array([config['init_var']]) * config['action_dim']
    # if config['model_type'] == "C":
    #     config['Cost_ensemble'] = Cost_ensemble_C

    if config['script'] == "collect":
        collect(config)
    elif config['script'] == "compare":
        compare(config)
    else:
        main(config)
