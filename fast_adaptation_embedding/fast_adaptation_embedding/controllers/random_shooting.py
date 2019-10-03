
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import scipy.stats as stats
import numpy as np

class RS_opt(object):
    def __init__(self, config):
        self.initial_boost = config["initial_boost"]#1
        self.lb, self.ub = config["lb"], config["ub"]#-1, 1
        self.popsize = config["popsize"] #500
        self.sol_dim = config["sol_dim"] #8*20 #action dim*horizon
        self.cost_function = config["cost_fn"]
        self.horizon = config['horizon']
        self.omega = config["omega"]
        self.dt = config["control_time_step"]
        self.controller = config["controller"]

    def obtain_solution(self, init_mean=None, init_var=None, t0=0):
        """Optimizes the cost function using the provided initial candidate distribution
        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if init_mean is None or init_var is None:
            samples = np.random.uniform(self.lb, self.ub, size=(self.initial_boost*self.popsize, self.sol_dim))
        else:
            assert init_mean is not None and init_var is not None, "init mean and var must be provided"
            samples = np.random.normal(init_mean, init_var, size=(self.popsize, self.sol_dim))
            samples = np.clip(samples, self.lb, self.ub)
        if self.controller is not None:
            actions = np.concatenate([samples[:, 0:8] * np.sin(self.omega * (t0 + t * self.dt) + samples[:, 8:16])
                                      for t in range(self.horizon)], axis=1)
            costs = self.cost_function(np.array(actions))
        else:
            costs = self.cost_function(samples)
        return samples[np.argmin(costs)]
