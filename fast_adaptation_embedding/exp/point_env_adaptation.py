from fast_adaptation_embedding.env.point_env import Point
import fast_adaptation_embedding.models.embedding_nn as nn_model
from fast_adaptation_embedding.models.ffnn import FFNN_Ensemble_Model
from fast_adaptation_embedding.controllers.cem import CEM_opt 
import numpy as np 
import copy

class Point_env(Point):
    def __init__(self, goal, mismatch=np.ones(2)):
        self.mismatch  = mismatch
        super().__init__(goal)
    
    def step(self, action):
        self.state = self.state + action*self.mismatch
        cost = np.linalg.norm(self.state-self.goal)
        return self.state.copy(), cost, False, {}

class Cost_meta(object):
    def __init__(self, model, init_state, horizon, action_dim, goal, task_likelihoods):
       self.__model = model
       self.__init_state = init_state
       self.__horizon = horizon
       self.__action_dim = action_dim
       self.__goal = goal
       self.__task_likelihoods = task_likelihoods
    
    def cost_given_task(self, samples, task_id):
        all_costs = []
        for s in samples:
            state = self.__init_state.copy()
            episode_cost = 0
            for i in range(self.__horizon):
                action = s[self.__action_dim*i : self.__action_dim*i + self.__action_dim]
                x = np.concatenate((state, action), axis=1).reshape(1, self.__action_dim) 
                diff_state = self.__model.predict(x, np.array([[task_id]])).data.cpu().numpy().flatten()
                state += diff_state
                episode_cost += np.linalg.norm(self.__goal - state)            
            all_costs.append(episode_cost * self.__task_likelihoods[task_id])
        return np.array(all_costs)

    def cost_fn(self, samples):
        all_costs = np.zeros(len(samples))
        for i in range(len(self.__task_likelihoods)):
            all_costs += self.cost_given_task(samples, i)
        return all_costs / np.sum(self.__task_likelihoods)


class Cost_ensemble(object):
    def __init__(self, ensemble_model, init_state, horizon, action_dim, goal):
       self.__ensemble_model = ensemble_model
       self.__init_state = init_state
       self.__horizon = horizon
       self.__action_dim = action_dim
       self.__goal = goal
       self.__models = self.__ensemble_model.get_models()
    
    def cost_given_model(self, samples, model):
        all_costs = []
        for s in samples:
            state = self.__init_state.copy()
            episode_cost = 0
            for i in range(self.__horizon):
                action = s[self.__action_dim*i : self.__action_dim*i + self.__action_dim]
                x = np.concatenate((state, action)).reshape(1, -1) 
                diff_state = model.predict(x)
                state += diff_state.flatten()
                episode_cost += np.linalg.norm(self.__goal - state)            
            all_costs.append(episode_cost)
        return np.array(all_costs)

    def __cost_fn(self, samples):
        all_costs = np.zeros(len(samples))
        for i in range(len(self.__models)):
            all_costs += self.cost_given_model(samples, self.__models[i])
        return all_costs / len(self.__models)
    
    def cost_fn(self, samples):
        all_costs = []
        for s in samples:
            state = self.__init_state.copy()
            episode_cost = 0
            model = self.__models[np.random.randint(0, len(self.__models))]
            for i in range(self.__horizon):
                action = s[self.__action_dim*i : self.__action_dim*i + self.__action_dim]
                x = np.concatenate((state, action)).reshape(1, -1) 
                diff_state = model.predict(x)
                state += diff_state.flatten()
                episode_cost += np.linalg.norm(self.__goal - state)            
            all_costs.append(episode_cost)
        return np.array(all_costs)

def train_meta(tasks_in, tasks_out, config):
    model = nn_model.Embedding_NN(dim_in=config["dim_in"], hidden=config["hidden_layers"], dim_out=config["dim_out"], embedding_dim=config["embedding_size"], num_tasks=len(tasks_in), CUDA=config["cuda"], SEED=None, input_limit=1.0, dropout=0.0)
    nn_model.train_meta(model, tasks_in, tasks_out, meta_iter=config["meta_iter"], inner_iter=config["inner_iter"], inner_step=config["inner_step"], meta_step=config["meta_step"], minibatch=config["meta_batch_size"])
    return model

def train_model(model, train_in, train_out, task_id, config):
    cloned_model =copy.deepcopy(model)
    nn_model.train(cloned_model, train_in, train_out, task_id=task_id, inner_iter=config["epoch"], inner_lr=config["learning_rate"], minibatch=config["training_batch_size"])
    return cloned_model

def train_ensemble_model(train_in, train_out, sampling_size, config):
    model = FFNN_Ensemble_Model(dim_in=config["ensemble_dim_in"], hidden=config["ensemble_hidden"], dim_out=config["ensemble_dim_out"], CUDA=config["ensemble_cuda"], SEED=config["ensemble_seed"], output_limit=config["ensemble_output_limit"], dropout=config["ensemble_dropout"], n_ensembles=config["n_ensembles"])
    model.train(epochs=config["ensemble_epoch"], training_inputs=train_in, training_targets=train_out, batch_size=config["ensemble_batch_size"], logInterval=config["ensemble_log_interval"], sampling_size=sampling_size)
    return model

def process_data(data):
    '''Assuming dada: an array containing [state, action, state_transition, cost] '''
    training_in = []
    training_out = []
    for d in data:
        s = d[0]
        a = d[1]
        training_in.append(np.concatenate((s,a)))
        training_out.append(d[2])
    return np.array(training_in), np.array(training_out)

def execute_random(env, steps, init_state):
    current_state = env.reset(init_state)
    trajectory = []
    traject_cost = 0
    for i in range(steps):
        a = env.action_space.sample()
        next_state, cost, _, _ = env.step(a)
        trajectory.append([current_state.copy(), a.copy(), next_state-current_state, cost])
        current_state = next_state
        traject_cost += cost
    return np.array(trajectory), traject_cost

def execute(env, init_state, steps, init_mean, init_var, model, config):
    current_state = env.reset(init_state)
    trajectory = []
    traject_cost = 0
    for i in range(steps):
        cost_object = Cost_ensemble(ensemble_model=model, init_state=current_state, horizon=config["horizon"], action_dim=env.action_space.shape[0], goal=goal) 
        config["cost_fn"] = cost_object.cost_fn   
        optimizer = CEM_opt(config)
        a = optimizer.obtain_solution(init_mean, init_var)[0:env.action_space.shape[0]]
        next_state, cost, _, _ = env.step(a)
        trajectory.append([current_state.copy(), a.copy(), next_state-current_state, cost])
        current_state = next_state
        traject_cost += cost
        env.render(0.3)
    return np.array(trajectory), traject_cost

config = {
    #exp parameters:
    "horizon": 4,
    "iterations": 100,
    "episode_length": 50,
    "init_state": np.array([0.,0.]),

    #Model_parameters
    "dim_in": 2+2,
    "dim_out": 2,
    "hidden_layers": [10, 10],
    "embedding_size": 5,
    "cuda": False,

    #Meta learning parameters
    "meta_iter": 5000,
    "meta_step": 0.1,
    "inner_iter": 10,
    "inner_step": 0.02,

    #Model learning parameters
    "epoch": 1000,
    "learning_rate": 1e-3,
    "minibatch_size": 32,

    #Ensemble model params
    "ensemble_epoch": 1000,
    "ensemble_dim_in": 2+2,
    "ensemble_dim_out": 2,
    "ensemble_hidden": [10, 10],
    "ensemble_cuda": False,
    "ensemble_seed": None, 
    "ensemble_output_limit": 2.0, 
    "ensemble_dropout": 0.0, 
    "n_ensembles": 8,
    "ensemble_batch_size": 32,
    "ensemble_log_interval":None,

    #Optimizer parameters
    "max_iters": 10, 
    "epsilon": 0.001, 
    "lb": -1, 
    "ub": 1,
    "popsize": 100,
    "sol_dim": 2*10, 
    "num_elites": 10,
    "cost_fn": None, 
    "alpha": 0.1
}

#************************************************

# mismatches = np.array([[1., 1.], [1., 0.8], [1.2, 0.7], [0.8, 1.3], [0.7, 0.8], [0.6, 0.9], [1.4, 1.2], [0.6, 1.4]])
mismatches = np.array([[1., 1.]])
n_task = len(mismatches)
goal = [-10, 15]
envs = [Point_env(goal, mismatches[i]) for i in range(n_task)]
real_env = Point_env(goal, np.array([0.8, 0.9]))
data = n_task * [None]
models = n_task * [None]

for index_iter in range(config["iterations"]):
    '''Pick a random environment'''
    env_index = np.random.randint(0, n_task)
    env = envs[env_index]

    print("Episode: ", index_iter)
    print("Env index: ", env_index)
    
    if data[env_index] is None:
        print("Execution (Random actions)...")
        trajectory, c = execute_random(env=env, steps= config["episode_length"], init_state= config["init_state"])
        data[env_index] = trajectory
        print("Cost : ", c)
    else:    
        print("Execution...")
        trajectory, c = execute(env=env, init_state=config["init_state"], model=models[env_index], steps= config["episode_length"], init_mean=np.zeros(config["sol_dim"]), init_var=0.1 * np.ones(config["sol_dim"]), config=config)
        data[env_index] = np.concatenate((data[env_index], trajectory), axis=0)
        print("Cost : ", c)


    '''------------Update models------------'''
    x, y = process_data(data[env_index])
    print("Learning model...")
    models[env_index] = train_ensemble_model(train_in=x, train_out=y, sampling_size=len(x), config=config)
    print("-------------------------------\n")