import fast_adaptation_embedding.env
import fast_adaptation_embedding.models.embedding_nn_normalized as nn_model
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

class Cost_ensemble(object):
    def __init__(self, ensemble_model, init_state, horizon, action_dim, goal, pred_high, pred_low):
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
        action_samples = torch.FloatTensor(samples).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(samples)
        init_states = torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0)).cuda() if  self.__ensemble_model.CUDA else torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0))
        all_costs = torch.FloatTensor(np.zeros(len(samples))).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(np.zeros(len(samples)))
        
        n_model = len(self.__models)
        # n_batch = min(n_model, int(len(samples)/1024))
        n_batch = max(1, int(len(samples)/1024))
        per_batch = len(samples)/n_batch

        for i in range(n_batch): 
            start_index = int(i*per_batch)
            end_index = len(samples) if i == n_batch-1 else int(i*per_batch + per_batch)
            action_batch = action_samples[start_index:end_index]
            start_states = init_states[start_index:end_index]
            dyn_model = self.__models[np.random.randint(0, len(self.__models))]  
            for h in range(self.__horizon):
                actions = action_batch[:,h*self.__action_dim : h*self.__action_dim + self.__action_dim]
                model_input = torch.cat((start_states, actions), dim=1)
                diff_state = dyn_model.predict_tensor(model_input)
                start_states += diff_state
                for dim in range(self.__obs_dim):
                    start_states[:,dim].clamp_(self.__pred_low[dim], self.__pred_high[dim])

                action_cost = torch.sum(actions * actions, dim=1) * 0.0
                x_vel_cost = -start_states[:,13]
                survive_cost = (start_states[:,0]<0.26).type(start_states.dtype) * 2.0
                all_costs[start_index: end_index] +=  x_vel_cost * config["discount"]**h + action_cost * config["discount"]**h + survive_cost * config["discount"]**h
        return all_costs.cpu().detach().numpy()

class Cost_meta(object):
    def __init__(self,  model, init_state, horizon, action_dim, goal, task_likelihoods, pred_high, pred_low):
        self.__model = model
        self.__init_state = init_state
        self.__horizon = horizon
        self.__action_dim = action_dim
        self.__goal = torch.FloatTensor(goal).cuda() if self.__model.cuda_enabled else torch.FloatTensor(goal)
        self.__task_likelihoods = np.array(task_likelihoods)
        self.__n_tasks = len(task_likelihoods) 

        self.__pred_high = pred_high
        self.__pred_low = pred_low
        self.__obs_dim = len(init_state)

    def cost_fn(self, samples):
        action_samples = torch.FloatTensor(samples).cuda() if self.__model.cuda_enabled else torch.FloatTensor(samples)
        init_states = torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0)).cuda() if self.__model.cuda_enabled else torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0))
        all_costs = torch.FloatTensor(np.zeros(len(samples))).cuda() if self.__model.cuda_enabled else torch.FloatTensor(np.zeros(len(samples)))

        n_batch = max(1, int(len(samples)/1024))
        per_batch = len(samples)/n_batch
        task_ids = np.random.randint(0, self.__n_tasks, size=(len(samples), 1))
        task_ids_tensor = torch.LongTensor(task_ids).cuda() if self.__model.cuda_enabled else torch.LongTensor(task_ids)
        likelihoods = self.__task_likelihoods[task_ids.flatten()]
        likelihood_tensor = torch.FloatTensor(likelihoods).cuda() if self.__model.cuda_enabled else torch.FloatTensor(likelihoods)

        for i in range(n_batch): 
            start_index = int(i*per_batch)
            end_index = len(samples) if i == n_batch-1 else int(i*per_batch + per_batch)
            action_batch = action_samples[start_index:end_index]
            start_states = init_states[start_index:end_index]
            tasks = task_ids_tensor[start_index:end_index]
            __cost = 0
            for h in range(self.__horizon):
                actions = action_batch[:,h*self.__action_dim : h*self.__action_dim + self.__action_dim]
                model_input = torch.cat((start_states, actions), dim=1)
                diff_state = self.__model.predict_tensor(model_input, tasks)
                start_states += diff_state
                
                for dim in range(self.__obs_dim):
                    start_states[:,dim].clamp_(self.__pred_low[dim], self.__pred_high[dim])
                
                action_cost = torch.sum(actions * actions, dim=1) * 0.0
                x_vel_cost = -start_states[:,13]
                survive_cost = (start_states[:,0]<0.26).type(start_states.dtype) * 2.0
                all_costs[start_index: end_index] +=  x_vel_cost * config["discount"]**h + action_cost * config["discount"]**h + survive_cost * config["discount"]**h    

        return (all_costs * likelihood_tensor).cpu().detach().numpy()

def train_meta(tasks_in, tasks_out, config):
    model = nn_model.Embedding_NN(dim_in=config["dim_in"], 
                                hidden=config["hidden_layers"], 
                                dim_out=config["dim_out"], 
                                embedding_dim=config["embedding_size"], 
                                num_tasks=len(tasks_in), 
                                CUDA=config["cuda"], 
                                SEED=None, 
                                output_limit=config["output_limit"], 
                                dropout=0.0,
                                hidden_activation=config["hidden_activation"])
    nn_model.train_meta(model, 
                        tasks_in, 
                        tasks_out, 
                        meta_iter=config["meta_iter"], 
                        inner_iter=config["inner_iter"], 
                        inner_step=config["inner_step"], 
                        meta_step=config["meta_step"], 
                        minibatch=config["meta_batch_size"])
    return  model

def train_model(model, train_in, train_out, task_id, config):
    cloned_model =copy.deepcopy(model)
    nn_model.train(cloned_model, 
                train_in, 
                train_out, 
                task_id=task_id, 
                inner_iter=config["epoch"], 
                inner_lr=config["learning_rate"], 
                minibatch=config["minibatch_size"])
    return cloned_model

def train_ensemble_model(train_in, train_out, sampling_size, config, model= None):
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
    
    network.train(epochs=config["ensemble_epoch"], 
                training_inputs=train_in, 
                training_targets=train_out, 
                batch_size=config["ensemble_batch_size"], 
                logInterval=config["ensemble_log_interval"], 
                sampling_size=sampling_size)
    return copy.deepcopy(network)

def process_data(data):
    '''Assuming dada: an array containing [state, action, state_transition, cost] '''
    training_in = []
    training_out = []
    for d in data:
        s = d[0]
        a = d[1]
        training_in.append(np.concatenate((s,a)))
        training_out.append(d[2])
    return np.array(training_in), np.array(training_out), np.max(training_in, axis=0), np.min(training_in, axis=0)

def execute_random(env, steps, init_state):
    current_state = env.reset()
    trajectory = []
    traject_cost = 0
    for i in range(steps):
        a = env.action_space.sample()
        next_state, r = 0, 0 
        for k in range(1):
            next_state, r, _, _ = env.step(a)
            env.render()
        trajectory.append([current_state.copy(), a.copy(), next_state-current_state, -r])
        current_state = next_state
        traject_cost += -r
    return np.array(trajectory), traject_cost

def execute_2(env, init_state, steps, init_mean, init_var, model, config, last_action_seq, task_likelihoods, pred_high, pred_low):
    current_state = env.reset()
    trajectory = []
    traject_cost = 0
    sliding_mean = np.zeros(config["sol_dim"])
    for i in range(steps):
        cost_object = Cost_meta(model=model, init_state=current_state, horizon=config["horizon"], action_dim=env.action_space.shape[0], goal=config["goal"], task_likelihoods=task_likelihoods, pred_high=pred_high, pred_low=pred_low) 
        config["cost_fn"] = cost_object.cost_fn   
        optimizer = RS_opt(config)
        sol = optimizer.obtain_solution()         
        a = sol[0:env.action_space.shape[0]]
        next_state, r = 0, 0 
        for k in range(1):
            next_state, r, _, _ = env.step(a)
        trajectory.append([current_state.copy(), a.copy(), next_state-current_state, -r])
        current_state = next_state
        traject_cost += -r
        sliding_mean[0:-len(a)] = sol[len(a)::]

    return np.array(trajectory), traject_cost

def extract_action_seq(data):
    actions = []
    for d in data:
        actions += d[1].tolist()
    return np.array(actions)


config = {
    #exp parameters:
    "horizon": 20, #NOTE: "sol_dim" must be adjusted
    "iterations": 1000,
    "episode_length":1000,
    "init_state": None, #Must be updated before passing config as param
    "action_dim": 8,

    #Model_parameters
    "dim_in": 8+27,
    "dim_out": 27,
    "hidden_layers": [200, 200, 100],
    "embedding_size": 5,
    "cuda": True,
    "output_limit": 10.0,

    #Meta learning parameters
    "meta_iter": 1000,#5000,
    "meta_step": 0.1,
    "inner_iter": 20,#10,
    "inner_step": 0.0001,
    "meta_batch_size": 32,

    #Model learning parameters
    "epoch": 10,
    "learning_rate": 1e-3,
    "minibatch_size": 64,

    #Ensemble model params
    # "ensemble_epoch": 10,
    # "ensemble_dim_in": 5+7,
    # "ensemble_dim_out": 7,
    # "ensemble_hidden": [20, 20],
    "hidden_activation": "relu",
    # "ensemble_cuda": True,
    # "ensemble_seed": None, 
    # "ensemble_output_limit": 1, 
    # "ensemble_dropout": 0.1, 
    # "n_ensembles": 5,
    # "ensemble_batch_size": 64,
    # "ensemble_log_interval":500,

    #Optimizer parameters
    "max_iters": 5, 
    "epsilon": 0.0001, 
    "lb": -1., 
    "ub": 1.,
    "popsize": 500,
    "sol_dim": 8*20, #NOTE: Depends on Horizon 
    "num_elites": 50,
    "cost_fn": None, 
    "alpha": 0.1,
    "discount": 1.0
}

#************************************************
trained_mismatches = np.array([[1., 1., 1., 1., 1., 1., 1., 1.], [1., 0., 1., 1., 1., 1., 1., 1.], [1., 1., 0., 1., 1., 1., 1., 1.], [1., 1., 1., 0., 1., 1., 1., 1.], [1., 1., 1., 1., 0., 1., 1., 1.]])
n_training_tasks = 5

mismatches = trained_mismatches[0:1] + np.array([[0., 0.0, 0.0, 0.0, 0., 0.0, 0., 0.]])
n_task = len(mismatches)
envs = [gym.make("AntMuJoCoEnv_fastAdapt-v0") for i in range(n_task)]
for i,e in enumerate(envs):
    e.set_mismatch(mismatches[i])
    e.render(mode="human")

data = n_task * [None]
models = n_task * [None]
best_action_seq = np.random.rand(config["sol_dim"])*2.0 - 1.0
best_cost = 10000 
last_action_seq = None
all_action_seq = []
all_costs = []

'''--------------------Meta learn the models---------------------------'''
#Arrange data
meta_model = None
if not path.exists("ant_meta_model.pickle"):
    print("Model not found. Learning from data...")
    meta_data = np.load("./ant_data/trajectories_ant.npy")
    tasks_in, tasks_out = [], []
    for n in range(n_training_tasks):
        x, y, high, low = process_data(meta_data[n])
        tasks_in.append(x)
        tasks_out.append(y)
        print("task ", n, " data: ", len(tasks_in[n]), len(tasks_out[n]))
    meta_model = train_meta(tasks_in, tasks_out, config)

    pickle_out = open("ant_meta_model.pickle","wb")
    pickle.dump(meta_model, pickle_out)
    pickle_out.close()
else:
    print("Model found. Loading from pickle")
    pickle_in = open("ant_meta_model.pickle","rb")
    meta_model = pickle.load(pickle_in)

models =  [copy.deepcopy(meta_model) for _ in range(n_task)]

'''------------------------Test time------------------------------------'''
with open("costs.txt","w+") as f:
    f.write("")

for index_iter in range(config["iterations"]):
    
    '''Pick a random environment'''
    env_index = np.random.randint(0, n_task)
    env = envs[env_index]
    config["goal"] = np.random.uniform(-0.5, 0.5, size=2)

    print("Episode: ", index_iter)
    print("Env index: ", env_index)
    
    if data[env_index] is None or index_iter<1:
        print("Execution (Random actions)...")
        trajectory, c = execute_random(env=env, steps= config["episode_length"], init_state= config["init_state"])
        if data[env_index] is None:
            data[env_index] = trajectory
        else:
            data[env_index] = np.concatenate((data[env_index], trajectory), axis=0)
        print("Cost : ", c)
        
        if c<best_cost:
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
        x, y, high, low = process_data(data[env_index])
        print("Learning model...")
        models[env_index] = train_model(model=models[env_index], train_in=x, train_out=y, task_id=env_index, config=config)
        print("Execution...")
        task_likelihoods = np.ones(n_training_tasks) * 0.1/n_training_tasks
        task_likelihoods[0] = 1.0 - 0.1
        trajectory, c = execute_2(env=env, 
                                init_state=config["init_state"], 
                                model=models[env_index], 
                                steps= config["episode_length"], 
                                init_mean=best_action_seq[0:config["sol_dim"]] , 
                                init_var=0.1 * np.ones(config["sol_dim"]), 
                                config=config,
                                last_action_seq=all_action_seq[np.random.randint(0, len(all_action_seq))],
                                task_likelihoods=task_likelihoods,
                                pred_high= high,
                                pred_low=low)
        data[env_index] = np.concatenate((data[env_index], trajectory), axis=0)
        print("Cost : ", c)
        with open("costs.txt","a+") as f:
            f.write(str(c)+"\n")
    
        if c<best_cost:
            best_cost = c
            best_action_seq = []
            for d in trajectory:
                best_action_seq += d[1].tolist()
            best_action_seq = np.array(best_action_seq)
            last_action_seq = extract_action_seq(trajectory)

        all_action_seq.append(extract_action_seq(trajectory))
        all_costs.append(c)

        np.save("trajectories.npy", data)
        np.save("best_cost.npy", best_cost)
        np.save("best_action_seq.npy", best_action_seq)

    print("-------------------------------\n")