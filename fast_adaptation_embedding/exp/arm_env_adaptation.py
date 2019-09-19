import fast_adaptation_embedding.env
import fast_adaptation_embedding.models.embedding_nn as nn_model
from fast_adaptation_embedding.models.ffnn import FFNN_Ensemble_Model
from fast_adaptation_embedding.controllers.cem import CEM_opt 
from fast_adaptation_embedding.controllers.random_shooting import RS_opt 
from fast_adaptation_embedding.env.kinematic_arm import Arm_env
import torch
import numpy as np 
import copy
import gym
import time

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
       self.__goal = torch.FloatTensor(goal).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(goal)
       self.__models = self.__ensemble_model.get_models()
    
    def cost_fn(self, samples):
        action_samples = torch.FloatTensor(samples).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(samples)
        init_states = torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0)).cuda() if  self.__ensemble_model.CUDA else torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0))
        all_costs = torch.FloatTensor(np.zeros(len(samples))).cuda() if self.__ensemble_model.CUDA else torch.FloatTensor(np.zeros(len(samples)))
        
        n_model = len(self.__models)
        n_batch = min(n_model, len(samples))
        per_batch = len(samples)/n_batch

        for i in range(n_batch): 
            start_index = int(i*per_batch)
            end_index = len(samples) if i == n_batch-1 else int(i*per_batch + per_batch)
            action_batch = action_samples[start_index:end_index]
            start_states = init_states[start_index:end_index]
            __cost = 0
            dyn_model = self.__models[np.random.randint(0, len(self.__models))]  
            for h in range(self.__horizon):
                actions = action_batch[:,h*self.__action_dim : h*self.__action_dim + self.__action_dim]
                model_input = torch.cat((start_states, actions), dim=1)
                diff_state = dyn_model.predict_tensor(model_input)
                start_states += diff_state
                all_costs[start_index: end_index] += torch.sum(torch.pow(start_states[:,5::]-self.__goal, 2), dim=1) # * config["discount"]**h #+ torch.sum(actions * actions, dim=1) * 0.1 * config["discount"]**h
                
                # '''verify'''
                # print("start sample index: ", start_index)
                # print("end sample index: ", end_index)
                # print("action start index: ", h*self.__action_dim)
                # print("action end index: ", h*self.__action_dim + self.__action_dim)
                # print("current_state: ", start_states[3])
                # print("eff_state: ", start_states[3][5::])
                # print("cost: ", all_costs[3])
                # __state =  start_states[3].cpu().detach().numpy()
                # __goal = self.__goal.cpu().detach().numpy()
                # __cost += -np.sum(np.power(__state[5::]-__goal,2))
                # print("manual cost: ", __cost)
                # input()
        # print("Cost evaluations done...back to optimizer")
        return all_costs.cpu().detach().numpy()

def train_meta(tasks_in, tasks_out, config):
    model = nn_model.Embedding_NN(dim_in=config["dim_in"], hidden=config["hidden_layers"], dim_out=config["dim_out"], embedding_dim=config["embedding_size"], num_tasks=len(tasks_in), CUDA=config["cuda"], SEED=None, input_limit=1.0, dropout=0.0)
    nn_model.train_meta(model, tasks_in, tasks_out, meta_iter=config["meta_iter"], inner_iter=config["inner_iter"], inner_step=config["inner_step"], meta_step=config["meta_step"], minibatch=config["meta_batch_size"])
    return model

def train_model(model, train_in, train_out, task_id, config):
    cloned_model =copy.deepcopy(model)
    nn_model.train(cloned_model, train_in, train_out, task_id=task_id, inner_iter=config["epoch"], inner_lr=config["learning_rate"], minibatch=config["training_batch_size"])
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
    network.train(epochs=config["ensemble_epoch"], training_inputs=train_in, training_targets=train_out, batch_size=config["ensemble_batch_size"], logInterval=config["ensemble_log_interval"], sampling_size=sampling_size)
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
    return np.array(training_in), np.array(training_out)

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
            # env.render("human")
            # time.sleep(0.01)
        trajectory.append([current_state.copy(), a.copy(), next_state-current_state, -r])
        current_state = next_state
        traject_cost += -r
    return np.array(trajectory), traject_cost

def execute_2(env, init_state, steps, init_mean, init_var, model, config, last_action_seq):
    current_state = env.reset()
    trajectory = []
    traject_cost = 0
    model_error = 0
    sliding_mean = np.zeros(config["sol_dim"])
    # sliding_mean = init_mean
    for i in range(steps):
        # seq = last_action_seq[i*config["action_dim"] : i*config["action_dim"]+config["sol_dim"]]
        # sliding_mean[0:len(seq)] = seq
        cost_object = Cost_ensemble(ensemble_model=model, init_state=current_state, horizon=config["horizon"], action_dim=env.action_space.shape[0], goal=config["goal"]) 
        config["cost_fn"] = cost_object.cost_fn   
        optimizer = CEM_opt(config)
        sol = optimizer.obtain_solution(sliding_mean, init_var)
        # optimizer = RS_opt(config)
        # sol = optimizer.obtain_solution()
        a = sol[0:env.action_space.shape[0]]
        next_state, r = 0, 0 
        for k in range(1):
            next_state, r, _, _ = env.step(a)
            env.render()
        trajectory.append([current_state.copy(), a.copy(), next_state-current_state, -r])
        model_error += test_model(model, current_state.copy(), a.copy(), next_state-current_state)
        current_state = next_state
        traject_cost += -r
        sliding_mean[0:-len(a)] = sol[len(a)::]

    print("Model error: ", model_error)
    # full_model_propagation(model, env.reset(), np.array(trajectory), steps)
    return np.array(trajectory), traject_cost

def test_model(ensemble_model, init_state, action, state_diff):
    x = np.concatenate(([init_state], [action]), axis=1)
    y = state_diff.reshape(1,-1)
    # print("True: ", y.flatten()[5::] + init_state[5::])
    for m in ensemble_model.get_models():
        y_pred = m.predict(x)
        # print("pred: ", y_pred.flatten()[5::] + init_state[5::])
    # print("\n")

    return np.power(y-y_pred,2).sum()

def full_model_propagation(ensemble_model, init_state, trajectory, steps):
    curr_state = np.array([init_state])
    for t in range(steps):
        a = np.array([trajectory[t][1]])
        x = np.concatenate((curr_state, a), axis=1)
        diff = ensemble_model.get_models()[0].predict(x)
        curr_state +=diff
        print("action: ", a[0].tolist())
        print(curr_state.flatten()[5::])

def extract_action_seq(data):
    actions = []
    for d in data:
        actions += d[1].tolist()
    return np.array(actions)


config = {
    #exp parameters:
    "horizon": 10, #NOTE: "sol_dim" must be adjusted
    "iterations": 1000,
    "episode_length":50,
    "init_state": None, #Must be updated before passing config as param
    "action_dim":5,
    "goal":[0, 0.5],

    #Model_parameters
    "dim_in": 5+7,
    "dim_out": 7,
    "hidden_layers": [128, 128],
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
    "ensemble_epoch": 10,
    "ensemble_dim_in": 5+7,
    "ensemble_dim_out": 7,
    "ensemble_hidden": [20, 20],
    "hidden_activation": "tanh",
    "ensemble_cuda": True,
    "ensemble_seed": None, 
    "ensemble_output_limit": 1, 
    "ensemble_dropout": 0.1, 
    "n_ensembles": 5,
    "ensemble_batch_size": 64,
    "ensemble_log_interval":500,

    #Optimizer parameters
    "max_iters": 5, 
    "epsilon": 0.0001, 
    "lb": -1., 
    "ub": 1.,
    "popsize": 500,
    "sol_dim": 5*10, #NOTE: Depends on Horizon 
    "num_elites": 50,
    "cost_fn": None, 
    "alpha": 0.1,
    "discount":0.9
}

#************************************************

joint_mismatch = np.array([
                        [1., 1., 1., 1., 1.], 
                        [1., 0.8, 1., 1., 1.], 
                        [1., 1.4, 1., 1., 1.], 
                        [0.5, 0.9, 1., 1., 1.], 
                        [0.8, 1.0, 1., 1.4, 1.]])
np.save("tasks.npy", joint_mismatch)
n_task = len(joint_mismatch)
envs = [Arm_env(goal=tuple(config["goal"]), joint_mismatch=joint_mismatch[i]) for i in range(n_task)]
for e in envs:
    e.render(mode="human")
data = n_task * [None]
models = n_task * [None]
best_action_seq = np.random.rand(config["sol_dim"])*2.0 - 1.0
best_cost = 10000 
last_action_seq = None
all_action_seq = []
all_costs = []

for index_iter in range(config["iterations"]):
    
    '''Pick a random environment'''
    env_index = np.random.randint(0, n_task)
    env = envs[env_index]
    config["goal"] = np.random.uniform(-0.5, 0.5, size=2)
    env.set_goal(config["goal"])

    print("Episode: ", index_iter)
    print("Env index: ", env_index)
    
    if data[env_index] is None or index_iter<10:
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
        x, y = process_data(data[env_index])
        print("Learning model...")
        models[env_index] = train_ensemble_model(train_in=x, train_out=y, sampling_size=-1, config=config, model=models[env_index])
        print("Execution...")
        
        trajectory, c = execute_2(env=env, 
                                init_state=config["init_state"], 
                                model=models[env_index], 
                                steps= config["episode_length"], 
                                init_mean=best_action_seq[0:config["sol_dim"]] , 
                                init_var=0.1 * np.ones(config["sol_dim"]), 
                                config=config,
                                last_action_seq=all_action_seq[np.random.randint(0, len(all_action_seq))])
                                # last_action_seq=all_action_seq[np.random.randint(0, len(all_action_seq))])
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

        np.save("trajectorires.npy", data)
        np.save("best_cost.npy", best_cost)
        np.save("best_action_seq.npy", best_action_seq)

    print("-------------------------------\n")