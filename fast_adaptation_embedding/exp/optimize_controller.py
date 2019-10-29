import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import fast_adaptation_embedding.env
import gym
import numpy as np
import time
import fast_adaptation_embedding.env
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import cma
import pickle

def controller(t, w, params):
    a = [params[0] * np.sin(w * t + params[8]*2*np.pi),   #s FL
         params[1] * np.sin(w * t + params[9]*2*np.pi),   #s BL
         params[2] * np.sin(w * t + params[10]*2*np.pi),  #s FR
         params[3] * np.sin(w * t + params[11]*2*np.pi),  #s BR
         params[4] * np.sin(w * t + params[12]*2*np.pi),  #e FL
         params[5] * np.sin(w * t + params[13]*2*np.pi),  #e BR
         params[6] * np.sin(w * t + params[14]*2*np.pi),  #e FR
         params[7] * np.sin(w * t + params[15]*2*np.pi)   #e BR
         ]
    return a

def evaluate(params):
    system = gym.make("MinitaurGymEnv_fastAdapt-v0", control_time_step=0.006, accurate_motor_model_enabled=1)
    system.reset()
    rew = 0
    for i in range(166 * 5):
        t = system.minitaur.GetTimeSinceReset()
        w = 4 * np.pi
        a = controller(t, w, params)
        obs, r, done, _ = system.step(a)
        rew += r
    return -rew


init = np.array([0.15196979,  0.44120215,  0.44734516,  0.4737556,  0.4331324,
        0.39177511,  0.32945941,  0.19081727, 0.85514155,  0.31171731,
        0.46467034,  0.03487151,  0.0941874,  0.52535486,  0.41979745,
        0.08657507])
es = cma.CMAEvolutionStrategy(init, 0.3, {'bounds': [np.zeros(16), np.ones(16)]})
i = 0
while not es.stop():
    i+= 1
    solutions = es.ask()
    es.tell(solutions, [evaluate(x) for x in solutions])
    es.logger.add()  # write data to disc to be plotted
    es.disp()
    res = es.result
    with open("outcmaes/best.pk", 'wb') as f:
        pickle.dump(res, f)
    print(i, res[1], res[0])

# 0 xbest best solution evaluated
# 1 fbest objective function value of best solution
# 2 evals_best evaluation count when xbest was evaluated
# 3 evaluations evaluations overall done
# 4 iterations
# 5 xfavorite distribution mean in "phenotype" space, to be considered as current best estimate of the optimum
# 6 stds effective standard deviations, can be used to compute a lower bound on the expected coordinate-wise distance to the true optimum, which is (very) approximately stds[i] * dimension**0.5 / min(mueff, dimension) / 1.5 / 5 ~ std_i * dimension**0.5 / min(popsize / 2, dimension) / 5, where dimension = CMAEvolutionStrategy.N and mueff = CMAEvolutionStrategy.sp.weights.mueff ~ 0.3 * popsize.
# 7 stop termination conditions in a dictionary

