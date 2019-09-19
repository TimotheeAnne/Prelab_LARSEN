# Fast adaptation in robotics using task embedding in Meta learning

Basic idea: 
* The goal is to find an initial set of parameters for the neural network from which the dynamics of a system can be learnt with a few small number of samples.

* For this we want to use meta-learning, the 'Reptile' implementation that avoids 2nd degree differential and makes meta leaning easier.

* The problem here is if for different situation of the robot the dynamics is very diffferent, then it is not possible to find an initial set of parameters from where the learning can be done with small number of data.

* We propose, task embedding which will make the NN a conditional dynamical model conditioned on the task.


## Dependencies
* gym
* pybulletgym
* pytorch

## Preparation
* Need to include the directory in PYTHONPATH
```shell
export PYTHONPATH=${pwd}:$PYTHONPATH 
```