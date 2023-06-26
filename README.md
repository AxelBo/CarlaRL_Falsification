# Local Reinforcement Learning for Carla Falsification 
This repository contains code and resources for the project "Carla RL Falsification," which focuses on the falsification and optimization of the Carla Traffic Manager using Reinforcement Learning (RL) techniques. The goal of this project is to identify and evaluate critical traffic situations for autonomous vehicles.

## Installation and prequesites:
Please ensure the following software and libraries are installed:
- Install [Carla](https://carla.readthedocs.io/en/latest/start_quickstart/)
- Upgrade pip: ` pip3 install --upgrade pip`
- Install numpy: `pip3 install numpy`
- Install matplotlib: `pip3 install matplotlib`
- Install sklearn: `pip3 install sklearn`
- Install pandas: `pip3 install pandas`
- Install Shapely: `pip3 install shapely`
- Install gym: `pip3 install gym`
- Install stable_baselines3: `pip3 install stable_baselines3`
- Install tensorboard: `pip3 install tensorboard`
- Install torch: `pip3 install torch`
- Install optuna: `pip3 install optuna`

## Supported Versions
This repository has been tested with the following software versions:
- Python 3.8
- Carla 0.9.13
- pip 23.0.1
- numpy 1.23.5
- matplotlib 3.6.3
- sklearn 0.0.post1
- pandas 1.5.3
- Shapely 2.0.1
- gym 0.21.0
- stable-baselines3 1.7.0
- tensorboard 2.12.0
- torch 1.13.1
- optuna 3.1.1


## How to use
1. Configure the parameters in the file `config.py` (e.g. start positions, time steps, reward distribution, ...)
2. Start Carla
3. 
   a) Use existing Hyperparameter and train the Agent with `local_reinforcement_learning.py`

   b) Use the `optuna_optimization.py` to optimize the parameters with Optuna and Train the agent. 
4. Use `PowerAnalysis.py` to analyze the results of the training. And calculate the power of the test.
5. Run the saved Model or Action Sequence with `predictActions_runActions.py` to replay the results in Carla.

## Files and functions
A brief description of the files and functions in this repository is provided below.
- `config.py`: Contains all parameters for the training and the simulation.
- `connectionCarla.py`: Manages the connection to Carla.
- `envirionment.py`: Contains the environment for the agent.
- `helper_functions.py`: Contains helper functions for the different porpuses.
- `local_reinforcement_learning.py`: Trains the agent with the given parameters.
- `optuna_optimization.py`: Optimizes the parameters with Optuna and trains the agent.
- `PowerAnalysis.py`: Analyzes the results of the training and calculates the power of the test.
- `predictActions_runActions.py`: Runs the saved Model or Action Sequence in Carla.
- `walker_vehicle.py`: Contains the walker and vehicle class for the environment.
## Possible Extensions
- Train on different Parameters
- Select different starting positions
- Use different metrics
- Optimize for a different porpuse
- ...