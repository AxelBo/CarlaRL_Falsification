import datetime
import math
import carla
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

import connectionCarla
from config_file import var_lrl, var_startpos, var_connection, var_paths
from envirionment import CustomEnv
import optuna

# Gets the Connection to Carla with some default parameters
def get_connection(townName="Town03", host="localhost", reloadWorld=var_connection['reloadWorld'],
                   camaraLocation=var_startpos['camLocation'], camRotation=None, render=var_connection['render'],
                   syncMode=var_connection['syncMode'],
                   port=2000, delta=0.05):
    conncetion = connectionCarla.CarlaConnection()
    conncetion.__int__(townName=townName, host=host, reloadWorld=reloadWorld,
                       camaraLocation=camaraLocation, camRotation=camRotation, render=render, syncMode=syncMode,
                       port=port, delta=delta)
    return conncetion


# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5

# Gets the x, y and z pos of an actor
def get_actor_position(world, vehicle_id):
    vehicle = world.get_actor(vehicle_id)
    if vehicle is not None:
        vehicle_location = vehicle.get_location()
        x_pos = vehicle_location.x
        y_pos = vehicle_location.y
        z_pos = vehicle_location.z

        return [x_pos, y_pos, z_pos]
    else:
        print("Actor not found")
        return None

# Retruns distance beween two actors in meters
def distance_actor(actor1, actor2):
    try:
        pos1 = actor1.get_location()
        pos2 = actor2.get_location()
        distance = math.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)
        return distance
    except:
        print("Fail")
        return 999

# Returns all near walkers within a set distance
def find_nearWalkers(world, vehicle, max_distance=30, special_id=None):
    walkers_list = world.get_actors().filter('walker.pedestrian.*')
    near_walkers = [walker for walker in walkers_list if
                    distance_actor(vehicle, walker) < max_distance]
    if special_id:
        selected_actor = None
        for walker in near_walkers:
            if walker.id == special_id:
                selected_actor = walker
                break
        return [selected_actor]
    else:
        return near_walkers

# Predicts a given model and returns the mean reward
def predictModel(env, model, steps, num_runs=3, render_afterwards=False, render_mode="human"):
    rewards_all = []

    for run in range(num_runs):
        obs = env.reset()
        rewards = 0

        for i in range(steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

            if not done:
                rewards += reward
            else:
                break

        rewards_all.append(rewards)
        print(f"Run {run + 1}: Rewards = {rewards}")

    mean_rewards = np.mean(rewards_all)

    # Render the results
    env.render(render_mode) if render_afterwards else env.render("nope")

    return mean_rewards

# Returns the environment with some default parameters
def getEnvirionment(envName="CustomEnv", time_steps_per_training=var_lrl['time_steps_per_trainings'], syncMode=True,
                    render=True,
                    spawnInfo=var_startpos['spawnInfo'], reloadMap=var_connection['reloadMap'],
                    camaraLocation=var_startpos['camLocation'],
                    camRotation=carla.Rotation(pitch=-60), port=2000):
    env = CustomEnv(time_steps_per_training=time_steps_per_training, syncMode=syncMode, render=render,
                        spawnInfo=spawnInfo, reloadMap=reloadMap, camaraLocation=camaraLocation,
                        camRotation=camRotation, port=port)
    return env

# Returns the current time as a string
def get_current_timestamp():
    x = datetime.datetime.now()
    return f"{x.year}_{x.month}_{x.day}_{x.hour}_{x.minute}"

# Function to create an optuna study
def create_optuna_storageMySQL(user, pw, ip, dbName, studyName, direction='maximize'):
    storage = optuna.storages.RDBStorage(
        url=f'mysql://{user}:{pw}@{ip}/{dbName}',
        engine_kwargs={
            'pool_size': 20,
            'max_overflow': 0
        }
    )
    return optuna.create_study(study_name=studyName, storage=storage, direction=direction)

# Function to load an optuna study
def load_optuna_studyMySQL(user, pw, ip, dbName, studyName):
    storage = optuna.storages.RDBStorage(
        url=f'mysql://{user}:{pw}@{ip}/{dbName}',
        engine_kwargs={
            'pool_size': 20,
            'max_overflow': 0
        }
    )
    study = optuna.load_study(
        study_name=studyName, storage=storage
    )
    return study

# Function to get or create an optuna study (if it does not exist)
def get_or_create_optuna_study(user, pw, ip, dbName, studyName, direction='maximize'):
    storage = optuna.storages.RDBStorage(
        url=f'mysql://{user}:{pw}@{ip}/{dbName}',
        engine_kwargs={
            'pool_size': 20,
            'max_overflow': 0
        }
    )
    study = optuna.create_study(study_name=studyName, storage=storage, direction=direction, load_if_exists=True)
    return study

# Funktin to visualize an optuna study as a plot
def plot_optuna_study(study, params_list=None, type="history"):
    if type == "history":
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
    elif type == "plot_slice":
        # Example: params_list = [["lr", "gamma"], ["lr", "epsilon"]]
        for params in params_list:
            fig = optuna.visualization.plot_slice(study, params=params)
            fig.show()
    elif type == "plot_parallel_coordinate":
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.show()
    else:
        print("Wrong type")

# Build the policy for the PPO algorithm containing the activation function and the network architecture
def build_policy(sel_policy, activation_function):
    if activation_function == "relu":
        activation_function = nn.ReLU
    elif activation_function == "tanh":
        activation_function = nn.Tanh
    elif activation_function == "elu":
        activation_function = nn.ELU
    else:
        print("Wrong activation function")
        raise ValueError

    if sel_policy == 1:
        policy_kwargs = dict(activation_fn=activation_function,
                             net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    elif sel_policy == 2:
        policy_kwargs = dict(activation_fn=activation_function,
                             net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    elif sel_policy == 3:
        policy_kwargs = dict(activation_fn=activation_function,
                             net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    else:
        print("Wrong policy")
        raise ValueError
        policy_kwargs = None
    return policy_kwargs

# Function to pre-train a model and save it
def pre_train_model_imitation(steps, tb_path, n_steps, policy_kwargs, batch_size, model_path, timesteps):
    spawnInfo = var_startpos['spawnInfo']
    camLocation = var_startpos['camLocation']
    env = CustomEnv(time_steps_per_training=steps, syncMode=True, render=True,
                       spawnInfo=spawnInfo, camaraLocation=camLocation)
    env.rewardDistribution = [0, 0, 0, 0, 1]
    new_logger = configure(tb_path, ["tensorboard"])
    model = PPO("MlpPolicy", env, verbose=1, n_steps=n_steps, policy_kwargs=policy_kwargs)
    model.set_logger(new_logger)
    model.learn(total_timesteps=timesteps, log_interval=batch_size, reset_num_timesteps=True)
    model.save(model_path)
    return model, env
