import time

import carla
import numpy as np
from numpy import array, float32
from stable_baselines3 import PPO

from config_file import var_recording, var_startpos, var_paths
from helper_functions import getEnvirionment

# Simulate the actions from the actionList and return the x and y coordinates of the walker and the vehicle
def runAction(actionList=None):
    if actionList == None:
        actionList = var_recording['positions_walker_rl']
    else:
        actionList = actionList

    index = 1
    location = carla.Location(x=142.484970, y=-120.983490, z=52.080902)
    rotation = carla.Rotation(pitch=-90, yaw=2, roll=0)
    env = getEnvirionment(port=2000, camaraLocation=location, camRotation=rotation)
    env.reset()
    walker_x_y = []
    vehicle_x_y = []
    for action in actionList:
        index += 1
        env.step(action)
        walker_x_y.append(env.walker.getPositionXY())
        vehicle_x_y.append(env.vehicle.getPositionXY())
        time.sleep(0.01)

    # Do 3 more steps to see the final position
    for _ in range(3):
        if index >= 128:
            break
        index += 1
        env.step(actionList[-1])
        walker_x_y.append(env.walker.getPositionXY())
        vehicle_x_y.append(env.vehicle.getPositionXY())
        time.sleep(0.5)
        print(env.ttc())

    print("Walker:", walker_x_y)
    print("Vehicle:", vehicle_x_y)
    return walker_x_y

# Run the model and return the rewards of the runs
def runPrediction(path=var_paths['path_model'],
                  modelname="TimeStepsPerTraining128_totalTimesteps500000_Run0_Policy2.zip", env="CustomEnv",
                  deterministic=True, runs=3):
    if env == "CustomEnv":
        location = carla.Location(x=142.484970, y=-120.983490, z=52.080902)
        rotation = carla.Rotation(pitch=-90, yaw=2, roll=0)
        env = getEnvirionment(envName="CustomEnv", port=2010, camaraLocation=location, camRotation=rotation)
    else:
        env = env
    model = PPO.load(path + modelname, env=env)
    rewards_list = []
    for _ in range(runs):
        obs = env.reset()
        rewards = 0

        walker_x_y = []
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            time.sleep(0.01)
            walker_x_y.append(env.walker.getPositionXY())
            rewards += reward
            print(env.ttc())
            if done:
                break
        print(f"Reward = {rewards}")
        print(walker_x_y)
        print(env.actionsList)
        rewards_list.append(rewards)
    return rewards_list


if __name__ == '__main__':

    # Actio from imitation (example)
    actions = [array([ 0.35298356, -1.        ,  1.        ], dtype=float32), array([-0.16977055, -1.        ,  1.        ], dtype=float32), array([-0.90804327, -1.        ,  1.        ], dtype=float32), array([-0.4664857, -1.       ,  1.       ], dtype=float32), array([ 0.22497602, -1.        ,  1.        ], dtype=float32), array([ 0.00138909, -1.        ,  1.        ], dtype=float32), array([-0.07081927, -1.        ,  1.        ], dtype=float32), array([-0.54275155, -1.        ,  1.        ], dtype=float32), array([-0.6372821, -1.       ,  1.       ], dtype=float32), array([ 0.04234619, -1.        ,  1.        ], dtype=float32), array([-0.7065077, -1.       ,  1.       ], dtype=float32), array([-0.10052156, -1.        ,  1.        ], dtype=float32), array([-0.08880828, -1.        ,  1.        ], dtype=float32), array([-0.18111947, -1.        ,  0.934341  ], dtype=float32), array([ 0.12217896, -1.        ,  1.        ], dtype=float32), array([-0.15297037, -1.        ,  1.        ], dtype=float32), array([-0.00958559, -1.        ,  1.        ], dtype=float32), array([-0.15160733, -1.        ,  1.        ], dtype=float32), array([-0.8425284, -1.       ,  1.       ], dtype=float32), array([ 0.12269302, -1.        ,  1.        ], dtype=float32), array([-0.21751808, -1.        ,  1.        ], dtype=float32), array([-0.5887422, -1.       ,  1.       ], dtype=float32), array([ 0.04202831, -1.        ,  1.        ], dtype=float32), array([-0.50504744, -1.        ,  1.        ], dtype=float32), array([ 0.17769274, -1.        ,  1.        ], dtype=float32), array([-0.46488675, -1.        ,  1.        ], dtype=float32), array([-0.22337118, -1.        ,  1.        ], dtype=float32), array([-0.65389645, -1.        ,  1.        ], dtype=float32), array([-0.28146434, -1.        ,  1.        ], dtype=float32), array([-0.42493856, -1.        ,  1.        ], dtype=float32), array([-0.4026921, -1.       ,  1.       ], dtype=float32), array([-0.30296293, -1.        ,  1.        ], dtype=float32), array([-0.01986247, -1.        ,  1.        ], dtype=float32), array([-0.9864947, -1.       ,  1.       ], dtype=float32), array([-1.       , -0.9718017,  1.       ], dtype=float32), array([-1., -1.,  1.], dtype=float32), array([-0.9727144, -1.       ,  1.       ], dtype=float32), array([-1.        , -0.48563582,  1.        ], dtype=float32), array([-1.,  1.,  1.], dtype=float32), array([-1., -1.,  1.], dtype=float32), array([-1.        , -0.17879075,  1.        ], dtype=float32), array([-1.        ,  0.64872056,  1.        ], dtype=float32), array([-1.        , -0.05283491,  1.        ], dtype=float32), array([-1.       ,  0.2695533,  1.       ], dtype=float32), array([-1.       , -0.6759476,  1.       ], dtype=float32), array([-1.        , -0.17817838,  1.        ], dtype=float32), array([-1.       ,  0.3336872,  1.       ], dtype=float32), array([-1.       , -0.5679555,  1.       ], dtype=float32), array([-1.        , -0.44304764,  1.        ], dtype=float32), array([-1.        , -0.02140289,  1.        ], dtype=float32), array([-1.,  1.,  1.], dtype=float32), array([-1.       ,  0.8041854,  1.       ], dtype=float32), array([-1.       ,  0.7584962,  1.       ], dtype=float32), array([-1.        ,  0.39255255,  1.        ], dtype=float32), array([-1.        ,  0.26315218,  1.        ], dtype=float32), array([-1.       ,  0.5331332,  1.       ], dtype=float32), array([-1.        , -0.35278618,  1.        ], dtype=float32), array([-1.       ,  0.1557769,  1.       ], dtype=float32), array([-1.       ,  0.3058533,  1.       ], dtype=float32), array([-1.       , -0.7629307,  1.       ], dtype=float32), array([-1.        , -0.33128178,  1.        ], dtype=float32), array([-0.901779,  1.      ,  1.      ], dtype=float32), array([-1.        ,  0.15904233,  1.        ], dtype=float32), array([-1.        , -0.25702703,  1.        ], dtype=float32), array([-1.       ,  0.5968555,  1.       ], dtype=float32), array([-1.        , -0.52718914,  1.        ], dtype=float32)]
    runAction(actionList=actions)
    path = "./model/baseline/"
    modelname = "NameModel.zip"
    print(f"Mean = {np.mean(runPrediction(modelname=modelname, path=path, deterministic=True, runs=2))}")
