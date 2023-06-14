#!/usr/bin/env python

import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from torch import nn

from config_file import var_lrl, var_startpos, var_connection
from envirionments import CustomEnv
from helper_functions import get_current_timestamp, build_policy

from stable_baselines3.common.callbacks import BaseCallback


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomTensorboardCallback, self).__init__(verbose)
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # Logge den Wert von highest_basic_reward in Tensorboard
        # self.logger.record(f"highest_basic_reward", self.training_env.get_attr('highest_basic_reward')[0])
        self.logger.record(f"basicReward", self.training_env.get_attr('lastBasicReward')[0])
        self.logger.record(f"Best MinTTC", self.training_env.get_attr('min_ttc_env')[0])
        self.logger.record(f"Last MinTTC", self.training_env.get_attr('lastmin_ttc')[0])

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def select_policy_by_index(index):
    select_policy = index
    if select_policy == 0:
        print("Die default Richtlinie wird ausgewählt. dict(pi=[64, 64], vf=[64, 64])")
        policy_kwargs = None
    elif select_policy == 1:
        print("Die erste Richtlinie wird ausgewählt. net_arch=dict(pi=[128, 64, 64], vf=[128, 64, 64])")
        policy_kwargs = dict(net_arch=dict(pi=[128, 64, 64], vf=[128, 64, 64]))
    elif select_policy == 2:
        print("Die zweite Richtlinie wird ausgewählt.256, 256, dict(vf=[512, 256, 128], pi=[512, 256, 128])")
        policy_kwargs = {
            'net_arch': [256, 256, dict(vf=[512, 256, 128], pi=[512, 256, 128])],
            'activation_fn': nn.ReLU
        }
    elif select_policy == 3:
        print("Die dritte Richtlinie wird ausgewählt.")
        policy_kwargs = {
            'net_arch': [64, 64, dict(vf=[256, 128], pi=[256, 128])],
            'activation_fn': nn.ELU
        }

    return policy_kwargs


def saveBasic_reward(basicReward, index, file_name="basic_rewards.txt"):
    with open(file_name, "a") as file:
        file.write(f"[{index},{basicReward}]\n")


def trainBaseline(n=2, steps=128, total_timesteps=1000000, nameOfRuns="default", distribute=[0, 0, 0, 0, 1.0]):
    # Parameter
    log_interval = 8
    policy = build_policy(3, 'tanh')

    # Iterations for training multiple models
    for i in range(n):
        tb_path = f"./tensorboard/baseline/BaseLine_{total_timesteps}_Run{i}_{nameOfRuns}_{steps}.zip"
        model_path = f"./model/baseline/BaseLine_{total_timesteps}_Run{i}_{nameOfRuns}_{steps}.zip"
        if os.path.isfile(model_path):
            print(f"Die Datei unter {model_path} existiert.")
            continue
        new_logger = configure(tb_path, ["tensorboard", "stdout"])
        env = CustomEnv(time_steps_per_training=steps, syncMode=True, render=False,
                        spawnInfo=spawnInfo, camaraLocation=camLocation)
        # Angle; TTC; Dist; Min TTC; Dist Imitation
        env.rewardDistribution = distribute
        model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, policy_kwargs=policy, learning_rate=0.00015)
        # model = PPO("MlpPolicy", env, verbose=1)
        model.set_logger(new_logger)
        custom_tb_callback = CustomTensorboardCallback()
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval, reset_num_timesteps=True,
                    callback=custom_tb_callback)
        saveBasic_reward(env.highest_basic_reward, i)
        model.save(model_path)


if __name__ == '__main__':
    time_steps_per_training = var_lrl["time_steps_per_training"]
    log_interval = var_lrl["log_interval"]
    spawnLocationWalker = var_startpos["spawnLocationWalker"]
    spawnLocationVehicle = var_startpos["spawnLocationVehicle"]
    rotationVehicle = var_startpos["rotationVehicle"]
    rotationWalker = var_startpos["rotationWalker"]
    camLocation = var_startpos["camLocation"]
    spawnInfo = var_startpos["spawnInfo"]
    testPolicys = var_lrl["testPolicys"]
    time_steps_per_trainings = var_lrl["time_steps_per_trainings"]
    steps = time_steps_per_trainings
    validations = var_lrl["validations"]
    rewardDistribution = var_lrl["rewardDistribution"]
    timesteps_preTrain = var_lrl["timesteps_preTrain"]
    timesteps_Train = var_lrl["timesteps_Train"]
    batch_size_preTrain = var_lrl["batch_size_preTrain"]
    batch_size_Train = var_lrl["batch_size_Train"]
    syncMode = var_connection["syncMode"]
    render_env = var_connection["render"]
    n_steps_size_faktor = var_lrl["n_steps_size_faktor"]

    # Angle; TTC; Dist; Min TTC; Dist Imitation
    distribute = [0.0, 0.0, 0.0, 0.0, 1.0]
    steps = 128
    n = 10
    total_timesteps = 1000000
    nameOfRuns = "Imitation"
    trainBaseline(n=n, steps=steps, total_timesteps=total_timesteps, nameOfRuns=nameOfRuns, distribute=distribute)


    exit(0)