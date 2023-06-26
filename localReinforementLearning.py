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
    # A custom callback that derives from `BaseCallback` for saving training stats in TensorBoard.
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


# Saves the basic reward in a file
def saveBasic_reward(basicReward, index, file_name="basic_rewards.txt"):
    with open(file_name, "a") as file:
        file.write(f"[{index},{basicReward}]\n")

# Train the model with the given parameters
def trainBaseline(n=2, steps=128, total_timesteps=1000000, nameOfRuns="default", distribute=[0, 0, 0, 0, 1.0]):
    # Parameter
    log_interval = var_lrl["log_interval"]
    policy = build_policy(3, 'tanh')

    # Iterations for training multiple models
    for i in range(n):
        tb_path = f"./tensorboard/baseline/BaseLine_{total_timesteps}_Run{i}_{nameOfRuns}_{steps}.zip"
        model_path = f"./model/baseline/BaseLine_{total_timesteps}_Run{i}_{nameOfRuns}_{steps}.zip"
        if os.path.isfile(model_path):
            print(f"Die Datei unter {model_path} existiert.")
            continue
        new_logger = configure(tb_path, ["tensorboard", "stdout"])
        # Camera Location
        camLocation = var_startpos["camLocation"]
        # Spawn Info (Spawn Locations, Spawn Rotations)
        spawnInfo = var_startpos["spawnInfo"]

        syncMode = var_connection["syncMode"]
        render_env = var_connection["render"]

        env = CustomEnv(time_steps_per_training=steps, syncMode=syncMode, render=render_env,
                        spawnInfo=spawnInfo, camaraLocation=camLocation)
        # Angle; TTC; Dist; Min TTC; Dist Imitation
        env.rewardDistribution = distribute

        #TODO change Hyperparameter if needed
        model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, policy_kwargs=policy, learning_rate=0.00015)
        # model = PPO("MlpPolicy", env, verbose=1)
        model.set_logger(new_logger)
        custom_tb_callback = CustomTensorboardCallback()
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval, reset_num_timesteps=True,
                    callback=custom_tb_callback)
        saveBasic_reward(env.highest_basic_reward, i)
        model.save(model_path)


if __name__ == '__main__':

    # Parameter from config_file
    steps = var_lrl["time_steps_per_training"]
    validations = var_lrl["validations"]
    rewardDistribution = var_lrl["rewardDistribution"]
    timesteps_Train = var_lrl["timesteps_Train"]
    n_steps_size_faktor = var_lrl["n_steps_size_faktor"]

    nameOfRuns = "Imitation"

    trainBaseline(n=validations, steps=steps, total_timesteps=timesteps_Train,
                  nameOfRuns=nameOfRuns, distribute=rewardDistribution)


    exit(0)