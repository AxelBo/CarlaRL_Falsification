import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from config_file import var_lrl, var_paths, var_startpos
from envirionment import CustomEnv
from helper_functions import get_or_create_optuna_study, getEnvirionment, \
    get_current_timestamp, build_policy, load_optuna_studyMySQL


# Optuna Trial to find the best hyperparameters
def optuna_trial(trial):
    """Trial for Optuna with all variables"""
    # paths
    path_tb = var_paths['path_tb']
    path_model = var_paths['path_model']

    # Hyperparameters
    network_architecture = trial.suggest_categorical("network_architecture", [1, 2, 3])
    activation_function = trial.suggest_categorical("activation_function", ["relu", "tanh", "elu"])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_categorical("learning_rate", [0.00015, 0.0003, 0.0006])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.4])

    steps_total_train = 200000
    policy_kwargs = build_policy(network_architecture, activation_function)
    highest_rewards = []
    for i in range(3):
        timestamp = get_current_timestamp()
        param_sum = f"NA-{network_architecture}_AF-{activation_function}_NS-{n_steps}_BS-{batch_size}_LR-{learning_rate}_CR-{clip_range}_run-{i}"
        spawnInfo = var_startpos['spawnInfo']
        camLocation = var_startpos['camLocation']
        env = CustomEnv(time_steps_per_training=128, syncMode=True, render=False,
                        spawnInfo=spawnInfo, camaraLocation=camLocation)
        # Angle; TTC; Dist; Min TTC; Dist Imitation
        env.rewardDistribution = [0, 0, 0, 0, 1.0]
        model = PPO("MlpPolicy", env, learning_rate=learning_rate, batch_size=batch_size, clip_range=clip_range,
                    verbose=1, n_steps=n_steps, policy_kwargs=policy_kwargs)
        tb_path_train = f"{path_tb}TRAIN_{param_sum}_TS-{timestamp}"
        model_path_train = f"{path_model}_TRAIN_{param_sum}"
        new_logger = configure(tb_path_train, ["tensorboard"])
        model.set_logger(new_logger)
        model.learn(total_timesteps=steps_total_train, log_interval=32, reset_num_timesteps=False)
        model.save(model_path_train)
        highest_rewards.append(env.min_ttc_env)
    return np.mean(highest_rewards)

# Optuna Traial to find the best reward distribution (not used at the moment)
def optuna_trail2(trial):
    """Trial for Optuna with all variables"""
    # paths
    path_tb = var_paths['path_tb']
    path_model = var_paths['path_model']

    # variables
    steps_total_train = 500000

    # Hyperparameters
    rewardAngle = trial.suggest_int("rewardAngle", low=0, high=100)
    ttc_now = trial.suggest_int("ttc_now", low=0, high=100)
    dist_car = trial.suggest_int("dist_car", low=0, high=100)
    ttc_min = trial.suggest_int("ttc_min", low=0, high=100)
    dist_walker = trial.suggest_int("dist_walker", low=0, high=100)

    summe = rewardAngle + ttc_now + dist_car + ttc_min + dist_walker
    if summe == 0:
        reward_distributuion = [0.2, 0.2, 0.2, 0.2, 0.2]
    else:
        reward_distributuion = [rewardAngle/summe, ttc_now/summe, dist_car/summe, ttc_min/summe, dist_walker/summe]
    highest_rewards = []
    for i in range(3):
        timestamp = get_current_timestamp()
        param_sum = f"{rewardAngle}_{ttc_now}_{dist_car}_{ttc_min}_{dist_walker}_run-{i}"
        env_new = getEnvirionment(syncMode=True, render=False, port=2000)
        env_new.rewardDistribution = reward_distributuion

        tb_path_train = f"{path_tb}TRAIN_{param_sum}_TS-{timestamp}"
        model_path_train = f"{path_model}_TRAIN_{param_sum}"

        # PreTraining
        model = PPO("MlpPolicy", env_new, verbose=1)
        new_logger = configure(tb_path_train, ["tensorboard"])
        model.set_logger(new_logger)
        model.learn(total_timesteps=steps_total_train, log_interval=64)
        #model.save(model_path_train)

        # Training
        env_new.highest_basic_reward = -999
        env_new.rewardDistribution = [0.0, 0.0, 0.0, 1.0, 0.0]
        model.set_env(env_new)
        model.learn(total_timesteps=steps_total_train, log_interval=64)
        model.save(model_path_train)

        highest_rewards.append(env_new.highest_basic_reward)
    return np.mean(highest_rewards)


def opt_training(n_trials, hostname, user, password, db_name, study_name):
    study = load_optuna_studyMySQL(user, password, hostname, db_name, study_name)
    # start to optimize study with optuna_trial
    study.optimize(optuna_trial, n_trials=n_trials)



if __name__ == '__main__':
    user = 'test'
    pw = 'pw123'
    ip = '123.456.789.10'
    dbName = 'optuna'
    studyName = 'STUDY_NAME'

    # Deside if you want to maximize or minimize the function
    direction = 'minimize'
    n_trials = 100

    study = get_or_create_optuna_study(user, pw, ip, dbName, studyName, direction=direction)
    study.optimize(optuna_trial, n_trials=n_trials)