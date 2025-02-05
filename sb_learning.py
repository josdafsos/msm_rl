import numpy as np

import custom_matlab_env
import msm_model
# from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
import os
import mujoco_viewer
import torch
from datetime import datetime
import matplotlib.pyplot as plt


import utils


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        return True
        rewards = self.training_env.get_attr("total_reward")
        if rewards:
            print(f"Episode Rewards: {rewards}")
        return True

def make_env():
    global reward_log_list, log_dir
    # if Monitor class is used, access environment via env.get_env()
    return msm_model.MSM_Environment(randomize_setpoint=True)
    # return Monitor(msm_model.MSM_Environment(randomize_setpoint=True), log_dir)

def plot_rewards_history(vec_env):
    min_len = 0
    reward_matrix = vec_env.get_attr("episode_reward_list")
    for i in range(len(reward_matrix)):  # need to equalize sizes of the vector
        min_len = min(min_len, len(reward_matrix[i]))
    for i in range(len(reward_matrix)):
        reward_matrix[i] = reward_matrix[i][-min_len:]
    rewards = np.transpose(np.array(reward_matrix))
    mean_rewards = np.mean(rewards, axis=1)
    mean_rewards = mean_rewards[1:]
    reward_steps = [i for i in range(len(mean_rewards))]

    plt.figure(figsize=(10, 5))
    plt.plot(reward_steps, mean_rewards)
    plt.xlabel("Episodes x Number of Cores")
    plt.ylabel("Average reward per episode set")
    plt.title("Reward history")
    plt.grid()
    plt.show()


if __name__ == '__main__':

    # TODO plot reward history

    test_model = False
    model_name = ""  # leave empty if a new model must be trained
    # model_name = "30000000_network_02_01_25_time_11_25"
    device = 'cpu'

    log_dir = 'logs'

    # path = os.path.join('sb_neural_networks', 'sb_neural_network')
    # env = msm_model.MSM_Environment()  # randomize_setpoint=False
    env = make_env()
    policy_kwargs = dict(
        net_arch=[256, 256, 512],  # hidden layers with VALUE neurons each
        activation_fn=torch.nn.ReLU
    )

    if test_model:
        #model = RecurrentPPO.load(os.path.join('sb_neural_networks', model_name))
        model = PPO.load(os.path.join('sb_neural_networks', model_name))

        viewer = mujoco_viewer.MujocoViewer(env.environment.model, env.environment.data)
        viewer.cam.azimuth = 180
        viewer.cam.distance = 0.005

        obs, _ = env.reset()
        # env.velocity_setpoint = 0.006  # if not set explicitly the default constant setpoint will be used

        while viewer.is_alive:
            action, _states = model.predict(obs)
            observation, reward, terminated, truncated, info = env.step(action)
            #env.render("human")
            viewer.render()
        viewer.close()

        env.environment.plot_rack_instant_velocity()
        env.environment.plot_rack_average_velocity()
        env.environment.plot_control_value()

    else:

        # Vectorized environment setup
        num_envs = 30  # Number of parallel environments
        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])  # Or use DummyVecEnv([make_env])
        vec_env = VecMonitor(vec_env)
        timesteps = int(3e6)
        if num_envs == 1:
            vec_env = env
        # Instantiate the agent
        #model = PPO('MlpLstmPolicy', vec_env, learning_rate=1e-3, verbose=1)  # n_steps=10 for rollout. n_step also define minum lear steps
        if model_name == "":
            print("creating new model")
            # model = RecurrentPPO("MlpLstmPolicy", vec_env, learning_rate=1e-3, verbose=1)
            model = PPO("MlpPolicy",
                        vec_env,
                        device=device,
                        learning_rate=3e-4,
                        policy_kwargs=policy_kwargs,
                        verbose=1) # ,
            #model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, learning_rate=1e-4, verbose=1)
        else:
            print("Loading model for training")
            #model = RecurrentPPO.load(os.path.join('sb_neural_networks', model_name))
            model = PPO.load(os.path.join('sb_neural_networks', model_name))

            model.set_env(vec_env)

        #model.save(os.path.join('sb_neural_networks', "original_" + network_name))
        #model = SAC('MlpPolicy', env, learning_rate=1e-3, verbose=1)
        # Train the agent
        model.learn(total_timesteps=timesteps,
                    progress_bar=True,
                    #callback=RewardLoggerCallback()
                    )
        # Save the agent
        if model_name == "":
            model_name = f"{timesteps}_network_" + datetime.now().strftime("%D_time_%H_%M").replace("/", "_")
        else:
            model_name = model_name + "_new"
        model.save(os.path.join('sb_neural_networks', model_name))

        plot_rewards_history(vec_env)



        # print(utils.reward_list)
        # msm_model.MSM_Environment.plot_expected_reward_history(utils.reward_list)


