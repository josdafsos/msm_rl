import custom_matlab_env
import msm_model
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os
import mujoco_viewer
import torch
from datetime import datetime

import utils


def make_env():
    global reward_log_list
    return msm_model.MSM_Environment(randomize_setpoint=True)


if __name__ == '__main__':

    test_model = False
    model_name = ""  # leave empty if a new model must be trained
    # model_name = "1e5_setpoint_0_012_singlecore"

    # path = os.path.join('sb_neural_networks', 'sb_neural_network')
    env = msm_model.MSM_Environment()  # randomize_setpoint=False
    policy_kwargs = dict(
        net_arch=[256, 256],  # hidden layers with VALUE neurons each
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
        num_envs = 10  # Number of parallel environments
        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])  # Or use DummyVecEnv([make_env])
        timesteps = int(1e7)
        if num_envs == 1:
            vec_env = env
        # Instantiate the agent
        #model = PPO('MlpLstmPolicy', vec_env, learning_rate=1e-3, verbose=1)  # n_steps=10 for rollout. n_step also define minum lear steps
        if model_name == "":
            print("creating new model")
            # model = RecurrentPPO("MlpLstmPolicy", vec_env, learning_rate=1e-3, verbose=1)
            model = PPO("MlpPolicy", vec_env, learning_rate=1e-3, verbose=1)
            #model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, learning_rate=1e-4, verbose=1)
        else:
            print("Loading model for training")
            #model = RecurrentPPO.load(os.path.join('sb_neural_networks', model_name))
            model = PPO.load(os.path.join('sb_neural_networks', model_name))

            model.set_env(vec_env)

        #model.save(os.path.join('sb_neural_networks', "original_" + network_name))
        #model = SAC('MlpPolicy', env, learning_rate=1e-3, verbose=1)
        # Train the agent
        model.learn(total_timesteps=timesteps, progress_bar=True)
        # Save the agent
        if model_name == "":
            model_name = f"{timesteps}_network_" + datetime.now().strftime("%D_time_%H_%M").replace("/", "_")
        else:
            model_name = model_name + "_new"
        model.save(os.path.join('sb_neural_networks', model_name))


        print(utils.reward_list)
        msm_model.MSM_Environment.plot_expected_reward_history(utils.reward_list)

