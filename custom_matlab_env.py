import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import utils
import sim_plant


class MatlabEnv(gym.Env):
    """Custom Environment that follows gym interface."""


    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(utils.features_cnt,), dtype=np.float32)


        self.plant = sim_plant.SimPool.get_instance()
        #self.plant.set_desired_velocity(type="random")
        self.iteration = 0
        #self.plant.reset_plant()

    def step(self, action):
        info = {}
        print("NN action: ", action)
        observation, reward, terminated, truncated = self.plant.step(action)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.plant.reset_plant()
        self.plant.set_desired_velocity(type="random")
        info = {}
        observation = np.float32(np.array([-100, 0, 0, 0]))
        print("new iteration: ", self.iteration)
        self.iteration += 1

        return observation, info

    def render(self):
        print('render function was called')

    def close(self):
        print('close function was called')
        #sim_plant.SimPool.delete_instance(self.plant)

if __name__ == '__main__':
    print("checking matlab environment")
    sim_plant.SimulinkPlant.set_debug_mode(True)
    env = MatlabEnv()

    # env.reset()
    # i = 0
    # terminated = False
    # while not terminated:
    #     observation, reward, terminated, truncated, info = env.step(0)
    #     print('environment\'s step ', i, ' has finished')
    #     i += 1

    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)
