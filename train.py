# Local
from src.environment_rl import Environment
import controller

# External
import os
import csv
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

# RL Libraries
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure


## Cursed file loading
targets = []
with open("targets.csv", "r") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        if (
            float(row[0]) > 8
            or float(row[1]) > 8
            or float(row[0]) < 0
            or float(row[1]) < 0
        ):
            print(
                "WARNING: Target outside of environment bounds (0, 0) to (8, 8), not loading target"
            )
        else:
            targets.append((float(row[0]), float(row[1])))


## Environment setup
def make_env(rank, log_dir, env_base, seed=0):
    def _init():
        env = EnvWrapper(env_base)
        env.seed(controller.group_number + rank)
        env = Monitor(env, filename=f'{log_dir}training_data_{rank}')
        return env
    set_random_seed(seed)
    return _init

class EnvWrapper(gym.Env):
    def __init__(self, env_base):
        self.env = env_base
        self.num_steps = 161
        self.action_space = gym.spaces.MultiDiscrete(np.array([self.num_steps, self.num_steps]))
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf, -np.pi, -np.inf]),
            high=np.array([8, 8, np.inf, np.inf, np.pi, np.inf]),
            dtype=np.float32
        )  # Define the observation space
        self.time = 0.0  # Initialize time variable
        self.seed()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(controller.group_number, controller.wind_active)
        self.time = 0.0  # Reset time variable
        
        # Randomly select a target position
        x_targ = np.random.uniform(0.5, 7.5)
        y_targ = np.random.uniform(0.5, 7.5)
        self.target_pos = (x_targ, y_targ)
        
        observation = self.env.drone.get_state()
        info = {}  # Add any additional information if needed
        return observation, info

    def step(self, action):
        u_1, u_2 = self._convert_action(action)
        self.env.step((u_1, u_2))
        state = self.env.drone.get_state()
        x, y = state[0], state[1]

        # Calculate time_delta and update time variable
        sim_time = self.env.time / 60

        # Calculate the reward
        reward = self.calc_reward(state, self.target_pos, sim_time)
        
        # Check if the episode is done based on certain conditions
        done = False
        if not (-8 <= x <= 8 and -8 <= y <= 8):
            # Episode is done if the drone goes out of bounds
            done = True
        elif sim_time >= 20:
            # Episode is done if the time exceeds 20 seconds
            done = True
    
        truncated = False
        info = {}
        return state, reward, done, truncated, info

    # Define the reward function
    def calc_reward(self, state, target_pos, t):
        x, y = state[0], state[1]
        x_targ, y_targ = target_pos[0], target_pos[1]

        dist = ((x - x_targ)**2 + (y - y_targ)**2)**0.5
        dist_fact = (1 / (1 + dist**2))
        
        # Punish for leaving the game area
        if not (-8 <= x <= 8 and -8 <= y <= 8):
            return -20
        
        # Reward for being close to the target
        reward =  dist_fact * 2
        
        # Reward for time spent alive
        # reward += t*0.05
        
        # Reward for being at the target during the last 10 seconds
        if 10 <= t <= 20 and dist < 0.1:
            reward += dist_fact * 1
            
        return reward

    def _convert_action(self, action):
        u_1 = action[0] / 160 if action[0] != 0 else 0
        u_2 = action[1] / 160 if action[1] != 0 else 0
        
        return u_1, u_2

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]


## Training
def train(time_steps=1e6, save_dir='./models/', save_freq=1e5, log_dir='./logs/'):    
    callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f'{save_dir}/training/',
        name_prefix="PPO_Drone",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.learn(total_timesteps=time_steps, callback=callback)
    model.set_logger(new_logger)
    model.save(f'{save_dir}PPO_Drone')


## Main
if __name__ == "__main__":
    # Config vars
    num_cpu = 14
    time_steps = 1_000_000_000
    save_freq = 1_000_000
    save_dir = './models/'
    log_dir = './logs/'
    
    # Create environment & define model
    env_base = Environment(
        render_mode=None,
        render_path=False,
        screen_width=1000,
        ui_width=200,
        rand_dynamics_seed=controller.group_number,
        wind_active=controller.wind_active,
    )
    
    env_vec = SubprocVecEnv([make_env(i, log_dir, env_base) for i in range(num_cpu)])
    model = PPO('MlpPolicy', env=env_vec, verbose=1, learning_rate=1e-5,
                batch_size=2048*num_cpu, seed=18)
        
    # Train model
    train(time_steps, save_dir, max(save_freq // num_cpu, 1))
    print('Finished :D')