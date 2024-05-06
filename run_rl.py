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
from stable_baselines3.common.results_plotter import load_results, ts2xy


## Logging or something idk
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

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
env_base = Environment(
    render_mode=None,
    render_path=False,
    screen_width=1000,
    ui_width=200,
    rand_dynamics_seed=controller.group_number,
    wind_active=controller.wind_active,
)

def make_env(rank, seed=0):
    def _init():
        env = EnvWrapper(env_base)
        env.seed(controller.group_number + rank)
        env = Monitor(env, filename=f'./logs/training_data_{rank}')
        return env
    set_random_seed(seed)
    return _init

class EnvWrapper(gym.Env):
    def __init__(self, env_base):
        self.env = env_base
        self.num_steps = 161
        self.action_space = gym.spaces.MultiDiscrete(np.array([self.num_steps, self.num_steps]))
        self.observation_space = gym.spaces.Box(
            low=np.array([-8, -8, -np.inf, -np.inf, -np.pi, -np.inf]),
            high=np.array([8, 8, np.inf, np.inf, np.pi, np.inf]),
            dtype=np.float32
        )  # Define the observation space
        self.time = 0.0  # Initialize time variable
        self.seed()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(controller.group_number, controller.wind_active)
        self.time = 0.0  # Reset time variable
        
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
        reward = self.calc_reward(state, target_pos, sim_time)
        
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
        
        # Punish for leaving the game area
        if not (-8 <= x <= 8 and -8 <= y <= 8):
            return -20
        
        # Reward for being close to the target
        dist = ((x - x_targ)**2 + (y - y_targ)**2)**0.5
        reward = 1 / (1 + dist)
        
        # Reward for time spent alive
        reward += t*0.05
        
        # Reward for being at the target during the last 10 seconds
        if 10 <= t <= 20 and dist < 0.01:
            reward += 1
            
        return reward

    def _convert_action(self, action):
        u_1 = action[0] / 160 if action[0] != 0 else 0
        u_2 = action[1] / 160 if action[1] != 0 else 0
        
        return u_1, u_2

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]


## Training
target_pos = targets[0]
def train(time_steps=1e6):
    model.learn(total_timesteps=time_steps)
    model.save("PPO_Drone")
    

## Plotting
def plot_training_data():
    log_dir = './logs/'
    
    # Get all the monitor.csv files in the log directory
    monitor_files = [file for file in os.listdir(log_dir) if file.endswith('.monitor.csv')]
    
    # Load the data from all monitor files
    data_frames = []
    for file in monitor_files:
        file_path = os.path.join(log_dir, file)
        data = pd.read_csv(file_path, skiprows=1)
        data_frames.append(data)
    
    # Concatenate the data from all monitor files
    all_data = pd.concat(data_frames)
    
    # Extract the desired parameters
    timesteps = all_data['l'].cumsum()
    rewards = all_data['r']
    episode_lengths = all_data['l']
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot rewards
    ax1.plot(timesteps, rewards, color='blue', linewidth=1)
    ax1.set_ylabel('Rewards')
    ax1.set_title('Training Rewards')
    ax1.grid(True)
    
    # Plot episode lengths (survival time)
    ax2.plot(timesteps, episode_lengths, color='green', linewidth=1)
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths (Survival Time)')
    ax2.grid(True)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the plot as an image file
    plt.savefig(f'{log_dir}training_data.png')
    plt.close()


## Main
if __name__ == "__main__":
    # Create the PPO agents
    num_cpu = 14
    time_steps = 100_000_000
    env_vec = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    model = PPO('MlpPolicy', env=env_vec, verbose=1, learning_rate=1e-5,
                batch_size=2048*num_cpu, seed=18)
        
    # Train the model & plot
    train(time_steps)
    plot_training_data()
    print('Finished :D')