from src.environment import Environment
import controller
import csv
import sys
import importlib
import pathlib
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
import logging
from tqdm import tqdm

from stable_baselines3.common.logger import configure

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import multiprocessing

# Logging or something idk
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_env(rank, seed=0):
    def _init():
        env = EnvironmentWrapper(environment)
        env.seed(controller.group_number + rank)
        return env
    set_random_seed(seed)
    return _init

targets = []
time = 0

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

environment = Environment(
    render_mode=None,
    render_path=False,
    screen_width=1000,
    ui_width=200,
    rand_dynamics_seed=controller.group_number,
    wind_active=controller.wind_active,
)

class EnvironmentWrapper(gym.Env):
    def __init__(self, environment):
        self.environment = environment
        self.num_steps = 161
        self.action_space = gym.spaces.Discrete(self.num_steps ** 2)
        self.observation_space = gym.spaces.Box(
            low=np.array([-8, -8, -np.inf, -np.inf, -np.pi, -np.inf]),
            high=np.array([8, 8, np.inf, np.inf, np.pi, np.inf]),
            dtype=np.float32
        )  # Define the observation space
        self.time = 0.0  # Initialize time variable
        self.seed()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.environment.reset(controller.group_number, controller.wind_active)
        self.time = 0.0  # Reset time variable
        
        observation = self.environment.drone.get_state()
        info = {}  # Add any additional information if needed
        return observation, info

    def step(self, action):
        u_1, u_2 = self._convert_action(action)
        self.environment.step((u_1, u_2))
        state = self.environment.drone.get_state()

        # Calculate time_delta and update time variable
        sim_time = self.environment.time / 60

        reward = calc_reward(state, target_pos, sim_time)
        
        # Check if the episode is done based on certain conditions
        done = False
        if not self.in_bounds(state):
            # Episode is done if the drone goes out of bounds
            done = True
        elif sim_time >= 20:
            # Episode is done if the time exceeds 20 seconds
            done = True
    
        truncated = False
        info = {}
    
        # logging.info(f"Left Thrust: {u_1}, Right Thrust: {u_2}, State: {state}, Reward: {reward}, Done: {done}")
        return state, reward, done, truncated, info

    def _convert_action(self, action):
        u_1_idx = action // self.num_steps
        u_2_idx = action % self.num_steps
        u_1 = u_1_idx * 0.00625
        u_2 = u_2_idx * 0.00625
        return u_1, u_2

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def in_bounds(self, state):
        x, y = state[0], state[1]
        return -8 <= x <= 8 and -8 <= y <= 8

## RL Training
# Define the state space
state_low = np.array([-8, -8, -np.inf, -np.inf, -np.pi, -np.inf])
state_high = np.array([8, 8, np.inf, np.inf, np.pi, np.inf])
state_space = gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32)

# Define the reward function
def calc_reward(state, target_pos, t):
    x, y = state[0], state[1]
    x_targ, y_targ = target_pos[0], target_pos[1]
    
    # Punish for leaving the game area
    if not (-8 <= x <= 8 and -8 <= y <= 8):
        return -500
    
    # Reward for survival
    reward = 0.1
    
    # Reward for being close to the target
    dist = ((x - x_targ)**2 + (y - y_targ)**2)**0.5
    reward += 1 / (1 + dist)
    
    # Reward for being at the target for 10 seconds
    if 10 <= t <= 20 and dist < 0.01:
        reward += 100
        
    # logging.info(f"Time: {t}, State: {state}, Target: {target_pos}, Reward: {reward}")
    
    return reward

# Create the wrapper for the environment
env_wrapper = EnvironmentWrapper(environment)

# Training loop
target_pos = targets[0]
def train(num_episodes):
    episode_rewards = []
    logger = logging.getLogger(__name__)
    
    for ep in tqdm(range(num_episodes), desc="Training Progress"): #range(num_episodes):
        states = env_vector.reset()
        done = [False] * num_cpu  # Initialize done for each environment
        episode_reward = 0
        
        while not all(done):
            actions = model.predict(states, deterministic=True)[0]
            next_states, rewards, dones, infos = env_vector.step(actions)
            
            model.replay_buffer.add(states, next_states, actions, rewards, dones, infos)
            states = next_states
            episode_reward += sum(rewards)
            done = [done[i] or dones[i] for i in range(num_cpu)]  # Update the done list
            
            if model.replay_buffer.size() >= model.batch_size:
                model.train(gradient_steps=1, batch_size=model.batch_size)
                
            if all(done):
                states = env_vector.reset()
                
            # Update the target position for the next episode
            # target_idx = (ep + 1) % len(targets)
            # target_pos = targets[target_idx]
            
            # Print episode information
            # print(f"Episode: {ep+1}, Reward: {episode_reward:.2f}")
        
        episode_rewards.append(episode_reward)
        logger.info(f"Episode: {ep+1}, Reward: {episode_reward:.2f}")
    
    # Save the trained model
    model.save("trained_model")
    
    return episode_rewards

def reload():
    # re importing the controller module without closing the program
    try:
        importlib.reload(controller)
        environment.reset(controller.group_number, controller.wind_active)

    except Exception as e:
        print("Error reloading controller.py")
        print(e)

if __name__ == "__main__":
    # Create the DQN agent
    num_cpu = int(multiprocessing.cpu_count()/2)
    env_vector = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    model = DQN('MlpPolicy', env=env_vector, verbose=2, learning_starts=1000, target_update_interval=500)

    # Configure the logger
    if not hasattr(model, '_logger') or model._logger is None:
        model._logger = configure()
        
    # Train the model
    num_episodes = 10000
    episode_rewards = train(num_episodes)

    # Print the average reward over the last 100 episodes
    avg_reward = np.mean(episode_rewards[-100:])
    print(f"\nAverage reward over the last 100 episodes: {avg_reward:.2f}")