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
        self.num_throttle_steps = 41
        self.action_space = gym.spaces.Discrete(self.num_throttle_steps ** 2)  # Define the discrete action space
        self.observation_space = gym.spaces.Box(
            low=np.array([-8, -8, -np.inf, -np.inf, -np.pi, -np.inf]),
            high=np.array([8, 8, np.inf, np.inf, np.pi, np.inf]),
            dtype=np.float32
        )  # Define the observation space
        self.time = 0.0  # Initialize time variable

    def reset(self):
        self.environment.reset(controller.group_number, controller.wind_active)
        self.time = 0.0  # Reset time variable
        return self.environment.drone.get_state()

    def step(self, action):
        u_1, u_2 = self._convert_action(action)
        self.environment.step((u_1, u_2))
        state = self.environment.drone.get_state()

        # Calculate time_delta and update time variable
        time_delta = self.environment.clock.tick(60) / 1000.0
        self.time += time_delta

        reward = calculate_reward(state, target_pos, self.time)
        
        # Check if the episode is done based on certain conditions
        done = False
        if not self.is_within_bounds(state):
            # Episode is done if the drone goes out of bounds
            done = True
        elif self.time >= 20.0:
            # Episode is done if the time exceeds 20 seconds
            done = True
    
        return state, reward, done, {}

    def _convert_action(self, action):
        u_1_idx = action // self.num_throttle_steps
        u_2_idx = action % self.num_throttle_steps
        u_1 = u_1_idx * 0.025
        u_2 = u_2_idx * 0.025
        return u_1, u_2

    def render(self, mode='human'):
        # You can implement rendering if needed
        pass
    
    def is_within_bounds(self, state):
        x, y = state[0], state[1]
        return -8 <= x <= 8 and -8 <= y <= 8

running = True
target_pos = targets[0]

## RL Training
# Define the state space
state_low = np.array([-8, -8, -np.inf, -np.inf, -np.pi, -np.inf])
state_high = np.array([8, 8, np.inf, np.inf, np.pi, np.inf])
state_space = gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32)

# Define the reward function
def calculate_reward(state, target_pos, t):
    x, y = state[0], state[1]
    x_targ, y_targ = target_pos[0], target_pos[1]
    
    # Punish for leaving the game area
    if not (-8 <= x <= 8 and -8 <= y <= 8):
        return -100
    
    # Reward for survival
    reward = 0.1
    
    # Reward for being close to the target
    dist = ((x - x_targ)**2 + (y - y_targ)**2)**0.5
    reward += 1 / (1 + dist)
    
    # Reward for being at the target for 10 seconds
    if 10 <= t <= 20 and dist < 0.01:
        reward += 100
    
    return reward

# Create the wrapper for the environment
env_wrapper = EnvironmentWrapper(environment)

# Create the DQN agent
model = DQN('MlpPolicy', env=env_wrapper, verbose=1, learning_starts=1000, target_update_interval=500)

# Training loop
def train(num_episodes):
    episode_rewards = []

    for ep in range(num_episodes):
        state = env_wrapper.reset()
        done = False
        episode_reward = 0
        while not done:
            action = model.predict(state, deterministic=True)[0]
            next_state, reward, done, info = env_wrapper.step(action)

            # Check if the drone is within bounds
            if not env_wrapper.is_within_bounds(next_state):
                reward = -100  # Apply punishment for breaching bounds
                done = True  # End the episode

            # Convert the action to the appropriate format
            action = np.array(action).reshape(1, -1)

            model.replay_buffer.add(state, action, next_state, reward, done, info)  # Store the experience in the replay buffer
            state = next_state
            episode_reward += reward

            if model.replay_buffer.size() >= model.batch_size:
                # Update the model using the collected experiences
                model.train(gradient_steps=1, batch_size=model.batch_size)

        # Print episode information
        print(f"Episode: {ep+1}, Reward: {episode_reward:.2f}, Position: ({state[0]:.2f}, {state[1]:.2f})")
        episode_rewards.append(episode_reward)

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


# def check_action(unchecked_action):
#     # Check if the action is a tuple or list and of length 2
#     if isinstance(unchecked_action, (tuple, list)):
#         if len(unchecked_action) != 2:
#             print(
#                 "WARNING: Controller returned an action of length "
#                 + str(len(unchecked_action))
#                 + ", expected 2"
#             )
#             checked_action = (0, 0)
#             sys.exit()
#         else:
#             checked_action = unchecked_action

#     else:
#         print(
#             "WARNING: Controller returned an action of type "
#             + str(type(unchecked_action))
#             + ", expected list or tuple"
#         )
#         checked_action = (0, 0)
#         sys.exit()

#     return checked_action
    
# Train the model
num_episodes = 1000
episode_rewards = train(num_episodes)

# Print the average reward over the last 100 episodes
avg_reward = np.mean(episode_rewards[-100:])
print(f"\nAverage reward over the last 100 episodes: {avg_reward:.2f}")