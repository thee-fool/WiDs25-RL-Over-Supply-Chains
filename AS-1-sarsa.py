import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Environment
# -----------------------------
env = gym.make(
    "FrozenLake-v1",
    map_name="6x6",
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n


alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

num_episodes = 20000
max_steps = 200

# -----------------------------
# Q-table
# -----------------------------
Q_sarsa = np.zeros((n_states, n_actions))

# -----------------------------
# Logging
# -----------------------------
episode_rewards = []
success_rate = []

success_window = 100
success_buffer = []

# -----------------------------
# ε-greedy policy
# -----------------------------
def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

# -----------------------------
# SARSA loop
# -----------------------------
for episode in range(num_episodes):
    state, _ = env.reset()
    action = epsilon_greedy(Q_sarsa, state, epsilon)
    total_reward = 0

    for step in range(max_steps):
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_action = epsilon_greedy(Q_sarsa, next_state, epsilon)

        # SARSA update
        td_target = reward + gamma * Q_sarsa[next_state, next_action]
        td_error = td_target - Q_sarsa[state, action]
        Q_sarsa[state, action] += alpha * td_error

        state, action = next_state, next_action
        total_reward += reward

        if done:
            break

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Logging
    episode_rewards.append(total_reward)
    success_buffer.append(total_reward)
    if len(success_buffer) > success_window:
        success_buffer.pop(0)

    success_rate.append(np.mean(success_buffer))

env.close()
