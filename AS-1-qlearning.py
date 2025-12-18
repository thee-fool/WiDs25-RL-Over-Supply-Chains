import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Environment setup
# -----------------------------
env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -----------------------------
# Hyperparameters
# -----------------------------
alpha = 0.1          
gamma = 0.99          
epsilon = 1.0            
epsilon_min = 0.05
epsilon_decay = 0.9995

num_episodes = 20000
max_steps = 200

# -----------------------------
# Q-table initialization
# -----------------------------
Q = np.zeros((n_states, n_actions))

# -----------------------------
# Logging
# -----------------------------
episode_rewards = []
success_rate = []

success_window = 100
success_buffer = []

# -----------------------------
# Q-learning loop
# -----------------------------
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):

        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state, best_next_action]
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state
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

# -----------------------------
# Plot success rate
# -----------------------------
plt.plot(success_rate)
plt.xlabel("episode")
plt.ylabel("Succes")
plt.title("qlearning")
plt.grid()
plt.show()
