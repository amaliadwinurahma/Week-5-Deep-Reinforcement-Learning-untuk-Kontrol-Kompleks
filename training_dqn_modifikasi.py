import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Pastikan backend mendukung tampilan grafik
matplotlib.use("TkAgg")

# === Inisialisasi environment ===
train_env = gym.make("Pendulum-v1", render_mode=None)       # training tanpa visual
eval_env  = gym.make("Pendulum-v1", render_mode="rgb_array") # evaluasi → simpan frame

# Parameter environment
state_size = train_env.observation_space.shape[0]
action_size = train_env.action_space.shape[0]
action_bound = train_env.action_space.high[0]

# === Hyperparameter DQN ===
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory = deque(maxlen=10000)

# === Model Deep Q-Network (DQN) ===
model = keras.Sequential([
    keras.Input(shape=(state_size,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(action_size, activation="tanh")  # aksi continuous
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

# === Fungsi aksi (eksplorasi/eksploitasi) ===
def select_action(state, epsilon_val):
    if np.random.rand() <= epsilon_val:
        return np.random.uniform(-action_bound, action_bound, size=action_size)
    q_values = model.predict(state, verbose=0)
    return q_values[0] * action_bound

# === Logging hasil reward ===
rewards_per_episode = []

# === Training loop ===
for episode in range(1000):  # bisa diperbesar
    state, _ = train_env.reset()
    state = np.array(state).reshape(1, state_size)
    total_reward = 0

    for t in range(200):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state).reshape(1, state_size)

        # Reward shaping
        reward -= 0.05 * np.square(action).sum()

        # Simpan pengalaman
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            break

    rewards_per_episode.append(total_reward)
    print(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    # Training replay memory
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for state_mb, action_mb, reward_mb, next_state_mb, done_mb in minibatch:
            target = reward_mb
            if not done_mb:
                target += gamma * np.amax(model.predict(next_state_mb, verbose=0)[0])
            target_f = model.predict(state_mb, verbose=0)
            target_f[0] = target
            model.fit(state_mb, target_f, epochs=1, verbose=0)

    # Kurangi epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training selesai!")

# === Grafik Reward ===
plt.figure(figsize=(10,5))
plt.plot(rewards_per_episode, label="Reward per Episode", alpha=0.5)

# Moving average
window = 10
if len(rewards_per_episode) >= window:
    moving_avg = np.convolve(rewards_per_episode, np.ones(window)/window, mode="valid")
    plt.plot(range(window-1, len(rewards_per_episode)), moving_avg,
             label=f"Moving Average (window={window})", color="red")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve DQN pada Pendulum-v1")
plt.legend()
plt.grid(True)
plt.show()

# === Evaluasi → Simpan Frame ===
frames = []
state, _ = eval_env.reset()
state = np.array(state).reshape(1, state_size)
for _ in range(200):
    action = select_action(state, epsilon_val=0.0)  # eksploitasi penuh
    next_state, reward, terminated, truncated, _ = eval_env.step(action)
    frames.append(eval_env.render())  # simpan frame RGB
    state = np.array(next_state).reshape(1, state_size)
    if terminated or truncated:
        break
eval_env.close()

# === Animasi dengan Matplotlib ===
fig = plt.figure()
plt.axis("off")
im = plt.imshow(frames[0])

def update(i):
    im.set_array(frames[i])
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
plt.show()
