import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

# Inisialisasi environment
env = gym.make("CartPole-v1")

# Parameter DRL
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

# Membangun model Deep Q-Network (DQN)
model = keras.Sequential([
    keras.Input(shape=(state_size,)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

# Fungsi memilih aksi (eksplorasi vs eksploitasi)
def select_action(state):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)  # eksplorasi
    q_values = model.predict(state, verbose=0)  # eksploitasi
    return np.argmax(q_values[0])

# Proses training
for episode in range(1000):
    state, _ = env.reset()
    state = np.array(state).reshape(1, state_size)

    for time in range(500):
        # Pilih aksi
        action = select_action(state)

        # Eksekusi aksi
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state).reshape(1, state_size)

        # Simpan ke memori
        memory.append((state, action, reward, next_state, done))
        state = next_state

        print(f"Episode: {episode}, Score: {time}, Epsilon: {epsilon:.2f}")

        if done:
            break

    # Training DQN dari pengalaman
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for state_mb, action_mb, reward_mb, next_state_mb, done_mb in minibatch:
            target = reward_mb
            if not done_mb:
                target += gamma * np.amax(model.predict(next_state_mb, verbose=0)[0])
            target_f = model.predict(state_mb, verbose=0)
            target_f[0][action_mb] = target
            model.fit(state_mb, target_f, epochs=1, verbose=0)

    # Kurangi epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training selesai!")
