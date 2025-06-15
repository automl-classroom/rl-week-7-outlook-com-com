import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from rl_exercises.week_4.dqn import DQNAgent, set_seed
import torch
import os

# --- CONFIG ---
SNAPSHOT_STEPS = [100, 1000, 10000]  # Example: early, mid, late
MODEL_PATHS = [
    'dqn_snapshot_100.pth',
    'dqn_snapshot_1000.pth',
    'dqn_snapshot_10000.pth',
]
ENV_NAME = 'LunarLander-v3'
SEED = 0

# --- SCRIPT ---
def plot_trajectory(agent, env, save_path):
    state, _ = env.reset()
    done = False
    xs, ys = [], []
    while not done:
        action, _ = agent.predict_action(state, evaluate=True)
        next_state, reward, done, truncated, info = env.step(action)
        xs.append(state[0])  # x position
        ys.append(state[1])  # y position
        state = next_state
        if done or truncated:
            break
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker='o')
    plt.title('Lander Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    set_seed(env, SEED)
    for steps, model_path in zip(SNAPSHOT_STEPS, MODEL_PATHS):
        if not os.path.exists(model_path):
            print(f"Model snapshot {model_path} not found. Please save your agent at these steps.")
            continue
        agent = DQNAgent(env)
        agent.load(model_path)
        save_path = f"trajectory_snapshot_{steps}.png"
        plot_trajectory(agent, env, save_path)
        print(f"Saved trajectory plot for {steps} steps as {save_path}")
