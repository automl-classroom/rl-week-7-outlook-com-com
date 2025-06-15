"""
Baseline DQN for comparison with RND DQN.
"""

from typing import Any, Dict, List, Tuple
import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed

@hydra.main(version_base=None, config_path="../../rl_exercises/configs/exercise", config_name="w6_dqn.yaml")
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env_name)
    set_seed(env, cfg.seed)
    agent = DQNAgent(
        env,
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=cfg.seed,
    )
    agent.train(num_frames=10000, eval_interval=1000)

if __name__ == "__main__":
    main()
