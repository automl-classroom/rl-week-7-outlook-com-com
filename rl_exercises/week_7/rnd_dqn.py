"""
Deep Q-Learning with RND implementation.
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
import os


class RNDNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128, n_layers=2):
        super().__init__()
        layers = []
        last_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.ReLU())
            last_dim = hidden_size
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.seed = seed
        # TODO: initialize the RND networks
        obs_dim = env.observation_space.shape[0]
        rnd_output_size = rnd_hidden_size  # embedding size

        self.rnd_target = RNDNetwork(obs_dim, rnd_output_size, rnd_hidden_size, rnd_n_layers)
        self.rnd_predictor = RNDNetwork(obs_dim, rnd_output_size, rnd_hidden_size, rnd_n_layers)

        # Freeze target network parameters
        for param in self.rnd_target.parameters():
            param.requires_grad = False

        self.rnd_optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=rnd_lr)
        self.rnd_update_freq = rnd_update_freq
        self.rnd_reward_weight = rnd_reward_weight

        # device setup (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnd_target.to(self.device)
        self.rnd_predictor.to(self.device)
        ...

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        """
        # TODO: get states and next_states from the batch
        # TODO: compute the MSE
        # TODO: update the RND network
        states = np.array([transition[0] for transition in training_batch])
        states_tensor = torch.FloatTensor(states).to(self.device)

        with torch.no_grad():
            target_emb = self.rnd_target(states_tensor)
        pred_emb = self.rnd_predictor(states_tensor)

        loss = nn.MSELoss()(pred_emb, target_emb)

        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()

        return loss.item()
        ...

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        # TODO: predict embeddings
        # TODO: get error
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            target_emb = self.rnd_target(state_tensor)
        pred_emb = self.rnd_predictor(state_tensor)
        error = (pred_emb - target_emb).pow(2).mean().item()
        return error
        ...

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []

        snapshot_steps = [100, 1000, 10000]
        saved_snapshots = set()
        # Get absolute path to main workspace root (rl-week-7-outlook-com-com)
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        snapshot_dir = os.path.join(workspace_root, 'rl_exercises', 'week_7')
        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # TODO: apply RND bonus
            rnd_bonus = self.get_rnd_bonus(state)
            reward += self.rnd_reward_weight * rnd_bonus    

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

                if self.total_steps % self.rnd_update_freq == 0:
                    self.update_rnd(batch)

            # --- Save model snapshots at key steps ---
            for snap in snapshot_steps:
                if frame >= snap and snap not in saved_snapshots:
                    os.makedirs(snapshot_dir, exist_ok=True)
                    import torch
                    snapshot_path = os.path.join(snapshot_dir, f"rnd_dqn_snapshot_{snap}.pth")
                    print(f"Saving snapshot at frame {frame} to {snapshot_path}")
                    print(f"[DEBUG] Saving snapshot at frame {frame} to absolute path: {os.path.abspath(snapshot_path)}")
                    torch.save(self.get_snapshot(), snapshot_path)
                    saved_snapshots.add(snap)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards)
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        # Saving to .csv for simplicity
        # Could also be e.g. npz
        print("Training complete.")
        training_data = pd.DataFrame({"steps": steps, "rewards": episode_rewards})
        training_data.to_csv(f"training_data_seed_{self.seed}.csv", index=False)

    def get_snapshot(self):
        return {
            'q': self.q.state_dict(),
            'target_q': self.target_q.state_dict(),
            'rnd_predictor': self.rnd_predictor.state_dict(),
            'rnd_target': self.rnd_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rnd_optimizer': self.rnd_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'seed': self.seed,
        }

    def load(self, path: str) -> None:
        """Load agent from a snapshot file."""
        import torch
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["q"])
        self.target_q.load_state_dict(checkpoint["target_q"])
        self.rnd_predictor.load_state_dict(checkpoint["rnd_predictor"])
        self.rnd_target.load_state_dict(checkpoint["rnd_target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.rnd_optimizer.load_state_dict(checkpoint["rnd_optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.seed = checkpoint.get("seed", 0)


@hydra.main(config_path="../../rl_exercises/configs/exercise", config_name="w6_dqn.yaml", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env_name)
    set_seed(env, cfg.seed)

    # 3) TODO: instantiate & train the agent
    agent = RNDDQNAgent(
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
    rnd_hidden_size=128,
    rnd_lr=1e-3,
    rnd_update_freq=1000,
    rnd_n_layers=2,
    rnd_reward_weight=0.1,
    )
    agent.train(num_frames=10000, eval_interval=1000)

if __name__ == "__main__":
    main()
