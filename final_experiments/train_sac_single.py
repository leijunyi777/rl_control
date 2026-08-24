"""独立单 gap SAC 训练脚本。

本文件按 code/python/main12_sac_train.py 的训练流程组织，但 SAC 网络、
ReplayBuffer 和训练超参数都写在本文件内，不再从 main11/main12 导入。
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from single_gap_env import (
    ACTION_DIM,
    ALPHA_LR,
    BATCH_SIZE,
    DT,
    EGO_X_BASE,
    EGO_X_RANDOM_RANGE,
    GAMMA,
    HIDDEN_SIZE,
    INITIAL_RANDOM_STEPS,
    Main12SacUEnv,
    NUM_EPISODES,
    POLICY_LR,
    Q_LR,
    RENDER_DURING_TRAINING,
    REPLAY_SIZE,
    RESULT_FIG_PATH,
    SEED,
    SIM_TIME,
    TAU,
    U_HIGH,
    U_LOW,
    UPDATES_PER_STEP,
)


POLICY_PATH = "main12_sac_u_policy.pth"
CSV_PATH = "single_gap_sac_train.csv"


def set_seed(seed):
    """设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ReplayBuffer:
    """SAC 经验回放池。"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.as_tensor(state, dtype=torch.float32, device=device),
            torch.as_tensor(action, dtype=torch.float32, device=device),
            torch.as_tensor(reward[:, None], dtype=torch.float32, device=device),
            torch.as_tensor(next_state, dtype=torch.float32, device=device),
            torch.as_tensor(done[:, None], dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    """SAC 高斯策略网络，输出 tanh 归一化 action。"""

    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20.0, 2.0)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class QNetwork(nn.Module):
    """SAC Q 网络。"""

    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent:
    """最小 SAC 实现，结构与原始 main11_sac_train.py 保持一致。"""

    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.policy = GaussianPolicy(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.q1 = QNetwork(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.q2 = QNetwork(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.target_q1 = QNetwork(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.target_q2 = QNetwork(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=POLICY_LR)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=Q_LR)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=Q_LR)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.policy.sample(state_tensor)
        chosen = mean_action if evaluate else action
        return chosen.squeeze(0).cpu().numpy()

    def update(self, replay_buffer):
        state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE, self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_q = torch.min(
                self.target_q1(next_state, next_action),
                self.target_q2(next_state, next_action),
            ) - self.alpha * next_log_prob
            target = reward + (1.0 - done) * GAMMA * target_q

        q1_loss = F.mse_loss(self.q1(state, action), target)
        q2_loss = F.mse_loss(self.q2(state, action), target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_action, log_prob, _ = self.policy.sample(state)
        min_q = torch.min(self.q1(state, new_action), self.q2(state, new_action))
        policy_loss = (self.alpha * log_prob - min_q).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update(self, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(TAU * source_param.data + (1.0 - TAU) * target_param.data)


def moving_average(values, window=10):
    """计算滑动平均曲线。"""
    if len(values) < window:
        return np.asarray(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_results(episode_rewards, episode_progress, episode_collisions, episode_successes, episode_mean_u, episode_ego_x0):
    """绘制与原始 main12 一致的训练结果图。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)
    episodes = np.arange(1, len(episode_rewards) + 1)

    axes[0].plot(episodes, episode_rewards, color="tab:blue", alpha=0.35, label="Episode reward")
    avg_reward = moving_average(episode_rewards, 10)
    axes[0].plot(np.arange(len(avg_reward)) + 1, avg_reward, color="tab:blue", linewidth=2, label="10-episode average")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Main12 SAC u(t) Training Result")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(episodes, episode_progress, color="tab:green", label="Final lane-change progress")
    axes[1].set_ylabel("Progress")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(episodes, episode_mean_u, color="tab:orange", label="Mean RL u(t)")
    axes[2].set_ylabel("Mean u(t)")
    axes[2].set_ylim(U_LOW - 0.1, U_HIGH + 0.1)
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(episodes, episode_ego_x0, color="tab:cyan", label="Initial ego x")
    axes[3].axhline(EGO_X_BASE, color="gray", linestyle="--", linewidth=1.5)
    axes[3].set_ylabel("ego x0 (m)")
    axes[3].legend()
    axes[3].grid(True)

    axes[4].plot(episodes, episode_collisions, color="tab:red", label="Collision")
    axes[4].set_ylabel("Collision")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].legend()
    axes[4].grid(True)

    axes[5].plot(episodes, episode_successes, color="tab:purple", label="Success")
    axes[5].set_xlabel("Episode")
    axes[5].set_ylabel("Success")
    axes[5].set_ylim(-0.05, 1.05)
    axes[5].legend()
    axes[5].grid(True)

    fig.tight_layout()
    fig.savefig(RESULT_FIG_PATH, dpi=180)
    plt.show()


def write_training_csv(rows, csv_out):
    """写出训练日志 CSV。"""
    if not csv_out or not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(episodes=NUM_EPISODES, policy_out=POLICY_PATH, csv_out=CSV_PATH, save_csv=True):
    """执行 SAC 训练。"""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Main12SacUEnv(render=RENDER_DURING_TRAINING)
    state_dim = env.reset().shape[0]
    action_dim = ACTION_DIM

    agent = SACAgent(state_dim, action_dim, device)
    replay_buffer = ReplayBuffer(REPLAY_SIZE)

    total_steps = 0
    episode_rewards = []
    episode_progress = []
    episode_collisions = []
    episode_successes = []
    episode_mean_u = []
    episode_ego_x0 = []
    rows = []
    last_losses = {"q1_loss": 0.0, "q2_loss": 0.0, "policy_loss": 0.0, "alpha": 0.0}

    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        u_values = []
        last_info = {"lane_progress": 0.0, "collided": False, "success": False, "u_t": 0.0}

        for _ in range(int(SIM_TIME / DT)):
            if total_steps < INITIAL_RANDOM_STEPS:
                action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            u_values.append(info["u_t"])
            last_info = info
            total_steps += 1

            if len(replay_buffer) >= BATCH_SIZE:
                for _ in range(UPDATES_PER_STEP):
                    last_losses = agent.update(replay_buffer)

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_progress.append(last_info["lane_progress"])
        episode_collisions.append(float(last_info["collided"]))
        episode_successes.append(float(last_info["success"]))
        episode_mean_u.append(float(np.mean(u_values)) if u_values else 0.0)
        episode_ego_x0.append(float(env.ego_x0))

        row = {
            "episode": episode,
            "reward": episode_reward,
            "progress": last_info["lane_progress"],
            "collision": float(last_info["collided"]),
            "success": float(last_info["success"]),
            "mean_u": episode_mean_u[-1],
            "ego_x0": float(env.ego_x0),
            "q1_loss": last_losses["q1_loss"],
            "q2_loss": last_losses["q2_loss"],
            "policy_loss": last_losses["policy_loss"],
            "alpha": last_losses["alpha"],
            "total_steps": total_steps,
        }
        rows.append(row)

        print(
            f"Episode {episode:04d} | reward={episode_reward:8.2f} | "
            f"progress={last_info['lane_progress']:.3f} | "
            f"mean_u={episode_mean_u[-1]:.3f} | ego_x0={env.ego_x0:.3f} | "
            f"q1_loss={last_losses['q1_loss']:.4f} | alpha={last_losses['alpha']:.4f} | "
            f"success={last_info['success']} | collision={last_info['collided']} | steps={total_steps}"
        )

    plot_training_results(episode_rewards, episode_progress, episode_collisions, episode_successes, episode_mean_u, episode_ego_x0)
    torch.save(
        {
            "policy_state_dict": agent.policy.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "u_low": U_LOW,
            "u_high": U_HIGH,
            "sim_time": SIM_TIME,
            "dt": DT,
            "ego_x_base": EGO_X_BASE,
            "ego_x_random_range": EGO_X_RANDOM_RANGE,
        },
        policy_out,
    )
    if save_csv:
        write_training_csv(rows, csv_out)
    print(f"Saved trained policy to {policy_out}")
    print(f"Saved training result figure to {RESULT_FIG_PATH}")
    if save_csv:
        print(f"Saved training CSV to {csv_out}")


def main():
    parser = argparse.ArgumentParser(description="independent main12-aligned single-gap SAC training")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="training episode count")
    parser.add_argument("--policy-out", default=POLICY_PATH, help="output policy path")
    parser.add_argument("--csv-out", default=CSV_PATH, help="output CSV path")
    parser.add_argument("--no-csv", action="store_true", help="disable CSV output")
    args = parser.parse_args()
    train(args.episodes, args.policy_out, args.csv_out, save_csv=not args.no_csv)


if __name__ == "__main__":
    main()
