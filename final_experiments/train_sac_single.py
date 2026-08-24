"""基于 single_gap_env.py 的 SAC 训练脚本。

只依赖 single_gap_env.py 和 torch。训练结束后输出：
1. 训练好的 policy checkpoint (.pth)
2. 每个 episode 的训练结果 CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import deque
from typing import Deque, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from single_gap_env import SingleGapEnv, U_HIGH, U_LOW


# =========================
# 与最新 main12 SAC 训练一致的参数
# =========================
NUM_EPISODES = 200
BATCH_SIZE = 256
REPLAY_SIZE = 200_000
INITIAL_RANDOM_STEPS = 1_000
UPDATES_PER_STEP = 1
GAMMA = 0.99
TAU = 0.005
POLICY_LR = 3e-4
Q_LR = 3e-4
ALPHA_LR = 3e-4
HIDDEN_SIZE = 256
ACTION_DIM = 1
SEED = 7


def set_seed(seed: int) -> None:
    """设置 Python 与 torch 的随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    """SAC 经验回放池。"""

    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[List[float], List[float], float, List[float], float]] = deque(maxlen=capacity)

    def push(self, state: List[float], action: List[float], reward: float, next_state: List[float], done: float) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.float32, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(next_state, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    """SAC 高斯策略网络，输出归一化动作 [-1, 1]。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
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
        h = self.net(state)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -20.0, 2.0)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class QNetwork(nn.Module):
    """SAC Q 网络。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


class SACAgent:
    """最小独立 SAC 实现。"""

    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
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
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)
        self.target_entropy = -float(action_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state: List[float], evaluate: bool = False) -> List[float]:
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(state_tensor)
            else:
                action, _, _ = self.policy.sample(state_tensor)
        return action.cpu()[0].tolist()

    def update(self, replay_buffer: ReplayBuffer) -> dict:
        state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE, self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_q = torch.min(
                self.target_q1(next_state, next_action),
                self.target_q2(next_state, next_action),
            ) - self.alpha * next_log_prob
            q_target = reward + (1.0 - done) * GAMMA * target_q

        q1_loss = F.mse_loss(self.q1(state, action), q_target)
        q2_loss = F.mse_loss(self.q2(state, action), q_target)

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

        self.soft_update(self.q1, self.target_q1)
        self.soft_update(self.q2, self.target_q2)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    @staticmethod
    def soft_update(source: nn.Module, target: nn.Module) -> None:
        """目标网络软更新。"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(TAU * source_param.data + (1.0 - TAU) * target_param.data)


def action_to_u(action: List[float]) -> float:
    """将 SAC 输出的 [-1,1] 动作映射到真实 u(t)。"""
    return U_LOW + 0.5 * (action[0] + 1.0) * (U_HIGH - U_LOW)


def train(
    episodes: int,
    seed: int,
    policy_out: str,
    csv_out: str,
) -> None:
    """执行 SAC 训练。"""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SingleGapEnv(seed=seed)
    state_dim = len(env.reset(seed=seed))
    agent = SACAgent(state_dim, ACTION_DIM, device)
    replay_buffer = ReplayBuffer(REPLAY_SIZE)

    rows = []
    total_steps = 0
    last_losses = {"q1_loss": 0.0, "q2_loss": 0.0, "policy_loss": 0.0, "alpha": 0.0}

    for episode in range(1, episodes + 1):
        state = env.reset(seed=seed + episode)
        episode_reward = 0.0
        done = False
        step_count = 0
        last_info = {"lane_progress": 0.0, "collided": False, "success": False, "u_t": 0.0, "min_distance": 0.0}
        u_sum = 0.0

        while not done:
            if total_steps < INITIAL_RANDOM_STEPS:
                action = [random.uniform(-1.0, 1.0)]
            else:
                action = agent.select_action(state)
            u_t = action_to_u(action)
            next_state, reward, done, info = env.step(u_t)
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            step_count += 1
            total_steps += 1
            u_sum += u_t
            last_info = info

            if len(replay_buffer) >= BATCH_SIZE:
                for _ in range(UPDATES_PER_STEP):
                    last_losses = agent.update(replay_buffer)

        row = {
            "episode": episode,
            "reward": episode_reward,
            "progress": last_info["lane_progress"],
            "collision": float(last_info["collided"]),
            "success": float(last_info["success"]),
            "time": last_info["time"],
            "steps": step_count,
            "mean_u": u_sum / max(step_count, 1),
            "ego_x0": env.ego_x0,
            "min_distance": last_info["min_distance"],
            **last_losses,
        }
        rows.append(row)
        print(
            f"Episode {episode:04d} | reward={episode_reward:9.3f} | "
            f"progress={row['progress']:.3f} | success={bool(row['success'])} | "
            f"collision={bool(row['collision'])} | mean_u={row['mean_u']:.3f}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(policy_out)), exist_ok=True)
    torch.save(
        {
            "state_dim": state_dim,
            "action_dim": ACTION_DIM,
            "hidden_size": HIDDEN_SIZE,
            "u_low": U_LOW,
            "u_high": U_HIGH,
            "policy_state_dict": agent.policy.state_dict(),
            "episodes": episodes,
            "seed": seed,
        },
        policy_out,
    )

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"训练完成，policy 已保存：{policy_out}")
    print(f"训练结果 CSV 已保存：{csv_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="单 gap SAC 训练")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="训练 episode 数")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    parser.add_argument("--policy-out", default="single_gap_sac_policy.pth", help="输出 policy 路径")
    parser.add_argument("--csv-out", default="single_gap_sac_train.csv", help="训练结果 CSV 路径")
    args = parser.parse_args()
    train(args.episodes, args.seed, args.policy_out, args.csv_out)


if __name__ == "__main__":
    main()
