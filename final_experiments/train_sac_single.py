"""单 gap SAC 训练脚本。

本文件把原始 main12_sac_train.py 中与训练结果相关的 SAC 结构、随机数流程、
action 尺度、环境调用顺序和 checkpoint 字段逐项复制到最终版独立环境中。
额外保留 CSV 输出，方便论文实验统计。
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
    DT,
    EGO_X_BASE,
    EGO_X_RANDOM_RANGE,
    SIM_TIME,
    U_HIGH,
    U_LOW,
    SingleGapEnv,
)


# =========================
# 与原始 main12_sac_train.py 保持一致的训练参数
# =========================
# SAC 对样本量比较敏感，50+ 轮通常只适合快速冒烟测试；默认给到 300 轮用于正式训练。
NUM_EPISODES = 300
RENDER_DURING_TRAINING = False
SEED = 7

BATCH_SIZE = 128
REPLAY_SIZE = 200_000
# 单维 action 问题不需要太长的纯随机阶段，降低该值可以更早进入策略学习。
INITIAL_RANDOM_STEPS = 300
UPDATES_PER_STEP = 1
GAMMA = 0.99
TAU = 0.005
POLICY_LR = 2e-4
Q_LR = 2e-4
ALPHA_LR = 1e-4
HIDDEN_SIZE = 256
REWARD_SCALE = 0.02
GRAD_CLIP_NORM = 5.0
MIN_ALPHA = 0.08
MAX_ALPHA = 1.0

POLICY_PATH = "single_gap_sac_policy.pth"
CSV_PATH = "single_gap_sac_train.csv"


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    """经验回放池，采样方式与原始 main11/main12 SAC 完全一致。"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.as_tensor(state, dtype=torch.float32, device=device),
            torch.as_tensor(action, dtype=torch.float32, device=device),
            torch.as_tensor(reward[:, None], dtype=torch.float32, device=device),
            torch.as_tensor(next_state, dtype=torch.float32, device=device),
            torch.as_tensor(done[:, None], dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    """SAC 高斯策略网络，输出归一化 action，范围为 [-1, 1]。"""

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
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent:
    """与原始 main11_sac_train.py 一致的最小 SAC 实现。"""

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

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)

    @property
    def alpha(self):
        return torch.clamp(self.log_alpha.exp(), MIN_ALPHA, MAX_ALPHA)

    def select_action(self, state, evaluate: bool = False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.policy.sample(state_tensor)
        chosen = mean_action if evaluate else action
        return chosen.squeeze(0).cpu().numpy()

    def update(self, replay_buffer: ReplayBuffer):
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
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), GRAD_CLIP_NORM)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), GRAD_CLIP_NORM)
        self.q2_opt.step()

        new_action, log_prob, _ = self.policy.sample(state)
        min_q = torch.min(self.q1(state, new_action), self.q2(state, new_action))
        policy_loss = (self.alpha * log_prob - min_q).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_CLIP_NORM)
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

    def _soft_update(self, source, target) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(TAU * source_param.data + (1.0 - TAU) * target_param.data)


def action_to_u(action) -> float:
    """仅用于记录 mean_u，不用于环境 step，避免改变原始训练 action 路径。"""
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return float(U_LOW + 0.5 * (action[0] + 1.0) * (U_HIGH - U_LOW))


def train(episodes: int, seed: int, policy_out: str, csv_out: str) -> None:
    """执行 SAC 训练，并输出 checkpoint 与每轮结果 CSV。"""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 保持原始 main12_sac_train.py 的调用顺序：
    # 构造环境时 reset 一次，随后再 reset 一次读取 state_dim。
    env = SingleGapEnv(render=RENDER_DURING_TRAINING)
    state_dim = env.reset().shape[0]
    action_dim = ACTION_DIM

    agent = SACAgent(state_dim, action_dim, device)
    replay_buffer = ReplayBuffer(REPLAY_SIZE)

    total_steps = 0
    rows = []
    last_losses = {"q1_loss": 0.0, "q2_loss": 0.0, "policy_loss": 0.0, "alpha": 0.0}

    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_train_reward = 0.0
        term_sums = {}
        u_values = []
        last_info = {"lane_progress": 0.0, "collided": False, "success": False, "u_t": 0.0, "time": 0.0, "min_distance": 0.0}
        step_count = 0

        for _ in range(int(SIM_TIME / DT)):
            if total_steps < INITIAL_RANDOM_STEPS:
                action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step_action(action)
            train_reward = reward * REWARD_SCALE
            replay_buffer.push(state, action, train_reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            episode_train_reward += train_reward
            u_values.append(info["u_t"])
            last_info = info
            for key, value in info.get("reward_terms", {}).items():
                term_sums[key] = term_sums.get(key, 0.0) + float(value)
            total_steps += 1
            step_count += 1

            if len(replay_buffer) >= BATCH_SIZE:
                for _ in range(UPDATES_PER_STEP):
                    last_losses = agent.update(replay_buffer)

            if done:
                break

        row = {
            "episode": episode,
            "reward": episode_reward,
            "train_reward": episode_train_reward,
            "progress": last_info["lane_progress"],
            "collision": float(last_info["collided"]),
            "success": float(last_info["success"]),
            "time": last_info["time"],
            "steps": step_count,
            "mean_u": float(np.mean(u_values)) if u_values else 0.0,
            "ego_x0": float(env.ego_x0),
            "min_distance": last_info["min_distance"],
            **last_losses,
        }
        for key in sorted(term_sums):
            row[f"term_{key}"] = term_sums[key]
        rows.append(row)

        print(
            f"Episode {episode:04d} | reward={episode_reward:8.2f} | "
            f"progress={last_info['lane_progress']:.3f} | "
            f"mean_u={row['mean_u']:.3f} | ego_x0={env.ego_x0:.3f} | "
            f"q1_loss={last_losses['q1_loss']:.2f} | "   # 新增
            f"alpha={last_losses['alpha']:.3f} | "   # 新增
            f"success={last_info['success']} | collision={last_info['collided']} | steps={total_steps}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(policy_out)), exist_ok=True)
    torch.save(
        {
            "policy_state_dict": agent.policy.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "u_low": U_LOW,
            "u_high": U_HIGH,
            "reward_scale": REWARD_SCALE,
            "min_alpha": MIN_ALPHA,
            "max_alpha": MAX_ALPHA,
            "sim_time": SIM_TIME,
            "dt": DT,
            "ego_x_base": EGO_X_BASE,
            "ego_x_random_range": EGO_X_RANDOM_RANGE,
        },
        policy_out,
    )

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved trained policy to {policy_out}")
    print(f"Saved training CSV to {csv_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="single-gap SAC training")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="training episode count")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed")
    parser.add_argument("--policy-out", default=POLICY_PATH, help="output policy path")
    parser.add_argument("--csv-out", default=CSV_PATH, help="output training CSV path")
    args = parser.parse_args()
    train(args.episodes, args.seed, args.policy_out, args.csv_out)


if __name__ == "__main__":
    main()
