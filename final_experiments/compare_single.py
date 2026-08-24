"""单 gap SAC policy 与手工 RBF u(t) 对比。

只依赖 single_gap_env.py；如果需要读取 SAC policy，则额外需要 torch。
输出 CSV 包含每次随机初始位置下两种方法的 reward、progress、collision、success 等结果。
"""

from __future__ import annotations

import argparse
import csv
import random
from typing import List

import torch
import torch.nn as nn

from single_gap_env import U_HIGH, U_LOW, SingleGapEnv, compute_rbf_u, run_episode


HIDDEN_SIZE = 256
ACTION_DIM = 1


class GaussianPolicy(nn.Module):
    """与训练脚本一致的策略网络结构，仅用于加载和推理。"""

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


class LoadedPolicy:
    """加载训练好的 policy，并输出真实 u(t)。"""

    def __init__(self, policy_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(policy_path, map_location=self.device)
        self.u_low = float(checkpoint.get("u_low", U_LOW))
        self.u_high = float(checkpoint.get("u_high", U_HIGH))
        self.policy = GaussianPolicy(
            int(checkpoint["state_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint.get("hidden_size", HIDDEN_SIZE)),
        ).to(self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.eval()

    def __call__(self, obs: List[float], info) -> float:
        with torch.no_grad():
            state = torch.tensor([obs], dtype=torch.float32, device=self.device)
            mean, _ = self.policy(state)
            action = torch.tanh(mean).cpu()[0, 0].item()
        return self.u_low + 0.5 * (float(action) + 1.0) * (self.u_high - self.u_low)


def run_with_ego_x(policy, ego_x0: float) -> dict:
    """在指定 ego 初始位置下运行一次仿真。"""
    env = SingleGapEnv(seed=0)
    env.reset(ego_x0=ego_x0)
    total_reward = 0.0
    obs = env.observation()
    info = {"formula_u_t": compute_rbf_u(env.ego, env.front, env.rear)["u_t"]}
    done = False
    steps = 0
    while not done:
        u_t = policy(obs, info)
        obs, reward, done, info = env.step(u_t)
        total_reward += reward
        steps += 1
    return {
        "reward": total_reward,
        "progress": info["lane_progress"],
        "success": float(info["success"]),
        "collision": float(info["collided"]),
        "time": info["time"],
        "steps": steps,
        "min_distance": info["min_distance"],
    }


def compare(policy_path: str, runs: int, seed: int, csv_out: str) -> None:
    """执行多次随机初始位置下的 policy vs RBF 对比。"""
    rng = random.Random(seed)
    policy = LoadedPolicy(policy_path)
    rows = []
    for run in range(1, runs + 1):
        ego_x0 = 20.0 + rng.uniform(-5.0, 5.0)
        sac = run_with_ego_x(policy, ego_x0)
        rbf = run_with_ego_x(lambda obs, info: info["formula_u_t"], ego_x0)
        row = {
            "run": run,
            "ego_x0": ego_x0,
            "sac_reward": sac["reward"],
            "rbf_reward": rbf["reward"],
            "reward_diff": sac["reward"] - rbf["reward"],
            "sac_progress": sac["progress"],
            "rbf_progress": rbf["progress"],
            "sac_success": sac["success"],
            "rbf_success": rbf["success"],
            "sac_collision": sac["collision"],
            "rbf_collision": rbf["collision"],
            "sac_time": sac["time"],
            "rbf_time": rbf["time"],
            "sac_min_distance": sac["min_distance"],
            "rbf_min_distance": rbf["min_distance"],
        }
        rows.append(row)
        print(
            f"Run {run:03d} | ego_x0={ego_x0:7.3f} | "
            f"SAC={row['sac_reward']:9.3f} | RBF={row['rbf_reward']:9.3f} | "
            f"diff={row['reward_diff']:9.3f}"
        )

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sac_mean = sum(row["sac_reward"] for row in rows) / len(rows)
    rbf_mean = sum(row["rbf_reward"] for row in rows) / len(rows)
    print(f"SAC mean reward: {sac_mean:.3f}")
    print(f"RBF mean reward: {rbf_mean:.3f}")
    print(f"CSV 已保存：{csv_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="单 gap SAC 与 RBF 对比")
    parser.add_argument("--policy", default="single_gap_sac_policy.pth", help="训练好的 policy 路径")
    parser.add_argument("--runs", type=int, default=10, help="重复评价次数")
    parser.add_argument("--seed", type=int, default=7825, help="随机种子")
    parser.add_argument("--csv-out", default="single_gap_compare.csv", help="输出 CSV")
    args = parser.parse_args()
    compare(args.policy, args.runs, args.seed, args.csv_out)


if __name__ == "__main__":
    main()
