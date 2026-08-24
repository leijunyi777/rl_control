"""多 gap 高层决策消融：意见动力学 vs 简单 max()。

与 multi_gap_env.py 环境一致，可设置 seed 和 runs。
同一轮中 opinion 与 max 使用相同 gap_seed 和 ego_seed，保证交通场景完全一致。
"""

from __future__ import annotations

import argparse
import csv
import random
from typing import List, Optional

from multi_gap_env import run_episode


# =========================
# 开头可调评价参数
# =========================
EVAL_RUNS = 100
EVAL_SEED = 7
DEFAULT_POLICY_PATH = "main12_sac_u_policy.pth"
DEFAULT_CSV_OUT = "multi_gap_opinion_vs_max.csv"


class LoadedPolicy:
    """加载单 gap SAC policy；不提供 policy 时脚本使用环境内置 RBF u(t)。"""

    def __init__(self, policy_path: str):
        import torch
        import torch.nn as nn

        class GaussianPolicy(nn.Module):
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
                return self.mean(h), self.log_std(h)

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(policy_path, map_location=self.device)
        self.u_low = float(checkpoint.get("u_low", 0.0))
        self.u_high = float(checkpoint.get("u_high", 3.0))
        self.policy = GaussianPolicy(
            int(checkpoint["state_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint.get("hidden_size", 256)),
        ).to(self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.eval()

    def __call__(self, obs: List[float], info) -> float:
        with self.torch.no_grad():
            state = self.torch.tensor([obs], dtype=self.torch.float32, device=self.device)
            mean, _ = self.policy(state)
            action = self.torch.tanh(mean).cpu()[0, 0].item()
        return self.u_low + 0.5 * (float(action) + 1.0) * (self.u_high - self.u_low)


def compare(runs: int, seed: Optional[int], policy_path: Optional[str], csv_out: str) -> None:
    """执行 opinion 高层与 max 高层的配对对比。"""
    rng = random.Random(seed)
    policy = LoadedPolicy(policy_path) if policy_path else None
    rows = []

    for run in range(1, runs + 1):
        gap_seed = rng.randint(0, 2**31 - 1) if seed is not None else None
        ego_seed = rng.randint(0, 2**31 - 1) if seed is not None else None

        opinion = run_episode(
            gap_seed=gap_seed,
            ego_seed=ego_seed,
            decision_method="opinion",
            low_level_policy=policy,
        )
        max_result = run_episode(
            gap_seed=gap_seed,
            ego_seed=ego_seed,
            decision_method="max",
            low_level_policy=policy,
        )

        row = {
            "run": run,
            "gap_seed": gap_seed if gap_seed is not None else -1,
            "ego_seed": ego_seed if ego_seed is not None else -1,
            "low_level": "SAC" if policy else "RBF",
            "opinion_reward": opinion["reward"],
            "max_reward": max_result["reward"],
            "reward_diff": opinion["reward"] - max_result["reward"],
            "opinion_progress": opinion["progress"],
            "max_progress": max_result["progress"],
            "opinion_success": opinion["success"],
            "max_success": max_result["success"],
            "opinion_collision": opinion["collision"],
            "max_collision": max_result["collision"],
            "opinion_time": opinion["time"],
            "max_time": max_result["time"],
            "opinion_min_distance": opinion["min_distance"],
            "max_min_distance": max_result["min_distance"],
            "opinion_switch_count": opinion["switch_count"],
            "max_switch_count": max_result["switch_count"],
        }
        rows.append(row)
        print(
            f"Run {run:03d} | opinion={row['opinion_reward']:9.3f} | "
            f"max={row['max_reward']:9.3f} | diff={row['reward_diff']:9.3f} | "
            f"switches=({row['opinion_switch_count']}, {row['max_switch_count']})"
        )

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    opinion_mean = sum(row["opinion_reward"] for row in rows) / len(rows)
    max_mean = sum(row["max_reward"] for row in rows) / len(rows)
    print(f"Opinion mean reward: {opinion_mean:.3f}")
    print(f"Max mean reward: {max_mean:.3f}")
    print(f"Mean reward diff: {opinion_mean - max_mean:.3f}")
    print(f"CSV 已保存：{csv_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="多 gap opinion vs max 高层决策对比")
    parser.add_argument("--runs", type=int, default=EVAL_RUNS, help="评价轮次")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="随机种子")
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="可选：单 gap SAC policy 路径")
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT, help="输出 CSV")
    args = parser.parse_args()
    compare(args.runs, args.seed, args.policy, args.csv_out)


if __name__ == "__main__":
    main()
