"""多轮多 gap 随机评价脚本。

基于 multi_gap_env.py，可设置 runs 和 seed。
默认使用手工 RBF u(t)；若提供 --policy，则使用单 gap SAC 训练出的 policy 作为底层 u(t)。
"""

from __future__ import annotations

import argparse
import csv
import random
from typing import List, Optional

from multi_gap_env import run_episode


class LoadedPolicy:
    """延迟加载 torch policy，避免没有 policy 时引入额外依赖。"""

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


def evaluate(runs: int, seed: Optional[int], policy_path: Optional[str], csv_out: str) -> None:
    """执行多轮随机评价。"""
    rng = random.Random(seed)
    policy = LoadedPolicy(policy_path) if policy_path else None
    rows = []
    for run in range(1, runs + 1):
        gap_seed = rng.randint(0, 2**31 - 1) if seed is not None else None
        ego_seed = rng.randint(0, 2**31 - 1) if seed is not None else None
        result = run_episode(
            gap_seed=gap_seed,
            ego_seed=ego_seed,
            decision_method="opinion",
            low_level_policy=policy,
        )
        row = {"run": run, "low_level": "SAC" if policy else "RBF", **result}
        rows.append(row)
        print(
            f"Run {run:03d} | reward={result['reward']:9.3f} | "
            f"progress={result['progress']:.3f} | success={bool(result['success'])} | "
            f"collision={bool(result['collision'])} | switches={result['switch_count']}"
        )

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_reward = sum(row["reward"] for row in rows) / len(rows)
    success_rate = sum(row["success"] for row in rows) / len(rows)
    collision_rate = sum(row["collision"] for row in rows) / len(rows)
    print(f"平均 reward: {mean_reward:.3f}")
    print(f"成功率: {success_rate:.2%}")
    print(f"碰撞率: {collision_rate:.2%}")
    print(f"CSV 已保存：{csv_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="多 gap 多轮随机评价")
    parser.add_argument("--runs", type=int, default=100, help="评价轮次")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--policy", default=None, help="可选：单 gap SAC policy 路径")
    parser.add_argument("--csv-out", default="multi_gap_eval.csv", help="输出 CSV")
    args = parser.parse_args()
    evaluate(args.runs, args.seed, args.policy, args.csv_out)


if __name__ == "__main__":
    main()
