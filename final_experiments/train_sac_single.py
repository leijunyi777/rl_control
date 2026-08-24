"""单 gap SAC 训练脚本。

训练流程按 ``code/python/main12_sac_train.py`` 重构：
1. 使用原始 ``Main12SacUEnv``，不使用兼容回放类。
2. 使用原始 main11 SAC 网络、ReplayBuffer、随机探索步数和学习率。
3. reward 原值直接进入 replay buffer，不做缩放、裁剪、课程学习或训练中评价。
4. 默认 policy 文件名与原始 main12 保持一致。
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from single_gap_env import (
    ACTION_DIM,
    BATCH_SIZE,
    DT,
    EGO_X_BASE,
    EGO_X_RANDOM_RANGE,
    INITIAL_RANDOM_STEPS,
    Main12SacUEnv,
    NUM_EPISODES,
    ORIGINAL_MAIN12_POLICY_PATH,
    ORIGINAL_MAIN12_RESULT_FIG_PATH,
    RENDER_DURING_TRAINING,
    REPLAY_SIZE,
    SACAgent,
    SEED,
    SIM_TIME,
    U_HIGH,
    U_LOW,
    UPDATES_PER_STEP,
    ReplayBuffer,
    moving_average,
    plot_training_results,
    set_seed,
)


POLICY_PATH = ORIGINAL_MAIN12_POLICY_PATH
RESULT_FIG_PATH = ORIGINAL_MAIN12_RESULT_FIG_PATH
CSV_PATH = "single_gap_sac_train.csv"


def write_training_csv(rows, csv_out: str) -> None:
    """保存每轮训练结果；该函数不影响训练中的随机数和 SAC 更新。"""
    if not csv_out or not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(episodes: int = NUM_EPISODES, policy_out: str = POLICY_PATH, csv_out: str = CSV_PATH, save_csv: bool = True) -> None:
    """执行与原始 main12_sac_train.py 一致的 SAC 训练。"""
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
                    agent.update(replay_buffer)

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_progress.append(last_info["lane_progress"])
        episode_collisions.append(float(last_info["collided"]))
        episode_successes.append(float(last_info["success"]))
        episode_mean_u.append(float(np.mean(u_values)) if u_values else 0.0)
        episode_ego_x0.append(float(env.ego_x0))

        rows.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "progress": last_info["lane_progress"],
                "collision": float(last_info["collided"]),
                "success": float(last_info["success"]),
                "mean_u": episode_mean_u[-1],
                "ego_x0": float(env.ego_x0),
                "total_steps": total_steps,
            }
        )

        print(
            f"Episode {episode:04d} | reward={episode_reward:8.2f} | "
            f"progress={last_info['lane_progress']:.3f} | "
            f"mean_u={episode_mean_u[-1]:.3f} | ego_x0={env.ego_x0:.3f} | "
            f"success={last_info['success']} | collision={last_info['collided']} | steps={total_steps}"
        )

    plot_training_results(
        episode_rewards,
        episode_progress,
        episode_collisions,
        episode_successes,
        episode_mean_u,
        episode_ego_x0,
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="main12-aligned single-gap SAC training")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="training episode count")
    parser.add_argument("--policy-out", default=POLICY_PATH, help="output policy path")
    parser.add_argument("--csv-out", default=CSV_PATH, help="output CSV path")
    parser.add_argument("--no-csv", action="store_true", help="disable CSV output")
    args = parser.parse_args()
    train(args.episodes, args.policy_out, args.csv_out, save_csv=not args.no_csv)


if __name__ == "__main__":
    main()
