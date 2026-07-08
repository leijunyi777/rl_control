import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from main_sac_train import Main5SacMergeEnv, SACAgent, SEED, set_seed


# =========================
# 回放参数
# =========================
POLICY_PATH = "sac_policy.pth"
RENDER = True
STEP_PAUSE = 0.02


def load_trained_agent(policy_path, device):
    """加载训练完成后保存的 SAC policy。"""
    checkpoint = torch.load(policy_path, map_location=device)
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])

    agent = SACAgent(state_dim, action_dim, device)
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy.eval()
    return agent


def run_policy_once():
    """使用训练好的 policy 进行一次带图像的完整仿真。"""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_trained_agent(POLICY_PATH, device)
    env = Main5SacMergeEnv(render=RENDER)

    state = env.reset()
    total_reward = 0.0
    last_info = {"success": False, "collided": False, "lane_progress": 0.0, "time": 0.0}

    for _ in range(int(env.sim_time / env.dt)):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        last_info = info

        params = info["params"]
        print(
            f"t={info['time']:5.2f}s | reward={reward:8.3f} | "
            f"progress={info['lane_progress']:.3f} | "
            f"k_mu={params[0]:.3f}, k={params[1]:.3f}, eps={params[2]:.4f}"
        )

        if STEP_PAUSE > 0:
            time.sleep(STEP_PAUSE)

        if done:
            break

    print(
        "\nReplay finished | "
        f"total_reward={total_reward:.2f} | "
        f"progress={last_info['lane_progress']:.3f} | "
        f"success={last_info['success']} | collision={last_info['collided']}"
    )

    if RENDER:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    run_policy_once()
