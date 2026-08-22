import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_07_main12_sac_train_u_random_ego import (
    ACTION_DIM,
    EGO_X_BASE,
    EGO_X_RANDOM_RANGE,
    POLICY_PATH,
    SEED,
    SIM_TIME,
    DT,
    U_HIGH,
    U_LOW,
    Main11SacUDynamics,
    Main12SacUEnv,
    SACAgent,
    set_seed,
)
from model_ode import EgoVehicleOdeModel, KinematicBicycleModel, front_velocity

SEED = 7825
NUM_EVAL_EPISODES = 10
RESULT_FIG_PATH = "main12_value_comparison.png"


class FixedInitialMain12SacUEnv(Main12SacUEnv):
    """Main12 evaluation environment with a fixed ego initial x."""

    def __init__(self, ego_x0, *args, **kwargs):
        self.fixed_ego_x0 = float(ego_x0)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.ego_x0 = self.fixed_ego_x0
        self.veh1 = KinematicBicycleModel(
            id="Veh 1 (Leader)",
            x=30.0,
            y=self.target_lane_y,
            v=15.0,
            L=self.vehicle_l,
            color="lightblue",
        )
        self.veh2 = KinematicBicycleModel(
            id="Veh 2 (Gap Control)",
            x=15.0,
            y=self.target_lane_y,
            v=15.0,
            L=self.vehicle_l,
            color="royalblue",
        )
        self.ego = EgoVehicleOdeModel(
            id="Veh 3 (Ego Main12 Value)",
            x=self.ego_x0,
            y=self.original_lane_y,
            v=15.0,
            L=self.vehicle_l,
            color="lightgreen",
        )
        self.dynamics = Main11SacUDynamics(
            self.veh1,
            self.veh2,
            self.ego,
            desired_gap=self.desired_gap,
            gap_safe=self.gap_safe,
        )
        self.collision_radius = self.ego.r
        self.state = self.dynamics.pack_state()
        self.t = 0.0
        self.prev_lane_progress = self._lane_progress()
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.prev_lateral_velocity = front_velocity(self.state[10:15], self.ego.L)[1]
        self.t_hist, self.z_hist, self.u_hist, self.formula_u_hist, self.bt_hist = [], [], [], [], []
        self.dist1_hist, self.dist2_hist, self.veh12_gap_hist = [], [], []
        return self._get_obs()


def load_trained_agent(policy_path, device):
    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"Cannot find {policy_path}. Run rl_07_main12_sac_train_u_random_ego.py first to generate the trained policy."
        )

    checkpoint = torch.load(policy_path, map_location=device)
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])

    agent = SACAgent(state_dim, action_dim, device)
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy.eval()
    return agent


def sample_eval_ego_x_values():
    return EGO_X_BASE + np.random.uniform(-EGO_X_RANDOM_RANGE, EGO_X_RANDOM_RANGE, size=NUM_EVAL_EPISODES)


def u_to_action(u_t):
    ratio = (float(u_t) - U_LOW) / (U_HIGH - U_LOW)
    action = 2.0 * ratio - 1.0
    return np.array([np.clip(action, -1.0, 1.0)], dtype=np.float32)


def trained_policy_action(agent, state):
    return agent.select_action(state, evaluate=True)


def formula_policy_action(env):
    diag = env.dynamics.diagnostics(env.state)
    return u_to_action(diag["formula_u_t"])


def run_episode(env, action_fn):
    state = env.reset()
    total_reward = 0.0
    last_info = {"lane_progress": 0.0, "success": False, "collided": False, "time": 0.0}

    for _ in range(int(SIM_TIME / DT)):
        action = action_fn(state, env)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        last_info = info
        if done:
            break

    return {
        "reward": float(total_reward),
        "progress": float(last_info["lane_progress"]),
        "success": bool(last_info["success"]),
        "collided": bool(last_info["collided"]),
        "time": float(last_info["time"]),
    }


def plot_value_comparison(ego_x_values, trained_results, formula_results):
    trained_rewards = np.array([item["reward"] for item in trained_results], dtype=float)
    formula_rewards = np.array([item["reward"] for item in formula_results], dtype=float)
    means = [float(np.mean(trained_rewards)), float(np.mean(formula_rewards))]
    stds = [float(np.std(trained_rewards)), float(np.std(formula_rewards))]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    labels = ["Trained SAC policy", "Original RBF u(t)"]
    x_pos = np.arange(len(labels))
    axes[0].bar(x_pos, means, yerr=stds, color=["tab:blue", "tab:orange"], alpha=0.8, capsize=8)
    axes[0].scatter(np.full_like(trained_rewards, x_pos[0], dtype=float), trained_rewards, color="navy", s=35, alpha=0.75)
    axes[0].scatter(np.full_like(formula_rewards, x_pos[1], dtype=float), formula_rewards, color="darkorange", s=35, alpha=0.75)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Main12 Value Comparison: 10 Random ego x0 Trials")
    axes[0].grid(True, axis="y")

    trial_index = np.arange(1, NUM_EVAL_EPISODES + 1)
    axes[1].plot(trial_index, trained_rewards, "o-", color="tab:blue", linewidth=2, label="Trained SAC policy")
    axes[1].plot(trial_index, formula_rewards, "o-", color="tab:orange", linewidth=2, label="Original RBF u(t)")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Episode reward")
    axes[1].set_title("Per-trial Rewards with Shared Random Initial Positions")
    axes[1].legend()
    axes[1].grid(True)

    for idx, ego_x0 in enumerate(ego_x_values, start=1):
        axes[1].annotate(f"{ego_x0:.1f}", (idx, min(trained_rewards[idx - 1], formula_rewards[idx - 1])), fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULT_FIG_PATH, dpi=180)
    plt.show()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_trained_agent(POLICY_PATH, device)
    ego_x_values = sample_eval_ego_x_values()

    trained_results = []
    formula_results = []

    for episode_index, ego_x0 in enumerate(ego_x_values, start=1):
        trained_env = FixedInitialMain12SacUEnv(ego_x0, render=False)
        formula_env = FixedInitialMain12SacUEnv(ego_x0, render=False)

        trained_result = run_episode(
            trained_env,
            lambda state, env: trained_policy_action(agent, state),
        )
        formula_result = run_episode(
            formula_env,
            lambda state, env: formula_policy_action(env),
        )
        trained_results.append(trained_result)
        formula_results.append(formula_result)

        print(
            f"Trial {episode_index:02d} | ego_x0={ego_x0:.3f} | "
            f"trained_reward={trained_result['reward']:8.2f}, progress={trained_result['progress']:.3f}, "
            f"success={trained_result['success']}, collision={trained_result['collided']} | "
            f"formula_reward={formula_result['reward']:8.2f}, progress={formula_result['progress']:.3f}, "
            f"success={formula_result['success']}, collision={formula_result['collided']}"
        )

    trained_mean = float(np.mean([item["reward"] for item in trained_results]))
    formula_mean = float(np.mean([item["reward"] for item in formula_results]))
    print("\nMain12 value comparison finished")
    print(f"Trained SAC policy mean reward: {trained_mean:.2f}")
    print(f"Original RBF u(t) mean reward: {formula_mean:.2f}")
    print(f"Saved comparison figure to {RESULT_FIG_PATH}")

    plot_value_comparison(ego_x_values, trained_results, formula_results)


if __name__ == "__main__":
    main()
