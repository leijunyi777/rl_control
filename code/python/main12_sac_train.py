import numpy as np
import matplotlib.pyplot as plt
import torch

from main11_sac_train import (
    ACTION_DIM,
    BATCH_SIZE,
    DT,
    INITIAL_RANDOM_STEPS,
    Main11SacUDynamics,
    Main11SacUEnv,
    NUM_EPISODES,
    RENDER_DURING_TRAINING,
    REPLAY_SIZE,
    SEED,
    SIM_TIME,
    U_HIGH,
    U_LOW,
    UPDATES_PER_STEP,
    ReplayBuffer,
    SACAgent,
    moving_average,
    set_seed,
)
from models_ode import EgoVehicleOdeModel, KinematicBicycleModel, front_velocity


EGO_X_BASE = 20.0
EGO_X_RANDOM_RANGE = 5.0
POLICY_PATH = "main12_sac_u_policy.pth"
RESULT_FIG_PATH = "main12_sac_u_training_result.png"


def sample_ego_x():
    return float(EGO_X_BASE + np.random.uniform(-EGO_X_RANDOM_RANGE, EGO_X_RANDOM_RANGE))


class Main12SacUEnv(Main11SacUEnv):
    """Main11 SAC-u environment with randomized ego initial x position."""

    def reset(self):
        ego_x0 = sample_ego_x()
        self.ego_x0 = ego_x0
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
            id="Veh 3 (Ego Main12 SAC-u)",
            x=ego_x0,
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


def plot_training_results(episode_rewards, episode_progress, episode_collisions, episode_successes, episode_mean_u, episode_ego_x0):
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


def train():
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

    for episode in range(1, NUM_EPISODES + 1):
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

        print(
            f"Episode {episode:04d} | reward={episode_reward:8.2f} | "
            f"progress={last_info['lane_progress']:.3f} | "
            f"mean_u={episode_mean_u[-1]:.3f} | ego_x0={env.ego_x0:.3f} | "
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
        POLICY_PATH,
    )
    print(f"Saved trained policy to {POLICY_PATH}")
    print(f"Saved training result figure to {RESULT_FIG_PATH}")


if __name__ == "__main__":
    train()
