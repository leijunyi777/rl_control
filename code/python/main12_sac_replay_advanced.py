import os

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import torch

from main12_sac_train import (
    ACTION_DIM,
    EGO_X_BASE,
    EGO_X_RANDOM_RANGE,
    POLICY_PATH,
    SEED,
    Main11SacUDynamics,
    Main12SacUEnv,
    SACAgent,
    set_seed,
)
from models_ode import EgoVehicleOdeModel, KinematicBicycleModel, front_velocity
from utils import draw_car, draw_environment


NUM_GHOST_EGOS = 5
EXPORT_ANIMATION = True
EXPORT_PATH = "main12_sac_u_multi_ego_replay.gif"
EXPORT_FPS = 20
EXPORT_FRAME_STRIDE = 2
SHOW_AFTER_EXPORT = True
PRINT_EVERY_STEP = True

GHOST_COLORS = ["#2ca02c", "#17becf", "#9467bd", "#ff7f0e", "#d62728"]


class FixedInitialMain12SacUEnv(Main12SacUEnv):
    """Main12 replay environment with a fixed sampled ego initial x."""

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
            id="Veh 3 (Ego Main12 SAC-u)",
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
            f"Cannot find {policy_path}. Run main12_sac_train.py first to generate the trained policy."
        )

    checkpoint = torch.load(policy_path, map_location=device)
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])

    agent = SACAgent(state_dim, action_dim, device)
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy.eval()
    return agent


def sample_ego_x_values(count):
    return EGO_X_BASE + np.random.uniform(-EGO_X_RANDOM_RANGE, EGO_X_RANDOM_RANGE, size=count)


def make_car_from_state(car_id, state, color, wheelbase):
    car = KinematicBicycleModel(id=car_id, L=wheelbase, color=color)
    car.set_state(state)
    return car


def draw_ghost_car(ax, state, label, color, wheelbase, alpha=0.38):
    x, y, theta, _, _ = state
    visual_length = float(wheelbase)
    visual_width = 0.6 * visual_length
    corner_radius = 0.18 * visual_length

    rear = np.array([x, y])
    heading = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])
    center = rear + 0.5 * visual_length * heading
    lower_left = center - 0.5 * visual_length * heading - 0.5 * visual_width * normal

    car_patch = patches.FancyBboxPatch(
        (lower_left[0], lower_left[1]),
        visual_length,
        visual_width,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        facecolor=color,
        edgecolor="black",
        linewidth=1.2,
        alpha=alpha,
    )
    car_patch.set_transform(
        transforms.Affine2D().rotate_around(lower_left[0], lower_left[1], theta) + ax.transData
    )
    ax.add_patch(car_patch)

    front = rear + visual_length * heading
    ax.plot([rear[0], front[0]], [rear[1], front[1]], color="black", linewidth=0.8, alpha=0.5)
    ax.text(x, y + 0.9, label, fontsize=8, color=color, fontweight="bold")


def snapshot(time_value, target_env, envs, last_infos, total_rewards, success_flags, collision_flags):
    egos = []
    for index, env in enumerate(envs):
        info = last_infos[index]
        egos.append({
            "state": env.state[10:15].copy(),
            "z_new": float(env.state[17]),
            "u_t": float(info.get("u_t", 0.0)),
            "b_t": float(info.get("b_t", 0.0)),
            "progress": float(info.get("lane_progress", env._lane_progress())),
            "reward": float(total_rewards[index]),
            "success": bool(success_flags[index]),
            "collided": bool(collision_flags[index]),
            "ego_x0": float(env.ego_x0),
        })

    return {
        "time": float(time_value),
        "veh1_state": target_env.state[0:5].copy(),
        "veh2_state": target_env.state[5:10].copy(),
        "egos": egos,
    }


def collect_replay_frames(agent, envs):
    states = [env.reset() for env in envs]
    print("Main12 advanced replay ego initial x values:")
    for index, env in enumerate(envs, start=1):
        print(f"  Ego {index}: x0 = {env.ego_x0:.3f} m")

    last_infos = [
        {"u_t": 0.0, "b_t": 0.0, "lane_progress": env._lane_progress(), "success": False, "collided": False}
        for env in envs
    ]
    total_rewards = np.zeros(len(envs), dtype=float)
    done_flags = np.zeros(len(envs), dtype=bool)
    success_flags = np.zeros(len(envs), dtype=bool)
    collision_flags = np.zeros(len(envs), dtype=bool)
    frames = [snapshot(0.0, envs[0], envs, last_infos, total_rewards, success_flags, collision_flags)]
    target_source = envs[0]

    for step_index in range(int(envs[0].sim_time / envs[0].dt)):
        if np.all(success_flags):
            break

        current_time = 0.0
        for ego_index, env in enumerate(envs):
            if done_flags[ego_index]:
                current_time = max(current_time, env.t)
                continue

            action = agent.select_action(states[ego_index], evaluate=True)
            next_state, reward, done, info = env.step(action)
            states[ego_index] = next_state
            total_rewards[ego_index] += reward
            last_infos[ego_index] = info
            success_flags[ego_index] = bool(info["success"])
            collision_flags[ego_index] = bool(info["collided"])
            done_flags[ego_index] = bool(done)
            current_time = max(current_time, info["time"])
            target_source = env

        if PRINT_EVERY_STEP:
            progress_text = ", ".join(
                f"E{idx + 1}:p={last_infos[idx].get('lane_progress', 0.0):.2f},u={last_infos[idx].get('u_t', 0.0):.2f}"
                for idx in range(len(envs))
            )
            print(
                f"t={current_time:5.2f}s | success={int(np.sum(success_flags))}/{len(envs)} | "
                f"collision={int(np.sum(collision_flags))}/{len(envs)} | {progress_text}"
            )

        if step_index % EXPORT_FRAME_STRIDE == 0 or np.all(success_flags) or np.all(done_flags):
            frames.append(
                snapshot(current_time, target_source, envs, last_infos, total_rewards, success_flags, collision_flags)
            )

        if np.all(done_flags) and not np.all(success_flags):
            break

    print("\nAdvanced replay finished")
    for index, env in enumerate(envs):
        status = "SUCCESS" if success_flags[index] else "COLLISION" if collision_flags[index] else "TIMEOUT"
        print(
            f"Ego {index + 1}: {status} | x0={env.ego_x0:.3f} | "
            f"progress={last_infos[index].get('lane_progress', 0.0):.3f} | reward={total_rewards[index]:.2f}"
        )
    return frames


def export_multi_replay_animation(frames, env_template, export_path):
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1)
    ax_progress = plt.subplot(2, 2, 3)
    ax_u = plt.subplot(2, 2, 4)

    times = np.array([frame["time"] for frame in frames])
    progress_hist = np.array([[ego["progress"] for ego in frame["egos"]] for frame in frames])
    u_hist = np.array([[ego["u_t"] for ego in frame["egos"]] for frame in frames])

    def draw_frame(index):
        frame = frames[index]
        veh1 = make_car_from_state("Veh 1 (Leader)", frame["veh1_state"], "lightblue", env_template.vehicle_l)
        veh2 = make_car_from_state("Veh 2 (Gap Control)", frame["veh2_state"], "royalblue", env_template.vehicle_l)
        ego_states = [ego["state"] for ego in frame["egos"]]
        ego_x_center = float(np.mean([state[0] for state in ego_states]))
        success_count = sum(ego["success"] for ego in frame["egos"])
        collision_count = sum(ego["collided"] for ego in frame["egos"])

        ax_anim.cla()
        draw_environment(ax_anim, env_template.lane_width)
        draw_car(ax_anim, veh1, wheelbase=env_template.collision_radius)
        draw_car(ax_anim, veh2, wheelbase=env_template.collision_radius)
        for ego_index, ego in enumerate(frame["egos"]):
            label = f"E{ego_index + 1}"
            color = GHOST_COLORS[ego_index % len(GHOST_COLORS)]
            alpha = 0.68 if ego["success"] else 0.28 if ego["collided"] else 0.42
            draw_ghost_car(ax_anim, ego["state"], label, color, env_template.collision_radius, alpha=alpha)

        ax_anim.set_xlim(ego_x_center - 18, ego_x_center + 48)
        ax_anim.set_ylim(-2, env_template.lane_width * 2 + 2)
        ax_anim.set_aspect("equal")
        ax_anim.set_title(
            f"Time: {frame['time']:.2f}s | Main12 advanced replay | "
            f"success={success_count}/{NUM_GHOST_EGOS}, collision={collision_count}/{NUM_GHOST_EGOS}"
        )

        ax_progress.cla()
        for ego_index in range(NUM_GHOST_EGOS):
            ax_progress.plot(
                times[: index + 1],
                progress_hist[: index + 1, ego_index],
                color=GHOST_COLORS[ego_index % len(GHOST_COLORS)],
                linewidth=2,
                label=f"Ego {ego_index + 1}",
            )
        ax_progress.axhline(0.95, color="black", linestyle="--", linewidth=1.5, label="Success Progress")
        ax_progress.set_xlim(0, env_template.sim_time)
        ax_progress.set_ylim(-0.05, 1.05)
        ax_progress.set_title("Lane-change Progress")
        ax_progress.legend(loc="lower right")
        ax_progress.grid(True)

        ax_u.cla()
        for ego_index in range(NUM_GHOST_EGOS):
            ax_u.plot(
                times[: index + 1],
                u_hist[: index + 1, ego_index],
                color=GHOST_COLORS[ego_index % len(GHOST_COLORS)],
                linewidth=2,
                label=f"Ego {ego_index + 1}",
            )
        ax_u.set_xlim(0, env_template.sim_time)
        ax_u.set_ylim(0, max(3.1, float(np.max(u_hist)) + 0.2))
        ax_u.set_title("Policy u(t)")
        ax_u.legend(loc="upper right")
        ax_u.grid(True)

    replay_animation = animation.FuncAnimation(fig, draw_frame, frames=len(frames), interval=1000 / EXPORT_FPS)
    extension = os.path.splitext(export_path)[1].lower()
    writer = animation.PillowWriter(fps=EXPORT_FPS) if extension == ".gif" else animation.FFMpegWriter(fps=EXPORT_FPS)
    replay_animation.save(export_path, writer=writer)
    print(f"Saved advanced replay animation to {export_path}")

    if SHOW_AFTER_EXPORT:
        plt.show()
    else:
        plt.close(fig)


def run_advanced_replay():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_trained_agent(POLICY_PATH, device)

    ego_x_values = sample_ego_x_values(NUM_GHOST_EGOS)
    envs = [FixedInitialMain12SacUEnv(ego_x0, render=False) for ego_x0 in ego_x_values]
    frames = collect_replay_frames(agent, envs)

    if EXPORT_ANIMATION:
        export_multi_replay_animation(frames, envs[0], EXPORT_PATH)


if __name__ == "__main__":
    run_advanced_replay()
