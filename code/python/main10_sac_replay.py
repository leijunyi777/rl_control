import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from main10_sac_train import (
    POLICY_PATH,
    SACAgent,
    SEED,
    Main10SacMergeEnv,
    set_seed,
)
from models_ode import KinematicBicycleModel
from utils import draw_car, draw_environment


EXPORT_ANIMATION = True
EXPORT_PATH = "main10_sac_replay.gif"
EXPORT_FPS = 20
EXPORT_FRAME_STRIDE = 2
SHOW_AFTER_EXPORT = True
PRINT_EVERY_STEP = True


def load_trained_agent(policy_path, device):
    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"Cannot find {policy_path}. Run main10_sac_train.py first to generate the trained policy."
        )

    checkpoint = torch.load(policy_path, map_location=device)
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])

    agent = SACAgent(state_dim, action_dim, device)
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy.eval()
    return agent


def make_car_from_state(car_id, state, color, wheelbase):
    car = KinematicBicycleModel(id=car_id, L=wheelbase, color=color)
    car.set_state(state)
    return car


def snapshot(env, info=None, reward=0.0, total_reward=0.0):
    dist1, dist2 = env._distances()
    veh12_gap = env._veh12_gap()
    initial_info = {
        "b_t": 0.0,
        "gap_formula_b_t": 0.0,
        "u_t": 0.0,
        "success": False,
        "collided": False,
        "lane_progress": env._lane_progress(),
    }
    data = initial_info if info is None else info

    return {
        "time": float(env.t),
        "veh1_state": env.state[0:5].copy(),
        "veh2_state": env.state[5:10].copy(),
        "ego_state": env.state[10:15].copy(),
        "z_new": float(env.state[17]),
        "b_t": float(data["b_t"]),
        "gap_formula_b_t": float(data["gap_formula_b_t"]),
        "u_t": float(data["u_t"]),
        "dist1": float(dist1),
        "dist2": float(dist2),
        "veh12_gap": float(veh12_gap),
        "reward": float(reward),
        "total_reward": float(total_reward),
        "success": bool(data["success"]),
        "collided": bool(data["collided"]),
        "progress": float(data["lane_progress"]),
    }


def collect_replay_frames(agent, env):
    state = env.reset()
    frames = [snapshot(env)]
    total_reward = 0.0
    last_info = {"success": False, "collided": False, "lane_progress": 0.0, "time": 0.0, "b_t": 0.0}

    for step_index in range(int(env.sim_time / env.dt)):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        last_info = info

        if PRINT_EVERY_STEP:
            print(
                f"t={info['time']:5.2f}s | reward={reward:8.3f} | "
                f"progress={info['lane_progress']:.3f} | "
                f"b(t)={info['b_t']:+.3f} | z={info['z_new']:+.3f} | u(t)={info['u_t']:.3f}"
            )

        if step_index % EXPORT_FRAME_STRIDE == 0 or done:
            frames.append(snapshot(env, info=info, reward=reward, total_reward=total_reward))

        if done:
            break

    print(
        "\nReplay finished | "
        f"total_reward={total_reward:.2f} | "
        f"progress={last_info['lane_progress']:.3f} | "
        f"b(t)={last_info['b_t']:+.3f} | "
        f"success={last_info['success']} | collision={last_info['collided']}"
    )
    return frames


def export_replay_animation(frames, env, export_path):
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1)
    ax_z = plt.subplot(2, 2, 3)
    ax_dist = plt.subplot(2, 2, 4)

    times = np.array([frame["time"] for frame in frames])
    z_hist = np.array([frame["z_new"] for frame in frames])
    bt_hist = np.array([frame["b_t"] for frame in frames])
    gap_bt_hist = np.array([frame["gap_formula_b_t"] for frame in frames])
    ut_hist = np.array([frame["u_t"] for frame in frames])
    dist1_hist = np.array([frame["dist1"] for frame in frames])
    dist2_hist = np.array([frame["dist2"] for frame in frames])
    veh12_gap_hist = np.array([frame["veh12_gap"] for frame in frames])

    def draw_frame(index):
        frame = frames[index]
        veh1 = make_car_from_state("Veh 1 (Leader)", frame["veh1_state"], "lightblue", env.vehicle_l)
        veh2 = make_car_from_state("Veh 2 (Gap Control)", frame["veh2_state"], "royalblue", env.vehicle_l)
        ego = make_car_from_state("Veh 3 (Ego Main10 SAC)", frame["ego_state"], "lightgreen", env.vehicle_l)

        ax_anim.cla()
        draw_environment(ax_anim, env.lane_width)
        draw_car(ax_anim, veh1, wheelbase=env.collision_radius)
        draw_car(ax_anim, veh2, wheelbase=env.collision_radius)
        draw_car(ax_anim, ego, wheelbase=env.collision_radius)
        ax_anim.set_xlim(ego.x - 15, ego.x + 45)
        ax_anim.set_ylim(-2, env.lane_width * 2 + 2)
        ax_anim.set_aspect("equal")
        title = f"Time: {frame['time']:.2f}s | Main10 SAC policy replay"
        if frame["success"]:
            title += " | SUCCESS"
        if frame["collided"]:
            title += " | COLLISION"
        ax_anim.set_title(title)

        ax_z.cla()
        ax_z.plot(times[: index + 1], z_hist[: index + 1], "m-", linewidth=2.5, label="New Formula z")
        ax_z.plot(times[: index + 1], bt_hist[: index + 1], "g--", linewidth=1.8, label="Policy b(t)")
        ax_z.plot(times[: index + 1], gap_bt_hist[: index + 1], "gray", linestyle="--", linewidth=1.5, label="Gap Formula b(t)")
        ax_z.plot(times[: index + 1], ut_hist[: index + 1], "purple", linestyle=":", linewidth=1.8, label="u(t)")
        ax_z.axhline(0, color="gray", linestyle="--")
        ax_z.axvline(20.0, color="black", linestyle=":", linewidth=1.5, label="Gap Control Starts")
        ax_z.set_xlim(0, env.sim_time)
        ax_z.set_title("Policy Bias and New Opinion Dynamics")
        ax_z.legend(loc="upper left")
        ax_z.grid(True)

        ax_dist.cla()
        ax_dist.plot(times[: index + 1], dist1_hist[: index + 1], "purple", linewidth=2, label="Distance to Veh 1")
        ax_dist.plot(times[: index + 1], dist2_hist[: index + 1], "red", linewidth=2, label="Distance to Veh 2")
        ax_dist.plot(times[: index + 1], veh12_gap_hist[: index + 1], "gray", linestyle="-.", linewidth=2, label="Veh1-Veh2 Gap")
        ax_dist.axhline(env.collision_radius, color="black", linestyle="--", linewidth=2, label=f"Collision Threshold r={env.collision_radius:g}m")
        ax_dist.axhline(env.gap_safe, color="orange", linestyle="--", linewidth=1.8, label="Safe Gap 15m")
        ax_dist.axhline(env.desired_gap, color="green", linestyle=":", linewidth=2, label="Target Gap 20m")
        ax_dist.axvline(20.0, color="black", linestyle=":", linewidth=1.5)
        ax_dist.set_xlim(0, env.sim_time)
        upper_distance = max(
            env.collision_radius * 2.0,
            env.desired_gap * 1.2,
            float(np.max(dist1_hist)),
            float(np.max(dist2_hist)),
            float(np.max(veh12_gap_hist)),
        )
        ax_dist.set_ylim(0, upper_distance * 1.1)
        ax_dist.set_title("Relative Distance Monitoring")
        ax_dist.legend(loc="upper right")
        ax_dist.grid(True)

    replay_animation = animation.FuncAnimation(fig, draw_frame, frames=len(frames), interval=1000 / EXPORT_FPS)
    extension = os.path.splitext(export_path)[1].lower()
    writer = animation.PillowWriter(fps=EXPORT_FPS) if extension == ".gif" else animation.FFMpegWriter(fps=EXPORT_FPS)
    replay_animation.save(export_path, writer=writer)
    print(f"Saved replay animation to {export_path}")

    if SHOW_AFTER_EXPORT:
        plt.show()
    else:
        plt.close(fig)


def run_policy_once():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_trained_agent(POLICY_PATH, device)
    env = Main10SacMergeEnv(render=False)

    frames = collect_replay_frames(agent, env)
    if EXPORT_ANIMATION:
        export_replay_animation(frames, env, EXPORT_PATH)
    else:
        plt.figure(figsize=(10, 5))
        plt.plot([frame["time"] for frame in frames], [frame["z_new"] for frame in frames], label="New Formula z")
        plt.plot([frame["time"] for frame in frames], [frame["b_t"] for frame in frames], label="Policy b(t)")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    run_policy_once()
