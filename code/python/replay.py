import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from main_sac_train import Main7SacMergeEnv, SACAgent, SEED, set_seed
from models_ode import KinematicBicycleModel
from utils import draw_car, draw_environment


# =========================
# 回放与导出参数
# =========================
POLICY_PATH = "sac_policy.pth"
EXPORT_PATH = "sac_replay.gif"
EXPORT_FPS = 20
EXPORT_FRAME_STRIDE = 2
SHOW_AFTER_EXPORT = True


def load_trained_agent(policy_path, device):
    """加载训练完成后保存的 SAC policy。"""
    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"Cannot find {policy_path}. Run main_sac_train.py first to generate the trained policy."
        )

    checkpoint = torch.load(policy_path, map_location=device)
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])

    agent = SACAgent(state_dim, action_dim, device)
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy.eval()
    return agent


def make_car_from_state(car_id, state, color, wheelbase):
    """根据保存的车辆状态临时构造用于绘图的车辆对象。"""
    car = KinematicBicycleModel(id=car_id, L=wheelbase, color=color)
    car.set_state(state)
    return car


def snapshot(env, info=None, reward=0.0, total_reward=0.0):
    """保存当前仿真状态，用于后续生成动画帧。"""
    dist1, dist2 = env._distances()
    veh12_gap = env._veh12_gap()
    return {
        "time": env.t,
        "veh1_state": env.state[0:5].copy(),
        "veh2_state": env.state[5:10].copy(),
        "ego_state": env.state[10:15].copy(),
        "z": float(env.state[15]),
        "mu": float(env.state[16]),
        "dist1": float(dist1),
        "dist2": float(dist2),
        "veh12_gap": float(veh12_gap),
        "reward": float(reward),
        "total_reward": float(total_reward),
        "success": False if info is None else bool(info["success"]),
        "collided": False if info is None else bool(info["collided"]),
        "progress": env._lane_progress() if info is None else float(info["lane_progress"]),
    }


def collect_replay_frames(agent, env):
    """使用训练好的 policy 跑一轮仿真，并记录动画所需的数据。"""
    state = env.reset()
    frames = [snapshot(env)]
    total_reward = 0.0
    last_info = {"success": False, "collided": False, "lane_progress": 0.0, "time": 0.0}

    for step_index in range(int(env.sim_time / env.dt)):
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

        if step_index % EXPORT_FRAME_STRIDE == 0 or done:
            frames.append(snapshot(env, info=info, reward=reward, total_reward=total_reward))

        if done:
            break

    print(
        "\nReplay finished | "
        f"total_reward={total_reward:.2f} | "
        f"progress={last_info['lane_progress']:.3f} | "
        f"success={last_info['success']} | collision={last_info['collided']}"
    )
    return frames


def export_replay_animation(frames, env, export_path):
    """将车辆运动、决策状态曲线和距离曲线导出为 GIF 或视频。"""
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1)
    ax_mu_z = plt.subplot(2, 2, 3)
    ax_dist = plt.subplot(2, 2, 4)

    times = np.array([frame["time"] for frame in frames])
    mu_hist = np.array([frame["mu"] for frame in frames])
    z_hist = np.array([frame["z"] for frame in frames])
    dist1_hist = np.array([frame["dist1"] for frame in frames])
    dist2_hist = np.array([frame["dist2"] for frame in frames])
    veh12_gap_hist = np.array([frame["veh12_gap"] for frame in frames])

    def draw_frame(index):
        frame = frames[index]
        veh1 = make_car_from_state("Veh 1 (Leader)", frame["veh1_state"], "lightblue", env.vehicle_l)
        veh2 = make_car_from_state("Veh 2 (Gap Control)", frame["veh2_state"], "royalblue", env.vehicle_l)
        ego = make_car_from_state("Veh 3 (Ego RL)", frame["ego_state"], "lightgreen", env.vehicle_l)

        ax_anim.cla()
        draw_environment(ax_anim, env.lane_width)
        draw_car(ax_anim, veh1, wheelbase=env.collision_radius)
        draw_car(ax_anim, veh2, wheelbase=env.collision_radius)
        draw_car(ax_anim, ego, wheelbase=env.collision_radius)
        ax_anim.set_xlim(ego.x - 15, ego.x + 45)
        ax_anim.set_ylim(-2, env.lane_width * 2 + 2)
        ax_anim.set_aspect("equal")
        title = f"Time: {frame['time']:.2f}s | SAC policy replay in main7 environment"
        if frame["success"]:
            title += " | SUCCESS"
        if frame["collided"]:
            title += " | COLLISION"
        ax_anim.set_title(title)

        ax_mu_z.cla()
        ax_mu_z.plot(times[: index + 1], mu_hist[: index + 1], "c-", linewidth=2, label="Env Score ($\\mu$)")
        ax_mu_z.plot(times[: index + 1], z_hist[: index + 1], "b-", linewidth=3, label="Opinion State ($z$)")
        ax_mu_z.axhline(0, color="gray", linestyle="--")
        ax_mu_z.axvline(20.0, color="black", linestyle=":", linewidth=1.5, label="Gap Control Starts")
        ax_mu_z.set_xlim(0, env.sim_time)
        ax_mu_z.set_ylim(-1.0, 1.5)
        ax_mu_z.set_title("Decision Dynamics ($\\mu$ and $z$)")
        ax_mu_z.legend(loc="upper left")
        ax_mu_z.grid(True)

        ax_dist.cla()
        ax_dist.plot(times[: index + 1], dist1_hist[: index + 1], "purple", linewidth=2, label="Distance to Veh 1")
        ax_dist.plot(times[: index + 1], dist2_hist[: index + 1], "red", linewidth=2, label="Distance to Veh 2")
        ax_dist.plot(times[: index + 1], veh12_gap_hist[: index + 1], "gray", linestyle="-.", linewidth=2, label="Veh1-Veh2 Gap")
        ax_dist.axhline(
            env.collision_radius,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Collision Threshold r={env.collision_radius:g}m",
        )
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

    if extension == ".gif":
        writer = animation.PillowWriter(fps=EXPORT_FPS)
    else:
        writer = animation.FFMpegWriter(fps=EXPORT_FPS)

    replay_animation.save(export_path, writer=writer)
    print(f"Saved replay animation to {export_path}")

    if SHOW_AFTER_EXPORT:
        plt.show()
    else:
        plt.close(fig)


def run_policy_once():
    """加载训练好的 policy，运行一次仿真，并导出动画。"""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_trained_agent(POLICY_PATH, device)
    env = Main7SacMergeEnv(render=False)

    frames = collect_replay_frames(agent, env)
    export_replay_animation(frames, env, EXPORT_PATH)


if __name__ == "__main__":
    run_policy_once()
