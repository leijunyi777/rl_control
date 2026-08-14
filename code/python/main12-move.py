import importlib.util
import os
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from main7 import get_veh12_gap
from models_ode import EgoVehicleOdeModel, KinematicBicycleModel


_MAIN11_PATH = Path(__file__).with_name("main11-move.py")
_MAIN11_SPEC = importlib.util.spec_from_file_location("main11_move_module", _MAIN11_PATH)
_MAIN11 = importlib.util.module_from_spec(_MAIN11_SPEC)
_MAIN11_SPEC.loader.exec_module(_MAIN11)


SIM_TIME = _MAIN11.SIM_TIME
DT = _MAIN11.DT
RENDER_DURING_SIM = True
EXPORT_ANIMATION = True
EXPORT_PATH = "main12_move_random_ego_x.gif"
EXPORT_FPS = _MAIN11.EXPORT_FPS
EXPORT_FRAME_STRIDE = _MAIN11.EXPORT_FRAME_STRIDE
SHOW_AFTER_EXPORT = True

DESIRED_GAP = _MAIN11.DESIRED_GAP
GAP_SAFE = _MAIN11.GAP_SAFE
EGO_X_BASE = 20.0
EGO_X_RANDOM_RANGE = 5.0


def sample_ego_x():
    return float(EGO_X_BASE + np.random.uniform(-EGO_X_RANDOM_RANGE, EGO_X_RANDOM_RANGE))


def draw_scene(ax_anim, ax_z, ax_dist, frame, frames, lane_width, sim_time, collision_radius, desired_gap, safe_gap, vehicle_l):
    _MAIN11.draw_scene(ax_anim, ax_z, ax_dist, frame, frames, lane_width, sim_time, collision_radius, desired_gap, safe_gap, vehicle_l)
    title = f"Time: {frame['time']:.2f}s | Main12 RBF-u Control | ego x0 random +/-5m"
    if frame["collided"]:
        title += " | COLLISION"
    ax_anim.set_title(title)


def export_animation(frames, lane_width, sim_time, collision_radius, desired_gap, safe_gap, vehicle_l, export_path):
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1)
    ax_z = plt.subplot(2, 2, 3)
    ax_dist = plt.subplot(2, 2, 4)

    def draw_frame(index):
        draw_scene(ax_anim, ax_z, ax_dist, frames[index], frames[: index + 1], lane_width, sim_time, collision_radius, desired_gap, safe_gap, vehicle_l)

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(frames), interval=1000 / EXPORT_FPS)
    extension = os.path.splitext(export_path)[1].lower()
    writer = animation.PillowWriter(fps=EXPORT_FPS) if extension == ".gif" else animation.FFMpegWriter(fps=EXPORT_FPS)
    ani.save(export_path, writer=writer)
    print(f"Saved animation to {export_path}")

    if SHOW_AFTER_EXPORT:
        plt.show()
    else:
        plt.close(fig)


def main():
    lane_width = 4.0
    vehicle_l = 2.8
    ego_x0 = sample_ego_x()
    print(f"Main12 move ego initial x = {ego_x0:.3f} m")

    veh1 = KinematicBicycleModel(id="Veh 1 (Leader)", x=30.0, y=lane_width * 1.5, v=15.0, L=vehicle_l, color="lightblue")
    veh2 = KinematicBicycleModel(id="Veh 2 (Gap Control)", x=15.0, y=lane_width * 1.5, v=15.0, L=vehicle_l, color="royalblue")
    veh3 = EgoVehicleOdeModel(id="Veh 3 (Ego Main12 Control)", x=ego_x0, y=lane_width * 0.5, v=15.0, L=vehicle_l, color="lightgreen")

    dynamics = _MAIN11.Main11MoveDynamics(veh1, veh2, veh3, desired_gap=DESIRED_GAP, gap_safe=GAP_SAFE)
    collision_radius = veh3.r
    state = dynamics.pack_state()
    frames = []
    collided = False

    if RENDER_DURING_SIM:
        plt.ion()
        fig = plt.figure(figsize=(14, 8))
        ax_anim = plt.subplot(2, 1, 1)
        ax_z = plt.subplot(2, 2, 3)
        ax_dist = plt.subplot(2, 2, 4)

    for i in range(int(SIM_TIME / DT)):
        t = i * DT
        sol = solve_ivp(fun=dynamics.rhs, t_span=(t, t + DT), y0=state, method="RK45", rtol=1e-6, atol=1e-8, max_step=DT / 5.0)
        if not sol.success:
            raise RuntimeError(sol.message)

        state = sol.y[:, -1]
        dynamics.apply_state(state)
        state = dynamics.pack_state()

        diag = dynamics.diagnostics(state)
        dist1 = diag["sensor_data"]["veh1"]["dist"]
        dist2 = diag["sensor_data"]["veh2"]["dist"]
        veh12_gap = get_veh12_gap(state[:17], veh1.L, veh2.L)
        collided = min(dist1, dist2, veh12_gap) < collision_radius
        if collided:
            print(f"Collision detected at t={t + DT:.2f}s: ego distances=({dist1:.3f}, {dist2:.3f}), veh12_gap={veh12_gap:.3f}, r={collision_radius:.3f}")

        if i % EXPORT_FRAME_STRIDE == 0 or collided:
            frames.append(_MAIN11.snapshot(t + DT, state, diag, veh12_gap, collided))

        if RENDER_DURING_SIM and (i % 4 == 0 or collided):
            draw_scene(ax_anim, ax_z, ax_dist, frames[-1], frames, lane_width, SIM_TIME, collision_radius, dynamics.desired_gap, dynamics.gap_safe, vehicle_l)
            plt.pause(0.01)

        if collided:
            break

    if RENDER_DURING_SIM:
        plt.ioff()

    if EXPORT_ANIMATION and frames:
        export_animation(frames, lane_width, SIM_TIME, collision_radius, dynamics.desired_gap, dynamics.gap_safe, vehicle_l, EXPORT_PATH)
    elif RENDER_DURING_SIM:
        plt.show()


if __name__ == "__main__":
    main()
