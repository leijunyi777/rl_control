import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from main7 import Main7GapFollowingDynamics, get_veh12_gap, make_car_from_state
from models_ode import (
    EgoVehicleOdeModel,
    KinematicBicycleModel,
    compute_gap_opinion_z_dot,
    compute_gap_signals_from_states,
    rear_state_derivative,
)
from utils import draw_car, draw_environment


SIM_TIME = 40.0
DT = 0.05
RENDER_DURING_SIM = True
EXPORT_ANIMATION = False
EXPORT_PATH = "main10_nomove_z_compare.gif"
EXPORT_FPS = 20
EXPORT_FRAME_STRIDE = 2
SHOW_AFTER_EXPORT = True

DESIRED_GAP = 20.0
GAP_SAFE = 15.0
K_GAP = 0.25
K_VEL = 0.45
U_BASE = 0.4
U_GAIN = 0.25
Z_DAMPING = 1.0
Z_ALPHA = 2.0


class Main10NoMoveDynamics(Main7GapFollowingDynamics):
    """Main7 target-lane dynamics with both old and new z updates, but no ego control."""

    def __init__(self, *args, z_new0=0.01, gap_safe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_new = float(z_new0)
        self.gap_safe = float(self.desired_gap if gap_safe is None else gap_safe)

    def pack_state(self):
        return np.concatenate([super().pack_state(), np.array([self.z_new])])

    def apply_state(self, state):
        super().apply_state(state[:17])
        self.z_new = float(state[17])

    def _gap_signals(self, state):
        return compute_gap_signals_from_states(
            state[0:5],
            state[5:10],
            self.veh1.L,
            self.veh2.L,
            self.gap_safe,
            k_gap=K_GAP,
            k_vel=K_VEL,
            u_base=U_BASE,
            u_gain=U_GAIN,
        )

    def rhs(self, t, state):
        veh1_state = state[0:5]
        veh2_state = state[5:10]
        ego_state = state[10:15]
        z_old = state[15]
        mu = state[16]
        z_new = state[17]

        a1, omega1 = 0.0, 0.0
        a2, omega2 = self._veh2_acceleration(t, veh1_state, veh2_state), 0.0

        old_control = self.veh3.control_derivatives(ego_state, z_old, mu, self._target_states(state))
        gap_signals = self._gap_signals(state)
        z_new_dot = compute_gap_opinion_z_dot(
            z_new,
            gap_signals["b_t"],
            gap_signals["u_t"],
            damping=Z_DAMPING,
            alpha=Z_ALPHA,
        )

        return np.concatenate([
            rear_state_derivative(veh1_state, a1, omega1, self.veh1.L),
            rear_state_derivative(veh2_state, a2, omega2, self.veh2.L),
            rear_state_derivative(ego_state, 0.0, 0.0, self.veh3.L),
            np.array([old_control["z_dot"], old_control["mu_dot"], z_new_dot]),
        ])

    def diagnostics(self, state):
        diag = self.veh3.control_derivatives(state[10:15], state[15], state[16], self._target_states(state))
        gap_signals = self._gap_signals(state)
        diag.update(gap_signals)
        diag["z_new_dot"] = compute_gap_opinion_z_dot(
            state[17],
            gap_signals["b_t"],
            gap_signals["u_t"],
            damping=Z_DAMPING,
            alpha=Z_ALPHA,
        )
        return diag


def snapshot(t, state, diag, veh12_gap, collided):
    return {
        "time": float(t),
        "veh1_state": state[0:5].copy(),
        "veh2_state": state[5:10].copy(),
        "ego_state": state[10:15].copy(),
        "z_old": float(state[15]),
        "mu": float(state[16]),
        "z_new": float(state[17]),
        "gap": float(diag["gap"]),
        "gap_dot": float(diag["gap_dot"]),
        "b_t": float(diag["b_t"]),
        "u_t": float(diag["u_t"]),
        "dist1": float(diag["sensor_data"]["veh1"]["dist"]),
        "dist2": float(diag["sensor_data"]["veh2"]["dist"]),
        "veh12_gap": float(veh12_gap),
        "collided": bool(collided),
    }


def draw_scene(ax_anim, ax_z, ax_gap, frame, frames, lane_width, sim_time, collision_radius, desired_gap, safe_gap, vehicle_l):
    veh1 = make_car_from_state("Veh 1 (Leader)", frame["veh1_state"], "lightblue", vehicle_l)
    veh2 = make_car_from_state("Veh 2 (Gap Control)", frame["veh2_state"], "royalblue", vehicle_l)
    ego = make_car_from_state("Veh 3 (Ego Constant Speed)", frame["ego_state"], "lightgreen", vehicle_l)

    times = [item["time"] for item in frames]
    old_z_hist = [item["z_old"] for item in frames]
    new_z_hist = [item["z_new"] for item in frames]
    gap_hist = [item["gap"] for item in frames]
    gap_dot_hist = [item["gap_dot"] for item in frames]
    b_hist = [item["b_t"] for item in frames]
    u_hist = [item["u_t"] for item in frames]

    ax_anim.cla()
    draw_environment(ax_anim, lane_width)
    draw_car(ax_anim, veh1, wheelbase=collision_radius)
    draw_car(ax_anim, veh2, wheelbase=collision_radius)
    draw_car(ax_anim, ego, wheelbase=collision_radius)
    ax_anim.set_xlim(ego.x - 15, ego.x + 45)
    ax_anim.set_ylim(-2, lane_width * 2 + 2)
    ax_anim.set_aspect("equal")
    title = f"Time: {frame['time']:.2f}s | Main10 No-Control z Comparison"
    if frame["collided"]:
        title += " | COLLISION"
    ax_anim.set_title(title)

    ax_z.cla()
    ax_z.plot(times, old_z_hist, "b-", linewidth=2.5, label="Old Formula z")
    ax_z.plot(times, new_z_hist, "m-", linewidth=2.5, label="New Formula z")
    ax_z.axhline(0, color="gray", linestyle="--")
    ax_z.axvline(20.0, color="black", linestyle=":", linewidth=1.5, label="Gap Control Starts")
    ax_z.set_xlim(0, sim_time)
    ax_z.set_title("Opinion State Comparison")
    ax_z.legend(loc="upper left")
    ax_z.grid(True)

    ax_gap.cla()
    ax_gap.plot(times, gap_hist, "gray", linewidth=2, label="Veh1-Veh2 Gap")
    ax_gap.plot(times, gap_dot_hist, "orange", linewidth=2, label="Gap Rate")
    ax_gap.plot(times, b_hist, "green", linewidth=2, label="b(t)")
    ax_gap.plot(times, u_hist, "purple", linewidth=2, label="u(t)")
    ax_gap.axhline(safe_gap, color="green", linestyle=":", linewidth=2, label="Safe Gap")
    ax_gap.axhline(desired_gap, color="gray", linestyle="-.", linewidth=1.5, label="Target Gap 20m")
    ax_gap.axvline(20.0, color="black", linestyle=":", linewidth=1.5)
    ax_gap.set_xlim(0, sim_time)
    ax_gap.set_title("Gap Signals for New Formula")
    ax_gap.legend(loc="upper right")
    ax_gap.grid(True)


def export_animation(frames, lane_width, sim_time, collision_radius, desired_gap, safe_gap, vehicle_l, export_path):
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1)
    ax_z = plt.subplot(2, 2, 3)
    ax_gap = plt.subplot(2, 2, 4)

    def draw_frame(index):
        draw_scene(
            ax_anim,
            ax_z,
            ax_gap,
            frames[index],
            frames[: index + 1],
            lane_width,
            sim_time,
            collision_radius,
            desired_gap,
            safe_gap,
            vehicle_l,
        )

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

    veh1 = KinematicBicycleModel(
        id="Veh 1 (Leader)",
        x=30.0,
        y=lane_width * 1.5,
        v=15.0,
        L=vehicle_l,
        color="lightblue",
    )
    veh2 = KinematicBicycleModel(
        id="Veh 2 (Gap Control)",
        x=15.0,
        y=lane_width * 1.5,
        v=15.0,
        L=vehicle_l,
        color="royalblue",
    )
    veh3 = EgoVehicleOdeModel(
        id="Veh 3 (Ego Constant Speed)",
        x=20.0,
        y=lane_width * 0.5,
        v=15.0,
        L=vehicle_l,
        color="lightgreen",
    )

    dynamics = Main10NoMoveDynamics(veh1, veh2, veh3, desired_gap=DESIRED_GAP, gap_safe=GAP_SAFE)
    collision_radius = veh3.r
    state = dynamics.pack_state()
    frames = []
    collided = False

    if RENDER_DURING_SIM:
        plt.ion()
        fig = plt.figure(figsize=(14, 8))
        ax_anim = plt.subplot(2, 1, 1)
        ax_z = plt.subplot(2, 2, 3)
        ax_gap = plt.subplot(2, 2, 4)

    for i in range(int(SIM_TIME / DT)):
        t = i * DT
        sol = solve_ivp(
            fun=dynamics.rhs,
            t_span=(t, t + DT),
            y0=state,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            max_step=DT / 5.0,
        )
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
            print(
                f"Collision detected at t={t + DT:.2f}s: "
                f"ego distances=({dist1:.3f}, {dist2:.3f}), veh12_gap={veh12_gap:.3f}, r={collision_radius:.3f}"
            )

        if i % EXPORT_FRAME_STRIDE == 0 or collided:
            frames.append(snapshot(t + DT, state, diag, veh12_gap, collided))

        if RENDER_DURING_SIM and (i % 4 == 0 or collided):
            draw_scene(
                ax_anim,
                ax_z,
                ax_gap,
                frames[-1],
                frames,
                lane_width,
                SIM_TIME,
                collision_radius,
                dynamics.desired_gap,
                dynamics.gap_safe,
                vehicle_l,
            )
            plt.pause(0.01)

        if collided:
            break

    if RENDER_DURING_SIM:
        plt.ioff()

    if EXPORT_ANIMATION and frames:
        export_animation(
            frames,
            lane_width,
            SIM_TIME,
            collision_radius,
            dynamics.desired_gap,
            dynamics.gap_safe,
            vehicle_l,
            EXPORT_PATH,
        )
    elif RENDER_DURING_SIM:
        plt.show()


if __name__ == "__main__":
    main()
