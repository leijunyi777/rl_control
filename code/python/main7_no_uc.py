import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from models_ode import (
    KinematicBicycleModel,
    EgoVehicleOdeModel,
    Main4OdeDynamics,
    front_position,
    front_velocity,
    rear_state_derivative,
)
from utils import draw_car, draw_environment


SIM_TIME = 40.0
DT = 0.05
RENDER_DURING_SIM = True
EXPORT_ANIMATION = False
EXPORT_PATH = "main7_no_uc.gif"
EXPORT_FPS = 20
EXPORT_FRAME_STRIDE = 2
SHOW_AFTER_EXPORT = True


class Main7GapFollowingDynamics(Main4OdeDynamics):
    """main5 scenario with a 20 m rear-car gap controller after 20 s."""

    def __init__(
        self,
        veh1,
        veh2,
        veh3,
        a_vel=4.0,
        period=6.0,
        yield_time=20.0,
        desired_gap=20.0,
    ):
        super().__init__(veh1, veh2, veh3, a_vel=a_vel, period=period, yield_time=yield_time)
        self.desired_gap = desired_gap

    def _veh2_acceleration(self, t, veh1_state, veh2_state):
        if t <= self.yield_time:
            wave_omega = 2.0 * np.pi / self.period
            return self.a_vel * wave_omega * np.cos(wave_omega * t)

        p1 = front_position(veh1_state, self.veh1.L)
        p2 = front_position(veh2_state, self.veh2.L)
        v1 = front_velocity(veh1_state, self.veh1.L)
        v2 = front_velocity(veh2_state, self.veh2.L)

        gap = p1[0] - p2[0]
        gap_error = gap - self.desired_gap
        closing_speed = v2[0] - v1[0]
        a2 = 0.35 * gap_error - 1.1 * closing_speed
        return float(np.clip(a2, -5.0, 2.0))

    def rhs(self, t, state):
        veh1_state = state[0:5]
        veh2_state = state[5:10]
        ego_state = state[10:15]
        z = state[15]
        mu = state[16]

        a1, omega1 = 0.0, 0.0
        a2, omega2 = self._veh2_acceleration(t, veh1_state, veh2_state), 0.0
        control = self.veh3.control_derivatives(ego_state, z, mu, self._target_states(state))

        return np.concatenate([
            rear_state_derivative(veh1_state, a1, omega1, self.veh1.L),
            rear_state_derivative(veh2_state, a2, omega2, self.veh2.L),
            rear_state_derivative(ego_state, control["a"], control["omega"], self.veh3.L),
            np.array([control["z_dot"], control["mu_dot"]]),
        ])


def make_car_from_state(car_id, state, color, wheelbase):
    car = KinematicBicycleModel(id=car_id, L=wheelbase, color=color)
    car.set_state(state)
    return car


def get_veh12_gap(state, veh1_l, veh2_l):
    p1 = front_position(state[0:5], veh1_l)
    p2 = front_position(state[5:10], veh2_l)
    return float(p1[0] - p2[0])


def snapshot(t, state, diag, veh12_gap, collided):
    return {
        "time": float(t),
        "veh1_state": state[0:5].copy(),
        "veh2_state": state[5:10].copy(),
        "ego_state": state[10:15].copy(),
        "z": float(state[15]),
        "mu": float(state[16]),
        "dist1": float(diag["sensor_data"]["veh1"]["dist"]),
        "dist2": float(diag["sensor_data"]["veh2"]["dist"]),
        "veh12_gap": float(veh12_gap),
        "collided": bool(collided),
    }


def draw_scene(ax_anim, ax_mu_z, ax_dist, frame, frames, lane_width, sim_time, collision_radius, desired_gap, vehicle_l):
    veh1 = make_car_from_state("Veh 1 (Leader)", frame["veh1_state"], "lightblue", vehicle_l)
    veh2 = make_car_from_state("Veh 2 (Gap Control)", frame["veh2_state"], "royalblue", vehicle_l)
    ego = make_car_from_state("Veh 3 (Ego RK45)", frame["ego_state"], "lightgreen", vehicle_l)

    times = [item["time"] for item in frames]
    mu_hist = [item["mu"] for item in frames]
    z_hist = [item["z"] for item in frames]
    dist1_hist = [item["dist1"] for item in frames]
    dist2_hist = [item["dist2"] for item in frames]
    gap_hist = [item["veh12_gap"] for item in frames]

    ax_anim.cla()
    draw_environment(ax_anim, lane_width)
    draw_car(ax_anim, veh1, wheelbase=collision_radius)
    draw_car(ax_anim, veh2, wheelbase=collision_radius)
    draw_car(ax_anim, ego, wheelbase=collision_radius)
    ax_anim.set_xlim(ego.x - 15, ego.x + 45)
    ax_anim.set_ylim(-2, lane_width * 2 + 2)
    ax_anim.set_aspect("equal")
    title = f"Time: {frame['time']:.2f}s | Rear car maintains 20m gap after 20s"
    if frame["collided"]:
        title += " | COLLISION"
    ax_anim.set_title(title)

    ax_mu_z.cla()
    ax_mu_z.plot(times, mu_hist, "c-", linewidth=2, label="Env Score ($\\mu$)")
    ax_mu_z.plot(times, z_hist, "b-", linewidth=3, label="Opinion State ($z$)")
    ax_mu_z.axhline(0, color="gray", linestyle="--")
    ax_mu_z.axvline(20.0, color="black", linestyle=":", linewidth=1.5, label="Gap Control Starts")
    ax_mu_z.set_xlim(0, sim_time)
    ax_mu_z.set_ylim(-1.0, 1.5)
    ax_mu_z.set_title("Decision Dynamics ($\\mu$ and $z$)")
    ax_mu_z.legend(loc="upper left")
    ax_mu_z.grid(True)

    ax_dist.cla()
    ax_dist.plot(times, dist1_hist, "purple", linewidth=2, label="Distance to Veh 1")
    ax_dist.plot(times, dist2_hist, "red", linewidth=2, label="Distance to Veh 2")
    ax_dist.plot(times, gap_hist, "gray", linestyle="-.", linewidth=2, label="Veh1-Veh2 Gap")
    ax_dist.axhline(collision_radius, color="black", linestyle="--", linewidth=2, label=f"Collision Threshold r={collision_radius:g}m")
    ax_dist.axhline(desired_gap, color="green", linestyle=":", linewidth=2, label="Target Gap 20m")
    ax_dist.axvline(20.0, color="black", linestyle=":", linewidth=1.5)
    ax_dist.set_xlim(0, sim_time)
    upper_distance = max(collision_radius * 2.0, desired_gap * 1.2, max(dist1_hist), max(dist2_hist), max(gap_hist))
    ax_dist.set_ylim(0, upper_distance * 1.1)
    ax_dist.set_title("Relative Distance Monitoring")
    ax_dist.legend(loc="upper right")
    ax_dist.grid(True)


def export_animation(frames, lane_width, sim_time, collision_radius, desired_gap, vehicle_l, export_path):
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1)
    ax_mu_z = plt.subplot(2, 2, 3)
    ax_dist = plt.subplot(2, 2, 4)

    def draw_frame(index):
        draw_scene(
            ax_anim,
            ax_mu_z,
            ax_dist,
            frames[index],
            frames[: index + 1],
            lane_width,
            sim_time,
            collision_radius,
            desired_gap,
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
        x=17.0,
        y=lane_width * 1.5,
        v=15.0,
        L=vehicle_l,
        color="royalblue",
    )
    veh3 = EgoVehicleOdeModel(
        id="Veh 3 (Ego RK45)",
        x=20.0,
        y=lane_width * 0.5,
        v=15.0,
        L=vehicle_l,
        color="lightgreen",
    )

    #veh3.k_o = 0.0
    veh3.r_rho = -8.0

    dynamics = Main7GapFollowingDynamics(veh1, veh2, veh3)
    collision_radius = veh3.r
    state = dynamics.pack_state()
    frames = []
    collided = False

    fig = None
    if RENDER_DURING_SIM:
        plt.ion()
        fig = plt.figure(figsize=(14, 8))
        ax_anim = plt.subplot(2, 1, 1)
        ax_mu_z = plt.subplot(2, 2, 3)
        ax_dist = plt.subplot(2, 2, 4)

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
        veh12_gap = get_veh12_gap(state, veh1.L, veh2.L)
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
                ax_mu_z,
                ax_dist,
                frames[-1],
                frames,
                lane_width,
                SIM_TIME,
                collision_radius,
                dynamics.desired_gap,
                vehicle_l,
            )
            plt.pause(0.01)

        if collided:
            break

    if RENDER_DURING_SIM:
        plt.ioff()

    if EXPORT_ANIMATION and frames:
        export_animation(frames, lane_width, SIM_TIME, collision_radius, dynamics.desired_gap, vehicle_l, EXPORT_PATH)
    elif RENDER_DURING_SIM:
        plt.show()


if __name__ == "__main__":
    main()
