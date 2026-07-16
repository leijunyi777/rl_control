import numpy as np
import matplotlib.pyplot as plt
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


class Main6RandomYieldDynamics(Main4OdeDynamics):
    """后车采用随机加减速度，并在 20s 后切换为让行行为。"""

    def __init__(
        self,
        veh1,
        veh2,
        veh3,
        yield_time=20.0,
        yield_speed=12.0,
        random_update_period=0.5,
        random_accel_limit=2.5,
        seed=7,
    ):
        super().__init__(veh1, veh2, veh3, yield_time=yield_time, yield_speed=yield_speed)
        self.random_update_period = random_update_period
        self.random_accel_limit = random_accel_limit
        self.rng = np.random.default_rng(seed)
        self.random_accel = 0.0
        self.next_random_update = 0.0

    def update_random_accel(self, t):
        """按固定周期更新一次随机加速度，避免 ODE 求解期间随机项频繁跳变。"""
        if t >= self.next_random_update:
            self.random_accel = self.rng.uniform(-self.random_accel_limit, self.random_accel_limit)
            self.next_random_update = t + self.random_update_period

    def _veh2_acceleration(self, t, veh1_state, veh2_state, ego_state):
        """计算后车纵向加速度：前 20s 随机贴近 ego，20s 后主动让行。"""
        if t > self.yield_time:
            return -3.0 if veh2_state[3] > self.yield_speed else 0.0

        p1 = front_position(veh1_state, self.veh1.L)
        p2 = front_position(veh2_state, self.veh2.L)
        pe = front_position(ego_state, self.veh3.L)
        v2 = front_velocity(veh2_state, self.veh2.L)
        ve = front_velocity(ego_state, self.veh3.L)

        x_error = pe[0] - p2[0]
        v_error = ve[0] - v2[0]
        align_accel = 0.55 * x_error + 0.9 * v_error
        a2 = align_accel + self.random_accel

        front_gap = p1[0] - p2[0]
        front_rel_speed = v2[0] - front_velocity(veh1_state, self.veh1.L)[0]
        desired_front_gap = max(3.0 * self.veh3.r, 4.5)
        if front_gap < desired_front_gap or front_rel_speed > 0.0:
            brake_accel = -1.8 * max(desired_front_gap - front_gap, 0.0) - 1.2 * max(front_rel_speed, 0.0)
            a2 = min(a2, brake_accel)

        return np.clip(a2, -6.0, 3.0)

    def rhs(self, t, state):
        veh1_state = state[0:5]
        veh2_state = state[5:10]
        ego_state = state[10:15]
        z = state[15]
        mu = state[16]

        a1, omega1 = 0.0, 0.0
        a2 = self._veh2_acceleration(t, veh1_state, veh2_state, ego_state)
        omega2 = 0.0

        control = self.veh3.control_derivatives(ego_state, z, mu, self._target_states(state))

        return np.concatenate([
            rear_state_derivative(veh1_state, a1, omega1, self.veh1.L),
            rear_state_derivative(veh2_state, a2, omega2, self.veh2.L),
            rear_state_derivative(ego_state, control["a"], control["omega"], self.veh3.L),
            np.array([control["z_dot"], control["mu_dot"]]),
        ])


def main():
    lane_width = 4.0
    L = 2.8

    veh1 = KinematicBicycleModel(
        id="Veh 1 (Leader)",
        x=32.0,
        y=lane_width * 1.5,
        v=15.0,
        L=L,
        color="lightblue",
    )
    veh2 = KinematicBicycleModel(
        id="Veh 2 (Random)",
        x=25.0,
        y=lane_width * 1.5,
        v=15.0,
        L=L,
        color="royalblue",
    )
    veh3 = EgoVehicleOdeModel(
        id="Veh 3 (Ego RK45)",
        x=20.0,
        y=lane_width * 0.5,
        v=15.0,
        L=L,
        color="lightgreen",
    )

    dynamics = Main6RandomYieldDynamics(veh1, veh2, veh3)
    collision_radius = veh3.r

    dt = 0.05
    sim_time = 35.0
    steps = int(sim_time / dt)
    state = dynamics.pack_state()

    t_hist, mu_hist, z_hist, dist1_hist, dist2_hist, veh12_dist_hist = [], [], [], [], [], []
    collided = False

    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1)
    ax_mu_z = plt.subplot(2, 2, 3)
    ax_dist = plt.subplot(2, 2, 4)

    for i in range(steps):
        t = i * dt
        dynamics.update_random_accel(t)
        sol = solve_ivp(
            fun=dynamics.rhs,
            t_span=(t, t + dt),
            y0=state,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            max_step=dt / 5.0,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        state = sol.y[:, -1]
        dynamics.apply_state(state)
        state = dynamics.pack_state()

        diag = dynamics.diagnostics(state)
        dist1 = diag["sensor_data"]["veh1"]["dist"]
        dist2 = diag["sensor_data"]["veh2"]["dist"]
        veh12_dist = np.linalg.norm(
            front_position(state[0:5], veh1.L) - front_position(state[5:10], veh2.L)
        )

        ego_min_distance = min(dist1, dist2)
        if ego_min_distance < collision_radius:
            collided = True
            collided_with = "Veh 1" if dist1 <= dist2 else "Veh 2"
            print(
                f"Collision detected at t={t + dt:.2f}s with {collided_with}: "
                f"distance={ego_min_distance:.3f}m < r={collision_radius:.3f}m"
            )

        if veh12_dist < collision_radius:
            collided = True
            print(
                f"Collision detected at t={t + dt:.2f}s between Veh 1 and Veh 2: "
                f"distance={veh12_dist:.3f}m < r={collision_radius:.3f}m"
            )

        t_hist.append(t + dt)
        mu_hist.append(state[16])
        z_hist.append(state[15])
        dist1_hist.append(dist1)
        dist2_hist.append(dist2)
        veh12_dist_hist.append(veh12_dist)

        if i % 4 == 0 or collided:
            ax_anim.cla()
            draw_environment(ax_anim, lane_width)
            draw_car(ax_anim, veh1, wheelbase=collision_radius)
            draw_car(ax_anim, veh2, wheelbase=collision_radius)
            draw_car(ax_anim, veh3, wheelbase=collision_radius)
            ax_anim.set_xlim(veh3.x - 15, veh3.x + 45)
            ax_anim.set_ylim(-2, lane_width * 2 + 2)
            ax_anim.set_aspect("equal")
            title = f"Time: {t + dt:.2f}s | Random rear-car acceleration"
            if collided:
                title += " | COLLISION"
            ax_anim.set_title(title)

            ax_mu_z.cla()
            ax_mu_z.plot(t_hist, mu_hist, "c-", linewidth=2, label="Env Score ($\\mu$)")
            ax_mu_z.plot(t_hist, z_hist, "b-", linewidth=3, label="Opinion State ($z$)")
            ax_mu_z.axhline(0, color="gray", linestyle="--")
            ax_mu_z.axvline(dynamics.yield_time, color="black", linestyle=":", linewidth=1.5, label="Yield Starts")
            ax_mu_z.set_xlim(0, sim_time)
            ax_mu_z.set_ylim(-1.0, 1.5)
            ax_mu_z.set_title("Decision Dynamics ($\\mu$ and $z$)")
            ax_mu_z.legend(loc="upper left")
            ax_mu_z.grid(True)

            ax_dist.cla()
            ax_dist.plot(t_hist, dist1_hist, "purple", linewidth=2, label="Ego to Veh 1")
            ax_dist.plot(t_hist, dist2_hist, "red", linewidth=2, label="Ego to Veh 2")
            ax_dist.plot(t_hist, veh12_dist_hist, "gray", linewidth=1.5, linestyle="-.", label="Veh 1 to Veh 2")
            ax_dist.axhline(
                collision_radius,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Collision Threshold r={collision_radius:g}m",
            )
            ax_dist.set_xlim(0, sim_time)
            upper_distance = max([collision_radius * 2.0, *dist1_hist, *dist2_hist, *veh12_dist_hist])
            ax_dist.set_ylim(0, upper_distance * 1.1)
            ax_dist.set_title("Relative Distance Monitoring")
            ax_dist.legend(loc="upper right")
            ax_dist.grid(True)

            plt.pause(0.01)

        if collided:
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
