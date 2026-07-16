import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models_ode import KinematicBicycleModel, EgoVehicleOdeModel, Main4OdeDynamics
from utils import draw_car, draw_environment


def main():
    lane_width = 4.0
    L = 2.8

    veh1 = KinematicBicycleModel(
        id="Veh 1 (Leader)",
        x=30.0,
        y=lane_width * 1.5,
        v=15.0,
        L=L,
        color="lightblue",
    )
    veh2 = KinematicBicycleModel(
        id="Veh 2 (Aggressive)",
        x=15.0,
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

    dynamics = Main4OdeDynamics(veh1, veh2, veh3)
    collision_radius = veh3.r

    dt = 0.05
    sim_time = 35.0
    steps = int(sim_time / dt)
    state = dynamics.pack_state()

    t_hist, mu_hist, z_hist, dist1_hist, dist2_hist, ego_y_hist = [], [], [], [], [], []
    uc_x_hist, uc_y_hist = [], []
    collided = False
    collision_message = ""

    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1)
    ax_mu_z = plt.subplot(2, 2, 3)
    ax_dist = plt.subplot(2, 2, 4)

    for i in range(steps):
        t = i * dt
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
        min_distance = min(dist1, dist2)
        if min_distance < collision_radius:
            collided = True
            collided_with = "Veh 1" if dist1 <= dist2 else "Veh 2"
            collision_message = (
                f"Collision detected at t={t + dt:.2f}s with {collided_with}: "
                f"distance={min_distance:.3f}m < r={collision_radius:.3f}m"
            )
            print(collision_message)

        t_hist.append(t + dt)
        mu_hist.append(state[16])
        z_hist.append(state[15])
        dist1_hist.append(dist1)
        dist2_hist.append(dist2)
        ego_y_hist.append(veh3.y)
        uc_x_hist.append(diag["u_c"][0])
        uc_y_hist.append(diag["u_c"][1])

        if i % 4 == 0 or collided:
            ax_anim.cla()
            draw_environment(ax_anim, lane_width)
            draw_car(ax_anim, veh1, wheelbase=collision_radius)
            draw_car(ax_anim, veh2, wheelbase=collision_radius)
            draw_car(ax_anim, veh3, wheelbase=collision_radius)
            ax_anim.set_xlim(veh3.x - 15, veh3.x + 45)
            ax_anim.set_ylim(-2, lane_width * 2 + 2)
            ax_anim.set_aspect("equal")
            title = f"Time: {t + dt:.2f}s | RK45 ODE control"
            if collided:
                title += " | COLLISION"
            ax_anim.set_title(title)

            ax_mu_z.cla()
            ax_mu_z.plot(t_hist, mu_hist, "c-", linewidth=2, label="Env Score ($\\mu$)")
            ax_mu_z.plot(t_hist, z_hist, "b-", linewidth=3, label="Opinion State ($z$)")
            ax_mu_z.axhline(0, color="gray", linestyle="--")
            ax_mu_z.set_xlim(0, sim_time)
            ax_mu_z.set_ylim(-1.0, 1.5)
            ax_mu_z.set_title("Decision Dynamics ($\\mu$ and $z$)")
            ax_mu_z.legend(loc="upper left")
            ax_mu_z.grid(True)

            ax_dist.cla()
            ax_dist.plot(t_hist, dist1_hist, "purple", linewidth=2, label="Distance to Veh 1")
            ax_dist.plot(t_hist, dist2_hist, "red", linewidth=2, label="Distance to Veh 2")
            ax_dist.axhline(
                collision_radius,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Collision Threshold r={collision_radius:g}m",
            )
            ax_dist.set_xlim(0, sim_time)
            upper_distance = max([collision_radius * 2.0, *dist1_hist, *dist2_hist])
            ax_dist.set_ylim(0, upper_distance * 1.1)
            ax_dist.set_title("Ego Relative Distance Monitoring")
            ax_dist.legend(loc="upper right")
            ax_dist.grid(True)

            plt.pause(0.01)

        if collided:
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
