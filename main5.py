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

    dt = 0.05
    sim_time = 35.0
    steps = int(sim_time / dt)
    state = dynamics.pack_state()

    t_hist, mu_hist, z_hist, d1_hist, d2_hist, ego_y_hist = [], [], [], [], [], []
    uc_x_hist, uc_y_hist = [], []

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
        t_hist.append(t + dt)
        mu_hist.append(state[16])
        z_hist.append(state[15])
        d1_hist.append(diag["d1"])
        d2_hist.append(diag["d2"])
        ego_y_hist.append(veh3.y)
        uc_x_hist.append(diag["u_c"][0])
        uc_y_hist.append(diag["u_c"][1])

        if i % 4 == 0:
            ax_anim.cla()
            draw_environment(ax_anim, lane_width)
            draw_car(ax_anim, veh1)
            draw_car(ax_anim, veh2)
            draw_car(ax_anim, veh3)
            ax_anim.set_xlim(veh3.x - 15, veh3.x + 45)
            ax_anim.set_ylim(-2, lane_width * 2 + 2)
            ax_anim.set_aspect("equal")
            ax_anim.set_title(f"Time: {t + dt:.2f}s | RK45 ODE control with front-axle u_c")

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
            ax_dist.plot(t_hist, d1_hist, "purple", linewidth=2, label="Safe Dist to Veh 1 ($d_1$)")
            ax_dist.plot(t_hist, d2_hist, "red", linewidth=2, label="Safe Dist to Veh 2 ($d_2$)")
            ax_dist.axhline(0, color="black", linestyle="--", linewidth=2, label="Collision Threshold")
            ax_dist.set_xlim(0, sim_time)
            ax_dist.set_ylim(-2, 35)
            ax_dist.set_title("Safety Distance Monitoring ($d_1, d_2$)")
            ax_dist.legend(loc="upper right")
            ax_dist.grid(True)

            plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
