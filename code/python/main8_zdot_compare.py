import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import main8_multigap_test as base
from models_ode import rear_state_derivative, _signed_safe
from utils import draw_car, draw_environment


class MultiGapZDotCompareDynamics(base.MultiGapOdeDynamics):
    """基于 main8 的 z_dot 新旧公式对比动力学。"""

    def __init__(self, target_vehicles, ego, base_gap):
        super().__init__(target_vehicles, ego, base_gap)
        self.z_old0 = np.full(self.num_gaps, 0.01)
        self.z_new0 = np.full(self.num_gaps, 0.01)

    def pack_state(self):
        vehicle_states = [vehicle.get_state() for vehicle in self.target_vehicles]
        return np.concatenate(
            vehicle_states
            + [self.ego.get_state(), self.z_old0, self.z_new0, self.mu0, self.gap_error_integral0]
        )

    def apply_state(self, state):
        for index, vehicle in enumerate(self.target_vehicles):
            vehicle.set_state(state[5 * index: 5 * (index + 1)])

        ego_start = 5 * len(self.target_vehicles)
        z_old_start = ego_start + 5
        z_new_start = z_old_start + self.num_gaps
        mu_start = z_new_start + self.num_gaps
        integral_start = mu_start + self.num_gaps

        self.ego.set_state(state[ego_start: ego_start + 5])
        self.z_old0 = state[z_old_start:z_new_start].copy()
        self.z_new0 = state[z_new_start:mu_start].copy()
        self.z0 = self.z_old0.copy()
        self.mu0 = state[mu_start:integral_start].copy()
        self.gap_error_integral0 = state[integral_start:integral_start + self.num_gaps].copy()

    def _split_state(self, state):
        target_states = [
            state[5 * index: 5 * (index + 1)]
            for index in range(len(self.target_vehicles))
        ]
        ego_start = 5 * len(self.target_vehicles)
        z_old_start = ego_start + 5
        z_new_start = z_old_start + self.num_gaps
        mu_start = z_new_start + self.num_gaps
        integral_start = mu_start + self.num_gaps

        ego_state = state[ego_start:ego_start + 5]
        z_old = state[z_old_start:z_new_start]
        z_new = state[z_new_start:mu_start]
        mu = state[mu_start:integral_start]
        gap_error_integral = state[integral_start:integral_start + self.num_gaps]
        return target_states, ego_state, z_old, z_new, mu, gap_error_integral

    def compute_gap_updates(self, target_states, ego_state, z_old, z_new, mu, gap_error_integral, t):
        mu_dot = np.zeros(self.num_gaps)
        z_old_dot = np.zeros(self.num_gaps)
        z_new_dot = np.zeros(self.num_gaps)
        gap_error_integral_dot = np.zeros(self.num_gaps)
        u_nominal = np.zeros((self.num_gaps, 2))
        u_safe = np.zeros((self.num_gaps, 2))
        u_total = np.zeros((self.num_gaps, 2))
        gap_distances = np.zeros(self.num_gaps)
        gap_errors = np.zeros(self.num_gaps)
        gap_pid_accels = np.zeros(self.num_gaps)
        desired_gaps = self.desired_gaps_at(t)

        for gap_index in range(self.num_gaps):
            front_state = target_states[gap_index]
            rear_state = target_states[gap_index + 1]
            data = self._gap_pair_data(front_state, rear_state, ego_state)

            g_front = data["e_front"] / _signed_safe(np.linalg.norm(data["e_front"]))
            g_rear = data["e_rear"] / _signed_safe(np.linalg.norm(data["e_rear"]))
            g_rear_front = data["e_rear_front"] / _signed_safe(np.linalg.norm(data["e_rear_front"]))

            gap_distance = np.linalg.norm(data["e_rear_front"])
            gap_along_road = data["p_front"][0] - data["p_rear"][0]
            gap_error = gap_along_road - desired_gaps[gap_index]
            gap_error_derivative = data["v_front"][0] - data["v_rear"][0]
            gap_error_integral_dot[gap_index] = gap_error
            gap_errors[gap_index] = gap_error
            gap_pid_accels[gap_index] = np.clip(
                base.GAP_PID_KP * gap_error
                + base.GAP_PID_KI * gap_error_integral[gap_index]
                + base.GAP_PID_KD * gap_error_derivative,
                -base.GAP_ACCEL_LIMIT,
                base.GAP_ACCEL_LIMIT,
            )

            d_rear_front = gap_distance - self.r
            phi_rear_front = np.dot(g_rear_front, data["v_rear_front"]) / _signed_safe(d_rear_front)
            gap_distances[gap_index] = gap_distance

            tanh_arg = (
                -self.k
                * np.dot(self.rho, g_front)
                * np.dot(self.rho, g_rear)
                * (d_rear_front - 2.0 * self.r)
                * (phi_rear_front + self.eps2)
            )
            mu_dot[gap_index] = -self.k_mu * mu[gap_index] + np.tanh(tanh_arg)
            z_old_dot[gap_index] = self.ego.compute_z_dot(z_old[gap_index], mu[gap_index])
            z_new_dot[gap_index] = self.ego.compute_z_dot_new(z_new[gap_index], mu[gap_index])

            w = np.tanh(self.k_w * z_old[gap_index])
            desired_relative_position = 0.5 * data["e_rear_front"] + self.eta * ((1.0 - w) * self.r_eta)
            u_nominal[gap_index] = (
                -self.k_p * (data["e_front"] - desired_relative_position)
                - self.k_v * data["v_ego_front"]
            )

            d_front = np.linalg.norm(data["e_front"]) - self.r
            d_rear = np.linalg.norm(data["e_rear"]) - self.r
            phi_front = np.dot(g_front, data["v_ego_front"]) / _signed_safe(d_front)
            phi_rear = np.dot(g_rear, data["v_ego_rear"]) / _signed_safe(d_rear)
            u_safe[gap_index] = -self.k_o * g_front * phi_front - self.k_o * g_rear * phi_rear
            u_total[gap_index] = u_nominal[gap_index] + u_safe[gap_index]

        return {
            "mu_dot": mu_dot,
            "z_old_dot": z_old_dot,
            "z_new_dot": z_new_dot,
            "gap_error_integral_dot": gap_error_integral_dot,
            "u_nominal": u_nominal,
            "u_safe": u_safe,
            "u_total": u_total,
            "gap_distances": gap_distances,
            "desired_gaps": desired_gaps,
            "gap_errors": gap_errors,
            "gap_pid_accels": gap_pid_accels,
        }

    def rhs(self, t, state):
        target_states, ego_state, z_old, z_new, mu, gap_error_integral = self._split_state(state)
        updates = self.compute_gap_updates(target_states, ego_state, z_old, z_new, mu, gap_error_integral, t)

        target_derivatives = [rear_state_derivative(target_states[0], 0.0, 0.0, base.VEHICLE_L)]
        for gap_index in range(self.num_gaps):
            rear_state = target_states[gap_index + 1]
            target_derivatives.append(
                rear_state_derivative(rear_state, updates["gap_pid_accels"][gap_index], 0.0, base.VEHICLE_L)
            )

        ego_derivative = rear_state_derivative(ego_state, 0.0, 0.0, self.ego.L)

        return np.concatenate(
            target_derivatives
            + [
                ego_derivative,
                updates["z_old_dot"],
                updates["z_new_dot"],
                updates["mu_dot"],
                updates["gap_error_integral_dot"],
            ]
        )

    def diagnostics(self, state, t=0.0):
        target_states, ego_state, z_old, z_new, mu, gap_error_integral = self._split_state(state)
        updates = self.compute_gap_updates(target_states, ego_state, z_old, z_new, mu, gap_error_integral, t)
        return {
            "target_states": target_states,
            "ego_state": ego_state,
            "z_old": z_old,
            "z_new": z_new,
            "mu": mu,
            "gap_error_integral": gap_error_integral,
            **updates,
        }


def draw_compare_scene(ax_scene, ax_mu, ax_z, dynamics, state, histories, target_gap, current_time):
    diagnostics = dynamics.diagnostics(state, current_time)
    dynamics.apply_state(state)

    ax_scene.cla()
    draw_environment(ax_scene, base.LANE_WIDTH)
    for vehicle in dynamics.target_vehicles:
        draw_car(ax_scene, vehicle, wheelbase=dynamics.r)
    draw_car(ax_scene, dynamics.ego, wheelbase=dynamics.r)

    ego_x = dynamics.ego.x
    ax_scene.set_xlim(ego_x - 15, ego_x + 45)
    ax_scene.set_ylim(-2, base.LANE_WIDTH * 2 + 2)
    ax_scene.set_aspect("equal")
    ax_scene.set_title(
        f"Z-dot comparison | base gap={target_gap:.2f}m | switch period={base.GAP_SWITCH_PERIOD:.1f}s | ego control disabled"
    )

    ax_mu.cla()
    for gap_index in range(dynamics.num_gaps):
        ax_mu.plot(histories["time"], histories["mu"][:, gap_index], linewidth=2, label=f"mu gap {gap_index + 1}")
    ax_mu.axhline(0, color="gray", linestyle="--")
    ax_mu.set_xlim(0, base.SIM_TIME)
    ax_mu.set_title("Bifurcation Parameters for Gaps")
    ax_mu.legend(loc="upper left")
    ax_mu.grid(True)

    ax_z.cla()
    for gap_index in range(dynamics.num_gaps):
        old_line = ax_z.plot(
            histories["time"],
            histories["z_old"][:, gap_index],
            linewidth=2,
            linestyle="-",
            label=f"old z gap {gap_index + 1}",
        )[0]
        ax_z.plot(
            histories["time"],
            histories["z_new"][:, gap_index],
            linewidth=2,
            linestyle="--",
            color=old_line.get_color(),
            label=f"new z gap {gap_index + 1}",
        )
    ax_z.axhline(0, color="gray", linestyle="--")
    ax_z.set_xlim(0, base.SIM_TIME)
    ax_z.set_title("Opinion States: Old Solid, New Dashed")
    ax_z.legend(loc="upper left")
    ax_z.grid(True)


def main():
    target_vehicles, ego, target_gap = base.build_scene()
    dynamics = MultiGapZDotCompareDynamics(target_vehicles, ego, target_gap)
    state = dynamics.pack_state()

    print("Dynamic desired-gap schedule by period:")
    for period_index, desired_gaps in enumerate(dynamics.gap_schedule):
        period_start = period_index * base.GAP_SWITCH_PERIOD
        period_end = period_start + base.GAP_SWITCH_PERIOD
        print(f"  {period_start:5.1f}s - {period_end:5.1f}s: {np.round(desired_gaps, 3)}")

    histories = {
        "time": np.empty((0,), dtype=float),
        "mu": np.empty((0, dynamics.num_gaps), dtype=float),
        "z_old": np.empty((0, dynamics.num_gaps), dtype=float),
        "z_new": np.empty((0, dynamics.num_gaps), dtype=float),
    }

    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax_scene = plt.subplot(2, 1, 1)
    ax_mu = plt.subplot(2, 2, 3)
    ax_z = plt.subplot(2, 2, 4)

    for step_index in range(int(base.SIM_TIME / base.DT)):
        t = step_index * base.DT
        sol = solve_ivp(
            fun=dynamics.rhs,
            t_span=(t, t + base.DT),
            y0=state,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            max_step=base.DT / 5.0,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        state = sol.y[:, -1]
        diagnostics = dynamics.diagnostics(state, t + base.DT)

        histories["time"] = np.append(histories["time"], t + base.DT)
        histories["mu"] = np.vstack([histories["mu"], diagnostics["mu"]])
        histories["z_old"] = np.vstack([histories["z_old"], diagnostics["z_old"]])
        histories["z_new"] = np.vstack([histories["z_new"], diagnostics["z_new"]])

        if step_index % 4 == 0:
            draw_compare_scene(ax_scene, ax_mu, ax_z, dynamics, state, histories, target_gap, t + base.DT)
            plt.pause(0.01)

    plt.ioff()
    plt.show()

    final_diagnostics = dynamics.diagnostics(state, base.SIM_TIME)
    print("Final gap distances:", np.round(final_diagnostics["gap_distances"], 3))
    print("Final desired gaps:", np.round(final_diagnostics["desired_gaps"], 3))
    print("Final gap errors:", np.round(final_diagnostics["gap_errors"], 3))
    print("Final PID accelerations:", np.round(final_diagnostics["gap_pid_accels"], 3))
    print("Final mu:", np.round(final_diagnostics["mu"], 3))
    print("Final old z:", np.round(final_diagnostics["z_old"], 3))
    print("Final new z:", np.round(final_diagnostics["z_new"], 3))
    print("Computed u_total for each gap, based on old z and not applied to ego:")
    print(np.round(final_diagnostics["u_total"], 3))


if __name__ == "__main__":
    main()
