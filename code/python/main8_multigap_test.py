import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models_ode import (
    KinematicBicycleModel,
    EgoVehicleOdeModel,
    front_position,
    front_velocity,
    rear_state_derivative,
    _signed_safe,
)
from utils import draw_car, draw_environment


SIM_TIME = 40.0
DT = 0.05
LANE_WIDTH = 4.0
VEHICLE_L = 2.8
NUM_TARGET_VEHICLES = 4
TARGET_SPEED = 15.0
GAP_SWITCH_PERIOD = 6.0
MAX_CHANGED_GAPS_PER_PERIOD = 2
GAP_MULTIPLIERS = np.array([0.5, 1.0, 1.3])
RANDOM_SEED = 778

GAP_PID_KP = 0.7
GAP_PID_KI = 0.04
GAP_PID_KD = 1.2
GAP_ACCEL_LIMIT = 4.0


class MultiGapOdeDynamics:
    """Test-only dynamics for five target-lane vehicles and four candidate gaps."""

    def __init__(self, target_vehicles, ego, base_gap):
        self.target_vehicles = target_vehicles
        self.ego = ego
        self.num_gaps = len(target_vehicles) - 1
        self.base_gap = base_gap

        self.r = ego.r
        self.rho = ego.rho
        self.eta = ego.eta
        self.k_mu = ego.k_mu
        self.k = ego.k
        self.k_w = ego.k_w
        self.eps = ego.eps
        self.eps2 = ego.eps2
        self.k_p = ego.k_p
        self.k_v = ego.k_v
        self.k_o = ego.k_o
        self.r_eta = ego.r_eta

        self.z0 = np.full(self.num_gaps, 0.01)
        self.mu0 = np.zeros(self.num_gaps)
        self.gap_error_integral0 = np.zeros(self.num_gaps)
        self.gap_schedule = self._build_gap_schedule()

    def _build_gap_schedule(self):
        schedule = []
        num_periods = int(np.ceil(SIM_TIME / GAP_SWITCH_PERIOD)) + 1
        rng = np.random.default_rng(RANDOM_SEED)

        for period_index in range(num_periods):
            multipliers = np.ones(self.num_gaps)
            min_changed_count = 1 if period_index == 0 and MAX_CHANGED_GAPS_PER_PERIOD > 0 else 0
            changed_count = int(rng.integers(min_changed_count, MAX_CHANGED_GAPS_PER_PERIOD + 1))
            if changed_count > 0:
                changed_gaps = rng.choice(self.num_gaps, size=changed_count, replace=False)
                multipliers[changed_gaps] = rng.choice(GAP_MULTIPLIERS, size=changed_count, replace=True)
            schedule.append(self.base_gap * multipliers)
        return np.array(schedule)

    def desired_gaps_at(self, t):
        period_index = min(int(t // GAP_SWITCH_PERIOD), len(self.gap_schedule) - 1)
        return self.gap_schedule[period_index]

    def pack_state(self):
        vehicle_states = [vehicle.get_state() for vehicle in self.target_vehicles]
        return np.concatenate(vehicle_states + [self.ego.get_state(), self.z0, self.mu0, self.gap_error_integral0])

    def apply_state(self, state):
        for index, vehicle in enumerate(self.target_vehicles):
            vehicle.set_state(state[5 * index: 5 * (index + 1)])

        ego_start = 5 * len(self.target_vehicles)
        self.ego.set_state(state[ego_start: ego_start + 5])
        self.z0 = state[ego_start + 5: ego_start + 5 + self.num_gaps].copy()
        self.mu0 = state[ego_start + 5 + self.num_gaps: ego_start + 5 + 2 * self.num_gaps].copy()
        self.gap_error_integral0 = state[
            ego_start + 5 + 2 * self.num_gaps: ego_start + 5 + 3 * self.num_gaps
        ].copy()

    def _split_state(self, state):
        target_states = [
            state[5 * index: 5 * (index + 1)]
            for index in range(len(self.target_vehicles))
        ]
        ego_start = 5 * len(self.target_vehicles)
        ego_state = state[ego_start: ego_start + 5]
        z = state[ego_start + 5: ego_start + 5 + self.num_gaps]
        mu = state[ego_start + 5 + self.num_gaps: ego_start + 5 + 2 * self.num_gaps]
        gap_error_integral = state[
            ego_start + 5 + 2 * self.num_gaps: ego_start + 5 + 3 * self.num_gaps
        ]
        return target_states, ego_state, z, mu, gap_error_integral

    def _gap_pair_data(self, front_state, rear_state, ego_state):
        p_front = front_position(front_state, VEHICLE_L)
        p_rear = front_position(rear_state, VEHICLE_L)
        p_ego = front_position(ego_state, self.ego.L)

        v_front = front_velocity(front_state, VEHICLE_L)
        v_rear = front_velocity(rear_state, VEHICLE_L)
        v_ego = front_velocity(ego_state, self.ego.L)

        e_front = p_ego - p_front
        e_rear = p_ego - p_rear
        v_ego_front = v_ego - v_front
        v_ego_rear = v_ego - v_rear

        e_rear_front = p_rear - p_front
        v_rear_front = v_rear - v_front

        return {
            "p_front": p_front,
            "p_rear": p_rear,
            "p_ego": p_ego,
            "v_front": v_front,
            "v_rear": v_rear,
            "v_ego": v_ego,
            "e_front": e_front,
            "e_rear": e_rear,
            "v_ego_front": v_ego_front,
            "v_ego_rear": v_ego_rear,
            "e_rear_front": e_rear_front,
            "v_rear_front": v_rear_front,
        }

    def compute_gap_updates(self, target_states, ego_state, z, mu, gap_error_integral, t):
        mu_dot = np.zeros(self.num_gaps)
        z_dot = np.zeros(self.num_gaps)
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
                GAP_PID_KP * gap_error
                + GAP_PID_KI * gap_error_integral[gap_index]
                + GAP_PID_KD * gap_error_derivative,
                -GAP_ACCEL_LIMIT,
                GAP_ACCEL_LIMIT,
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
            z_dot[gap_index] = (1.0 / self.eps) * (-z[gap_index] ** 2 + mu[gap_index] * z[gap_index])

            w = np.tanh(self.k_w * z[gap_index])
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
            "z_dot": z_dot,
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
        target_states, ego_state, z, mu, gap_error_integral = self._split_state(state)
        updates = self.compute_gap_updates(target_states, ego_state, z, mu, gap_error_integral, t)

        target_derivatives = [rear_state_derivative(target_states[0], 0.0, 0.0, VEHICLE_L)]
        for gap_index in range(self.num_gaps):
            rear_state = target_states[gap_index + 1]
            target_derivatives.append(
                rear_state_derivative(rear_state, updates["gap_pid_accels"][gap_index], 0.0, VEHICLE_L)
            )

        # Test mode: all four gap controls are computed above, but the real ego input is disabled.
        # To enable actual control later, select one row from updates["u_total"] and convert it
        # through ego.u_to_physical_inputs(...).
        ego_derivative = rear_state_derivative(ego_state, 0.0, 0.0, self.ego.L)

        return np.concatenate(
            target_derivatives
            + [ego_derivative, updates["z_dot"], updates["mu_dot"], updates["gap_error_integral_dot"]]
        )

    def diagnostics(self, state, t=0.0):
        target_states, ego_state, z, mu, gap_error_integral = self._split_state(state)
        updates = self.compute_gap_updates(target_states, ego_state, z, mu, gap_error_integral, t)
        return {
            "target_states": target_states,
            "ego_state": ego_state,
            "z": z,
            "mu": mu,
            "gap_error_integral": gap_error_integral,
            **updates,
        }


def build_scene():
    target_lane_y = LANE_WIDTH * 1.5
    ego_lane_y = LANE_WIDTH * 0.5

    temp_ego = EgoVehicleOdeModel(id="ego_template", L=VEHICLE_L)
    target_gap = max(VEHICLE_L + 0.5, 3.2 * temp_ego.r)
    base_x = 36.0

    target_vehicles = []
    colors = ["lightblue", "cornflowerblue", "royalblue", "steelblue", "deepskyblue"]
    for index in range(NUM_TARGET_VEHICLES):
        target_vehicles.append(
            KinematicBicycleModel(
                id=f"Veh {index + 1}",
                x=base_x - index * target_gap,
                y=target_lane_y,
                v=TARGET_SPEED,
                L=VEHICLE_L,
                color=colors[index % len(colors)],
            )
        )

    ego = EgoVehicleOdeModel(
        id="Ego Test",
        x=base_x - 2.5 * target_gap,
        y=ego_lane_y,
        v=TARGET_SPEED,
        L=VEHICLE_L,
        color="lightgreen",
    )

    return target_vehicles, ego, target_gap


def draw_multigap_scene(ax_scene, ax_mu, ax_z, dynamics, state, histories, target_gap, current_time):
    diagnostics = dynamics.diagnostics(state, current_time)
    dynamics.apply_state(state)

    ax_scene.cla()
    draw_environment(ax_scene, LANE_WIDTH)
    for vehicle in dynamics.target_vehicles:
        draw_car(ax_scene, vehicle, wheelbase=dynamics.r)
    draw_car(ax_scene, dynamics.ego, wheelbase=dynamics.r)

    ego_x = dynamics.ego.x
    ax_scene.set_xlim(ego_x - 15, ego_x + 45)
    ax_scene.set_ylim(-2, LANE_WIDTH * 2 + 2)
    ax_scene.set_aspect("equal")
    ax_scene.set_title(
        f"Five target-lane vehicles | base gap={target_gap:.2f}m | switch period={GAP_SWITCH_PERIOD:.1f}s | ego control disabled"
    )

    ax_mu.cla()
    for gap_index in range(dynamics.num_gaps):
        ax_mu.plot(histories["time"], histories["mu"][:, gap_index], linewidth=2, label=f"mu gap {gap_index + 1}")
    ax_mu.axhline(0, color="gray", linestyle="--")
    ax_mu.set_xlim(0, SIM_TIME)
    ax_mu.set_title("Bifurcation Parameters for Four Gaps")
    ax_mu.legend(loc="upper left")
    ax_mu.grid(True)

    ax_z.cla()
    for gap_index in range(dynamics.num_gaps):
        ax_z.plot(histories["time"], histories["z"][:, gap_index], linewidth=2, label=f"z gap {gap_index + 1}")
    ax_z.axhline(0, color="gray", linestyle="--")
    ax_z.set_xlim(0, SIM_TIME)
    ax_z.set_title("Opinion States for Four Gaps")
    ax_z.legend(loc="upper left")
    ax_z.grid(True)


def main():
    target_vehicles, ego, target_gap = build_scene()
    dynamics = MultiGapOdeDynamics(target_vehicles, ego, target_gap)
    state = dynamics.pack_state()
    print("Dynamic desired-gap schedule by period:")
    for period_index, desired_gaps in enumerate(dynamics.gap_schedule):
        period_start = period_index * GAP_SWITCH_PERIOD
        period_end = period_start + GAP_SWITCH_PERIOD
        print(f"  {period_start:5.1f}s - {period_end:5.1f}s: {np.round(desired_gaps, 3)}")

    histories = {
        "time": np.empty((0,), dtype=float),
        "mu": np.empty((0, dynamics.num_gaps), dtype=float),
        "z": np.empty((0, dynamics.num_gaps), dtype=float),
    }

    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax_scene = plt.subplot(2, 1, 1)
    ax_mu = plt.subplot(2, 2, 3)
    ax_z = plt.subplot(2, 2, 4)

    for step_index in range(int(SIM_TIME / DT)):
        t = step_index * DT
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
        diagnostics = dynamics.diagnostics(state, t + DT)

        histories["time"] = np.append(histories["time"], t + DT)
        histories["mu"] = np.vstack([histories["mu"], diagnostics["mu"]])
        histories["z"] = np.vstack([histories["z"], diagnostics["z"]])

        if step_index % 4 == 0:
            draw_multigap_scene(ax_scene, ax_mu, ax_z, dynamics, state, histories, target_gap, t + DT)
            plt.pause(0.01)

    plt.ioff()
    plt.show()

    final_diagnostics = dynamics.diagnostics(state, SIM_TIME)
    print("Final gap distances:", np.round(final_diagnostics["gap_distances"], 3))
    print("Final desired gaps:", np.round(final_diagnostics["desired_gaps"], 3))
    print("Final gap errors:", np.round(final_diagnostics["gap_errors"], 3))
    print("Final PID accelerations:", np.round(final_diagnostics["gap_pid_accels"], 3))
    print("Final mu:", np.round(final_diagnostics["mu"], 3))
    print("Final z:", np.round(final_diagnostics["z"], 3))
    print("Computed u_total for each gap, not applied to ego:")
    print(np.round(final_diagnostics["u_total"], 3))


if __name__ == "__main__":
    main()
