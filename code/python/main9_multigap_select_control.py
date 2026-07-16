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
RANDOM_SEED = None

EGO_INITIAL_GAP_FACTOR = 2.5
EGO_RANDOM_POSITION_ENABLED = True
EGO_RANDOM_X_OFFSET_RANGE = (-2.5, 2.5)
EGO_RANDOM_Y_OFFSET_RANGE = (0.0, 0.0)
EGO_RANDOM_POSITION_SEED = None

ENABLE_EGO_CONTROL = True
ENABLE_GLOBAL_UC = True
STOP_ON_COLLISION = True
STOP_ON_SUCCESS = True

SUCCESS_LATERAL_TOL = 0.25
SUCCESS_MIN_DISTANCE_FACTOR = 0.1

GAP_PID_KP = 0.8
GAP_PID_KI = 0.03
GAP_PID_KD = 1.2
GAP_ACCEL_LIMIT = 4.0

CONTROL_ACCEL_LIMIT = 5.0
CONTROL_STEER_RATE_LIMIT = 0.8

SCORE_Z_WEIGHT = 4.0
SCORE_MU_WEIGHT = 1.0
SCORE_SPACE_WEIGHT = 1.5
SCORE_DISTANCE_WEIGHT = 0.08
SCORE_RISK_WEIGHT = 8.0


class MultiGapSelectDynamics:
    """五辆目标车、四个候选 gap 的实际并道控制仿真。"""

    def __init__(self, target_vehicles, ego, base_gap):
        self.target_vehicles = target_vehicles
        self.ego = ego
        self.num_gaps = len(target_vehicles) - 1
        self.base_gap = base_gap
        self.safe_margin = 2.5 * ego.r

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

        p_center = 0.5 * (p_front + p_rear)
        v_center = 0.5 * (v_front + v_rear)

        return {
            "p_front": p_front,
            "p_rear": p_rear,
            "p_ego": p_ego,
            "p_center": p_center,
            "v_front": v_front,
            "v_rear": v_rear,
            "v_ego": v_ego,
            "v_center": v_center,
            "e_front": p_ego - p_front,
            "e_rear": p_ego - p_rear,
            "v_ego_front": v_ego - v_front,
            "v_ego_rear": v_ego - v_rear,
            "e_rear_front": p_rear - p_front,
            "v_rear_front": v_rear - v_front,
        }

    def _compute_target_gap_accels(self, target_states, gap_error_integral, t):
        gap_error_integral_dot = np.zeros(self.num_gaps)
        gap_errors = np.zeros(self.num_gaps)
        gap_accels = np.zeros(self.num_gaps)
        desired_gaps = self.desired_gaps_at(t)

        for gap_index in range(self.num_gaps):
            front_state = target_states[gap_index]
            rear_state = target_states[gap_index + 1]
            p_front = front_position(front_state, VEHICLE_L)
            p_rear = front_position(rear_state, VEHICLE_L)
            v_front = front_velocity(front_state, VEHICLE_L)
            v_rear = front_velocity(rear_state, VEHICLE_L)

            gap_error = (p_front[0] - p_rear[0]) - desired_gaps[gap_index]
            gap_error_dot = v_front[0] - v_rear[0]
            gap_error_integral_dot[gap_index] = gap_error
            gap_errors[gap_index] = gap_error
            gap_accels[gap_index] = np.clip(
                GAP_PID_KP * gap_error
                + GAP_PID_KI * gap_error_integral[gap_index]
                + GAP_PID_KD * gap_error_dot,
                -GAP_ACCEL_LIMIT,
                GAP_ACCEL_LIMIT,
            )

        return gap_accels, gap_errors, gap_error_integral_dot, desired_gaps

    def _compute_global_safe_control(self, target_states, ego_state):
        p_ego = front_position(ego_state, self.ego.L)
        v_ego = front_velocity(ego_state, self.ego.L)

        u_c = np.zeros(2)
        distances = np.zeros(len(target_states))
        clearances = np.zeros(len(target_states))

        for index, target_state in enumerate(target_states):
            p_target = front_position(target_state, VEHICLE_L)
            v_target = front_velocity(target_state, VEHICLE_L)

            e_ego_target = p_ego - p_target
            v_ego_target = v_ego - v_target
            dist = np.linalg.norm(e_ego_target)
            g_ego_target = e_ego_target / _signed_safe(dist)
            clearance = dist - self.ego.r
            phi = np.dot(g_ego_target, v_ego_target) / _signed_safe(clearance)

            u_c += -self.ego.k_o * g_ego_target * phi
            distances[index] = dist
            clearances[index] = clearance

        return u_c, distances, clearances

    def compute_gap_updates(self, target_states, ego_state, z, mu, gap_error_integral, t):
        mu_dot = np.zeros(self.num_gaps)
        z_dot = np.zeros(self.num_gaps)
        u_nominal = np.zeros((self.num_gaps, 2))
        gap_scores = np.zeros(self.num_gaps)
        gap_distances = np.zeros(self.num_gaps)
        gap_center_distances = np.zeros(self.num_gaps)
        gap_risks = np.zeros(self.num_gaps)
        gap_centers = np.zeros((self.num_gaps, 2))

        for gap_index in range(self.num_gaps):
            data = self._gap_pair_data(target_states[gap_index], target_states[gap_index + 1], ego_state)

            g_front = data["e_front"] / _signed_safe(np.linalg.norm(data["e_front"]))
            g_rear = data["e_rear"] / _signed_safe(np.linalg.norm(data["e_rear"]))
            g_rear_front = data["e_rear_front"] / _signed_safe(np.linalg.norm(data["e_rear_front"]))

            gap_distance = np.linalg.norm(data["e_rear_front"])
            d_rear_front = gap_distance - self.ego.r
            phi_rear_front = np.dot(g_rear_front, data["v_rear_front"]) / _signed_safe(d_rear_front)

            tanh_arg = (
                -self.ego.k
                * np.dot(self.ego.rho, g_front)
                * np.dot(self.ego.rho, g_rear)
                * (d_rear_front - 2.0 * self.ego.r)
                * (phi_rear_front + self.ego.eps2)
            )
            mu_dot[gap_index] = -self.ego.k_mu * mu[gap_index] + np.tanh(tanh_arg)
            z_dot[gap_index] = (1.0 / self.ego.eps) * (-z[gap_index] ** 2 + mu[gap_index] * z[gap_index])

            w = np.tanh(self.ego.k_w * z[gap_index])
            p_goal = data["p_center"] + self.ego.eta * ((1.0 - w) * self.ego.r_eta)
            u_nominal[gap_index] = (
                -self.ego.k_p * (data["p_ego"] - p_goal)
                - self.ego.k_v * (data["v_ego"] - data["v_center"])
            )

            min_gap_vehicle_distance = min(np.linalg.norm(data["e_front"]), np.linalg.norm(data["e_rear"]))
            risk = max(0.0, (self.safe_margin - min_gap_vehicle_distance) / self.safe_margin) ** 2
            space_score = np.clip((gap_distance - 2.0 * self.ego.r) / _signed_safe(self.base_gap), 0.0, 2.0)
            center_distance = np.linalg.norm(data["p_ego"] - data["p_center"])

            gap_scores[gap_index] = (
                SCORE_Z_WEIGHT * z[gap_index]
                + SCORE_MU_WEIGHT * mu[gap_index]
                + SCORE_SPACE_WEIGHT * space_score
                - SCORE_DISTANCE_WEIGHT * center_distance
                - SCORE_RISK_WEIGHT * risk
            )
            gap_distances[gap_index] = gap_distance
            gap_center_distances[gap_index] = center_distance
            gap_risks[gap_index] = risk
            gap_centers[gap_index] = data["p_center"]

        gap_accels, gap_errors, gap_error_integral_dot, desired_gaps = self._compute_target_gap_accels(
            target_states,
            gap_error_integral,
            t,
        )
        u_c_global, ego_distances, ego_clearances = self._compute_global_safe_control(target_states, ego_state)
        selected_gap = int(np.argmax(gap_scores))

        return {
            "mu_dot": mu_dot,
            "z_dot": z_dot,
            "gap_error_integral_dot": gap_error_integral_dot,
            "gap_accels": gap_accels,
            "gap_errors": gap_errors,
            "desired_gaps": desired_gaps,
            "u_nominal": u_nominal,
            "u_c_global": u_c_global,
            "gap_scores": gap_scores,
            "gap_distances": gap_distances,
            "gap_center_distances": gap_center_distances,
            "gap_risks": gap_risks,
            "gap_centers": gap_centers,
            "ego_distances": ego_distances,
            "ego_clearances": ego_clearances,
            "selected_gap": selected_gap,
        }

    def rhs(self, t, state):
        target_states, ego_state, z, mu, gap_error_integral = self._split_state(state)
        updates = self.compute_gap_updates(target_states, ego_state, z, mu, gap_error_integral, t)

        target_derivatives = [rear_state_derivative(target_states[0], 0.0, 0.0, VEHICLE_L)]
        for gap_index in range(self.num_gaps):
            target_derivatives.append(
                rear_state_derivative(target_states[gap_index + 1], updates["gap_accels"][gap_index], 0.0, VEHICLE_L)
            )

        if ENABLE_EGO_CONTROL:
            selected_gap = updates["selected_gap"]
            u_total = updates["u_nominal"][selected_gap].copy()
            if ENABLE_GLOBAL_UC:
                u_total += updates["u_c_global"]
            a, omega = self.ego.u_to_physical_inputs(u_total, ego_state)
            a = np.clip(a, -CONTROL_ACCEL_LIMIT, CONTROL_ACCEL_LIMIT)
            omega = np.clip(omega, -CONTROL_STEER_RATE_LIMIT, CONTROL_STEER_RATE_LIMIT)
        else:
            a, omega = 0.0, 0.0

        ego_derivative = rear_state_derivative(ego_state, a, omega, self.ego.L)

        return np.concatenate(
            target_derivatives
            + [ego_derivative, updates["z_dot"], updates["mu_dot"], updates["gap_error_integral_dot"]]
        )

    def diagnostics(self, state, t=0.0):
        target_states, ego_state, z, mu, gap_error_integral = self._split_state(state)
        updates = self.compute_gap_updates(target_states, ego_state, z, mu, gap_error_integral, t)
        min_ego_distance = float(np.min(updates["ego_distances"]))
        target_lane_y = LANE_WIDTH * 1.5
        lane_error = abs(ego_state[1] - target_lane_y)
        success = (
            lane_error <= SUCCESS_LATERAL_TOL
            and min_ego_distance > SUCCESS_MIN_DISTANCE_FACTOR * self.ego.r
        )
        return {
            "target_states": target_states,
            "ego_state": ego_state,
            "z": z,
            "mu": mu,
            "gap_error_integral": gap_error_integral,
            "min_ego_distance": min_ego_distance,
            "lane_error": lane_error,
            "success": success,
            "collision": min_ego_distance < self.ego.r,
            **updates,
        }


def build_scene():
    target_lane_y = LANE_WIDTH * 1.5
    ego_lane_y = LANE_WIDTH * 0.5

    temp_ego = EgoVehicleOdeModel(id="ego_template", L=VEHICLE_L)
    base_gap = max(VEHICLE_L + 0.5, 3.2 * temp_ego.r)
    base_x = 38.0

    rng = np.random.default_rng(EGO_RANDOM_POSITION_SEED)
    if EGO_RANDOM_POSITION_ENABLED:
        ego_x_offset = rng.uniform(EGO_RANDOM_X_OFFSET_RANGE[0], EGO_RANDOM_X_OFFSET_RANGE[1])
        ego_y_offset = rng.uniform(EGO_RANDOM_Y_OFFSET_RANGE[0], EGO_RANDOM_Y_OFFSET_RANGE[1])
    else:
        ego_x_offset = 0.0
        ego_y_offset = 0.0

    ego_gap_factor = EGO_INITIAL_GAP_FACTOR + ego_x_offset
    ego_x = base_x - ego_gap_factor * base_gap
    ego_y = ego_lane_y + ego_y_offset

    colors = ["lightblue", "cornflowerblue", "royalblue", "steelblue", "deepskyblue"]
    target_vehicles = []
    for index in range(NUM_TARGET_VEHICLES):
        target_vehicles.append(
            KinematicBicycleModel(
                id=f"Veh {index + 1}",
                x=base_x - index * base_gap,
                y=target_lane_y,
                v=TARGET_SPEED,
                L=VEHICLE_L,
                color=colors[index % len(colors)],
            )
        )

    ego = EgoVehicleOdeModel(
        id="Ego",
        x=ego_x,
        y=ego_y,
        v=TARGET_SPEED,
        L=VEHICLE_L,
        color="lightgreen",
    )

    return target_vehicles, ego, base_gap


def draw_selected_gap(ax, diagnostics):
    selected_gap = diagnostics["selected_gap"]
    center = diagnostics["gap_centers"][selected_gap]
    ax.plot(center[0], center[1], marker="*", markersize=16, color="gold", markeredgecolor="black")
    ax.text(center[0], center[1] + 0.7, f"Gap {selected_gap + 1}", fontsize=10, color="black", ha="center")


def draw_multigap_scene(ax_scene, ax_score, ax_z, ax_dist, dynamics, state, histories, current_time):
    diagnostics = dynamics.diagnostics(state, current_time)
    dynamics.apply_state(state)

    ax_scene.cla()
    draw_environment(ax_scene, LANE_WIDTH)
    for vehicle in dynamics.target_vehicles:
        draw_car(ax_scene, vehicle, wheelbase=dynamics.ego.r)
    draw_car(ax_scene, dynamics.ego, wheelbase=dynamics.ego.r)
    draw_selected_gap(ax_scene, diagnostics)

    ego_x = dynamics.ego.x
    ax_scene.set_xlim(ego_x - 20, ego_x + 45)
    ax_scene.set_ylim(-2, LANE_WIDTH * 2 + 2)
    ax_scene.set_aspect("equal")
    ax_scene.set_title(
        f"Multi-gap selection control | t={current_time:.2f}s | selected gap={diagnostics['selected_gap'] + 1} | switch period={GAP_SWITCH_PERIOD:.1f}s"
    )

    ax_score.cla()
    for gap_index in range(dynamics.num_gaps):
        ax_score.plot(
            histories["time"],
            histories["score"][:, gap_index],
            linewidth=2,
            label=f"score gap {gap_index + 1}",
        )
    ax_score.set_xlim(0, SIM_TIME)
    ax_score.set_title("Gap Selection Scores")
    ax_score.legend(loc="upper left")
    ax_score.grid(True)

    ax_z.cla()
    for gap_index in range(dynamics.num_gaps):
        ax_z.plot(histories["time"], histories["z"][:, gap_index], linewidth=2, label=f"z gap {gap_index + 1}")
    ax_z.axhline(0, color="gray", linestyle="--")
    ax_z.set_xlim(0, SIM_TIME)
    ax_z.set_title("Opinion States")
    ax_z.legend(loc="upper left")
    ax_z.grid(True)

    ax_dist.cla()
    for vehicle_index in range(NUM_TARGET_VEHICLES):
        ax_dist.plot(
            histories["time"],
            histories["ego_distances"][:, vehicle_index],
            linewidth=2,
            label=f"Ego-Veh {vehicle_index + 1}",
        )
    ax_dist.axhline(dynamics.ego.r, color="red", linestyle="--", label="Collision threshold r")
    ax_dist.set_xlim(0, SIM_TIME)
    ax_dist.set_title("Distances to All Target Vehicles")
    ax_dist.legend(loc="upper right", fontsize=8)
    ax_dist.grid(True)


def main():
    target_vehicles, ego, base_gap = build_scene()
    dynamics = MultiGapSelectDynamics(target_vehicles, ego, base_gap)
    state = dynamics.pack_state()

    print(f"Base gap: {base_gap:.3f} m")
    print(f"Initial ego position: x={ego.x:.3f} m, y={ego.y:.3f} m")
    print(
        f"Ego initial gap factor: {(target_vehicles[0].x - ego.x) / base_gap:.3f} "
        f"(base={EGO_INITIAL_GAP_FACTOR:.3f})"
    )
    print(f"Ego control: {ENABLE_EGO_CONTROL}")
    print(f"Global u_c: {ENABLE_GLOBAL_UC}")
    print(f"Stop on success: {STOP_ON_SUCCESS}")
    print(
        f"Success condition: |ego_y - target_lane_y| <= {SUCCESS_LATERAL_TOL:.2f}m "
        f"and min distance > {SUCCESS_MIN_DISTANCE_FACTOR:.1f}r"
    )
    print("Gap score = 4*z + 1*mu + 1.5*space - 0.08*center_distance - 8*risk")
    print("Dynamic desired-gap schedule by period:")
    for period_index, desired_gaps in enumerate(dynamics.gap_schedule):
        period_start = period_index * GAP_SWITCH_PERIOD
        period_end = period_start + GAP_SWITCH_PERIOD
        print(f"  {period_start:5.1f}s - {period_end:5.1f}s: {np.round(desired_gaps, 3)}")

    histories = {
        "time": np.empty((0,), dtype=float),
        "z": np.empty((0, dynamics.num_gaps), dtype=float),
        "mu": np.empty((0, dynamics.num_gaps), dtype=float),
        "score": np.empty((0, dynamics.num_gaps), dtype=float),
        "selected_gap": np.empty((0,), dtype=int),
        "ego_distances": np.empty((0, NUM_TARGET_VEHICLES), dtype=float),
    }

    plt.ion()
    fig = plt.figure(figsize=(14, 9))
    ax_scene = plt.subplot(2, 1, 1)
    ax_score = plt.subplot(2, 3, 4)
    ax_z = plt.subplot(2, 3, 5)
    ax_dist = plt.subplot(2, 3, 6)

    collision_time = None
    success_time = None
    total_steps = int(SIM_TIME / DT)
    for step_index in range(total_steps):
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
        histories["z"] = np.vstack([histories["z"], diagnostics["z"]])
        histories["mu"] = np.vstack([histories["mu"], diagnostics["mu"]])
        histories["score"] = np.vstack([histories["score"], diagnostics["gap_scores"]])
        histories["selected_gap"] = np.append(histories["selected_gap"], diagnostics["selected_gap"])
        histories["ego_distances"] = np.vstack([histories["ego_distances"], diagnostics["ego_distances"]])

        if diagnostics["collision"]:
            collision_time = t + DT
            print(f"Collision detected at t={collision_time:.2f}s, min distance={diagnostics['min_ego_distance']:.3f}m")
            if STOP_ON_COLLISION:
                break

        if diagnostics["success"]:
            success_time = t + DT
            print(
                f"Lane merge success at t={success_time:.2f}s, "
                f"lane error={diagnostics['lane_error']:.3f}m, "
                f"min distance={diagnostics['min_ego_distance']:.3f}m"
            )
            if STOP_ON_SUCCESS:
                break

        if step_index % 4 == 0:
            draw_multigap_scene(ax_scene, ax_score, ax_z, ax_dist, dynamics, state, histories, t + DT)
            plt.pause(0.01)

    draw_multigap_scene(
        ax_scene,
        ax_score,
        ax_z,
        ax_dist,
        dynamics,
        state,
        histories,
        histories["time"][-1] if histories["time"].size else 0.0,
    )
    plt.ioff()
    plt.show()

    final_diagnostics = dynamics.diagnostics(state, histories["time"][-1] if histories["time"].size else 0.0)
    print("Final selected gap:", final_diagnostics["selected_gap"] + 1)
    print("Final gap scores:", np.round(final_diagnostics["gap_scores"], 3))
    print("Final desired gaps:", np.round(final_diagnostics["desired_gaps"], 3))
    print("Final gap errors:", np.round(final_diagnostics["gap_errors"], 3))
    print("Final z:", np.round(final_diagnostics["z"], 3))
    print("Final mu:", np.round(final_diagnostics["mu"], 3))
    print("Final ego distances:", np.round(final_diagnostics["ego_distances"], 3))
    print(f"Final lane error: {final_diagnostics['lane_error']:.3f} m")
    if success_time is not None:
        print(f"Stopped after successful lane merge at t={success_time:.2f}s.")
    if collision_time is None:
        print("No collision detected.")


if __name__ == "__main__":
    main()
