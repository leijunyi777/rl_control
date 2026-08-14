import importlib.util
import os
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from models_ode import (
    EgoVehicleOdeModel,
    KinematicBicycleModel,
    compute_gap_bias_bt,
    compute_gap_confidence_signals_from_states,
    compute_gap_opinion_z_dot,
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
TARGET_SPEED = 15.0

NUM_TARGET_VEHICLES = 5
BASE_GAP = 8.0
GAP_SWITCH_PERIOD = 4.0
GAP_MULTIPLIERS = np.array([0.75, 1.0, 1.25, 1.5])
MAX_CHANGED_GAPS_PER_PERIOD = 2
GAP_RANDOM_SEED = 7788
GAP_PID_KP = 0.55
GAP_PID_KD = 1.05
GAP_ACCEL_LIMIT = 4.0

EGO_X_BASE = 20.0
EGO_X_RANDOM_RANGE = 1.0
EGO_RANDOM_SEED = None

USE_RL_U = True
MAIN12_POLICY_PATH = "main12_sac_u_policy.pth"
RL_U_LOW = 0.0
RL_U_HIGH = 3.0

GAP_SAFE = 5.0
K_GAP = 0.2
K_VEL = 0.1
U_BASE = 0.2
U_AMP = 2.5
SIGMA_D = 2.0
SIGMA_V = 1.5
Z_DAMPING = 2.0
Z_ALPHA = 2.0

HIGH_DAMPING = 0.5
HIGH_ALPHA = 5.0
HIGH_U_TAU = 1.0
HIGH_U_MAX = 2.0
HIGH_HILL_K = 0.15
HIGH_HILL_N = 2.0
HIGH_DECISION_THRESHOLD = 0.18

ENABLE_GLOBAL_UC = True
CONTROL_ACCEL_LIMIT = 5.0
CONTROL_STEER_RATE_LIMIT = 0.8
STOP_ON_COLLISION = True
STOP_ON_SUCCESS = True

EXPORT_FPS = 20
EXPORT_FRAME_STRIDE = 2
SHOW_AFTER_EXPORT = True


class Main12PolicyWrapper:
    def __init__(self, policy_path=MAIN12_POLICY_PATH):
        self.policy_path = policy_path
        self.agent = None
        self.device = None
        self.u_low = RL_U_LOW
        self.u_high = RL_U_HIGH

    def _load(self):
        if self.agent is not None:
            return
        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(
                f"Cannot find {self.policy_path}. Set USE_RL_U=False or train main12_sac_train.py first."
            )
        import torch

        spec = importlib.util.spec_from_file_location(
            "main12_sac_train_module",
            Path(__file__).with_name("main12_sac_train.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        checkpoint = torch.load(self.policy_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = module.SACAgent(int(checkpoint["state_dim"]), int(checkpoint["action_dim"]), self.device)
        self.agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.agent.policy.eval()
        self.u_low = float(checkpoint.get("u_low", RL_U_LOW))
        self.u_high = float(checkpoint.get("u_high", RL_U_HIGH))

    def predict_u(self, obs):
        self._load()
        normalized_action = self.agent.select_action(obs.astype(np.float32), evaluate=True)
        ratio = 0.5 * (float(normalized_action[0]) + 1.0)
        return float(self.u_low + ratio * (self.u_high - self.u_low))


def sample_ego_x(rng):
    return float(EGO_X_BASE + rng.uniform(-EGO_X_RANDOM_RANGE, EGO_X_RANDOM_RANGE))


def make_car_from_state(car_id, state, color, wheelbase):
    car = KinematicBicycleModel(id=car_id, L=wheelbase, color=color)
    car.set_state(state)
    return car


def nearest_three_indices(target_states, ego_state):
    ego_x = front_position(ego_state, VEHICLE_L)[0]
    target_x = np.array([front_position(state, VEHICLE_L)[0] for state in target_states])
    closest = np.argsort(np.abs(target_x - ego_x))[:3]
    return sorted(closest, key=lambda idx: target_x[idx], reverse=True)


def confidence_to_gap(ego_state, front_state, rear_state):
    ego_pos = front_position(ego_state, VEHICLE_L)
    front_pos = front_position(front_state, VEHICLE_L)
    rear_pos = front_position(rear_state, VEHICLE_L)
    ego_vel = front_velocity(ego_state, VEHICLE_L)
    front_vel = front_velocity(front_state, VEHICLE_L)
    rear_vel = front_velocity(rear_state, VEHICLE_L)

    x_gap = 0.5 * (front_pos[0] + rear_pos[0])
    v_gap = 0.5 * (front_vel[0] + rear_vel[0])
    d_gap = float(x_gap - ego_pos[0])
    dv_gap = float(ego_vel[0] - v_gap)
    if d_gap < 0.0:
        confidence = 0.0
    else:
        exponent = -0.5 * (d_gap / max(SIGMA_D, 1e-6)) ** 2 - 0.5 * (dv_gap / max(SIGMA_V, 1e-6)) ** 2
        confidence = float(np.exp(np.clip(exponent, -700.0, 0.0)))
    return {
        "x_gap": float(x_gap),
        "v_gap": float(v_gap),
        "d_gap": d_gap,
        "dv_gap": dv_gap,
        "confidence": confidence,
    }


def build_pair_obs(ego_state, front_state, rear_state):
    p_ego = front_position(ego_state, VEHICLE_L)
    v_ego = front_velocity(ego_state, VEHICLE_L)
    p_front = front_position(front_state, VEHICLE_L)
    v_front = front_velocity(front_state, VEHICLE_L)
    p_rear = front_position(rear_state, VEHICLE_L)
    v_rear = front_velocity(rear_state, VEHICLE_L)
    obs = np.concatenate([
        p_ego - p_front,
        v_ego - v_front,
        p_ego - p_rear,
        v_ego - v_rear,
    ]).astype(np.float32)
    obs_scale = np.array([40.0, 8.0, 20.0, 10.0, 40.0, 8.0, 20.0, 10.0], dtype=np.float32)
    return np.clip(obs / obs_scale, -5.0, 5.0).astype(np.float32)


class Main13HighLevelDynamics:
    def __init__(self, target_vehicles, ego, enable_ego_control=False, use_rl_u=USE_RL_U):
        if len(target_vehicles) < 3:
            raise ValueError("Main13 requires at least 3 target-lane vehicles.")
        self.target_vehicles = target_vehicles
        self.ego = ego
        self.enable_ego_control = enable_ego_control
        self.use_rl_u = use_rl_u
        self.num_gaps = len(target_vehicles) - 1
        self.rng = np.random.default_rng(GAP_RANDOM_SEED)
        self.desired_gap_schedule = self._build_gap_schedule()
        self.policy = Main12PolicyWrapper() if use_rl_u else None

    def _build_gap_schedule(self):
        num_periods = int(np.ceil(SIM_TIME / GAP_SWITCH_PERIOD)) + 2
        schedule = np.full((num_periods, self.num_gaps), BASE_GAP, dtype=float)
        for period_index in range(num_periods):
            changed_count = int(self.rng.integers(0, min(MAX_CHANGED_GAPS_PER_PERIOD, self.num_gaps) + 1))
            if changed_count > 0:
                changed = self.rng.choice(self.num_gaps, size=changed_count, replace=False)
                schedule[period_index, changed] = BASE_GAP * self.rng.choice(GAP_MULTIPLIERS, size=changed_count)
        return schedule

    def desired_gaps_at(self, t):
        period_index = min(int(t // GAP_SWITCH_PERIOD), len(self.desired_gap_schedule) - 1)
        return self.desired_gap_schedule[period_index]

    def pack_state(self):
        return np.concatenate([vehicle.get_state() for vehicle in self.target_vehicles] + [self.ego.get_state(), np.array([0.0, 0.0, 0.01])])

    def apply_state(self, state):
        for index, vehicle in enumerate(self.target_vehicles):
            vehicle.set_state(state[5 * index: 5 * (index + 1)])
        ego_start = 5 * len(self.target_vehicles)
        self.ego.set_state(state[ego_start: ego_start + 5])

    def _split_state(self, state):
        target_states = [state[5 * index: 5 * (index + 1)] for index in range(len(self.target_vehicles))]
        ego_start = 5 * len(self.target_vehicles)
        ego_state = state[ego_start: ego_start + 5]
        y_high, u_high, z_low = state[ego_start + 5: ego_start + 8]
        return target_states, ego_state, float(y_high), float(u_high), float(z_low)

    def _target_accels(self, target_states, t):
        desired_gaps = self.desired_gaps_at(t)
        accels = np.zeros(len(target_states))
        gap_values = np.zeros(self.num_gaps)
        for gap_index in range(self.num_gaps):
            front_state = target_states[gap_index]
            rear_state = target_states[gap_index + 1]
            p_front = front_position(front_state, VEHICLE_L)
            p_rear = front_position(rear_state, VEHICLE_L)
            v_front = front_velocity(front_state, VEHICLE_L)
            v_rear = front_velocity(rear_state, VEHICLE_L)
            gap = p_front[0] - p_rear[0]
            gap_error = gap - desired_gaps[gap_index]
            gap_error_dot = v_front[0] - v_rear[0]
            accels[gap_index + 1] = np.clip(
                GAP_PID_KP * gap_error + GAP_PID_KD * gap_error_dot,
                -GAP_ACCEL_LIMIT,
                GAP_ACCEL_LIMIT,
            )
            gap_values[gap_index] = gap
        return accels, desired_gaps, gap_values

    def _global_safe_control(self, target_states, ego_state):
        p_ego = front_position(ego_state, VEHICLE_L)
        v_ego = front_velocity(ego_state, VEHICLE_L)
        u_c = np.zeros(2)
        distances = np.zeros(len(target_states))
        for index, target_state in enumerate(target_states):
            p_target = front_position(target_state, VEHICLE_L)
            v_target = front_velocity(target_state, VEHICLE_L)
            rel = p_ego - p_target
            rel_v = v_ego - v_target
            dist = np.linalg.norm(rel)
            g = rel / _signed_safe(dist)
            clearance = dist - self.ego.r
            phi = np.dot(g, rel_v) / _signed_safe(clearance)
            u_c += -self.ego.k_o * g * phi
            distances[index] = dist
        return u_c, distances

    def _high_level(self, target_states, ego_state, y_high, u_high):
        triple = nearest_three_indices(target_states, ego_state)
        front_pair = (triple[0], triple[1])
        rear_pair = (triple[1], triple[2])
        front_conf = confidence_to_gap(ego_state, target_states[front_pair[0]], target_states[front_pair[1]])
        rear_conf = confidence_to_gap(ego_state, target_states[rear_pair[0]], target_states[rear_pair[1]])
        b_high = front_conf["confidence"] - rear_conf["confidence"]
        y_dot = -HIGH_DAMPING * y_high + u_high * np.tanh(HIGH_ALPHA * y_high) + b_high
        y_power = max(y_high * y_high, 0.0)
        hill_input = y_power ** HIGH_HILL_N
        hill = HIGH_U_MAX * hill_input / _signed_safe(HIGH_HILL_K ** HIGH_HILL_N + hill_input)
        u_high_dot = (-u_high + hill) / max(HIGH_U_TAU, 1e-6)

        if y_high > HIGH_DECISION_THRESHOLD:
            decision = "FORWARD"
            selected_pair = front_pair
            selected_conf = front_conf
        elif y_high < -HIGH_DECISION_THRESHOLD:
            decision = "BACKWARD"
            selected_pair = rear_pair
            selected_conf = rear_conf
        else:
            decision = "WAIT"
            selected_pair = None
            selected_conf = None

        return {
            "triple": triple,
            "front_pair": front_pair,
            "rear_pair": rear_pair,
            "front_conf": front_conf,
            "rear_conf": rear_conf,
            "b_high": float(b_high),
            "y_dot": float(y_dot),
            "u_high_dot": float(u_high_dot),
            "decision": decision,
            "selected_pair": selected_pair,
            "selected_conf": selected_conf,
        }

    def _low_level(self, target_states, ego_state, z_low, high):
        if high["selected_pair"] is None:
            return {
                "z_dot": -Z_DAMPING * z_low,
                "b_t": 0.0,
                "u_t": 0.0,
                "formula_u_t": 0.0,
                "target_point": front_position(ego_state, VEHICLE_L).copy(),
                "control": np.zeros(2),
                "pair_signals": {},
            }

        front_index, rear_index = high["selected_pair"]
        front_state = target_states[front_index]
        rear_state = target_states[rear_index]
        signals = compute_gap_confidence_signals_from_states(
            ego_state,
            front_state,
            rear_state,
            VEHICLE_L,
            VEHICLE_L,
            VEHICLE_L,
            GAP_SAFE,
            k_gap=K_GAP,
            k_vel=K_VEL,
            u_base=U_BASE,
            u_amp=U_AMP,
            sigma_d=SIGMA_D,
            sigma_v=SIGMA_V,
        )
        formula_u_t = signals["u_t"]
        if self.use_rl_u:
            u_t = self.policy.predict_u(build_pair_obs(ego_state, front_state, rear_state))
        else:
            u_t = formula_u_t

        z_dot = compute_gap_opinion_z_dot(z_low, signals["b_t"], u_t, damping=Z_DAMPING, alpha=Z_ALPHA)
        w = np.tanh(self.ego.k_w * z_low)
        p_front = front_position(front_state, VEHICLE_L)
        p_rear = front_position(rear_state, VEHICLE_L)
        v_front = front_velocity(front_state, VEHICLE_L)
        v_rear = front_velocity(rear_state, VEHICLE_L)
        p_ego = front_position(ego_state, VEHICLE_L)
        v_ego = front_velocity(ego_state, VEHICLE_L)
        gap_center = 0.5 * (p_front + p_rear)
        gap_velocity = 0.5 * (v_front + v_rear)
        target_point = gap_center + self.ego.eta * ((1.0 - w) * self.ego.r_eta)
        control = -self.ego.k_p * (p_ego - target_point) - self.ego.k_v * (v_ego - gap_velocity)
        return {
            "z_dot": z_dot,
            "b_t": signals["b_t"],
            "u_t": u_t,
            "formula_u_t": formula_u_t,
            "target_point": target_point,
            "control": control,
            "pair_signals": signals,
        }

    def diagnostics(self, state, t=0.0):
        target_states, ego_state, y_high, u_high, z_low = self._split_state(state)
        accels, desired_gaps, gap_values = self._target_accels(target_states, t)
        high = self._high_level(target_states, ego_state, y_high, u_high)
        low = self._low_level(target_states, ego_state, z_low, high)
        u_c, ego_distances = self._global_safe_control(target_states, ego_state)
        min_distance = float(np.min(ego_distances))
        lane_error = abs(ego_state[1] - LANE_WIDTH * 1.5)
        success = lane_error <= 0.25 and min_distance > 1.5 * self.ego.r
        return {
            "target_states": target_states,
            "ego_state": ego_state,
            "y_high": y_high,
            "u_high": u_high,
            "z_low": z_low,
            "target_accels": accels,
            "desired_gaps": desired_gaps,
            "gap_values": gap_values,
            "ego_distances": ego_distances,
            "min_distance": min_distance,
            "collision": min_distance < self.ego.r,
            "success": success,
            "lane_error": lane_error,
            **high,
            **low,
        }

    def rhs(self, t, state):
        target_states, ego_state, y_high, u_high, z_low = self._split_state(state)
        diag = self.diagnostics(state, t)
        target_derivatives = [
            rear_state_derivative(target_states[index], diag["target_accels"][index], 0.0, VEHICLE_L)
            for index in range(len(target_states))
        ]
        if self.enable_ego_control and diag["decision"] != "WAIT":
            control = diag["control"].copy()
            if ENABLE_GLOBAL_UC:
                control += self._global_safe_control(target_states, ego_state)[0]
            a, omega = self.ego.u_to_physical_inputs(control, ego_state)
            a = np.clip(a, -CONTROL_ACCEL_LIMIT, CONTROL_ACCEL_LIMIT)
            omega = np.clip(omega, -CONTROL_STEER_RATE_LIMIT, CONTROL_STEER_RATE_LIMIT)
        else:
            a, omega = 0.0, 0.0
        ego_derivative = rear_state_derivative(ego_state, a, omega, self.ego.L)
        return np.concatenate(target_derivatives + [ego_derivative, np.array([diag["y_dot"], diag["u_high_dot"], diag["z_dot"]])])


def build_scene(enable_ego_control=False, use_rl_u=USE_RL_U):
    target_lane_y = LANE_WIDTH * 1.5
    ego_lane_y = LANE_WIDTH * 0.5
    target_vehicles = []
    colors = ["lightblue", "cornflowerblue", "royalblue", "steelblue", "deepskyblue", "skyblue", "dodgerblue"]
    base_x = 48.0
    for index in range(NUM_TARGET_VEHICLES):
        target_vehicles.append(
            KinematicBicycleModel(
                id=f"Veh {index + 1}",
                x=base_x - index * BASE_GAP,
                y=target_lane_y,
                v=TARGET_SPEED,
                L=VEHICLE_L,
                color=colors[index % len(colors)],
            )
        )

    ego_rng = np.random.default_rng(EGO_RANDOM_SEED)
    ego_x = sample_ego_x(ego_rng)
    ego = EgoVehicleOdeModel(id="Ego", x=ego_x, y=ego_lane_y, v=TARGET_SPEED, L=VEHICLE_L, color="lightgreen")
    dynamics = Main13HighLevelDynamics(target_vehicles, ego, enable_ego_control=enable_ego_control, use_rl_u=use_rl_u)
    return dynamics


def snapshot(t, state, diag):
    return {
        "time": float(t),
        "target_states": [item.copy() for item in diag["target_states"]],
        "ego_state": diag["ego_state"].copy(),
        "decision": diag["decision"],
        "triple": list(diag["triple"]),
        "selected_pair": None if diag["selected_pair"] is None else list(diag["selected_pair"]),
        "target_point": diag["target_point"].copy(),
        "y_high": float(diag["y_high"]),
        "u_high": float(diag["u_high"]),
        "z_low": float(diag["z_low"]),
        "b_high": float(diag["b_high"]),
        "cf": float(diag["front_conf"]["confidence"]),
        "cr": float(diag["rear_conf"]["confidence"]),
        "b_t": float(diag["b_t"]),
        "u_t": float(diag["u_t"]),
        "formula_u_t": float(diag["formula_u_t"]),
        "min_distance": float(diag["min_distance"]),
        "collision": bool(diag["collision"]),
        "success": bool(diag["success"]),
    }


def draw_frame(ax_scene, ax_high, ax_low, frame, frames, enable_ego_control):
    ax_scene.cla()
    draw_environment(ax_scene, LANE_WIDTH)
    for index, target_state in enumerate(frame["target_states"]):
        color = "gold" if index in frame["triple"] else ["lightblue", "cornflowerblue", "royalblue", "steelblue", "deepskyblue", "skyblue"][index % 6]
        car = make_car_from_state(f"Veh {index + 1}", target_state, color, VEHICLE_L)
        draw_car(ax_scene, car, wheelbase=1.5)
    ego = make_car_from_state("Ego", frame["ego_state"], "lightgreen", VEHICLE_L)
    draw_car(ax_scene, ego, wheelbase=1.5)
    tp = frame["target_point"]
    ax_scene.plot(tp[0], tp[1], marker="*", color="red", markersize=16, markeredgecolor="black")
    ax_scene.text(tp[0], tp[1] + 0.7, "Target Point", color="red", ha="center", fontsize=9)
    if frame["selected_pair"] is not None:
        pair_text = f"pair {frame['selected_pair'][0] + 1}-{frame['selected_pair'][1] + 1}"
    else:
        pair_text = "no target pair"
    ax_scene.set_xlim(ego.x - 22, ego.x + 52)
    ax_scene.set_ylim(-2, LANE_WIDTH * 2 + 2)
    ax_scene.set_aspect("equal")
    mode = "MOVE" if enable_ego_control else "NOMOVE"
    ax_scene.set_title(
        f"Main13 {mode} | t={frame['time']:.2f}s | decision={frame['decision']} | {pair_text} | "
        f"use_u={'RL' if USE_RL_U else 'formula'}"
    )

    times = [item["time"] for item in frames]
    y_hist = [item["y_high"] for item in frames]
    uh_hist = [item["u_high"] for item in frames]
    b_hist = [item["b_high"] for item in frames]
    cf_hist = [item["cf"] for item in frames]
    cr_hist = [item["cr"] for item in frames]

    ax_high.cla()
    ax_high.plot(times, y_hist, "b-", linewidth=2, label="High opinion y")
    ax_high.plot(times, uh_hist, "purple", linewidth=2, label="High attention u")
    ax_high.plot(times, b_hist, "k--", linewidth=1.5, label="B=Cf-Cr")
    ax_high.plot(times, cf_hist, "g:", linewidth=2, label="Cf")
    ax_high.plot(times, cr_hist, "r:", linewidth=2, label="Cr")
    ax_high.axhline(HIGH_DECISION_THRESHOLD, color="gray", linestyle="--")
    ax_high.axhline(-HIGH_DECISION_THRESHOLD, color="gray", linestyle="--")
    ax_high.set_xlim(0, SIM_TIME)
    ax_high.set_title("High-level Decision Dynamics")
    ax_high.legend(loc="upper left")
    ax_high.grid(True)

    z_hist = [item["z_low"] for item in frames]
    bt_hist = [item["b_t"] for item in frames]
    ut_hist = [item["u_t"] for item in frames]
    min_dist_hist = [item["min_distance"] for item in frames]
    ax_low.cla()
    ax_low.plot(times, z_hist, "m-", linewidth=2, label="Low opinion z")
    ax_low.plot(times, bt_hist, "g--", linewidth=1.8, label="Low b(t)")
    ax_low.plot(times, ut_hist, "purple", linestyle="--", linewidth=1.8, label="Low u(t)")
    ax_low.plot(times, min_dist_hist, "tab:red", linewidth=1.8, label="Min distance")
    ax_low.axhline(1.5, color="black", linestyle=":", linewidth=1.5, label="Collision r")
    ax_low.set_xlim(0, SIM_TIME)
    ax_low.set_title("Low-level Main12 Decision Signals")
    ax_low.legend(loc="upper right")
    ax_low.grid(True)


def run_main13(enable_ego_control=False, export_animation_enabled=False, export_path="main13.gif"):
    dynamics = build_scene(enable_ego_control=enable_ego_control, use_rl_u=USE_RL_U)
    state = dynamics.pack_state()
    frames = []

    print(f"Main13 target vehicle count: {NUM_TARGET_VEHICLES}")
    print(f"Base gap: {BASE_GAP:.3f} m | switch period: {GAP_SWITCH_PERIOD:.3f} s")
    print(f"Ego initial x: {dynamics.ego.x:.3f} m")
    print(f"Ego control enabled: {enable_ego_control}")
    print(f"Low-level u source: {'main12 RL policy' if USE_RL_U else 'RBF equation'}")
    print("Desired-gap schedule:")
    for period_index, gaps in enumerate(dynamics.desired_gap_schedule[: int(np.ceil(SIM_TIME / GAP_SWITCH_PERIOD)) + 1]):
        print(f"  {period_index * GAP_SWITCH_PERIOD:5.1f}s: {np.round(gaps, 3)}")

    plt.ion()
    fig = plt.figure(figsize=(14, 9))
    ax_scene = plt.subplot(2, 1, 1)
    ax_high = plt.subplot(2, 2, 3)
    ax_low = plt.subplot(2, 2, 4)

    for step_index in range(int(SIM_TIME / DT)):
        t = step_index * DT
        sol = solve_ivp(dynamics.rhs, (t, t + DT), state, method="RK45", rtol=1e-6, atol=1e-8, max_step=DT / 5.0)
        if not sol.success:
            raise RuntimeError(sol.message)
        state = sol.y[:, -1]
        dynamics.apply_state(state)
        state = np.concatenate([vehicle.get_state() for vehicle in dynamics.target_vehicles] + [dynamics.ego.get_state(), state[-3:]])
        diag = dynamics.diagnostics(state, t + DT)
        if step_index % EXPORT_FRAME_STRIDE == 0 or diag["collision"] or diag["success"]:
            frames.append(snapshot(t + DT, state, diag))
        if step_index % 4 == 0 or diag["collision"] or diag["success"]:
            draw_frame(ax_scene, ax_high, ax_low, frames[-1], frames, enable_ego_control)
            plt.pause(0.01)
        if diag["collision"]:
            print(f"Collision at t={t + DT:.2f}s, min distance={diag['min_distance']:.3f}m")
            if STOP_ON_COLLISION:
                break
        if diag["success"]:
            print(f"Success at t={t + DT:.2f}s, decision={diag['decision']}")
            if STOP_ON_SUCCESS:
                break

    plt.ioff()
    if export_animation_enabled and frames:
        ani_fig = plt.figure(figsize=(14, 9))
        ani_ax_scene = plt.subplot(2, 1, 1)
        ani_ax_high = plt.subplot(2, 2, 3)
        ani_ax_low = plt.subplot(2, 2, 4)

        def animate(index):
            draw_frame(ani_ax_scene, ani_ax_high, ani_ax_low, frames[index], frames[: index + 1], enable_ego_control)

        ani = animation.FuncAnimation(ani_fig, animate, frames=len(frames), interval=1000 / EXPORT_FPS)
        extension = os.path.splitext(export_path)[1].lower()
        writer = animation.PillowWriter(fps=EXPORT_FPS) if extension == ".gif" else animation.FFMpegWriter(fps=EXPORT_FPS)
        ani.save(export_path, writer=writer)
        print(f"Saved animation to {export_path}")
        if SHOW_AFTER_EXPORT:
            plt.show()
        else:
            plt.close(ani_fig)
    else:
        plt.show()
