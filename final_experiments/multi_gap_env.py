"""最终版多 gap 环境：独立复刻最新 main13 高层意见动力学逻辑。

说明：
1. 不 import 原有 main13、models_ode、utils 文件。
2. 复用最终版 single_gap_env.py 中已经复制的自行车模型和控制器基础函数。
3. 运行本文件会显示一次带图仿真；评价脚本可导入 run_episode。
"""

from __future__ import annotations

import argparse
import csv
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

from single_gap_env import (
    ACTION_SMOOTH_PENALTY_GAIN,
    COLLISION_PENALTY,
    DIRECTION_FLIP_PENALTY,
    EgoVehicleOdeModel,
    HESITATION_PENALTY_GAIN,
    KinematicBicycleModel,
    LANE_PROGRESS_STEP_GAIN,
    OPPORTUNITY_PROGRESS_GAIN,
    PROGRESS_REWARD_GAIN,
    REVERSE_PROGRESS_PENALTY_GAIN,
    SAFETY_MARGIN_FACTOR,
    SAFETY_PENALTY_GAIN,
    SUCCESS_BONUS_BASE,
    SUCCESS_TIME_PENALTY_GAIN,
    TIMEOUT_PROGRESS_PENALTY_GAIN,
    TIME_PENALTY_GAIN,
    _signed_safe,
    compute_gap_bias_bt,
    compute_gap_confidence_signals_from_states,
    compute_gap_opinion_z_dot,
    draw_car,
    draw_environment,
    front_position,
    front_velocity,
    rear_state_derivative,
    require_matplotlib,
)


# =========================
# 与最新 main13_common.py 保持一致的参数
# =========================
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
GAP_RANDOM_SEED = 77888
GAP_PID_KP = 0.55
GAP_PID_KD = 1.05
GAP_ACCEL_LIMIT = 4.0

EGO_X_BASE = 30.0
EGO_X_RANDOM_RANGE = 10.0
EGO_RANDOM_SEED = None

RL_U_LOW = 0.0
RL_U_HIGH = 3.0

GAP_SAFE = 5.0
K_GAP = 0.2
K_VEL = 0.1
U_BASE = 0.2
U_AMP = 2.5
SIGMA_D = 4.0
SIGMA_V = 2.5
Z_DAMPING = 2.0
Z_ALPHA = 2.0

HIGH_DAMPING = 2.5
HIGH_ALPHA = 10.0
HIGH_U_TAU = 1.0
HIGH_U_MAX = 1.5
HIGH_HILL_K = 0.2
HIGH_HILL_N = 2.0
HIGH_DECISION_THRESHOLD = 0.18

ENABLE_GLOBAL_UC = True
CONTROL_ACCEL_LIMIT = 5.0
CONTROL_STEER_RATE_LIMIT = 0.8
STOP_ON_COLLISION = True
STOP_ON_SUCCESS = True

EXPORT_FRAME_STRIDE = 2


def sample_ego_x(rng):
    """采样多 gap 环境 ego 初始 x。"""
    return float(EGO_X_BASE + rng.uniform(-EGO_X_RANDOM_RANGE, EGO_X_RANDOM_RANGE))


def nearest_three_indices(target_states, ego_state):
    """按 ego 前轴纵向距离选择最近三辆目标车，并按从前到后排序。"""
    ego_x = front_position(ego_state, VEHICLE_L)[0]
    target_x = np.array([front_position(state, VEHICLE_L)[0] for state in target_states])
    closest = np.argsort(np.abs(target_x - ego_x))[:3]
    return sorted(closest, key=lambda idx: target_x[idx], reverse=True)


def confidence_to_gap(ego_state, front_state, rear_state):
    """计算高层候选 gap 置信度。"""
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
    exponent = -0.5 * (d_gap / max(SIGMA_D, 1e-6)) ** 2 - 0.5 * (dv_gap / max(SIGMA_V, 1e-6)) ** 2
    confidence = float(np.exp(np.clip(exponent, -700.0, 0.0)))
    return {"x_gap": float(x_gap), "v_gap": float(v_gap), "d_gap": d_gap, "dv_gap": dv_gap, "confidence": confidence}


def build_pair_obs(ego_state, front_state, rear_state):
    """构造与单 gap SAC policy 一致的 8 维观测。"""
    p_ego = front_position(ego_state, VEHICLE_L)
    v_ego = front_velocity(ego_state, VEHICLE_L)
    p_front = front_position(front_state, VEHICLE_L)
    v_front = front_velocity(front_state, VEHICLE_L)
    p_rear = front_position(rear_state, VEHICLE_L)
    v_rear = front_velocity(rear_state, VEHICLE_L)
    obs = np.concatenate([p_ego - p_front, v_ego - v_front, p_ego - p_rear, v_ego - v_rear]).astype(np.float32)
    obs_scale = np.array([40.0, 8.0, 20.0, 10.0, 40.0, 8.0, 20.0, 10.0], dtype=np.float32)
    return np.clip(obs / obs_scale, -5.0, 5.0).astype(np.float32)


def lane_progress(ego_state):
    """计算 ego 并道进度。"""
    return float(np.clip((ego_state[1] - LANE_WIDTH * 0.5) / LANE_WIDTH, 0.0, 1.0))


def normalized_u(u_t):
    """将真实 u(t) 映射到 [-1, 1]，用于 reward 动作平滑项。"""
    return float(np.clip(2.0 * (u_t - RL_U_LOW) / max(RL_U_HIGH - RL_U_LOW, 1e-6) - 1.0, -1.0, 1.0))


class Main13HighLevelDynamics:
    """复制 main13_common.Main13HighLevelDynamics，增加 max 高层基线开关。"""

    def __init__(self, target_vehicles, ego, enable_ego_control=True, low_level_policy=None, decision_method="opinion", gap_seed=GAP_RANDOM_SEED):
        if len(target_vehicles) < 3:
            raise ValueError("Main13 requires at least 3 target-lane vehicles.")
        self.target_vehicles = target_vehicles
        self.ego = ego
        self.enable_ego_control = enable_ego_control
        self.low_level_policy = low_level_policy
        self.decision_method = decision_method
        self.num_gaps = len(target_vehicles) - 1
        self.rng = np.random.default_rng(gap_seed)
        self.desired_gap_schedule = self._build_gap_schedule()

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
            accels[gap_index + 1] = np.clip(GAP_PID_KP * gap_error + GAP_PID_KD * gap_error_dot, -GAP_ACCEL_LIMIT, GAP_ACCEL_LIMIT)
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

        if self.decision_method == "max":
            if front_conf["confidence"] >= rear_conf["confidence"]:
                decision, selected_pair, selected_conf = "FORWARD", front_pair, front_conf
            else:
                decision, selected_pair, selected_conf = "BACKWARD", rear_pair, rear_conf
        elif y_high > HIGH_DECISION_THRESHOLD:
            decision, selected_pair, selected_conf = "FORWARD", front_pair, front_conf
        elif y_high < -HIGH_DECISION_THRESHOLD:
            decision, selected_pair, selected_conf = "BACKWARD", rear_pair, rear_conf
        else:
            decision, selected_pair, selected_conf = "WAIT", None, None

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
        if self.low_level_policy is None:
            u_t = formula_u_t
        else:
            u_t = self.low_level_policy(build_pair_obs(ego_state, front_state, rear_state), signals)
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
        return {"z_dot": z_dot, "b_t": signals["b_t"], "u_t": u_t, "formula_u_t": formula_u_t, "target_point": target_point, "control": control, "pair_signals": signals}

    def diagnostics(self, state, t=0.0):
        target_states, ego_state, y_high, u_high, z_low = self._split_state(state)
        accels, desired_gaps, gap_values = self._target_accels(target_states, t)
        high = self._high_level(target_states, ego_state, y_high, u_high)
        low = self._low_level(target_states, ego_state, z_low, high)
        _, ego_distances = self._global_safe_control(target_states, ego_state)
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
        target_states, ego_state, _, _, _ = self._split_state(state)
        diag = self.diagnostics(state, t)
        target_derivatives = [rear_state_derivative(target_states[index], diag["target_accels"][index], 0.0, VEHICLE_L) for index in range(len(target_states))]
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


def build_scene(gap_seed=GAP_RANDOM_SEED, ego_seed=EGO_RANDOM_SEED, decision_method="opinion", low_level_policy=None):
    """构建独立多 gap 场景。"""
    target_lane_y = LANE_WIDTH * 1.5
    ego_lane_y = LANE_WIDTH * 0.5
    colors = ["lightblue", "cornflowerblue", "royalblue", "steelblue", "deepskyblue"]
    target_vehicles = []
    for index in range(NUM_TARGET_VEHICLES):
        target_vehicles.append(KinematicBicycleModel(f"Veh {index + 1}", x=48.0 - index * BASE_GAP, y=target_lane_y, v=TARGET_SPEED, L=VEHICLE_L, color=colors[index % len(colors)]))
    ego_rng = np.random.default_rng(ego_seed)
    ego_x = sample_ego_x(ego_rng)
    ego = EgoVehicleOdeModel("Ego", x=ego_x, y=ego_lane_y, v=TARGET_SPEED, L=VEHICLE_L, color="lightgreen")
    return Main13HighLevelDynamics(target_vehicles, ego, enable_ego_control=True, low_level_policy=low_level_policy, decision_method=decision_method, gap_seed=gap_seed)


class MultiGapEnv:
    """评价脚本使用的环境包装器。"""

    def __init__(self, seed=None, gap_seed=None, ego_seed=None, decision_method="opinion", low_level_policy=None, render=False):
        if seed is not None:
            gap_seed = seed if gap_seed is None else gap_seed
            ego_seed = seed if ego_seed is None else ego_seed
        self.render_enabled = render
        self.dynamics = build_scene(gap_seed=gap_seed, ego_seed=ego_seed, decision_method=decision_method, low_level_policy=low_level_policy)
        self.state = self.dynamics.pack_state()
        self.t = 0.0
        self.ego_x0 = self.dynamics.ego.x
        self.prev_progress = lane_progress(self.dynamics.ego.get_state())
        self.prev_u_norm = 0.0
        self.prev_lateral_velocity = front_velocity(self.dynamics.ego.get_state(), VEHICLE_L)[1]
        self.prev_selected_pair = None
        self.switch_count = 0
        self.frames = []
        self.fig = None

    def step(self):
        t0 = self.t
        t1 = t0 + DT
        sol = solve_ivp(self.dynamics.rhs, (t0, t1), self.state, method="RK45", rtol=1e-6, atol=1e-8, max_step=DT / 5.0)
        if not sol.success:
            raise RuntimeError(sol.message)
        self.state = sol.y[:, -1]
        self.dynamics.apply_state(self.state)
        self.state = np.concatenate([vehicle.get_state() for vehicle in self.dynamics.target_vehicles] + [self.dynamics.ego.get_state(), self.state[-3:]])
        self.t = t1
        diag = self.dynamics.diagnostics(self.state, self.t)
        if diag["selected_pair"] != self.prev_selected_pair and diag["selected_pair"] is not None and self.prev_selected_pair is not None:
            self.switch_count += 1
        if diag["selected_pair"] is not None:
            self.prev_selected_pair = diag["selected_pair"]

        current_progress = lane_progress(diag["ego_state"])
        progress_delta = current_progress - self.prev_progress
        best_gap = float(np.max(diag["gap_values"])) if len(diag["gap_values"]) > 0 else 0.0
        opportunity = 1.0 if best_gap > GAP_SAFE else 0.0
        current_u_norm = normalized_u(float(diag["u_t"]))
        current_lateral_velocity = front_velocity(diag["ego_state"], VEHICLE_L)[1]
        lateral_flip = self.prev_lateral_velocity * current_lateral_velocity < 0.0 and abs(self.prev_lateral_velocity) > 1e-3 and abs(current_lateral_velocity) > 1e-3
        safety_margin = SAFETY_MARGIN_FACTOR * self.dynamics.ego.r
        safety_scale = max(safety_margin - self.dynamics.ego.r, 1e-6)
        safety_violation = max(0.0, (safety_margin - diag["min_distance"]) / safety_scale)
        timeout = (not diag["collision"]) and (not diag["success"]) and self.t >= SIM_TIME
        terms = {
            "progress": PROGRESS_REWARD_GAIN * max(progress_delta, 0.0),
            "lane_progress": LANE_PROGRESS_STEP_GAIN * current_progress,
            "opportunity": OPPORTUNITY_PROGRESS_GAIN * opportunity * max(progress_delta, 0.0),
            "reverse_progress": -REVERSE_PROGRESS_PENALTY_GAIN * max(-progress_delta, 0.0),
            "hesitation": -HESITATION_PENALTY_GAIN * opportunity * (1.0 - current_progress),
            "time": -TIME_PENALTY_GAIN * (1.0 - current_progress),
            "action_smooth": -ACTION_SMOOTH_PENALTY_GAIN * (current_u_norm - self.prev_u_norm) ** 2,
            "direction_flip": -DIRECTION_FLIP_PENALTY if lateral_flip else 0.0,
            "safety": -SAFETY_PENALTY_GAIN * safety_violation ** 2,
            "collision": -COLLISION_PENALTY if diag["collision"] else 0.0,
            "success": (SUCCESS_BONUS_BASE - SUCCESS_TIME_PENALTY_GAIN * self.t) if diag["success"] else 0.0,
            "timeout": -TIMEOUT_PROGRESS_PENALTY_GAIN * (1.0 - current_progress) if timeout else 0.0,
        }
        reward = float(sum(terms.values()))
        self.prev_progress = current_progress
        self.prev_u_norm = current_u_norm
        self.prev_lateral_velocity = current_lateral_velocity
        done = (diag["collision"] and STOP_ON_COLLISION) or (diag["success"] and STOP_ON_SUCCESS) or self.t >= SIM_TIME
        info = {
            **diag,
            "time": self.t,
            "reward": reward,
            "reward_terms": terms,
            "progress": current_progress,
            "safe_margin": safety_margin,
            "timeout": timeout,
            "switch_count": self.switch_count,
        }
        self.frames.append(snapshot(self.t, diag, self.switch_count))
        if self.render_enabled and (len(self.frames) % EXPORT_FRAME_STRIDE == 0 or done):
            self.render()
        return build_pair_obs(diag["ego_state"], diag["target_states"][diag["front_pair"][0]], diag["target_states"][diag["front_pair"][1]]), reward, done, info

    def render(self):
        plt = require_matplotlib()
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 9))
            self.ax_scene = plt.subplot(2, 1, 1)
            self.ax_high = plt.subplot(2, 2, 3)
            self.ax_low = plt.subplot(2, 2, 4)
        draw_frame(self.ax_scene, self.ax_high, self.ax_low, self.frames[-1], self.frames)
        plt.pause(0.001)


def snapshot(t, diag, switch_count):
    """保存一帧绘图/CSV 诊断数据。"""
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
        "min_distance": float(diag["min_distance"]),
        "collision": bool(diag["collision"]),
        "success": bool(diag["success"]),
        "switch_count": switch_count,
    }


def make_car_from_state(car_id, state, color, wheelbase):
    car = KinematicBicycleModel(id=car_id, L=wheelbase, color=color)
    car.set_state(state)
    return car


def draw_frame(ax_scene, ax_high, ax_low, frame, frames):
    """绘制多 gap 仿真画面。"""
    ax_scene.cla()
    draw_environment(ax_scene, LANE_WIDTH)
    for index, target_state in enumerate(frame["target_states"]):
        color = "gold" if index in frame["triple"] else ["lightblue", "cornflowerblue", "royalblue", "steelblue", "deepskyblue"][index % 5]
        draw_car(ax_scene, make_car_from_state(f"Veh {index + 1}", target_state, color, VEHICLE_L), wheelbase=1.5)
    ego = make_car_from_state("Ego", frame["ego_state"], "lightgreen", VEHICLE_L)
    draw_car(ax_scene, ego, wheelbase=1.5)
    tp = frame["target_point"]
    ax_scene.plot(tp[0], tp[1], marker="*", color="red", markersize=16, markeredgecolor="black")
    pair_text = "no target pair" if frame["selected_pair"] is None else f"pair {frame['selected_pair'][0] + 1}-{frame['selected_pair'][1] + 1}"
    ax_scene.set_xlim(ego.x - 22, ego.x + 52)
    ax_scene.set_ylim(-2, LANE_WIDTH * 2 + 2)
    ax_scene.set_aspect("equal")
    ax_scene.set_title(f"Final Multi-Gap Env | t={frame['time']:.2f}s | decision={frame['decision']} | {pair_text}")

    times = [item["time"] for item in frames]
    ax_high.cla()
    ax_high.plot(times, [item["y_high"] for item in frames], "b-", linewidth=2, label="High opinion y")
    ax_high.plot(times, [item["u_high"] for item in frames], "purple", linewidth=2, label="High attention u")
    ax_high.plot(times, [item["b_high"] for item in frames], "k--", linewidth=1.5, label="B=Cf-Cr")
    ax_high.plot(times, [item["cf"] for item in frames], "g:", linewidth=2, label="Cf")
    ax_high.plot(times, [item["cr"] for item in frames], "r:", linewidth=2, label="Cr")
    ax_high.axhline(HIGH_DECISION_THRESHOLD, color="gray", linestyle="--")
    ax_high.axhline(-HIGH_DECISION_THRESHOLD, color="gray", linestyle="--")
    ax_high.set_xlim(0, SIM_TIME)
    ax_high.set_title("High-level Decision Dynamics")
    ax_high.legend(loc="upper left")
    ax_high.grid(True)

    ax_low.cla()
    ax_low.plot(times, [item["z_low"] for item in frames], "m-", linewidth=2, label="Low opinion z")
    ax_low.plot(times, [item["b_t"] for item in frames], "g--", linewidth=1.8, label="Low b(t)")
    ax_low.plot(times, [item["u_t"] for item in frames], "purple", linestyle="--", linewidth=1.8, label="Low u(t)")
    ax_low.plot(times, [item["min_distance"] for item in frames], "tab:red", linewidth=1.8, label="Min distance")
    ax_low.axhline(1.5, color="black", linestyle=":", linewidth=1.5, label="Collision r")
    ax_low.set_xlim(0, SIM_TIME)
    ax_low.set_title("Low-level Signals")
    ax_low.legend(loc="upper right")
    ax_low.grid(True)


def run_episode(seed=None, gap_seed=None, ego_seed=None, decision_method="opinion", low_level_policy: Optional[Callable] = None, csv_path: Optional[str] = None, render=False):
    """运行一次多 gap 仿真。"""
    env = MultiGapEnv(seed=seed, gap_seed=gap_seed, ego_seed=ego_seed, decision_method=decision_method, low_level_policy=low_level_policy, render=render)
    rows = []
    total_reward = 0.0
    done = False
    info: Dict[str, object] = {}
    steps = 0
    while not done:
        _, reward, done, info = env.step()
        total_reward += reward
        steps += 1
        rows.append({
            "time": info["time"],
            "reward": reward,
            "total_reward": total_reward,
            "progress": info["progress"],
            "success": float(info["success"]),
            "collision": float(info["collision"]),
            "min_distance": info["min_distance"],
            "decision": info["decision"],
            "selected_pair": str(info["selected_pair"]),
            "switch_count": info["switch_count"],
            "best_gap": float(np.max(info["gap_values"])),
            "y_high": info["y_high"],
            "u_high": info["u_high"],
            "z_low": info["z_low"],
            "u_t": info["u_t"],
            "cf": info["front_conf"]["confidence"],
            "cr": info["rear_conf"]["confidence"],
        })
    if csv_path and rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if render:
        plt = require_matplotlib()
        plt.ioff()
        plt.show()
    return {
        "seed": -1 if seed is None else seed,
        "gap_seed": -1 if gap_seed is None else gap_seed,
        "ego_seed": -1 if ego_seed is None else ego_seed,
        "ego_x0": env.ego_x0,
        "decision_method": decision_method,
        "reward": total_reward,
        "progress": info["progress"],
        "success": float(info["success"]),
        "collision": float(info["collision"]),
        "time": info["time"],
        "steps": steps,
        "min_distance": info["min_distance"],
        "switch_count": env.switch_count,
    }


def main():
    parser = argparse.ArgumentParser(description="最终版多 gap 环境：运行一次带图仿真")
    parser.add_argument("--seed", type=int, default=None, help="统一随机种子")
    parser.add_argument("--gap-seed", type=int, default=None, help="gap 调度随机种子")
    parser.add_argument("--ego-seed", type=int, default=None, help="ego 初始位置随机种子")
    parser.add_argument("--decision", choices=["opinion", "max"], default="opinion", help="高层决策方法")
    parser.add_argument("--csv", default="multi_gap_rollout.csv", help="逐步仿真 CSV")
    parser.add_argument("--no-render", action="store_true", help="关闭图像显示")
    args = parser.parse_args()
    result = run_episode(seed=args.seed, gap_seed=args.gap_seed, ego_seed=args.ego_seed, decision_method=args.decision, csv_path=args.csv, render=not args.no_render)
    print("多 gap 仿真完成：")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print(f"逐步结果已保存：{args.csv}")


if __name__ == "__main__":
    main()
