"""独立单 gap 环境。

本文件直接复制 main12 训练所需的车辆模型、意见动力学、后车 gap 跟随逻辑和
reward 计算，不从 main11、main12、main7 文件导入模块或参数。训练脚本应使用
Main12SacUEnv；SingleGapEnv 只用于回放、RBF 对比和固定 ego 初始位置测试。
"""

from __future__ import annotations

import argparse
import csv
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp


# =========================
# 与原始 main11/main12 一致的训练和环境参数
# =========================
NUM_EPISODES = 300
RENDER_DURING_TRAINING = False
SIM_TIME = 40.0
DT = 0.05
SEED = 7

BATCH_SIZE = 256
REPLAY_SIZE = 200_000
INITIAL_RANDOM_STEPS = 1_000
UPDATES_PER_STEP = 1
GAMMA = 0.99
TAU = 0.005
POLICY_LR = 3e-4
Q_LR = 3e-4
ALPHA_LR = 3e-4
HIDDEN_SIZE = 256

ACTION_DIM = 1
U_LOW = 0.0
U_HIGH = 3.0

LANE_WIDTH = 4.0
VEHICLE_L = 2.8
TARGET_SPEED = 15.0
ORIGINAL_LANE_Y = LANE_WIDTH * 0.5
TARGET_LANE_Y = LANE_WIDTH * 1.5
FRONT_X0 = 30.0
REAR_X0 = 15.0
EGO_X_BASE = 22.0
EGO_X_RANDOM_RANGE = 2.0

DESIRED_GAP = 20.0
GAP_SAFE = 10.0
YIELD_TIME = 20.0
SINE_A_VEL = 4.0
SINE_PERIOD = 6.0

K_GAP = 0.2
K_VEL = 0.1
U_BASE = 0.2
U_AMP = 2.5
SIGMA_D = 2.0
SIGMA_V = 1.5
Z_DAMPING = 2.0
Z_ALPHA = 2.0

OBS_SCALE = np.array([40.0, 8.0, 20.0, 10.0, 40.0, 8.0, 20.0, 10.0], dtype=np.float32)

# 原始 main11/main12 reward 参数。
PROGRESS_REWARD_GAIN = 80.0
LANE_PROGRESS_STEP_GAIN = 0.15
OPPORTUNITY_PROGRESS_GAIN = 40.0
REVERSE_PROGRESS_PENALTY_GAIN = 30.0
HESITATION_PENALTY_GAIN = 0.25
TIME_PENALTY_GAIN = 0.05
ACTION_SMOOTH_PENALTY_GAIN = 0.5
DIRECTION_FLIP_PENALTY = 2.0
SAFETY_MARGIN_FACTOR = 2.5
SAFETY_PENALTY_GAIN = 5.0
COLLISION_PENALTY = 1000.0
SUCCESS_SAFE_FACTOR = 1.5
SUCCESS_BONUS_BASE = 200.0
SUCCESS_TIME_PENALTY_GAIN = 2.0
TIMEOUT_PROGRESS_PENALTY_GAIN = 0.0

POLICY_PATH = "main12_sac_u_policy.pth"
RESULT_FIG_PATH = "main12_sac_u_training_result.png"

plt = None


def require_matplotlib():
    """延迟导入 matplotlib，避免无图训练时提前加载绘图库。"""
    global plt
    if plt is None:
        import matplotlib.pyplot as _plt

        plt = _plt
    return plt


def _signed_safe(value, eps=1e-6):
    """带符号的除零保护。"""
    if abs(value) < eps:
        return eps if value >= 0.0 else -eps
    return value


def sample_ego_x():
    """采样 main12 的随机 ego 初始纵向位置。"""
    return float(EGO_X_BASE + np.random.uniform(-EGO_X_RANDOM_RANGE, EGO_X_RANDOM_RANGE))


class KinematicBicycleModel:
    """后轴自行车模型，控制参考点为前轴中心。"""

    def __init__(self, id, x=0.0, y=0.0, theta=0.0, v=0.0, delta=0.0, L=2.5, color="blue"):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.delta = delta
        self.L = L
        self.color = color

    def get_state(self):
        return np.array([self.x, self.y, self.theta, self.v, self.delta], dtype=float)

    def set_state(self, state):
        self.x, self.y, self.theta, self.v, self.delta = np.asarray(state, dtype=float)
        self.theta = (self.theta + np.pi) % (2.0 * np.pi) - np.pi
        self.delta = np.clip(self.delta, -np.pi / 4.0, np.pi / 4.0)
        self.v = max(0.0, self.v)


def front_position(state, L):
    """计算前轴点位置。"""
    x, y, theta, _, _ = state
    return np.array([x + L * np.cos(theta), y + L * np.sin(theta)])


def front_velocity(state, L):
    """计算前轴点速度。"""
    _, _, theta, vr, delta = state
    tan_delta = np.tan(delta)
    return vr * np.array([
        np.cos(theta) - np.sin(theta) * tan_delta,
        np.sin(theta) + np.cos(theta) * tan_delta,
    ])


def rear_state_derivative(state, a, omega, L):
    """计算后轴自行车模型状态导数。"""
    _, _, theta, vr, delta = state
    theta_dot = vr / L * (delta if abs(delta) < 1e-3 else np.tan(delta))
    return np.array([
        vr * np.cos(theta),
        vr * np.sin(theta),
        theta_dot,
        a,
        omega,
    ])


def compute_gap_bias_bt(gap, gap_dot, gap_safe, k_gap=0.25, k_vel=0.45):
    """由 gap 大小和 gap 变化率计算客观偏置 b(t)。"""
    return float(np.tanh(k_gap * (gap - gap_safe) + k_vel * gap_dot))


def compute_gap_opinion_z_dot(z, b_t, u_t, damping=1.0, alpha=2.0):
    """计算新意见动力学 z_dot = -d*z + u(t)*tanh(alpha*z) + b(t)。"""
    return float(-damping * z + u_t * np.tanh(alpha * z) + b_t)


def compute_gap_confidence_attention_ut(
    ego_state,
    front_state,
    rear_state,
    ego_l,
    front_l,
    rear_l,
    u_base=U_BASE,
    u_amp=U_AMP,
    sigma_d=SIGMA_D,
    sigma_v=SIGMA_V,
):
    """使用 RBF 置信度计算手工设计的注意力 u(t)。"""
    ego_pos = front_position(ego_state, ego_l)
    front_pos = front_position(front_state, front_l)
    rear_pos = front_position(rear_state, rear_l)
    ego_vel = front_velocity(ego_state, ego_l)
    front_vel = front_velocity(front_state, front_l)
    rear_vel = front_velocity(rear_state, rear_l)
    x_gap = 0.5 * (front_pos[0] + rear_pos[0])
    v_gap = 0.5 * (front_vel[0] + rear_vel[0])
    d_gap = float(x_gap - ego_pos[0])
    dv_gap = float(ego_vel[0] - v_gap)
    exponent = -0.5 * (d_gap / max(sigma_d, 1e-6)) ** 2 - 0.5 * (dv_gap / max(sigma_v, 1e-6)) ** 2
    confidence = float(np.exp(np.clip(exponent, -700.0, 0.0)))
    return {
        "x_gap": float(x_gap),
        "v_gap": float(v_gap),
        "d_gap": d_gap,
        "dv_gap": dv_gap,
        "confidence": confidence,
        "u_t": float(u_base + u_amp * confidence),
    }


def compute_gap_confidence_signals_from_states(
    ego_state,
    front_state,
    rear_state,
    ego_l,
    front_l,
    rear_l,
    gap_safe,
    k_gap=K_GAP,
    k_vel=K_VEL,
    u_base=U_BASE,
    u_amp=U_AMP,
    sigma_d=SIGMA_D,
    sigma_v=SIGMA_V,
):
    """返回 gap、gap_dot、b(t) 和 RBF u(t) 等信号。"""
    front_pos = front_position(front_state, front_l)
    rear_pos = front_position(rear_state, rear_l)
    front_vel = front_velocity(front_state, front_l)
    rear_vel = front_velocity(rear_state, rear_l)
    gap = float(front_pos[0] - rear_pos[0])
    gap_dot = float(front_vel[0] - rear_vel[0])
    b_t = compute_gap_bias_bt(gap, gap_dot, gap_safe, k_gap=k_gap, k_vel=k_vel)
    attention = compute_gap_confidence_attention_ut(
        ego_state,
        front_state,
        rear_state,
        ego_l,
        front_l,
        rear_l,
        u_base=u_base,
        u_amp=u_amp,
        sigma_d=sigma_d,
        sigma_v=sigma_v,
    )
    return {"gap": gap, "gap_dot": gap_dot, "b_t": b_t, **attention}


class EgoVehicleOdeModel(KinematicBicycleModel):
    """ego 车辆模型，包含论文控制律需要的意见状态和控制参数。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z = 0.01
        self.mu = 0.0
        self.r = 1.5
        self.rho = np.array([1.0, 0.0])
        self.eta = np.array([0.0, 1.0])
        self.k_mu = 5.0
        self.k = 20.0
        self.k_w = 40.0
        self.eps = 0.1
        self.eps2 = 0.5
        self.k_p = 0.7
        self.k_v = 2.0
        self.k_o = 1.0
        self.r_rho = -10.0
        self.r_eta = -4.0

    def read_sensor_from_states(self, ego_state, target_states):
        p_ego = front_position(ego_state, self.L)
        v_ego = front_velocity(ego_state, self.L)
        sensor_data = {}
        for name in ("veh1", "veh2"):
            target = target_states[name]
            p_target = front_position(target, target_states[name + "_L"])
            v_target = front_velocity(target, target_states[name + "_L"])
            rel_p = p_target - p_ego
            rel_v = v_target - v_ego
            sensor_data[name] = {"rel_p": rel_p, "rel_v": rel_v, "dist": np.linalg.norm(rel_p)}
        return sensor_data

    def compute_mu_dot(self, sensor_data, mu):
        dp1 = -sensor_data["veh1"]["rel_p"]
        dp2 = -sensor_data["veh2"]["rel_p"]
        g31 = dp1 / _signed_safe(np.linalg.norm(dp1))
        g32 = dp2 / _signed_safe(np.linalg.norm(dp2))
        e21 = sensor_data["veh2"]["rel_p"] - sensor_data["veh1"]["rel_p"]
        v21 = sensor_data["veh2"]["rel_v"] - sensor_data["veh1"]["rel_v"]
        d21 = np.linalg.norm(e21) - self.r
        g21 = e21 / _signed_safe(np.linalg.norm(e21))
        phi21 = np.dot(g21, v21) / _signed_safe(d21)
        tanh_arg = -self.k * np.dot(self.rho, g31) * np.dot(self.rho, g32) * (d21 - 2.0 * self.r) * (phi21 + self.eps2)
        return -self.k_mu * mu + np.tanh(tanh_arg)

    def compute_z_dot(self, z, mu):
        return (1.0 / self.eps) * (-z * z + mu * z)

    def compute_nominal_control(self, sensor_data, z):
        w = np.tanh(self.k_w * z)
        e31d = self.rho * self.r_rho + self.eta * ((1.0 - w) * self.r_eta)
        e31 = -sensor_data["veh1"]["rel_p"]
        v31 = -sensor_data["veh1"]["rel_v"]
        u_n = -self.k_p * (e31 - e31d) - self.k_v * v31
        return u_n, e31d

    def compute_safe_control(self, sensor_data):
        u_c = np.zeros(2)
        safe_distances = {}
        for name in ("veh1", "veh2"):
            e3j = -sensor_data[name]["rel_p"]
            v3j = -sensor_data[name]["rel_v"]
            dist = np.linalg.norm(e3j)
            g3j = e3j / _signed_safe(dist)
            d3j = dist - self.r
            phi3j = np.dot(g3j, v3j) / _signed_safe(d3j)
            u_c += -self.k_o * g3j * phi3j
            safe_distances[name] = d3j
        return u_c, safe_distances

    def u_to_physical_inputs(self, u, ego_state):
        _, _, theta, vr, delta = ego_state
        tan_delta = np.tan(delta)
        sec2_delta = 1.0 / (np.cos(delta) ** 2)
        vr_for_a = vr if abs(vr) >= 1e-6 else 1e-6
        a_matrix = np.array([
            [np.cos(theta) - np.sin(theta) * tan_delta, -vr_for_a * np.sin(theta) * sec2_delta],
            [np.sin(theta) + np.cos(theta) * tan_delta, vr_for_a * np.cos(theta) * sec2_delta],
        ])
        b_vector = -(vr * vr / self.L) * np.array([
            np.sin(theta) * tan_delta + np.cos(theta) * tan_delta * tan_delta,
            -np.cos(theta) * tan_delta + np.sin(theta) * tan_delta * tan_delta,
        ])
        try:
            return np.linalg.solve(a_matrix, u - b_vector)
        except np.linalg.LinAlgError:
            return np.zeros(2)

    def control_derivatives(self, ego_state, z, mu, target_states):
        sensor_data = self.read_sensor_from_states(ego_state, target_states)
        mu_dot = self.compute_mu_dot(sensor_data, mu)
        z_dot = self.compute_z_dot(z, mu)
        u_n, e31d = self.compute_nominal_control(sensor_data, z)
        u_c, safe_distances = self.compute_safe_control(sensor_data)
        a, omega = self.u_to_physical_inputs(u_n + u_c, ego_state)
        return {
            "sensor_data": sensor_data,
            "mu_dot": mu_dot,
            "z_dot": z_dot,
            "u_n": u_n,
            "u_c": u_c,
            "u_total": u_n + u_c,
            "e31d": e31d,
            "a": a,
            "omega": omega,
            "d1": safe_distances["veh1"],
            "d2": safe_distances["veh2"],
        }


def get_veh12_gap(state, veh1_l, veh2_l):
    """计算前后目标车前轴点之间的纵向 gap。"""
    p1 = front_position(state[0:5], veh1_l)
    p2 = front_position(state[5:10], veh2_l)
    return float(p1[0] - p2[0])


class Main7GapFollowingDynamics:
    """main7 的后车运动：20s 前正弦速度，20s 后跟踪 20m gap。"""

    def __init__(self, veh1, veh2, veh3, a_vel=SINE_A_VEL, period=SINE_PERIOD, yield_time=YIELD_TIME, desired_gap=DESIRED_GAP):
        self.veh1 = veh1
        self.veh2 = veh2
        self.veh3 = veh3
        self.a_vel = a_vel
        self.period = period
        self.yield_time = yield_time
        self.desired_gap = desired_gap

    def _target_states(self, state):
        return {"veh1": state[0:5], "veh2": state[5:10], "veh1_L": self.veh1.L, "veh2_L": self.veh2.L}

    def pack_state(self):
        return np.concatenate([self.veh1.get_state(), self.veh2.get_state(), self.veh3.get_state(), np.array([0.01, 0.0])])

    def apply_state(self, state):
        self.veh1.set_state(state[0:5])
        self.veh2.set_state(state[5:10])
        self.veh3.set_state(state[10:15])

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
        return float(np.clip(0.35 * gap_error - 1.1 * closing_speed, -5.0, 2.0))


class Main11MoveDynamics(Main7GapFollowingDynamics):
    """main11-move 的新 z 更新和基于 z 的车辆控制。"""

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
        return compute_gap_confidence_signals_from_states(
            state[10:15],
            state[0:5],
            state[5:10],
            self.veh3.L,
            self.veh1.L,
            self.veh2.L,
            self.gap_safe,
            k_gap=K_GAP,
            k_vel=K_VEL,
            u_base=U_BASE,
            u_amp=U_AMP,
            sigma_d=SIGMA_D,
            sigma_v=SIGMA_V,
        )

    def _new_z_control(self, ego_state, target_states, z_new):
        sensor_data = self.veh3.read_sensor_from_states(ego_state, target_states)
        u_n, e31d = self.veh3.compute_nominal_control(sensor_data, z_new)
        u_c, safe_distances = self.veh3.compute_safe_control(sensor_data)
        u_total = u_n + u_c
        a, omega = self.veh3.u_to_physical_inputs(u_total, ego_state)
        return {
            "sensor_data": sensor_data,
            "u_n": u_n,
            "u_c": u_c,
            "u_total": u_total,
            "e31d": e31d,
            "a": a,
            "omega": omega,
            "d1": safe_distances["veh1"],
            "d2": safe_distances["veh2"],
        }

    def diagnostics(self, state):
        z_new = state[17]
        old_diag = self.veh3.control_derivatives(state[10:15], state[15], state[16], self._target_states(state))
        new_diag = self._new_z_control(state[10:15], self._target_states(state), z_new)
        gap_signals = self._gap_signals(state)
        z_new_dot = compute_gap_opinion_z_dot(z_new, gap_signals["b_t"], gap_signals["u_t"], damping=Z_DAMPING, alpha=Z_ALPHA)
        return {
            **new_diag,
            "old_z_dot": old_diag["z_dot"],
            "old_mu_dot": old_diag["mu_dot"],
            "gap": gap_signals["gap"],
            "gap_dot": gap_signals["gap_dot"],
            "b_t": gap_signals["b_t"],
            "u_t": gap_signals["u_t"],
            "d_gap": gap_signals["d_gap"],
            "dv_gap": gap_signals["dv_gap"],
            "confidence": gap_signals["confidence"],
            "x_gap": gap_signals["x_gap"],
            "v_gap": gap_signals["v_gap"],
            "z_new_dot": z_new_dot,
        }


class Main11SacUDynamics(Main11MoveDynamics):
    """SAC 直接提供 u(t) 的 main11 动力学。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_u_t = 0.2

    def set_action_u_t(self, u_t):
        self.action_u_t = float(np.clip(u_t, U_LOW, U_HIGH))

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
        z_new_dot = compute_gap_opinion_z_dot(z_new, gap_signals["b_t"], self.action_u_t, damping=Z_DAMPING, alpha=Z_ALPHA)
        new_control = self._new_z_control(ego_state, self._target_states(state), z_new)
        return np.concatenate([
            rear_state_derivative(veh1_state, a1, omega1, self.veh1.L),
            rear_state_derivative(veh2_state, a2, omega2, self.veh2.L),
            rear_state_derivative(ego_state, new_control["a"], new_control["omega"], self.veh3.L),
            np.array([old_control["z_dot"], old_control["mu_dot"], z_new_dot]),
        ])

    def diagnostics(self, state):
        diag = super().diagnostics(state)
        formula_u_t = diag["u_t"]
        diag["formula_u_t"] = formula_u_t
        diag["u_t"] = self.action_u_t
        diag["z_new_dot"] = compute_gap_opinion_z_dot(state[17], diag["b_t"], self.action_u_t, damping=Z_DAMPING, alpha=Z_ALPHA)
        return diag


def draw_environment(ax, lane_width=LANE_WIDTH):
    """绘制两车道道路。"""
    ax.axhline(0.0, color="black", linewidth=1.2)
    ax.axhline(lane_width, color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(2.0 * lane_width, color="black", linewidth=1.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_car(ax, car, wheelbase=1.5):
    """绘制圆角矩形车辆。"""
    require_matplotlib()
    from matplotlib.patches import FancyBboxPatch

    length = 2.0 * wheelbase
    width = 1.2
    patch = FancyBboxPatch(
        (car.x - length / 2.0, car.y - width / 2.0),
        length,
        width,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        facecolor=car.color,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
    )
    ax.add_patch(patch)
    ax.text(car.x, car.y, car.id.split()[0], ha="center", va="center", fontsize=8)


def make_car_from_state(car_id, state, color, wheelbase):
    """根据状态生成绘图用车辆对象。"""
    car = KinematicBicycleModel(id=car_id, L=wheelbase, color=color)
    car.set_state(state)
    return car


class Main11SacUEnv:
    """原始 main11 SAC-u 环境。"""

    def __init__(self, render=False, sim_time=SIM_TIME, dt=DT):
        self.render_enabled = render
        self.sim_time = sim_time
        self.dt = dt
        self.lane_width = LANE_WIDTH
        self.vehicle_l = VEHICLE_L
        self.original_lane_y = ORIGINAL_LANE_Y
        self.target_lane_y = TARGET_LANE_Y
        self.desired_gap = DESIRED_GAP
        self.gap_safe = GAP_SAFE
        self.obs_scale = OBS_SCALE
        self.fig = None
        self.ax_anim = None
        self.ax_z = None
        self.ax_dist = None
        self.reset()

    def reset(self):
        self.veh1 = KinematicBicycleModel("Veh 1 (Leader)", x=FRONT_X0, y=self.target_lane_y, v=TARGET_SPEED, L=self.vehicle_l, color="lightblue")
        self.veh2 = KinematicBicycleModel("Veh 2 (Gap Control)", x=REAR_X0, y=self.target_lane_y, v=TARGET_SPEED, L=self.vehicle_l, color="royalblue")
        self.ego = EgoVehicleOdeModel("Veh 3 (Ego Main11 SAC-u)", x=EGO_X_BASE, y=self.original_lane_y, v=TARGET_SPEED, L=self.vehicle_l, color="lightgreen")
        self.dynamics = Main11SacUDynamics(self.veh1, self.veh2, self.ego, desired_gap=self.desired_gap, gap_safe=self.gap_safe)
        self.collision_radius = self.ego.r
        self.state = self.dynamics.pack_state()
        self.t = 0.0
        self.prev_lane_progress = self._lane_progress()
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.prev_lateral_velocity = front_velocity(self.state[10:15], self.ego.L)[1]
        self.t_hist, self.z_hist, self.u_hist, self.formula_u_hist, self.bt_hist = [], [], [], [], []
        self.dist1_hist, self.dist2_hist, self.veh12_gap_hist = [], [], []
        return self._get_obs()

    def _action_to_u(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return float(U_LOW + 0.5 * (action[0] + 1.0) * (U_HIGH - U_LOW))

    def _get_obs(self):
        diag = self.dynamics.diagnostics(self.state)
        rel1 = diag["sensor_data"]["veh1"]
        rel2 = diag["sensor_data"]["veh2"]
        ego_rel_pos_1 = -rel1["rel_p"]
        ego_rel_vel_1 = -rel1["rel_v"]
        ego_rel_pos_2 = -rel2["rel_p"]
        ego_rel_vel_2 = -rel2["rel_v"]
        obs = np.concatenate([ego_rel_pos_1, ego_rel_vel_1, ego_rel_pos_2, ego_rel_vel_2]).astype(np.float32)
        return np.clip(obs / self.obs_scale, -5.0, 5.0).astype(np.float32)

    def _lane_progress(self):
        relative_y = (self.ego.y - self.original_lane_y) / (self.target_lane_y - self.original_lane_y)
        return float(np.clip(relative_y, 0.0, 1.0))

    def _veh12_gap(self):
        return get_veh12_gap(self.state[:17], self.veh1.L, self.veh2.L)

    def _is_success(self, lane_progress, min_distance):
        return lane_progress > 0.95 and abs(self.ego.y - self.target_lane_y) < 0.2 and min_distance > SUCCESS_SAFE_FACTOR * self.collision_radius

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action_u_t = self._action_to_u(action)
        self.dynamics.set_action_u_t(action_u_t)
        sol = solve_ivp(
            fun=self.dynamics.rhs,
            t_span=(self.t, self.t + self.dt),
            y0=self.state,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            max_step=self.dt / 5.0,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        self.state = sol.y[:, -1]
        self.dynamics.apply_state(self.state)
        self.state = self.dynamics.pack_state()
        self.t += self.dt

        diag = self.dynamics.diagnostics(self.state)
        dist1 = diag["sensor_data"]["veh1"]["dist"]
        dist2 = diag["sensor_data"]["veh2"]["dist"]
        veh12_gap = self._veh12_gap()
        ego_min_distance = min(dist1, dist2)
        env_min_distance = min(ego_min_distance, veh12_gap)
        lane_progress = self._lane_progress()
        progress_delta = lane_progress - self.prev_lane_progress
        opportunity = 1.0 if self.state[16] > 0.1 or veh12_gap > self.gap_safe else 0.0
        current_lateral_velocity = front_velocity(self.state[10:15], self.ego.L)[1]
        lateral_direction_flip = (
            self.prev_lateral_velocity * current_lateral_velocity < 0.0
            and abs(self.prev_lateral_velocity) > 1e-3
            and abs(current_lateral_velocity) > 1e-3
        )
        safe_margin = SAFETY_MARGIN_FACTOR * self.collision_radius

        progress_reward = PROGRESS_REWARD_GAIN * progress_delta
        lane_progress_reward = LANE_PROGRESS_STEP_GAIN * lane_progress
        opportunity_reward = OPPORTUNITY_PROGRESS_GAIN * opportunity * max(progress_delta, 0.0)
        reverse_progress_penalty = -REVERSE_PROGRESS_PENALTY_GAIN * max(-progress_delta, 0.0)
        hesitation_penalty = -HESITATION_PENALTY_GAIN * opportunity * (1.0 - lane_progress)
        time_penalty = -TIME_PENALTY_GAIN * (1.0 - lane_progress)
        action_smooth_penalty = -ACTION_SMOOTH_PENALTY_GAIN * float(np.sum((action - self.prev_action) ** 2))
        direction_flip_penalty = -DIRECTION_FLIP_PENALTY if lateral_direction_flip else 0.0
        safety_penalty = -SAFETY_PENALTY_GAIN * max(0.0, (safe_margin - ego_min_distance) / safe_margin) ** 2
        collided = env_min_distance < self.collision_radius
        success = self._is_success(lane_progress, ego_min_distance)
        timeout = (not collided) and (not success) and self.t >= self.sim_time
        collision_penalty = -COLLISION_PENALTY if collided else 0.0
        success_bonus = (SUCCESS_BONUS_BASE - SUCCESS_TIME_PENALTY_GAIN * self.t * 0.5) if success else 0.0
        timeout_penalty = -TIMEOUT_PROGRESS_PENALTY_GAIN * (1.0 - lane_progress) if timeout else 0.0
        reward = (
            progress_reward
            + lane_progress_reward
            + opportunity_reward
            + reverse_progress_penalty
            + hesitation_penalty
            + time_penalty
            + action_smooth_penalty
            + direction_flip_penalty
            + safety_penalty
            + collision_penalty
            + success_bonus
            + timeout_penalty
        )

        self.prev_lane_progress = lane_progress
        self.prev_action = action.copy()
        self.prev_lateral_velocity = current_lateral_velocity
        done = collided or success or self.t >= self.sim_time
        obs = self._get_obs()
        info = {
            "u_t": action_u_t,
            "formula_u_t": diag["formula_u_t"],
            "b_t": diag["b_t"],
            "z_new": self.state[17],
            "z": self.state[17],
            "d_gap": diag["d_gap"],
            "dv_gap": diag["dv_gap"],
            "confidence": diag["confidence"],
            "lane_progress": lane_progress,
            "progress": lane_progress,
            "dist1": dist1,
            "dist2": dist2,
            "veh12_gap": veh12_gap,
            "gap": veh12_gap,
            "min_distance": ego_min_distance,
            "safe_margin": safe_margin,
            "success_safe_distance": SUCCESS_SAFE_FACTOR * self.collision_radius,
            "collided": collided,
            "collision": collided,
            "success": success,
            "timeout": timeout,
            "time": self.t,
            "reward": float(reward),
            "reward_terms": {
                "progress": progress_reward,
                "lane_progress": lane_progress_reward,
                "opportunity": opportunity_reward,
                "reverse_progress": reverse_progress_penalty,
                "hesitation": hesitation_penalty,
                "time": time_penalty,
                "action_smooth": action_smooth_penalty,
                "direction_flip": direction_flip_penalty,
                "safety": safety_penalty,
                "collision": collision_penalty,
                "success": success_bonus,
                "timeout": timeout_penalty,
            },
        }

        self.t_hist.append(self.t)
        self.z_hist.append(self.state[17])
        self.u_hist.append(action_u_t)
        self.formula_u_hist.append(diag["formula_u_t"])
        self.bt_hist.append(diag["b_t"])
        self.dist1_hist.append(dist1)
        self.dist2_hist.append(dist2)
        self.veh12_gap_hist.append(veh12_gap)
        if self.render_enabled:
            self.render(info)
        return obs, float(reward), done, info

    def render(self, info):
        require_matplotlib()
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 8))
            self.ax_anim = plt.subplot(2, 1, 1)
            self.ax_z = plt.subplot(2, 2, 3)
            self.ax_dist = plt.subplot(2, 2, 4)
        self.ax_anim.cla()
        draw_environment(self.ax_anim, self.lane_width)
        draw_car(self.ax_anim, self.veh1, wheelbase=self.collision_radius)
        draw_car(self.ax_anim, self.veh2, wheelbase=self.collision_radius)
        draw_car(self.ax_anim, self.ego, wheelbase=self.collision_radius)
        self.ax_anim.set_xlim(self.ego.x - 15, self.ego.x + 45)
        self.ax_anim.set_ylim(-2, self.lane_width * 2 + 2)
        self.ax_anim.set_aspect("equal")
        title = f"Time: {self.t:.2f}s | Main12 SAC u(t) training"
        if info["collided"]:
            title += " | COLLISION"
        if info["success"]:
            title += " | SUCCESS"
        self.ax_anim.set_title(title)

        self.ax_z.cla()
        self.ax_z.plot(self.t_hist, self.z_hist, "m-", linewidth=2.5, label="New Formula z")
        self.ax_z.plot(self.t_hist, self.u_hist, "purple", linewidth=1.8, label="RL u(t)")
        self.ax_z.plot(self.t_hist, self.formula_u_hist, "orange", linestyle="--", linewidth=1.8, label="RBF Formula u(t)")
        self.ax_z.plot(self.t_hist, self.bt_hist, "g--", linewidth=1.8, label="b(t)")
        self.ax_z.axhline(0, color="gray", linestyle="--")
        self.ax_z.axvline(20.0, color="black", linestyle=":", linewidth=1.5, label="Gap Control Starts")
        self.ax_z.set_xlim(0, self.sim_time)
        self.ax_z.set_title("RL Attention Injected into Main11 Opinion Dynamics")
        self.ax_z.legend(loc="upper left")
        self.ax_z.grid(True)

        self.ax_dist.cla()
        self.ax_dist.plot(self.t_hist, self.dist1_hist, "purple", linewidth=2, label="Distance to Veh 1")
        self.ax_dist.plot(self.t_hist, self.dist2_hist, "red", linewidth=2, label="Distance to Veh 2")
        self.ax_dist.plot(self.t_hist, self.veh12_gap_hist, "gray", linestyle="-.", linewidth=2, label="Veh1-Veh2 Gap")
        self.ax_dist.axhline(self.collision_radius, color="black", linestyle="--", linewidth=2, label=f"Collision Threshold r={self.collision_radius:g}m")
        self.ax_dist.axhline(self.gap_safe, color="orange", linestyle="--", linewidth=1.8, label="Safe Gap 10m")
        self.ax_dist.axhline(self.desired_gap, color="green", linestyle=":", linewidth=2, label="Target Gap 20m")
        self.ax_dist.axvline(20.0, color="black", linestyle=":", linewidth=1.5)
        self.ax_dist.set_xlim(0, self.sim_time)
        upper_distance = max([self.collision_radius * 2.0, self.desired_gap * 1.2, *self.dist1_hist, *self.dist2_hist, *self.veh12_gap_hist])
        self.ax_dist.set_ylim(0, upper_distance * 1.1)
        self.ax_dist.set_title("Relative Distance Monitoring")
        self.ax_dist.legend(loc="upper right")
        self.ax_dist.grid(True)
        plt.pause(0.001)


class Main12SacUEnv(Main11SacUEnv):
    """main12 SAC-u 环境：在 main11 基础上随机化 ego 初始 x 位置。"""

    def reset(self):
        ego_x0 = sample_ego_x()
        self.ego_x0 = ego_x0
        self.veh1 = KinematicBicycleModel("Veh 1 (Leader)", x=FRONT_X0, y=self.target_lane_y, v=TARGET_SPEED, L=self.vehicle_l, color="lightblue")
        self.veh2 = KinematicBicycleModel("Veh 2 (Gap Control)", x=REAR_X0, y=self.target_lane_y, v=TARGET_SPEED, L=self.vehicle_l, color="royalblue")
        self.ego = EgoVehicleOdeModel("Veh 3 (Ego Main12 SAC-u)", x=ego_x0, y=self.original_lane_y, v=TARGET_SPEED, L=self.vehicle_l, color="lightgreen")
        self.dynamics = Main11SacUDynamics(self.veh1, self.veh2, self.ego, desired_gap=self.desired_gap, gap_safe=self.gap_safe)
        self.collision_radius = self.ego.r
        self.state = self.dynamics.pack_state()
        self.t = 0.0
        self.prev_lane_progress = self._lane_progress()
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.prev_lateral_velocity = front_velocity(self.state[10:15], self.ego.L)[1]
        self.t_hist, self.z_hist, self.u_hist, self.formula_u_hist, self.bt_hist = [], [], [], [], []
        self.dist1_hist, self.dist2_hist, self.veh12_gap_hist = [], [], []
        return self._get_obs()


class SingleGapEnv(Main12SacUEnv):
    """兼容回放和对比脚本的单 gap 环境。"""

    def __init__(self, seed: Optional[int] = None, render: bool = False, sim_time: float = SIM_TIME, dt: float = DT):
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        super().__init__(render=render, sim_time=sim_time, dt=dt)
        self.front = self.veh1
        self.rear = self.veh2

    def reset(self, seed: Optional[int] = None, ego_x0: Optional[float] = None):
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        if ego_x0 is None:
            obs = super().reset()
        else:
            obs = self._reset_fixed_ego_x(float(ego_x0))
        self.front = self.veh1
        self.rear = self.veh2
        return obs

    def _reset_fixed_ego_x(self, ego_x0: float):
        self.ego_x0 = ego_x0
        self.veh1 = KinematicBicycleModel("Veh 1 (Leader)", x=FRONT_X0, y=self.target_lane_y, v=TARGET_SPEED, L=self.vehicle_l, color="lightblue")
        self.veh2 = KinematicBicycleModel("Veh 2 (Gap Control)", x=REAR_X0, y=self.target_lane_y, v=TARGET_SPEED, L=self.vehicle_l, color="royalblue")
        self.ego = EgoVehicleOdeModel("Veh 3 (Ego Main12 SAC-u)", x=ego_x0, y=self.original_lane_y, v=TARGET_SPEED, L=self.vehicle_l, color="lightgreen")
        self.dynamics = Main11SacUDynamics(self.veh1, self.veh2, self.ego, desired_gap=self.desired_gap, gap_safe=self.gap_safe)
        self.collision_radius = self.ego.r
        self.state = self.dynamics.pack_state()
        self.t = 0.0
        self.prev_lane_progress = self._lane_progress()
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.prev_lateral_velocity = front_velocity(self.state[10:15], self.ego.L)[1]
        self.t_hist, self.z_hist, self.u_hist, self.formula_u_hist, self.bt_hist = [], [], [], [], []
        self.dist1_hist, self.dist2_hist, self.veh12_gap_hist = [], [], []
        return self._get_obs()

    def observation(self):
        return self._get_obs().tolist()

    def step_action(self, action):
        return Main11SacUEnv.step(self, action)

    def step(self, u_t: float):
        action = np.array([2.0 * (float(u_t) - U_LOW) / (U_HIGH - U_LOW) - 1.0], dtype=np.float32)
        return Main11SacUEnv.step(self, action)


def compute_rbf_u(ego, front, rear):
    """计算 RBF 公式给出的手工 u(t)。"""
    return compute_gap_confidence_attention_ut(ego.get_state(), front.get_state(), rear.get_state(), ego.L, front.L, rear.L)


def run_episode(
    seed: Optional[int] = None,
    policy: Optional[Callable[[List[float], Dict[str, float]], float]] = None,
    csv_path: Optional[str] = None,
    render: bool = False,
):
    """运行一次单 gap 仿真；policy 输入 obs/info，输出真实 u(t)。"""
    env = SingleGapEnv(seed=seed, render=render)
    rows = []
    total_reward = 0.0
    obs = env.observation()
    info: Dict[str, float] = {"formula_u_t": compute_rbf_u(env.ego, env.veh1, env.veh2)["u_t"]}
    done = False
    steps = 0
    while not done:
        u_t = policy(obs, info) if policy is not None else info["formula_u_t"]
        obs, reward, done, info = env.step(u_t)
        total_reward += reward
        steps += 1
        rows.append({
            "time": info["time"],
            "reward": reward,
            "total_reward": total_reward,
            "progress": info["lane_progress"],
            "u_t": info["u_t"],
            "formula_u_t": info["formula_u_t"],
            "z": info["z_new"],
            "gap": info["veh12_gap"],
            "min_distance": info["min_distance"],
            "ego_x": env.ego.x,
            "ego_y": env.ego.y,
            "success": float(info["success"]),
            "collision": float(info["collided"]),
        })
    if csv_path and rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if render:
        require_matplotlib()
        plt.ioff()
        plt.show()
    return {
        "seed": -1 if seed is None else seed,
        "ego_x0": env.ego_x0,
        "reward": total_reward,
        "progress": info["lane_progress"],
        "success": float(info["success"]),
        "collision": float(info["collided"]),
        "time": info["time"],
        "steps": steps,
        "min_distance": info["min_distance"],
    }


def main():
    parser = argparse.ArgumentParser(description="single-gap environment rollout")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--csv", default="single_gap_rollout.csv", help="step-level rollout CSV")
    parser.add_argument("--no-render", action="store_true", help="disable figure display")
    args = parser.parse_args()
    result = run_episode(seed=args.seed, csv_path=args.csv, render=not args.no_render)
    print("Single-gap rollout finished:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print(f"Saved rollout CSV to {args.csv}")


if __name__ == "__main__":
    main()
