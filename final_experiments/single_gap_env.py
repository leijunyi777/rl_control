"""单 gap 并道实验环境。

本文件是最终版单 gap 仿真环境，参考最新 main12 设置，但不依赖任何旧环境代码。
环境本身只使用 Python 标准库，可直接运行，也可被 SAC 训练和对比脚本导入。
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


# =========================
# 1. 与最新 main12 一致的参数
# =========================
SIM_TIME = 40.0
DT = 0.05
LANE_WIDTH = 4.0
ORIGINAL_LANE_Y = 0.5 * LANE_WIDTH
TARGET_LANE_Y = 1.5 * LANE_WIDTH
VEHICLE_L = 2.8
TARGET_SPEED = 15.0

FRONT_X0 = 30.0
REAR_X0 = 15.0
EGO_X_BASE = 20.0
EGO_X_RANDOM_RANGE = 5.0

COLLISION_RADIUS = 1.5
DESIRED_GAP = 20.0
GAP_SAFE = 10.0
YIELD_TIME = 20.0
SINE_VEL_AMP = 4.0
SINE_PERIOD = 6.0
REAR_ACCEL_MIN = -5.0
REAR_ACCEL_MAX = 2.0

K_GAP = 0.2
K_VEL = 0.1
U_BASE = 0.2
U_AMP = 2.5
SIGMA_D = 2.0
SIGMA_V = 1.5
Z_DAMPING = 2.0
Z_ALPHA = 2.0
Z0 = 0.01
U_LOW = 0.0
U_HIGH = 3.0

EGO_K_P = 0.7
EGO_K_V = 2.0
EGO_K_O = 1.0
EGO_K_W = 40.0
EGO_R_ETA = -4.0
EGO_ACCEL_LIMIT = 5.0

OBS_SCALE = [40.0, 8.0, 20.0, 10.0, 40.0, 8.0, 20.0, 10.0]


def clip(value: float, low: float, high: float) -> float:
    """将数值限制在给定范围内。"""
    return max(low, min(high, value))


def safe_norm(dx: float, dy: float) -> float:
    """计算二维距离，并避免除零。"""
    return max(math.hypot(dx, dy), 1e-9)


@dataclass
class Vehicle:
    """简化车辆状态。

    为了让环境可独立运行，这里使用二维质点形式近似前轴点运动。
    目标车只沿纵向运动，ego 车同时具有纵向和横向速度。
    """

    x: float
    y: float
    vx: float
    vy: float = 0.0

    def step(self, ax: float, ay: float, dt: float) -> None:
        """使用半隐式 Euler 更新车辆状态。"""
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt


def compute_gap_bias(gap: float, gap_dot: float, gap_safe: float = GAP_SAFE) -> float:
    """计算底层客观偏置 b(t)。"""
    return math.tanh(K_GAP * (gap - gap_safe) + K_VEL * gap_dot)


def compute_rbf_u(
    ego: Vehicle,
    front: Vehicle,
    rear: Vehicle,
    sigma_d: float = SIGMA_D,
    sigma_v: float = SIGMA_V,
) -> Dict[str, float]:
    """根据 ego 与 gap 中心的对齐程度计算手工 RBF 注意力 u(t)。"""
    x_gap = 0.5 * (front.x + rear.x)
    v_gap = 0.5 * (front.vx + rear.vx)
    d_gap = x_gap - ego.x
    dv_gap = ego.vx - v_gap
    exponent = -0.5 * (d_gap / max(sigma_d, 1e-6)) ** 2 - 0.5 * (dv_gap / max(sigma_v, 1e-6)) ** 2
    confidence = math.exp(clip(exponent, -700.0, 0.0))
    return {
        "x_gap": x_gap,
        "v_gap": v_gap,
        "d_gap": d_gap,
        "dv_gap": dv_gap,
        "confidence": confidence,
        "u_t": U_BASE + U_AMP * confidence,
    }


def normalized_u(u_t: float) -> float:
    """把真实 u(t) 映射回 [-1, 1]，用于 reward 的动作平滑项。"""
    return clip(2.0 * (u_t - U_LOW) / max(U_HIGH - U_LOW, 1e-9) - 1.0, -1.0, 1.0)


class SingleGapEnv:
    """单 gap 并道环境。

    step(u_t) 输入真实注意力 u(t)，返回 obs, reward, done, info。
    obs 为 8 维相对位置和速度，与最新 SAC 训练设置保持一致。
    """

    def __init__(self, seed: Optional[int] = None, sim_time: float = SIM_TIME, dt: float = DT):
        self.rng = random.Random(seed)
        self.seed = seed
        self.sim_time = sim_time
        self.dt = dt
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None, ego_x0: Optional[float] = None) -> List[float]:
        """重置环境，可指定随机种子或 ego 初始位置。"""
        if seed is not None:
            self.rng.seed(seed)
            self.seed = seed
        if ego_x0 is None:
            ego_x0 = EGO_X_BASE + self.rng.uniform(-EGO_X_RANDOM_RANGE, EGO_X_RANDOM_RANGE)

        self.front = Vehicle(FRONT_X0, TARGET_LANE_Y, TARGET_SPEED)
        self.rear = Vehicle(REAR_X0, TARGET_LANE_Y, TARGET_SPEED)
        self.ego = Vehicle(ego_x0, ORIGINAL_LANE_Y, TARGET_SPEED)
        self.t = 0.0
        self.z = Z0
        self.prev_progress = self.lane_progress()
        self.prev_u_norm = normalized_u(U_BASE)
        self.prev_lateral_velocity = self.ego.vy
        self.ego_x0 = ego_x0
        return self.observation()

    def observation(self) -> List[float]:
        """返回 8 维相对状态：ego 相对于前车和后车的位置、速度。"""
        raw = [
            self.front.x - self.ego.x,
            self.front.y - self.ego.y,
            self.front.vx - self.ego.vx,
            self.front.vy - self.ego.vy,
            self.rear.x - self.ego.x,
            self.rear.y - self.ego.y,
            self.rear.vx - self.ego.vx,
            self.rear.vy - self.ego.vy,
        ]
        return [raw[i] / OBS_SCALE[i] for i in range(len(raw))]

    def lane_progress(self) -> float:
        """计算并道进度，0 表示原车道，1 表示目标车道。"""
        return clip((self.ego.y - ORIGINAL_LANE_Y) / (TARGET_LANE_Y - ORIGINAL_LANE_Y), 0.0, 1.0)

    def gap(self) -> float:
        """目标前后车之间的纵向 gap。"""
        return self.front.x - self.rear.x

    def gap_dot(self) -> float:
        """目标前后车之间 gap 的变化率。"""
        return self.front.vx - self.rear.vx

    def rear_acceleration(self, t: float) -> float:
        """后车分段加速度：20s 前正弦运动，20s 后跟踪 20m gap。"""
        if t <= YIELD_TIME:
            omega = 2.0 * math.pi / SINE_PERIOD
            return SINE_VEL_AMP * omega * math.cos(omega * t)
        gap_error = self.gap() - DESIRED_GAP
        closing_speed = self.rear.vx - self.front.vx
        return clip(0.35 * gap_error - 1.1 * closing_speed, REAR_ACCEL_MIN, REAR_ACCEL_MAX)

    def min_ego_distance(self) -> float:
        """ego 到两辆目标车的最小欧氏距离。"""
        d_front = math.hypot(self.ego.x - self.front.x, self.ego.y - self.front.y)
        d_rear = math.hypot(self.ego.x - self.rear.x, self.ego.y - self.rear.y)
        return min(d_front, d_rear)

    def safety_acceleration(self) -> Tuple[float, float]:
        """简化连续避障项，距离过近时给 ego 一个排斥加速度。"""
        ux, uy = 0.0, 0.0
        safe_distance = 2.5 * COLLISION_RADIUS
        for target in (self.front, self.rear):
            rx = self.ego.x - target.x
            ry = self.ego.y - target.y
            dist = safe_norm(rx, ry)
            if dist < safe_distance:
                strength = EGO_K_O * ((safe_distance - dist) / safe_distance) ** 2
                ux += strength * rx / dist
                uy += strength * ry / dist
        return ux, uy

    def control_ego(self, u_t: float) -> Dict[str, float]:
        """根据 u(t)、b(t)、z(t) 生成 ego 控制输入。"""
        gap = self.gap()
        gap_dot = self.gap_dot()
        b_t = compute_gap_bias(gap, gap_dot)
        rbf = compute_rbf_u(self.ego, self.front, self.rear)
        u_t = clip(float(u_t), U_LOW, U_HIGH)

        z_dot = -Z_DAMPING * self.z + u_t * math.tanh(Z_ALPHA * self.z) + b_t
        self.z += self.dt * z_dot

        w = math.tanh(EGO_K_W * self.z)
        gap_center_x = 0.5 * (self.front.x + self.rear.x)
        gap_center_v = 0.5 * (self.front.vx + self.rear.vx)
        target_x = gap_center_x
        target_y = TARGET_LANE_Y + (1.0 - w) * EGO_R_ETA

        ax_safe, ay_safe = self.safety_acceleration()
        ax = -EGO_K_P * (self.ego.x - target_x) - EGO_K_V * (self.ego.vx - gap_center_v) + ax_safe
        ay = -EGO_K_P * (self.ego.y - target_y) - EGO_K_V * self.ego.vy + ay_safe
        ax = clip(ax, -EGO_ACCEL_LIMIT, EGO_ACCEL_LIMIT)
        ay = clip(ay, -EGO_ACCEL_LIMIT, EGO_ACCEL_LIMIT)

        return {
            "ax": ax,
            "ay": ay,
            "b_t": b_t,
            "u_t": u_t,
            "formula_u_t": rbf["u_t"],
            "z": self.z,
            "z_dot": z_dot,
            "gap": gap,
            "gap_dot": gap_dot,
            **rbf,
        }

    def step(self, u_t: float) -> Tuple[List[float], float, bool, Dict[str, float]]:
        """推进一步仿真。"""
        diag = self.control_ego(u_t)

        # 目标车运动：前车匀速，后车按分段规则运动。
        self.front.step(0.0, 0.0, self.dt)
        self.rear.step(self.rear_acceleration(self.t), 0.0, self.dt)
        self.ego.step(diag["ax"], diag["ay"], self.dt)
        self.ego.vx = max(0.0, self.ego.vx)
        self.t += self.dt

        progress = self.lane_progress()
        progress_delta = progress - self.prev_progress
        ego_min_distance = self.min_ego_distance()
        env_min_distance = min(ego_min_distance, self.gap())
        opportunity = 1.0 if self.z > 0.1 or self.gap() > GAP_SAFE else 0.0
        lateral_flip = (
            self.prev_lateral_velocity * self.ego.vy < 0.0
            and abs(self.prev_lateral_velocity) > 1e-3
            and abs(self.ego.vy) > 1e-3
        )
        safe_margin = 2.5 * COLLISION_RADIUS

        progress_reward = 80.0 * progress_delta
        lane_progress_reward = 0.15 * progress
        opportunity_reward = 40.0 * opportunity * max(progress_delta, 0.0)
        reverse_progress_penalty = -30.0 * max(-progress_delta, 0.0)
        hesitation_penalty = -0.25 * opportunity * (1.0 - progress)
        time_penalty = -0.05 * (1.0 - progress)
        action_smooth_penalty = -0.5 * (normalized_u(diag["u_t"]) - self.prev_u_norm) ** 2
        direction_flip_penalty = -2.0 if lateral_flip else 0.0
        safety_penalty = -20.0 * max(0.0, (safe_margin - ego_min_distance) / safe_margin) ** 2

        collided = env_min_distance < COLLISION_RADIUS
        success = progress > 0.95 and abs(self.ego.y - TARGET_LANE_Y) < 0.2 and ego_min_distance > 1.5 * COLLISION_RADIUS
        collision_penalty = -1000.0 if collided else 0.0
        success_bonus = (100.0 - 2.0 * self.t) if success else 0.0

        reward_terms = {
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
        }
        reward = sum(reward_terms.values())

        self.prev_progress = progress
        self.prev_u_norm = normalized_u(diag["u_t"])
        self.prev_lateral_velocity = self.ego.vy

        info = {
            **diag,
            "time": self.t,
            "ego_x": self.ego.x,
            "ego_y": self.ego.y,
            "ego_vx": self.ego.vx,
            "ego_vy": self.ego.vy,
            "lane_progress": progress,
            "min_distance": ego_min_distance,
            "env_min_distance": env_min_distance,
            "collided": collided,
            "success": success,
            "reward": reward,
            "reward_terms": reward_terms,
        }
        done = collided or success or self.t >= self.sim_time
        return self.observation(), reward, done, info


def run_episode(
    seed: Optional[int] = None,
    policy: Optional[Callable[[List[float], Dict[str, float]], float]] = None,
    csv_path: Optional[str] = None,
) -> Dict[str, float]:
    """运行一次单 gap 仿真。

    policy 接收 observation 和上一步诊断信息，返回真实 u(t)。若不提供，则使用 RBF u(t)。
    """
    env = SingleGapEnv(seed=seed)
    rows: List[Dict[str, float]] = []
    total_reward = 0.0
    info: Dict[str, float] = {"formula_u_t": U_BASE}
    obs = env.observation()
    done = False
    steps = 0
    while not done:
        u_t = policy(obs, info) if policy is not None else compute_rbf_u(env.ego, env.front, env.rear)["u_t"]
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
            "z": info["z"],
            "gap": info["gap"],
            "min_distance": info["min_distance"],
            "ego_x": info["ego_x"],
            "ego_y": info["ego_y"],
            "success": float(info["success"]),
            "collision": float(info["collided"]),
        })

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="单 gap 独立仿真环境")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--csv", default="single_gap_rollout.csv", help="逐步仿真结果 CSV")
    args = parser.parse_args()
    result = run_episode(seed=args.seed, csv_path=args.csv)
    print("单 gap 仿真完成：")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print(f"逐步结果已保存：{args.csv}")


if __name__ == "__main__":
    main()
