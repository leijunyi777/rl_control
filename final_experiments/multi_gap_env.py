"""多 gap 并道实验环境。

本文件是最终版多 gap 仿真环境，参考最新 main13 设置，但不依赖任何旧环境代码。
环境本身只使用 Python 标准库；评价脚本可选择传入单 gap 训练得到的 policy。
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


# =========================
# 1. 与最新 main13 一致的参数
# =========================
SIM_TIME = 40.0
DT = 0.05
LANE_WIDTH = 4.0
ORIGINAL_LANE_Y = 0.5 * LANE_WIDTH
TARGET_LANE_Y = 1.5 * LANE_WIDTH
VEHICLE_L = 2.8
TARGET_SPEED = 15.0
COLLISION_RADIUS = 1.5

NUM_TARGET_VEHICLES = 5
BASE_GAP = 8.0
GAP_SWITCH_PERIOD = 4.0
GAP_MULTIPLIERS = [0.75, 1.0, 1.25, 1.5]
MAX_CHANGED_GAPS_PER_PERIOD = 2
GAP_PID_KP = 0.55
GAP_PID_KD = 1.05
GAP_ACCEL_LIMIT = 4.0

EGO_X_BASE = 30.0
EGO_X_RANDOM_RANGE = 10.0

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
Z0 = 0.01

HIGH_DAMPING = 2.5
HIGH_ALPHA = 10.0
HIGH_U_TAU = 1.0
HIGH_U_MAX = 1.5
HIGH_HILL_K = 0.2
HIGH_HILL_N = 2.0
HIGH_DECISION_THRESHOLD = 0.18

EGO_K_P = 0.7
EGO_K_V = 2.0
EGO_K_O = 1.0
EGO_K_W = 40.0
EGO_R_ETA = -4.0
EGO_ACCEL_LIMIT = 5.0

OBS_SCALE = [40.0, 8.0, 20.0, 10.0, 40.0, 8.0, 20.0, 10.0]


def clip(value: float, low: float, high: float) -> float:
    """将数值限制到指定区间。"""
    return max(low, min(high, value))


def safe_norm(dx: float, dy: float) -> float:
    """计算二维距离，并避免除零。"""
    return max(math.hypot(dx, dy), 1e-9)


def normalized_u(u_t: float) -> float:
    """将真实 u(t) 映射为 [-1, 1]。"""
    return clip(2.0 * (u_t - RL_U_LOW) / max(RL_U_HIGH - RL_U_LOW, 1e-9) - 1.0, -1.0, 1.0)


@dataclass
class Vehicle:
    """简化二维车辆状态。"""

    x: float
    y: float
    vx: float
    vy: float = 0.0

    def step(self, ax: float, ay: float, dt: float) -> None:
        """半隐式 Euler 更新。"""
        self.vx += ax * dt
        self.vy += ay * dt
        self.vx = max(0.0, self.vx)
        self.x += self.vx * dt
        self.y += self.vy * dt


def compute_gap_bias(gap: float, gap_dot: float) -> float:
    """底层 gap 偏置 b(t)。"""
    return math.tanh(K_GAP * (gap - GAP_SAFE) + K_VEL * gap_dot)


def gap_confidence(ego: Vehicle, front: Vehicle, rear: Vehicle) -> Dict[str, float]:
    """计算 ego 与候选 gap 的 RBF 置信度。"""
    x_gap = 0.5 * (front.x + rear.x)
    v_gap = 0.5 * (front.vx + rear.vx)
    d_gap = x_gap - ego.x
    dv_gap = ego.vx - v_gap
    exponent = -0.5 * (d_gap / max(SIGMA_D, 1e-6)) ** 2 - 0.5 * (dv_gap / max(SIGMA_V, 1e-6)) ** 2
    confidence = math.exp(clip(exponent, -700.0, 0.0))
    return {
        "x_gap": x_gap,
        "v_gap": v_gap,
        "d_gap": d_gap,
        "dv_gap": dv_gap,
        "confidence": confidence,
    }


def rbf_u(ego: Vehicle, front: Vehicle, rear: Vehicle) -> Dict[str, float]:
    """多 gap 环境中使用的手工 RBF 注意力。"""
    conf = gap_confidence(ego, front, rear)
    return {**conf, "u_t": U_BASE + U_AMP * conf["confidence"]}


def pair_observation(ego: Vehicle, front: Vehicle, rear: Vehicle) -> List[float]:
    """构建与单 gap SAC 训练一致的 8 维相对状态。"""
    raw = [
        front.x - ego.x,
        front.y - ego.y,
        front.vx - ego.vx,
        front.vy - ego.vy,
        rear.x - ego.x,
        rear.y - ego.y,
        rear.vx - ego.vx,
        rear.vy - ego.vy,
    ]
    return [raw[i] / OBS_SCALE[i] for i in range(len(raw))]


class MultiGapEnv:
    """五车四 gap 的多 gap 并道环境。"""

    def __init__(
        self,
        seed: Optional[int] = None,
        gap_seed: Optional[int] = None,
        ego_seed: Optional[int] = None,
        decision_method: str = "opinion",
        low_level_policy: Optional[Callable[[List[float], Dict[str, float]], float]] = None,
    ):
        self.seed = seed
        self.gap_seed = seed if gap_seed is None else gap_seed
        self.ego_seed = seed if ego_seed is None else ego_seed
        self.gap_rng = random.Random(self.gap_seed)
        self.ego_rng = random.Random(self.ego_seed)
        self.decision_method = decision_method
        self.low_level_policy = low_level_policy
        self.dt = DT
        self.reset(gap_seed=self.gap_seed, ego_seed=self.ego_seed)

    def reset(self, gap_seed: Optional[int] = None, ego_seed: Optional[int] = None) -> List[float]:
        """重置多 gap 环境。"""
        if gap_seed is not None:
            self.gap_rng.seed(gap_seed)
            self.gap_seed = gap_seed
        if ego_seed is not None:
            self.ego_rng.seed(ego_seed)
            self.ego_seed = ego_seed

        self.targets = [
            Vehicle(48.0 - i * BASE_GAP, TARGET_LANE_Y, TARGET_SPEED)
            for i in range(NUM_TARGET_VEHICLES)
        ]
        ego_x = EGO_X_BASE + self.ego_rng.uniform(-EGO_X_RANDOM_RANGE, EGO_X_RANDOM_RANGE)
        self.ego = Vehicle(ego_x, ORIGINAL_LANE_Y, TARGET_SPEED)
        self.ego_x0 = ego_x
        self.t = 0.0
        self.y_high = 0.0
        self.u_high = 0.01
        self.z_low = Z0
        self.prev_progress = self.lane_progress()
        self.prev_u_norm = normalized_u(0.0)
        self.prev_lateral_velocity = self.ego.vy
        self.prev_selected_pair: Optional[Tuple[int, int]] = None
        self.switch_count = 0
        self.schedule = self.build_gap_schedule()
        return self.current_observation()

    def build_gap_schedule(self) -> List[List[float]]:
        """构造每 4s 随机变化的期望 gap 日程。"""
        num_gaps = NUM_TARGET_VEHICLES - 1
        num_periods = int(math.ceil(SIM_TIME / GAP_SWITCH_PERIOD)) + 2
        schedule: List[List[float]] = []
        for _ in range(num_periods):
            gaps = [BASE_GAP for _ in range(num_gaps)]
            changed_count = self.gap_rng.randint(0, min(MAX_CHANGED_GAPS_PER_PERIOD, num_gaps))
            changed_indices = self.gap_rng.sample(range(num_gaps), changed_count) if changed_count > 0 else []
            for index in changed_indices:
                gaps[index] = BASE_GAP * self.gap_rng.choice(GAP_MULTIPLIERS)
            schedule.append(gaps)
        return schedule

    def desired_gaps_at(self, t: float) -> List[float]:
        """读取当前时刻的期望 gap。"""
        period_index = min(int(t // GAP_SWITCH_PERIOD), len(self.schedule) - 1)
        return self.schedule[period_index]

    def lane_progress(self) -> float:
        """并道进度。"""
        return clip((self.ego.y - ORIGINAL_LANE_Y) / (TARGET_LANE_Y - ORIGINAL_LANE_Y), 0.0, 1.0)

    def gap_values(self) -> List[float]:
        """当前四个物理 gap。"""
        return [self.targets[i].x - self.targets[i + 1].x for i in range(NUM_TARGET_VEHICLES - 1)]

    def target_accelerations(self) -> List[float]:
        """目标车队 gap 跟踪加速度。"""
        desired = self.desired_gaps_at(self.t)
        accels = [0.0 for _ in self.targets]
        for i in range(NUM_TARGET_VEHICLES - 1):
            gap = self.targets[i].x - self.targets[i + 1].x
            gap_dot = self.targets[i].vx - self.targets[i + 1].vx
            accels[i + 1] = clip(
                GAP_PID_KP * (gap - desired[i]) + GAP_PID_KD * gap_dot,
                -GAP_ACCEL_LIMIT,
                GAP_ACCEL_LIMIT,
            )
        return accels

    def nearest_three_indices(self) -> List[int]:
        """按照 ego 纵向位置选出最近三辆目标车，并按从前到后排序。"""
        nearest = sorted(range(len(self.targets)), key=lambda i: abs(self.targets[i].x - self.ego.x))[:3]
        return sorted(nearest, key=lambda i: self.targets[i].x, reverse=True)

    def high_level(self) -> Dict[str, object]:
        """高层 gap 决策，可选择 opinion 或 max 方法。"""
        triple = self.nearest_three_indices()
        front_pair = (triple[0], triple[1])
        rear_pair = (triple[1], triple[2])
        front_conf = gap_confidence(self.ego, self.targets[front_pair[0]], self.targets[front_pair[1]])
        rear_conf = gap_confidence(self.ego, self.targets[rear_pair[0]], self.targets[rear_pair[1]])
        b_high = front_conf["confidence"] - rear_conf["confidence"]

        y_dot = -HIGH_DAMPING * self.y_high + self.u_high * math.tanh(HIGH_ALPHA * self.y_high) + b_high
        y_power = max(self.y_high * self.y_high, 0.0)
        hill_input = y_power ** HIGH_HILL_N
        hill = HIGH_U_MAX * hill_input / max(HIGH_HILL_K ** HIGH_HILL_N + hill_input, 1e-9)
        u_high_dot = (-self.u_high + hill) / max(HIGH_U_TAU, 1e-9)

        if self.decision_method == "max":
            if front_conf["confidence"] >= rear_conf["confidence"]:
                decision = "FORWARD"
                selected_pair = front_pair
                selected_conf = front_conf
            else:
                decision = "BACKWARD"
                selected_pair = rear_pair
                selected_conf = rear_conf
        else:
            if self.y_high > HIGH_DECISION_THRESHOLD:
                decision = "FORWARD"
                selected_pair = front_pair
                selected_conf = front_conf
            elif self.y_high < -HIGH_DECISION_THRESHOLD:
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
            "b_high": b_high,
            "y_dot": y_dot,
            "u_high_dot": u_high_dot,
            "decision": decision,
            "selected_pair": selected_pair,
            "selected_conf": selected_conf,
        }

    def min_ego_distance(self) -> float:
        """ego 到目标车队的最小距离。"""
        return min(math.hypot(self.ego.x - car.x, self.ego.y - car.y) for car in self.targets)

    def safety_acceleration(self) -> Tuple[float, float]:
        """全局连续避障项。"""
        ux, uy = 0.0, 0.0
        safe_distance = 2.5 * COLLISION_RADIUS
        for target in self.targets:
            rx = self.ego.x - target.x
            ry = self.ego.y - target.y
            dist = safe_norm(rx, ry)
            if dist < safe_distance:
                strength = EGO_K_O * ((safe_distance - dist) / safe_distance) ** 2
                ux += strength * rx / dist
                uy += strength * ry / dist
        return ux, uy

    def low_level_control(self, high: Dict[str, object]) -> Dict[str, object]:
        """底层 z(t) 更新与 ego 控制。"""
        selected_pair = high["selected_pair"]
        if selected_pair is None:
            self.z_low += self.dt * (-Z_DAMPING * self.z_low)
            target_x = self.ego.x + 8.0
            target_y = ORIGINAL_LANE_Y
            target_vx = TARGET_SPEED
            u_t = 0.0
            b_t = 0.0
            pair_signals = {"confidence": 0.0, "d_gap": 0.0, "dv_gap": 0.0, "u_t": 0.0}
        else:
            front_i, rear_i = selected_pair
            front = self.targets[front_i]
            rear = self.targets[rear_i]
            gap = front.x - rear.x
            gap_dot = front.vx - rear.vx
            b_t = compute_gap_bias(gap, gap_dot)
            pair_signals = rbf_u(self.ego, front, rear)
            if self.low_level_policy is None:
                u_t = pair_signals["u_t"]
            else:
                u_t = self.low_level_policy(pair_observation(self.ego, front, rear), pair_signals)
            u_t = clip(float(u_t), RL_U_LOW, RL_U_HIGH)
            z_dot = -Z_DAMPING * self.z_low + u_t * math.tanh(Z_ALPHA * self.z_low) + b_t
            self.z_low += self.dt * z_dot

            w = math.tanh(EGO_K_W * self.z_low)
            target_x = 0.5 * (front.x + rear.x)
            target_y = TARGET_LANE_Y + (1.0 - w) * EGO_R_ETA
            target_vx = 0.5 * (front.vx + rear.vx)

        ax_safe, ay_safe = self.safety_acceleration()
        ax = -EGO_K_P * (self.ego.x - target_x) - EGO_K_V * (self.ego.vx - target_vx) + ax_safe
        ay = -EGO_K_P * (self.ego.y - target_y) - EGO_K_V * self.ego.vy + ay_safe
        ax = clip(ax, -EGO_ACCEL_LIMIT, EGO_ACCEL_LIMIT)
        ay = clip(ay, -EGO_ACCEL_LIMIT, EGO_ACCEL_LIMIT)

        return {
            "ax": ax,
            "ay": ay,
            "b_t": b_t,
            "u_t": u_t,
            "target_x": target_x,
            "target_y": target_y,
            "pair_signals": pair_signals,
        }

    def current_observation(self) -> List[float]:
        """返回当前所选 pair 的 observation；若无选择，则使用最近三车的前 gap。"""
        high = self.high_level()
        pair = high["selected_pair"] or high["front_pair"]
        return pair_observation(self.ego, self.targets[pair[0]], self.targets[pair[1]])

    def step(self) -> Tuple[List[float], float, bool, Dict[str, object]]:
        """推进一步多 gap 仿真。"""
        high = self.high_level()
        selected_pair = high["selected_pair"]
        if selected_pair != self.prev_selected_pair and selected_pair is not None and self.prev_selected_pair is not None:
            self.switch_count += 1
        if selected_pair is not None:
            self.prev_selected_pair = selected_pair

        low = self.low_level_control(high)
        self.y_high += self.dt * float(high["y_dot"])
        self.u_high += self.dt * float(high["u_high_dot"])
        self.u_high = max(0.0, self.u_high)

        target_accels = self.target_accelerations()
        for i, car in enumerate(self.targets):
            car.step(target_accels[i], 0.0, self.dt)
        self.ego.step(float(low["ax"]), float(low["ay"]), self.dt)
        self.t += self.dt

        progress = self.lane_progress()
        progress_delta = progress - self.prev_progress
        gaps = self.gap_values()
        best_gap = max(gaps) if gaps else 0.0
        opportunity = 1.0 if best_gap > GAP_SAFE else 0.0
        min_distance = self.min_ego_distance()
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
        action_smooth_penalty = -0.5 * (normalized_u(float(low["u_t"])) - self.prev_u_norm) ** 2
        direction_flip_penalty = -2.0 if lateral_flip else 0.0
        safety_penalty = -20.0 * max(0.0, (safe_margin - min_distance) / safe_margin) ** 2
        collision = min_distance < COLLISION_RADIUS
        success = abs(self.ego.y - TARGET_LANE_Y) <= 0.25 and min_distance > 1.5 * COLLISION_RADIUS
        collision_penalty = -1000.0 if collision else 0.0
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
        self.prev_u_norm = normalized_u(float(low["u_t"]))
        self.prev_lateral_velocity = self.ego.vy
        done = collision or success or self.t >= SIM_TIME

        info: Dict[str, object] = {
            "time": self.t,
            "reward": reward,
            "reward_terms": reward_terms,
            "progress": progress,
            "success": success,
            "collision": collision,
            "min_distance": min_distance,
            "ego_x": self.ego.x,
            "ego_y": self.ego.y,
            "ego_vx": self.ego.vx,
            "ego_vy": self.ego.vy,
            "gap_values": gaps,
            "best_gap": best_gap,
            "desired_gaps": self.desired_gaps_at(self.t),
            "decision": high["decision"],
            "selected_pair": selected_pair,
            "switch_count": self.switch_count,
            "y_high": self.y_high,
            "u_high": self.u_high,
            "z_low": self.z_low,
            "b_high": high["b_high"],
            "cf": high["front_conf"]["confidence"],
            "cr": high["rear_conf"]["confidence"],
            "b_t": low["b_t"],
            "u_t": low["u_t"],
        }
        return self.current_observation(), reward, done, info


def run_episode(
    seed: Optional[int] = None,
    gap_seed: Optional[int] = None,
    ego_seed: Optional[int] = None,
    decision_method: str = "opinion",
    low_level_policy: Optional[Callable[[List[float], Dict[str, float]], float]] = None,
    csv_path: Optional[str] = None,
) -> Dict[str, object]:
    """运行一次多 gap 仿真。"""
    env = MultiGapEnv(
        seed=seed,
        gap_seed=gap_seed,
        ego_seed=ego_seed,
        decision_method=decision_method,
        low_level_policy=low_level_policy,
    )
    rows: List[Dict[str, object]] = []
    total_reward = 0.0
    done = False
    info: Dict[str, object] = {}
    while not done:
        _, reward, done, info = env.step()
        total_reward += reward
        rows.append({
            "time": info["time"],
            "reward": reward,
            "total_reward": total_reward,
            "progress": info["progress"],
            "success": float(info["success"]),
            "collision": float(info["collision"]),
            "min_distance": info["min_distance"],
            "ego_x": info["ego_x"],
            "ego_y": info["ego_y"],
            "decision": info["decision"],
            "selected_pair": str(info["selected_pair"]),
            "switch_count": info["switch_count"],
            "best_gap": info["best_gap"],
            "y_high": info["y_high"],
            "u_high": info["u_high"],
            "z_low": info["z_low"],
            "u_t": info["u_t"],
            "cf": info["cf"],
            "cr": info["cr"],
        })

    if csv_path and rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

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
        "min_distance": info["min_distance"],
        "switch_count": env.switch_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="多 gap 独立仿真环境")
    parser.add_argument("--seed", type=int, default=None, help="统一随机种子")
    parser.add_argument("--gap-seed", type=int, default=None, help="gap 调度随机种子")
    parser.add_argument("--ego-seed", type=int, default=None, help="ego 初始位置随机种子")
    parser.add_argument("--decision", choices=["opinion", "max"], default="opinion", help="高层决策方法")
    parser.add_argument("--csv", default="multi_gap_rollout.csv", help="逐步仿真 CSV")
    args = parser.parse_args()
    result = run_episode(
        seed=args.seed,
        gap_seed=args.gap_seed,
        ego_seed=args.ego_seed,
        decision_method=args.decision,
        csv_path=args.csv,
    )
    print("多 gap 仿真完成：")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print(f"逐步结果已保存：{args.csv}")


if __name__ == "__main__":
    main()
