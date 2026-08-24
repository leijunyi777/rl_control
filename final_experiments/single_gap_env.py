"""单 gap 环境兼容层。

本文件不再手动复刻 main12/main11 的内部实现，而是直接从
``code/python/main12_sac_train.py`` 和 ``code/python/main11_sac_train.py``
导入原始类、函数和参数。这样可以保证最终实验中的单 gap 环境与原始
main12 训练环境使用完全相同的车辆动力学、reward、reset 顺序和 SAC 组件。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_PYTHON_DIR = ROOT_DIR / "code" / "python"
if str(CODE_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_PYTHON_DIR))

from main12_sac_train import (  # noqa: E402
    EGO_X_BASE,
    EGO_X_RANDOM_RANGE,
    Main12SacUEnv as _OriginalMain12SacUEnv,
    POLICY_PATH as ORIGINAL_MAIN12_POLICY_PATH,
    RESULT_FIG_PATH as ORIGINAL_MAIN12_RESULT_FIG_PATH,
    plot_training_results,
    sample_ego_x,
)
from main11_sac_train import (  # noqa: E402
    ACTION_DIM,
    ALPHA_LR,
    BATCH_SIZE,
    DT,
    GAMMA,
    HIDDEN_SIZE,
    INITIAL_RANDOM_STEPS,
    Main11SacUDynamics,
    Main11SacUEnv,
    NUM_EPISODES,
    POLICY_LR,
    Q_LR,
    RENDER_DURING_TRAINING,
    REPLAY_SIZE,
    SACAgent,
    SEED,
    SIM_TIME,
    TAU,
    U_HIGH,
    U_LOW,
    UPDATES_PER_STEP,
    GaussianPolicy,
    QNetwork,
    ReplayBuffer,
    moving_average,
    set_seed,
)
from main7 import get_veh12_gap  # noqa: E402
from models_ode import (  # noqa: E402
    EgoVehicleOdeModel,
    KinematicBicycleModel,
    _signed_safe,
    compute_gap_bias_bt,
    compute_gap_confidence_attention_ut,
    compute_gap_confidence_signals_from_states,
    compute_gap_opinion_z_dot,
    front_position,
    front_velocity,
    rear_state_derivative,
)
from utils import draw_car, draw_environment  # noqa: E402


# 原始 main11/main12 reward 参数。多 gap 脚本会从本文件读取这些名字。
PROGRESS_REWARD_GAIN = 80.0
LANE_PROGRESS_STEP_GAIN = 0.15
OPPORTUNITY_PROGRESS_GAIN = 40.0
REVERSE_PROGRESS_PENALTY_GAIN = 30.0
HESITATION_PENALTY_GAIN = 0.25
TIME_PENALTY_GAIN = 0.05
ACTION_SMOOTH_PENALTY_GAIN = 0.5
DIRECTION_FLIP_PENALTY = 2.0
SAFETY_MARGIN_FACTOR = 2.5
SAFETY_PENALTY_GAIN = 20.0
COLLISION_PENALTY = 1000.0
SUCCESS_SAFE_FACTOR = 1.5
SUCCESS_BONUS_BASE = 100.0
SUCCESS_TIME_PENALTY_GAIN = 2.0
TIMEOUT_PROGRESS_PENALTY_GAIN = 0.0


plt = None


def require_matplotlib():
    """延迟导入 matplotlib，保持无图训练时的依赖加载更轻。"""
    global plt
    if plt is None:
        import matplotlib.pyplot as _plt

        plt = _plt
    return plt


class Main12SacUEnv(_OriginalMain12SacUEnv):
    """原始 main12 SAC-u 环境。

    这个类不重写 ``reset`` 或 ``step``，训练脚本使用它时会得到与
    ``code/python/main12_sac_train.py`` 完全一致的行为。
    """


class SingleGapEnv(Main12SacUEnv):
    """给最终实验其它脚本使用的轻量兼容接口。

    训练时应直接使用 ``Main12SacUEnv``；本类只额外支持固定 ego 初始位置、
    ``step_action`` 和真实 ``u(t)`` 输入，方便单次回放和 RBF 对比。
    当 ``reset`` 不传入 ``ego_x0`` 时，它会调用原始 main12 的随机 reset。
    """

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
            obs = _OriginalMain12SacUEnv.reset(self)
        else:
            obs = self._reset_fixed_ego_x(float(ego_x0))
        self.front = self.veh1
        self.rear = self.veh2
        return obs

    def _reset_fixed_ego_x(self, ego_x0: float):
        self.ego_x0 = ego_x0
        self.veh1 = KinematicBicycleModel(
            id="Veh 1 (Leader)",
            x=30.0,
            y=self.target_lane_y,
            v=15.0,
            L=self.vehicle_l,
            color="lightblue",
        )
        self.veh2 = KinematicBicycleModel(
            id="Veh 2 (Gap Control)",
            x=15.0,
            y=self.target_lane_y,
            v=15.0,
            L=self.vehicle_l,
            color="royalblue",
        )
        self.ego = EgoVehicleOdeModel(
            id="Veh 3 (Ego Main12 SAC-u)",
            x=ego_x0,
            y=self.original_lane_y,
            v=15.0,
            L=self.vehicle_l,
            color="lightgreen",
        )
        self.dynamics = Main11SacUDynamics(
            self.veh1,
            self.veh2,
            self.ego,
            desired_gap=self.desired_gap,
            gap_safe=self.gap_safe,
        )
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
        obs, reward, done, info = super().step(action)
        return obs, reward, done, self._augment_info(info, reward)

    def step(self, u_t: float):
        action = np.array([2.0 * (float(u_t) - U_LOW) / (U_HIGH - U_LOW) - 1.0], dtype=np.float32)
        obs, reward, done, info = super().step(action)
        return obs, reward, done, self._augment_info(info, reward)

    def _augment_info(self, info: Dict[str, float], reward: float):
        info = dict(info)
        info["progress"] = info["lane_progress"]
        info["gap"] = info["veh12_gap"]
        info["min_distance"] = min(info["dist1"], info["dist2"])
        info["collision"] = info["collided"]
        info["reward"] = reward
        return info


def compute_rbf_u(ego, front, rear):
    """计算原始 RBF 公式给出的手工 ``u(t)``。"""
    return compute_gap_confidence_attention_ut(
        ego.get_state(),
        front.get_state(),
        rear.get_state(),
        ego.L,
        front.L,
        rear.L,
    )


def run_episode(
    seed: Optional[int] = None,
    policy: Optional[Callable[[List[float], Dict[str, float]], float]] = None,
    csv_path: Optional[str] = None,
    render: bool = False,
):
    """运行一次单 gap 仿真。

    ``policy`` 接收 ``obs`` 和上一时刻 ``info``，返回真实 ``u(t)``。
    未提供 policy 时，默认使用原始 RBF 公式的 ``u(t)``。
    """
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
        rows.append(
            {
                "time": info["time"],
                "reward": reward,
                "total_reward": total_reward,
                "progress": info["lane_progress"],
                "u_t": info["u_t"],
                "formula_u_t": info["formula_u_t"],
                "z": info["z_new"],
                "gap": info["veh12_gap"],
                "dist1": info["dist1"],
                "dist2": info["dist2"],
                "success": float(info["success"]),
                "collision": float(info["collided"]),
            }
        )

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
        "min_distance": min(info["dist1"], info["dist2"]),
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
