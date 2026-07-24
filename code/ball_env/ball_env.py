"""A 2D ball navigation environment with paper-style safety feedback.

The controlled ball has radius 1.0 and moves from the left side of the
workspace to a goal on the right side. Nine larger balls move on small circular
orbits. The obstacle layout is checked analytically so obstacle balls never
collide with each other.

Safe control follows the structure used in example.md:

    u = u_n + u_c
    u_c = -sum_j k_o * g_j * dot(d_j) / d_j

where d_j is the clearance between the controlled ball and obstacle j, g_j is
the unit vector from obstacle j to the controlled ball, and dot(d_j) is the
relative velocity projected onto g_j. A small tangential term is added from the
same barrier strength so the ball is deflected around obstacles instead of only
braking in front of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple


Vector = Tuple[float, float]


def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1])


def _mul(value: float, vec: Vector) -> Vector:
    return (value * vec[0], value * vec[1])


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross(a: Vector, b: Vector) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _norm(vec: Vector) -> float:
    return math.hypot(vec[0], vec[1])


def _unit_or_zero(vec: Vector, eps: float = 1e-9) -> Vector:
    length = _norm(vec)
    if length < eps:
        return (0.0, 0.0)
    return (vec[0] / length, vec[1] / length)


def _limit_norm(vec: Vector, max_norm: float) -> Vector:
    length = _norm(vec)
    if length <= max_norm:
        return vec
    return _mul(max_norm / length, vec)


def _signed_safe(value: float, eps: float = 1e-6) -> float:
    if abs(value) >= eps:
        return value
    return eps if value >= 0.0 else -eps


@dataclass(frozen=True)
class RotatingBall:
    """A circular obstacle whose center follows a small circular orbit."""

    center: Tuple[float, float]
    radius: float
    orbit_radius: float
    omega: float
    phase: float

    def position(self, t: float) -> Vector:
        angle = self.omega * t + self.phase
        return (
            self.center[0] + self.orbit_radius * math.cos(angle),
            self.center[1] + self.orbit_radius * math.sin(angle),
        )

    def velocity(self, t: float) -> Vector:
        angle = self.omega * t + self.phase
        return (
            -self.orbit_radius * self.omega * math.sin(angle),
            self.orbit_radius * self.omega * math.cos(angle),
        )


@dataclass
class EnvConfig:
    dt: float = 0.04
    max_time: float = 80.0
    ego_radius: float = 1.0
    start: Tuple[float, float] = (-48.0, 0.0)
    goal: Tuple[float, float] = (48.0, 0.0)
    goal_radius: float = 2.0
    k_goal: float = 0.65
    k_damping: float = 1.25
    k_o: float = 32.0
    tangential_gain: float = 1.50
    influence_distance: float = 14.0
    max_accel: float = 22.0
    max_speed: float = 10.0
    obstacles: List[RotatingBall] = field(default_factory=list)


class BallAvoidanceEnv:
    """Simple step/reset environment for the moving-ball obstacle scene."""

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        if not self.config.obstacles:
            self.config.obstacles = self._default_obstacles()
        self._validate_obstacles()
        self.t = 0.0
        self.pos = self.config.start
        self.vel = (0.0, 0.0)
        self.done = False
        self.stop_reason = "running"
        self.path: List[Vector] = []

    @staticmethod
    def _default_obstacles() -> List[RotatingBall]:
        return [
            RotatingBall((-32.0, 3.0), 3.0, 2.00, 0.50, 0.00),
            RotatingBall((-16.0, -3.5), 4.0, 2.30, -0.43, 0.70),
            RotatingBall((0.0, 3.0), 2.5, 2.60, 0.38, 1.30),
            RotatingBall((16.0, -3.5), 4.2, 2.10, -0.36, 2.40),
            RotatingBall((32.0, 3.0), 3.6, 2.40, 0.55, 3.10),
            RotatingBall((-24.0, 20.0), 2.8, 2.20, -0.48, 0.40),
            RotatingBall((8.0, 21.0), 5.0, 2.00, 0.42, 2.00),
            RotatingBall((24.0, -21.0), 2.0, 2.50, -0.52, 1.00),
            RotatingBall((-8.0, -21.0), 4.5, 2.10, 0.34, 2.80),
        ]

    def _validate_obstacles(self) -> None:
        obstacles = self.config.obstacles
        if len(obstacles) != 9:
            raise ValueError(f"Expected 9 obstacle balls, got {len(obstacles)}.")

        for index, ball in enumerate(obstacles):
            if not 2.0 <= ball.radius <= 5.0:
                raise ValueError(
                    f"Obstacle {index} radius must be in [2, 5], got {ball.radius}."
                )
            if ball.orbit_radius > 3.0:
                raise ValueError(
                    f"Obstacle {index} orbit radius is too large: {ball.orbit_radius}."
                )

        for i, first in enumerate(obstacles):
            for j, second in enumerate(obstacles[i + 1 :], start=i + 1):
                center_gap = _norm(_sub(first.center, second.center))
                worst_case_need = (
                    first.radius
                    + second.radius
                    + first.orbit_radius
                    + second.orbit_radius
                )
                if center_gap <= worst_case_need:
                    raise ValueError(
                        "Obstacle balls can collide: "
                        f"{i} and {j}, center_gap={center_gap:.2f}, "
                        f"need>{worst_case_need:.2f}."
                    )

        if not self._straight_path_blocked():
            raise ValueError(
                "At least one obstacle must block the straight path from start to goal."
            )

    def _straight_path_blocked(self) -> bool:
        start = self.config.start
        goal = self.config.goal
        path = _sub(goal, start)
        path_length_sq = max(_dot(path, path), 1e-9)

        for ball in self.config.obstacles:
            obstacle_pos = ball.position(0.0)
            offset = _sub(obstacle_pos, start)
            ratio = max(0.0, min(1.0, _dot(offset, path) / path_length_sq))
            closest = _add(start, _mul(ratio, path))
            distance_to_path = _norm(_sub(obstacle_pos, closest))
            if distance_to_path <= self.config.ego_radius + ball.radius:
                return True
        return False

    def reset(
        self,
        start: Optional[Tuple[float, float]] = None,
        velocity: Optional[Tuple[float, float]] = None,
    ) -> List[float]:
        self.t = 0.0
        self.pos = start or self.config.start
        self.vel = velocity or (0.0, 0.0)
        self.done = False
        self.stop_reason = "running"
        self.path = [self.pos]
        return self.observation()

    def obstacle_states(self, t: Optional[float] = None) -> List[Dict[str, object]]:
        time = self.t if t is None else t
        return [
            {
                "position": ball.position(time),
                "velocity": ball.velocity(time),
                "radius": ball.radius,
            }
            for ball in self.config.obstacles
        ]

    def observation(self) -> List[float]:
        goal_vec = _sub(self.config.goal, self.pos)
        values = [self.pos[0], self.pos[1], self.vel[0], self.vel[1], goal_vec[0], goal_vec[1]]
        for state in self.obstacle_states():
            px, py = state["position"]
            vx, vy = state["velocity"]
            values.extend([px, py, vx, vy, float(state["radius"])])
        return values

    def nominal_control(self) -> Vector:
        goal_vec = _sub(self.config.goal, self.pos)
        return _sub(_mul(self.config.k_goal, goal_vec), _mul(self.config.k_damping, self.vel))

    def safe_control(self) -> Tuple[Vector, List[Dict[str, float]]]:
        u_c = (0.0, 0.0)
        details: List[Dict[str, float]] = []
        path_vec = _sub(self.config.goal, self.config.start)

        for index, ball in enumerate(self.config.obstacles):
            obs_pos = ball.position(self.t)
            obs_vel = ball.velocity(self.t)
            rel_pos = _sub(self.pos, obs_pos)
            rel_vel = _sub(self.vel, obs_vel)
            dist = _norm(rel_pos)
            g_j = _unit_or_zero(rel_pos)
            clearance = dist - (self.config.ego_radius + ball.radius)
            dot_d = _dot(g_j, rel_vel)

            active_range = self.config.influence_distance
            active = clearance < active_range and dot_d < 0.0
            if active:
                clearance_safe = _signed_safe(max(clearance, 0.05))
                influence = ((active_range - clearance) / active_range) ** 2
                barrier_strength = self.config.k_o * (-dot_d / clearance_safe) * influence

                radial = _mul(barrier_strength, g_j)
                tangent = (-g_j[1], g_j[0])
                obstacle_side = _cross(path_vec, _sub(obs_pos, self.config.start))
                side = float(math.copysign(1.0, obstacle_side))
                if abs(obstacle_side) < 1e-6:
                    side = float(math.copysign(1.0, ball.omega)) if abs(ball.omega) > 1e-9 else 1.0
                tangential = _mul(self.config.tangential_gain * barrier_strength * side, tangent)
                u_c = _add(u_c, _add(radial, tangential))

            details.append(
                {
                    "index": float(index),
                    "distance": dist,
                    "clearance": clearance,
                    "dot_d": dot_d,
                    "active": float(active),
                }
            )

        return u_c, details

    def control(self, external_accel: Optional[Vector] = None) -> Tuple[Vector, Dict[str, object]]:
        u_n = self.nominal_control()
        u_c, safety = self.safe_control()
        u = _add(u_n, u_c)
        if external_accel is not None:
            u = _add(u, external_accel)

        u = _limit_norm(u, self.config.max_accel)

        info = {
            "u_n": u_n,
            "u_c": u_c,
            "safety": safety,
        }
        return u, info

    def step(self, external_accel: Optional[Vector] = None) -> Tuple[List[float], float, bool, Dict[str, object]]:
        if self.done:
            return self.observation(), 0.0, True, self._info({})

        accel, control_info = self.control(external_accel)
        self.vel = _add(self.vel, _mul(self.config.dt, accel))

        self.vel = _limit_norm(self.vel, self.config.max_speed)

        self.pos = _add(self.pos, _mul(self.config.dt, self.vel))
        self.t += self.config.dt
        self.path.append(self.pos)

        collided = self._collided()
        reached_goal = _norm(_sub(self.config.goal, self.pos)) <= self.config.goal_radius
        timed_out = self.t >= self.config.max_time

        self.done = collided or reached_goal or timed_out
        if collided:
            self.stop_reason = "collision"
        elif reached_goal:
            self.stop_reason = "success"
        elif timed_out:
            self.stop_reason = "timeout"
        else:
            self.stop_reason = "running"

        reward = self._reward(collided, reached_goal)
        return self.observation(), reward, self.done, self._info(control_info)

    def _collided(self) -> bool:
        for ball in self.config.obstacles:
            dist = _norm(_sub(self.pos, ball.position(self.t)))
            if dist <= self.config.ego_radius + ball.radius:
                return True
        return False

    def _reward(self, collided: bool, reached_goal: bool) -> float:
        goal_dist = _norm(_sub(self.config.goal, self.pos))
        reward = -0.01 * goal_dist - 0.02
        if collided:
            reward -= 100.0
        if reached_goal:
            reward += 100.0
        return reward

    def _info(self, control_info: Dict[str, object]) -> Dict[str, object]:
        min_clearance = min(
            _norm(_sub(self.pos, ball.position(self.t))) - (self.config.ego_radius + ball.radius)
            for ball in self.config.obstacles
        )
        return {
            "time": self.t,
            "position": self.pos,
            "velocity": self.vel,
            "stop_reason": self.stop_reason,
            "min_clearance": min_clearance,
            **control_info,
        }

    def render(self, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-55, 55)
        ax.set_ylim(-35, 35)
        ax.set_title("Ball obstacle avoidance")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)

        for ball in self.config.obstacles:
            pos = ball.position(self.t)
            orbit = Circle(ball.center, ball.orbit_radius, fill=False, color="0.7", linestyle="--")
            obstacle = Circle(pos, ball.radius, color="#3B82F6", alpha=0.85)
            ax.add_patch(orbit)
            ax.add_patch(obstacle)

        if len(self.path) > 1:
            xs = [point[0] for point in self.path]
            ys = [point[1] for point in self.path]
            ax.plot(xs, ys, color="#111827", linewidth=1.5)

        ego = Circle(self.pos, self.config.ego_radius, color="#EF4444", zorder=5)
        goal = Circle(self.config.goal, self.config.goal_radius, color="#22C55E", alpha=0.8)
        ax.add_patch(goal)
        ax.add_patch(ego)
        ax.text(self.config.goal[0], self.config.goal[1] + 3.0, "Goal", ha="center")
        ax.text(self.config.start[0], self.config.start[1] - 4.0, "Start", ha="center")
        ax.text(-54, 25, f"t={self.t:.2f}s, status={self.stop_reason}", ha="left")
        return ax
