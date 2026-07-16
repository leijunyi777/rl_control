import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp

from main7 import Main7GapFollowingDynamics, get_veh12_gap
from models_ode import KinematicBicycleModel, EgoVehicleOdeModel, front_velocity
from utils import draw_car, draw_environment


# =========================
# 训练与仿真参数
# =========================
NUM_EPISODES = 50
RENDER_DURING_TRAINING = False
SIM_TIME = 40.0
DT = 0.05
SEED = 7

# SAC 参数
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

# 动作对应的参数范围：action = [k_mu, k, eps]
ACTION_LOW = np.array([0.5, 1.0, 0.01], dtype=np.float32)
ACTION_HIGH = np.array([12.0, 50.0, 0.20], dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Main7SacMergeEnv:
    """基于 main7 仿真循环封装的强化学习环境。"""

    def __init__(self, render=False, sim_time=SIM_TIME, dt=DT):
        self.render_enabled = render
        self.sim_time = sim_time
        self.dt = dt
        self.lane_width = 4.0
        self.vehicle_l = 2.8
        self.original_lane_y = self.lane_width * 0.5
        self.target_lane_y = self.lane_width * 1.5
        self.desired_gap = 20.0
        self.obs_scale = np.array([40.0, 8.0, 20.0, 10.0, 40.0, 8.0, 20.0, 10.0], dtype=np.float32)
        self.fig = None
        self.ax_anim = None
        self.ax_mu_z = None
        self.ax_dist = None
        self.reset()

    def reset(self):
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
            id="Veh 3 (Ego RL)",
            x=20.0,
            y=self.original_lane_y,
            v=15.0,
            L=self.vehicle_l,
            color="lightgreen",
        )
        self.dynamics = Main7GapFollowingDynamics(self.veh1, self.veh2, self.ego, desired_gap=self.desired_gap)
        self.collision_radius = self.ego.r
        self.state = self.dynamics.pack_state()
        self.t = 0.0
        self.prev_lane_progress = self._lane_progress()
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_lateral_velocity = front_velocity(self.state[10:15], self.ego.L)[1]
        self.t_hist, self.mu_hist, self.z_hist = [], [], []
        self.dist1_hist, self.dist2_hist, self.veh12_gap_hist = [], [], []
        return self._get_obs()

    def _normalized_action_to_params(self, action):
        action = np.clip(action, -1.0, 1.0)
        ratio = 0.5 * (action + 1.0)
        return ACTION_LOW + ratio * (ACTION_HIGH - ACTION_LOW)

    def _set_controller_params(self, action):
        k_mu, k, eps = self._normalized_action_to_params(action)
        self.ego.k_mu = float(k_mu)
        self.ego.k = float(k)
        self.ego.eps = float(eps)
        return np.array([k_mu, k, eps], dtype=np.float32)

    def _get_obs(self):
        diag = self.dynamics.diagnostics(self.state)
        rel1 = diag["sensor_data"]["veh1"]
        rel2 = diag["sensor_data"]["veh2"]

        # 观测定义为 ego 相对于前车和后车的位置、速度。
        ego_rel_pos_1 = -rel1["rel_p"]
        ego_rel_vel_1 = -rel1["rel_v"]
        ego_rel_pos_2 = -rel2["rel_p"]
        ego_rel_vel_2 = -rel2["rel_v"]
        obs = np.concatenate([ego_rel_pos_1, ego_rel_vel_1, ego_rel_pos_2, ego_rel_vel_2]).astype(np.float32)
        return np.clip(obs / self.obs_scale, -5.0, 5.0).astype(np.float32)

    def _lane_progress(self):
        relative_y = (self.ego.y - self.original_lane_y) / (self.target_lane_y - self.original_lane_y)
        return float(np.clip(relative_y, 0.0, 1.0))

    def _distances(self):
        diag = self.dynamics.diagnostics(self.state)
        return diag["sensor_data"]["veh1"]["dist"], diag["sensor_data"]["veh2"]["dist"]

    def _veh12_gap(self):
        return get_veh12_gap(self.state, self.veh1.L, self.veh2.L)

    def _is_success(self, lane_progress, min_distance):
        return (
            lane_progress > 0.95
            and abs(self.ego.y - self.target_lane_y) < 0.2
            and min_distance > 1.5 * self.collision_radius
        )

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        params = self._set_controller_params(action)
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

        dist1, dist2 = self._distances()
        veh12_gap = self._veh12_gap()
        ego_min_distance = min(dist1, dist2)
        env_min_distance = min(ego_min_distance, veh12_gap)
        lane_progress = self._lane_progress()
        progress_delta = lane_progress - self.prev_lane_progress
        opportunity = 1.0 if self.state[16] > 0.1 else 0.0
        current_lateral_velocity = front_velocity(self.state[10:15], self.ego.L)[1]
        lateral_direction_flip = (
            self.prev_lateral_velocity * current_lateral_velocity < 0.0
            and abs(self.prev_lateral_velocity) > 1e-3
            and abs(current_lateral_velocity) > 1e-3
        )
        safe_margin = 2.5 * self.collision_radius

        progress_reward = 5.0 * progress_delta
        opportunity_reward = 2.0 * opportunity * max(progress_delta, 0.0)
        hesitation_penalty = -0.05 * opportunity * (1.0 - lane_progress)
        time_penalty = -0.02 * (1.0 - lane_progress)
        action_smooth_penalty = -0.5 * float(np.sum((action - self.prev_action) ** 2))
        direction_flip_penalty = -2.0 if lateral_direction_flip else 0.0
        safety_penalty = -20.0 * max(0.0, (safe_margin - ego_min_distance) / safe_margin) ** 2

        collided = env_min_distance < self.collision_radius
        success = self._is_success(lane_progress, ego_min_distance)
        collision_penalty = -1000.0 if collided else 0.0
        success_bonus = (100.0 - 2.0 * self.t) if success else 0.0
        reward = (
            progress_reward
            + opportunity_reward
            + hesitation_penalty
            + time_penalty
            + action_smooth_penalty
            + direction_flip_penalty
            + safety_penalty
            + collision_penalty
            + success_bonus
        )
        self.prev_lane_progress = lane_progress
        self.prev_action = action.copy()
        self.prev_lateral_velocity = current_lateral_velocity

        done = collided or success or self.t >= self.sim_time
        obs = self._get_obs()
        info = {
            "params": params,
            "lane_progress": lane_progress,
            "dist1": dist1,
            "dist2": dist2,
            "veh12_gap": veh12_gap,
            "collided": collided,
            "success": success,
            "time": self.t,
            "reward_terms": {
                "progress": progress_reward,
                "opportunity": opportunity_reward,
                "hesitation": hesitation_penalty,
                "time": time_penalty,
                "action_smooth": action_smooth_penalty,
                "direction_flip": direction_flip_penalty,
                "safety": safety_penalty,
                "collision": collision_penalty,
                "success": success_bonus,
            },
        }

        self.t_hist.append(self.t)
        self.mu_hist.append(self.state[16])
        self.z_hist.append(self.state[15])
        self.dist1_hist.append(dist1)
        self.dist2_hist.append(dist2)
        self.veh12_gap_hist.append(veh12_gap)
        if self.render_enabled:
            self.render(info)

        return obs, float(reward), done, info

    def render(self, info):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 8))
            self.ax_anim = plt.subplot(2, 1, 1)
            self.ax_mu_z = plt.subplot(2, 2, 3)
            self.ax_dist = plt.subplot(2, 2, 4)

        self.ax_anim.cla()
        draw_environment(self.ax_anim, self.lane_width)
        draw_car(self.ax_anim, self.veh1, wheelbase=self.collision_radius)
        draw_car(self.ax_anim, self.veh2, wheelbase=self.collision_radius)
        draw_car(self.ax_anim, self.ego, wheelbase=self.collision_radius)
        self.ax_anim.set_xlim(self.ego.x - 15, self.ego.x + 45)
        self.ax_anim.set_ylim(-2, self.lane_width * 2 + 2)
        self.ax_anim.set_aspect("equal")
        title = f"Time: {self.t:.2f}s | SAC training rollout"
        if info["collided"]:
            title += " | COLLISION"
        if info["success"]:
            title += " | SUCCESS"
        self.ax_anim.set_title(title)

        self.ax_mu_z.cla()
        self.ax_mu_z.plot(self.t_hist, self.mu_hist, "c-", linewidth=2, label="Env Score ($\\mu$)")
        self.ax_mu_z.plot(self.t_hist, self.z_hist, "b-", linewidth=3, label="Opinion State ($z$)")
        self.ax_mu_z.axhline(0, color="gray", linestyle="--")
        self.ax_mu_z.axvline(20.0, color="black", linestyle=":", linewidth=1.5, label="Gap Control Starts")
        self.ax_mu_z.set_xlim(0, self.sim_time)
        self.ax_mu_z.set_ylim(-1.0, 1.5)
        self.ax_mu_z.set_title("Decision Dynamics ($\\mu$ and $z$)")
        self.ax_mu_z.legend(loc="upper left")
        self.ax_mu_z.grid(True)

        self.ax_dist.cla()
        self.ax_dist.plot(self.t_hist, self.dist1_hist, "purple", linewidth=2, label="Distance to Veh 1")
        self.ax_dist.plot(self.t_hist, self.dist2_hist, "red", linewidth=2, label="Distance to Veh 2")
        self.ax_dist.plot(self.t_hist, self.veh12_gap_hist, "gray", linestyle="-.", linewidth=2, label="Veh1-Veh2 Gap")
        self.ax_dist.axhline(
            self.collision_radius,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Collision Threshold r={self.collision_radius:g}m",
        )
        self.ax_dist.axhline(self.desired_gap, color="green", linestyle=":", linewidth=2, label="Target Gap 20m")
        self.ax_dist.axvline(20.0, color="black", linestyle=":", linewidth=1.5)
        self.ax_dist.set_xlim(0, self.sim_time)
        upper_distance = max([
            self.collision_radius * 2.0,
            self.desired_gap * 1.2,
            *self.dist1_hist,
            *self.dist2_hist,
            *self.veh12_gap_hist,
        ])
        self.ax_dist.set_ylim(0, upper_distance * 1.1)
        self.ax_dist.set_title("Relative Distance Monitoring")
        self.ax_dist.legend(loc="upper right")
        self.ax_dist.grid(True)
        plt.pause(0.001)


class ReplayBuffer:
    """SAC 经验回放池。"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.as_tensor(state, dtype=torch.float32, device=device),
            torch.as_tensor(action, dtype=torch.float32, device=device),
            torch.as_tensor(reward[:, None], dtype=torch.float32, device=device),
            torch.as_tensor(next_state, dtype=torch.float32, device=device),
            torch.as_tensor(done[:, None], dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    """输出 tanh-squashed 高斯动作的策略网络。"""

    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20.0, 2.0)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class QNetwork(nn.Module):
    """SAC 的 Q 函数网络。"""

    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent:
    """Soft Actor-Critic 智能体。"""

    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.policy = GaussianPolicy(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.q1 = QNetwork(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.q2 = QNetwork(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.target_q1 = QNetwork(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.target_q2 = QNetwork(state_dim, action_dim, HIDDEN_SIZE).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=POLICY_LR)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=Q_LR)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=Q_LR)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.policy.sample(state_tensor)
        chosen = mean_action if evaluate else action
        return chosen.squeeze(0).cpu().numpy()

    def update(self, replay_buffer):
        state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE, self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_q = torch.min(
                self.target_q1(next_state, next_action),
                self.target_q2(next_state, next_action),
            ) - self.alpha * next_log_prob
            target = reward + (1.0 - done) * GAMMA * target_q

        q1_loss = F.mse_loss(self.q1(state, action), target)
        q2_loss = F.mse_loss(self.q2(state, action), target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_action, log_prob, _ = self.policy.sample(state)
        policy_loss = (self.alpha * log_prob - torch.min(self.q1(state, new_action), self.q2(state, new_action))).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update(self, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(TAU * source_param.data + (1.0 - TAU) * target_param.data)


def moving_average(values, window=10):
    if len(values) < window:
        return np.asarray(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_results(episode_rewards, episode_progress, episode_collisions, episode_successes):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    episodes = np.arange(1, len(episode_rewards) + 1)

    axes[0].plot(episodes, episode_rewards, color="tab:blue", alpha=0.35, label="Episode reward")
    avg_reward = moving_average(episode_rewards, 10)
    axes[0].plot(np.arange(len(avg_reward)) + 1, avg_reward, color="tab:blue", linewidth=2, label="10-episode average")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("SAC Training Result")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(episodes, episode_progress, color="tab:green", label="Final lane-change progress")
    axes[1].set_ylabel("Progress")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(episodes, episode_collisions, color="tab:red", label="Collision")
    axes[2].set_ylabel("Collision")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(episodes, episode_successes, color="tab:purple", label="Success")
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Success")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend()
    axes[3].grid(True)

    fig.tight_layout()
    fig.savefig("sac_training_result.png", dpi=180)
    plt.show()


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Main7SacMergeEnv(render=RENDER_DURING_TRAINING)
    state_dim = env.reset().shape[0]
    action_dim = 3

    agent = SACAgent(state_dim, action_dim, device)
    replay_buffer = ReplayBuffer(REPLAY_SIZE)

    total_steps = 0
    episode_rewards = []
    episode_progress = []
    episode_collisions = []
    episode_successes = []

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0
        last_info = {"lane_progress": 0.0, "collided": False, "success": False}

        for _ in range(int(SIM_TIME / DT)):
            if total_steps < INITIAL_RANDOM_STEPS:
                action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            last_info = info
            total_steps += 1

            if len(replay_buffer) >= BATCH_SIZE:
                for _ in range(UPDATES_PER_STEP):
                    agent.update(replay_buffer)

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_progress.append(last_info["lane_progress"])
        episode_collisions.append(float(last_info["collided"]))
        episode_successes.append(float(last_info["success"]))

        print(
            f"Episode {episode:04d} | reward={episode_reward:8.2f} | "
            f"progress={last_info['lane_progress']:.3f} | "
            f"success={last_info['success']} | collision={last_info['collided']} | steps={total_steps}"
        )

    plot_training_results(episode_rewards, episode_progress, episode_collisions, episode_successes)
    torch.save(
        {
            "policy_state_dict": agent.policy.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_low": ACTION_LOW,
            "action_high": ACTION_HIGH,
        },
        "sac_policy.pth",
    )
    print("Saved trained policy to sac_policy.pth")


if __name__ == "__main__":
    train()
