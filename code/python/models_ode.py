import numpy as np


def _signed_safe(value, eps=1e-6):
    if abs(value) < eps:
        return eps if value >= 0.0 else -eps
    return value


class KinematicBicycleModel:
    """自行车车辆模型，控制参考点取前轴中心。"""

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

    def get_front_axle(self):
        return front_position(self.get_state(), self.L)

    def get_velocity_vector(self):
        return front_velocity(self.get_state(), self.L)


def front_position(state, L):
    x, y, theta, _, _ = state
    return np.array([x + L * np.cos(theta), y + L * np.sin(theta)])


def front_velocity(state, L):
    _, _, theta, vr, delta = state
    tan_delta = np.tan(delta)
    return vr * np.array([
        np.cos(theta) - np.sin(theta) * tan_delta,
        np.sin(theta) + np.cos(theta) * tan_delta,
    ])


def rear_state_derivative(state, a, omega, L):
    _, _, theta, vr, delta = state
    delta_dot = omega
    theta_dot = vr / L * (delta if abs(delta) < 1e-3 else np.tan(delta))
    return np.array([
        vr * np.cos(theta),
        vr * np.sin(theta),
        theta_dot,
        a,
        delta_dot,
    ])


class EgoVehicleOdeModel(KinematicBicycleModel):
    """ego 车辆模型，包含意见动态、分岔参数动态和安全控制器。"""

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

    def set_decision_state(self, z, mu):
        self.z = float(z)
        self.mu = float(mu)

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
            sensor_data[name] = {
                "rel_p": rel_p,
                "rel_v": rel_v,
                "dist": np.linalg.norm(rel_p),
            }
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

    def compute_z_dot_new(self, z, mu):
        """计算意见状态 z 的导数 (饱和跨临界意见动力学模型)"""
        # 推荐将以下参数设为类的动态属性 (e.g., self.d, self.u)，实现可调灵敏度
        d = 10.0   # 惯性/阻力 (防止意见突变)
        u = 2.0   # 注意力强度 (打破僵局的驱动力)
        k = 1.0   # 灵敏度系数 (越大切换越锐利)
        b = 0.0   # 外部偏置 (导航意图)
    
        # 限制 z 的范围，防止 math.tanh 因极端输入溢出 (可选，视工程具体情况而定)
        z_clip = max(min(z, 50.0), -50.0)
        mu_clip = max(min(mu, 50.0), -50.0)

        # 核心公式: 饱和非线性动力学
        opinion_drive = np.tanh(k * (mu_clip + b)) - np.tanh(k * z_clip)
        z_dot = (1.0 / self.eps) * (-d * z + u * z * opinion_drive)
    
        return z_dot

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

        vr_for_a = vr
        if abs(vr_for_a) < 1e-6:
            vr_for_a = 1e-6 if vr_for_a >= 0.0 else -1e-6

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
        u_total = u_n + u_c
        a, omega = self.u_to_physical_inputs(u_total, ego_state)

        return {
            "sensor_data": sensor_data,
            "mu_dot": mu_dot,
            "z_dot": z_dot,
            "u_n": u_n,
            "u_c": u_c,
            "u_total": u_total,
            "e31d": e31d,
            "a": a,
            "omega": omega,
            "d1": safe_distances["veh1"],
            "d2": safe_distances["veh2"],
        }


class Main4OdeDynamics:
    """将三辆车和决策变量组合成一个连续 ODE 系统。"""

    def __init__(self, veh1, veh2, veh3, a_vel=4.0, period=6.0, yield_time=20.0, yield_speed=12.0):
        self.veh1 = veh1
        self.veh2 = veh2
        self.veh3 = veh3
        self.a_vel = a_vel
        self.period = period
        self.yield_time = yield_time
        self.yield_speed = yield_speed

    def pack_state(self):
        return np.concatenate([
            self.veh1.get_state(),
            self.veh2.get_state(),
            self.veh3.get_state(),
            np.array([self.veh3.z, self.veh3.mu]),
        ])

    def apply_state(self, state):
        self.veh1.set_state(state[0:5])
        self.veh2.set_state(state[5:10])
        self.veh3.set_state(state[10:15])
        self.veh3.set_decision_state(state[15], state[16])

    def _target_states(self, state):
        return {
            "veh1": state[0:5],
            "veh1_L": self.veh1.L,
            "veh2": state[5:10],
            "veh2_L": self.veh2.L,
        }

    def rhs(self, t, state):
        veh1_state = state[0:5]
        veh2_state = state[5:10]
        ego_state = state[10:15]
        z = state[15]
        mu = state[16]

        a1, omega1 = 0.0, 0.0
        wave_omega = 2.0 * np.pi / self.period
        a2 = self.a_vel * wave_omega * np.cos(wave_omega * t)
        if t > self.yield_time:
            a2 = -3.0 if veh2_state[3] > self.yield_speed else 0.0
        omega2 = 0.0

        control = self.veh3.control_derivatives(ego_state, z, mu, self._target_states(state))

        return np.concatenate([
            rear_state_derivative(veh1_state, a1, omega1, self.veh1.L),
            rear_state_derivative(veh2_state, a2, omega2, self.veh2.L),
            rear_state_derivative(ego_state, control["a"], control["omega"], self.veh3.L),
            np.array([control["z_dot"], control["mu_dot"]]),
        ])

    def diagnostics(self, state):
        return self.veh3.control_derivatives(state[10:15], state[15], state[16], self._target_states(state))
