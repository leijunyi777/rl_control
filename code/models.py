import numpy as np

class KinematicBicycleModel:
    """
    基础车辆动力学模型 (二阶非完整自行车模型)
    作为车1(前车)和车2(后车)的模型，不需要复杂的传感器功能。
    """
    def __init__(self, id, x=0.0, y=0.0, theta=0.0, v=0.0, delta=0.0, L=2.5, color='blue'):
        self.id = id
        self.x = x          # 后轴中心 x 坐标 (纵向)
        self.y = y          # 后轴中心 y 坐标 (横向)
        self.theta = theta  # 航向角
        self.v = v          # 后轴速度
        self.delta = delta  # 前轮转向角
        self.L = L          # 轴距
        self.color = color

    def update(self, a, omega, dt):
        """基于欧拉法的状态更新"""
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += (self.v / self.L) * np.tan(self.delta) * dt
        self.v += a * dt
        self.delta += omega * dt
        
        # 物理约束
        self.delta = np.clip(self.delta, -np.pi / 4.0, np.pi / 4.0)
        self.v = max(0.0, self.v)

    def get_front_axle(self):
        """获取前轴中心位置坐标矢量"""
        xh = self.x + self.L * np.cos(self.theta)
        yh = self.y + self.L * np.sin(self.theta)
        return np.array([xh, yh])

    def get_velocity_vector(self):
        """获取前轴速度矢量"""
        vx = self.v * np.cos(self.theta)
        vy = self.v * np.sin(self.theta)
        return np.array([vx, vy])


class EgoVehicleModel(KinematicBicycleModel):
    """
    带传感器和控制接口的自车模型 (车3)
    继承自 KinematicBicycleModel。
    """
    def __init__(self, id, x=0.0, y=0.0, theta=0.0, v=0.0, delta=0.0, L=2.5, color='lightgreen'):
        super().__init__(id, x, y, theta, v, delta, L, color)

        # --- 决策动力学参数 ---
        self.mu = 0.0     # 环境评分
        self.z = 0.01     # 变道意见 (初始给予微小的正值以打破 z=0 的不稳定平衡)
        self.eps = 0.5   # 意见演化速率常数 (来自论文数值实验参数)

    def read_sensor(self, target_vehicles):
        """
        模拟传感器感知
        :param target_vehicles: 字典形式的目标车辆 {'veh1': obj, 'veh2': obj}
        :return: 包含相对位置、相对速度等信息的字典
        """
        p_ego = self.get_front_axle()
        v_ego = self.get_velocity_vector()
        
        sensor_data = {}
        for name, veh in target_vehicles.items():
            p_target = veh.get_front_axle()
            v_target = veh.get_velocity_vector()
            
            # 严格计算相对状态矢量
            rel_p = p_target - p_ego  # 目标车相对于自车的位置向量
            rel_v = v_target - v_ego  # 目标车相对于自车的速度向量
            dist = np.linalg.norm(rel_p) # 欧氏距离
            
            sensor_data[name] = {
                'rel_p': rel_p,
                'rel_v': rel_v,
                'dist': dist
            }
        return sensor_data

    def u_to_physical_inputs(self, u):
        """
        将前轴的高层控制力 u = [ux, uy] 映射为底层物理控制 [a, omega]
        (严格对应论文中的公式 4)
        """
        vr = self.v
        theta = self.theta
        delta = self.delta
        L = self.L
        
        # 1. 低速奇异性保护 (论文提到：当 |vr| < eps_v 时，用 sign(vr)*eps_v 替代)
        eps_v = 0.1
        vr_safe = np.sign(vr) * max(abs(vr), eps_v) if vr != 0 else eps_v
        
        # 2. 组装雅可比矩阵 A
        # 计算 sec(delta)^2 = 1 / cos(delta)^2
        sec2_delta = 1.0 / (np.cos(delta)**2 + 1e-6)
        
        A = np.array([
            [np.cos(theta) - np.sin(theta)*np.tan(delta),  -vr_safe * np.sin(theta) * sec2_delta],
            [np.sin(theta) + np.cos(theta)*np.tan(delta),   vr_safe * np.cos(theta) * sec2_delta]
        ])
        
        # 3. 计算公式右侧的补偿项 (即减去的那一大堆包含了 v_r^2 / L 的非线性项)
        v_sq_L = (vr**2) / L
        tan_d = np.tan(delta)
        tan2_d = tan_d**2
        
        b_residual = np.array([
            -v_sq_L * np.sin(theta) * tan_d - v_sq_L * np.cos(theta) * tan2_d,
             v_sq_L * np.cos(theta) * tan_d - v_sq_L * np.sin(theta) * tan2_d
        ])
        
        # 4. 求解线性方程组 A * [a, omega]^T = (u - b_residual)
        target_vector = u - b_residual
        
        try:
            # 矩阵求逆求解 [a, omega]
            a_omega = np.linalg.solve(A, target_vector)
            a = a_omega[0]
            omega = a_omega[1]
        except np.linalg.LinAlgError:
            # 极端情况兜底：如果矩阵不可逆，暂时放弃控制输出
            a = 0.0
            omega = 0.0
            
        return a, omega
    
    def update_mu(self, sensor_data, dt):
        """
        根据论文公式 (9) 更新环境评估参数 \mu
        \dot{\mu} = -k_\mu \mu + \tanh(-k (\rho^T g_1) (\rho^T g_2) (\bar{d}_{21} - 2r) (\dot{\bar{d}}_{21} / \bar{d}_{21} + \epsilon_1))
        """
        # --- 论文 Sec 4 中的超参数 ---
        k_mu = 1.0
        k = 5.0
        r = 0.6       # 安全裕度
        eps1 = 0.5    # 速度发散流容忍度
        
        # 纵向单位方向 (假设道路沿 X 轴正向)
        rho = np.array([1.0, 0.0])
        
        # --- 1. 提取自车与车1、车2的相对状态 ---
        # 注意：公式中的 g_j 是从自车指向目标车的单位向量的反向 (即 p - p_j)。
        # 我们传感器获取的是 p_target - p_ego，所以需要加个负号。
        dp1 = -sensor_data['veh1']['rel_p']  # p - p1
        dp2 = -sensor_data['veh2']['rel_p']  # p - p2
        
        g1 = dp1 / (np.linalg.norm(dp1) + 1e-6)
        g2 = dp2 / (np.linalg.norm(dp2) + 1e-6)
        
        # --- 2. 计算目标车道中，车1与车2的相对状态 ---
        # 利用向量减法推导：(p2 - p_ego) - (p1 - p_ego) = p2 - p1
        p21 = sensor_data['veh2']['rel_p'] - sensor_data['veh1']['rel_p']
        v21 = sensor_data['veh2']['rel_v'] - sensor_data['veh1']['rel_v']
        
        dist21 = np.linalg.norm(p21)
        d21_bar = dist21 - r  # 扣除安全裕度后的两车间距 [cite: 133-134]
        
        g21_bar = p21 / (dist21 + 1e-6)
        d21_dot_bar = np.dot(g21_bar, v21) # 间距变化率
        
        # --- 3. 计算公式 (9) 中的各个逻辑判断项 ---
        # 条件1：位置是否合适？(rho^T g1 * rho^T g2 < 0 代表自车夹在两车之间)
        pos_term = np.dot(rho, g1) * np.dot(rho, g2) 
        
        # 条件2：空间是否足够？(d21_bar - 2r > 0)
        space_term = d21_bar - 2 * r
        
        # 条件3：两车相对运动趋势好不好？(大于 -eps1 说明没有在快速缩紧)
        flow_term = (d21_dot_bar / max(d21_bar, 1e-3)) + eps1
        
        # --- 4. 组装并积分 ---
        tanh_arg = -k * pos_term * space_term * flow_term
        mu_dot = -k_mu * self.mu + np.tanh(tanh_arg)
        
        # 欧拉法更新参数
        self.mu += mu_dot * dt
        return self.mu
    
    def update_z(self, dt):
        """
        核心更新：计算变道意见 z 的非线性动力学演化
        \dot{z} = (1/\epsilon) * (\mu * z - z^2)
        """
        z_dot = (1.0 / self.eps) * (self.mu * self.z - self.z**2)
        self.z += z_dot * dt
        #ode45

        # 不要让它等于绝对的 0.0，保留一个极其微小的正值(如 0.001)作为“火种”
        self.z = max(0.001, self.z)
        
        return self.z

    def compute_nominal_control(self, sensor_data, z):
        """
        计算名义控制器 (PD 控制)
        对应论文 公式(11) 和 公式(12)
        """
        # --- 论文 Sec 4 中的超参数 ---
        k_w = 5.0
        k_p = 0.7
        k_v = 2.0
        
        # 纵向与横向单位向量
        rho = np.array([1.0, 0.0])
        eta = np.array([0.0, 1.0])
        
        # 目标相对位置的基准偏置
        r_rho = -10.0  # 纵向上：希望跟在车1后方 14m 的位置
        r_eta = -4.0   # 横向上：初始在旁边车道，横向相差 4m (lane_width)
        
        # --- 1. 计算动态目标点 e1_star (Eq. 12) ---
        w_z = np.tanh(k_w * z)
        e1_star = rho * r_rho + (1.0 - w_z) * eta * r_eta
        
        # --- 2. 提取相对状态 (与车1的误差) ---
        # 注意：sensor_data['veh1']['rel_p'] 是 p1 - p
        # 论文中 e1 = p - p1，所以我们需要加个负号
        e1 = -sensor_data['veh1']['rel_p']
        nu1 = -sensor_data['veh1']['rel_v']
        
        # 车1的加速度 (假设领航车匀速直行，a=0)
        u1 = np.array([0.0, 0.0])
        
        # --- 3. 计算 PD 控制量 u_n (Eq. 11) ---
        u_n = -k_p * (e1 - e1_star) - k_v * nu1 + u1
        
        return u_n, e1_star

    def compute_safe_control(self, sensor_data):
        """
        计算耗散势垒安全防撞控制器 (u^c)
        严格对应论文 公式(13)
        """
        k_o = 1.0  # 防撞势垒增益
        r = 0.6    # 安全裕度
        
        u_c = np.array([0.0, 0.0])
        safe_distances = {} # 用于记录 d1, d2 供外部画图
        
        for name in ['veh1', 'veh2']:
            # 1. 提取自车到目标车的状态
            # 注意：sensor_data存的是 (目标 - 自车)
            # 论文中 p - pj 和 v - vj 是 (自车 - 目标)，所以要加负号
            dp_j = -sensor_data[name]['rel_p'] 
            dv_j = -sensor_data[name]['rel_v']
            
            dist = np.linalg.norm(dp_j)
            g_j = dp_j / (dist + 1e-6) # g_j 是从目标车指向自车的单位向量
            #需要更改
            
            d_j = dist - r             # 扣除安全裕度后的真实安全距离
            safe_distances[name] = d_j
            
            # 数值保护：防止距离过近时发生除以 0 的崩溃
            d_j_safe = max(d_j, 0.05)
            #需要更改
            
            # 2. 计算距离的变化率 \dot{d}_j = g_j^T \nu_j
            d_j_dot = np.dot(g_j, dv_j)
            
            # 3. 叠加势垒排斥力 (Eq. 13)
            # 如果 d_j_dot < 0 (正在靠近), u_c 会产生一个沿 g_j 方向(推开自车)的正向力
            u_c += -k_o * g_j * (d_j_dot / d_j_safe)
            
        return u_c, safe_distances

    def compute_control(self, sensor_data, dt):
        """
        【终极版】控制主循环：融合决策、追踪与防撞
        对应论文 Eq(7) 闭环架构
        """
        # 1. 高层具身意见动力学 (Eq.8 & 9)
        current_mu = self.update_mu(sensor_data, dt)
        current_z = self.update_z(dt)
        
        # 2. 中层目标生成与名义追踪控制 (Eq.11 & 12)
        u_n, e1_star = self.compute_nominal_control(sensor_data, current_z)
        
        # 3. 底层绝对安全防撞控制 (Eq.13)
        u_c, safe_distances = self.compute_safe_control(sensor_data)
        
        # 4. 总控制力叠加 (Eq.10)
        u_total = u_n + u_c
        
        # 5. 底层逆动力学映射 (Eq.4)
        a_ego, omega_ego = self.u_to_physical_inputs(u_total)
        
        # 返回丰富的系统状态供主程序可视化
        return a_ego, omega_ego, current_mu, current_z, safe_distances['veh1'], safe_distances['veh2']