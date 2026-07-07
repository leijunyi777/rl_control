import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. 车辆动力学模型 & 传感器
# ==========================================
class KinematicBicycleModel:
    def __init__(self, id, x=0.0, y=0.0, theta=0.0, v=0.0, delta=0.0, L=2.5, color='blue'):
        self.id = id
        self.x = x          
        self.y = y          
        self.theta = theta  
        self.v = v          
        self.delta = delta  
        self.L = L          
        self.color = color

    def update(self, a, omega, dt):
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += (self.v / self.L) * np.tan(self.delta) * dt
        self.v += a * dt
        self.delta += omega * dt
        self.delta = np.clip(self.delta, -np.pi / 4.0, np.pi / 4.0)
        self.v = max(0.0, self.v)

    def get_front_axle(self):
        xh = self.x + self.L * np.cos(self.theta)
        yh = self.y + self.L * np.sin(self.theta)
        return np.array([xh, yh])

    def get_velocity_vector(self):
        vx = self.v * np.cos(self.theta)
        vy = self.v * np.sin(self.theta)
        return np.array([vx, vy])

class EgoSensor:
    def __init__(self, ego_vehicle):
        self.ego = ego_vehicle
        
    def read_environment(self, veh1, veh2):
        return {
            'p': self.ego.get_front_axle(), 'v': self.ego.get_velocity_vector(),
            'p1': veh1.get_front_axle(), 'v1': veh1.get_velocity_vector(),
            'p2': veh2.get_front_axle(), 'v2': veh2.get_velocity_vector(),
            'vr': self.ego.v, 'theta': self.ego.theta, 'delta': self.ego.delta
        }

# ==========================================
# 2. 具身意见动力学控制器 (核心算法复现)
# ==========================================
class EmbodiedController:
    def __init__(self, L):
        self.L = L
        # 论文参数设置 (Sec. 4)
        self.kp = 0.7
        self.kv = 2.0
        self.ko = 1.0
        self.kw = 40.0
        self.k_mu = 5.0
        self.k = 20.0
        self.eps = 0.05
        self.r = 0.6
        self.eps1 = 0.5
        
        # 内部状态
        self.mu = 0.0
        self.z = 0.01  # 初始意见需要略大于0，以打破纯0的平衡态
        
        # 目标期望偏置 (相对 Veh 1)
        self.r_rho = -14.0 # 目标纵向位置：前车后方 14 米
        self.r_eta = -4.0  # 目标横向位置：未变道时距离前车 -4 米(相差一个车道宽度)

    def compute_control(self, sensor_data, dt):
        p, v = sensor_data['p'], sensor_data['v']
        p1, v1 = sensor_data['p1'], sensor_data['v1']
        p2, v2 = sensor_data['p2'], sensor_data['v2']
        
        rho = np.array([1.0, 0.0]) # 纵向单位向量
        eta = np.array([0.0, 1.0]) # 横向单位向量
        
        # 相对状态计算
        dp1 = p - p1
        dp2 = p - p2
        dist1 = np.linalg.norm(dp1)
        dist2 = np.linalg.norm(dp2)
        
        g1 = dp1 / (dist1 + 1e-6)
        g2 = dp2 / (dist2 + 1e-6)
        
        dp21 = p2 - p1
        dist21 = np.linalg.norm(dp21)
        d21_bar = dist21 - self.r
        g21_bar = dp21 / (dist21 + 1e-6)
        v21 = v2 - v1
        d21_dot_bar = np.dot(g21_bar, v21)
        
        # 1. 分岔参数动态 (Eq. 9)
        tanh_arg = -self.k * np.dot(rho, g1) * np.dot(rho, g2) * (d21_bar - 2*self.r) * (d21_dot_bar / max(d21_bar, 1e-3) + self.eps1)
        self.mu += dt * (-self.k_mu * self.mu + np.tanh(tanh_arg))
        
        # 2. 意见动态 (Eq. 8)
        self.z += dt * (1/self.eps) * (self.mu * self.z - self.z**2)
        self.z = max(0.0, self.z) # 意见不为负
        
        # 3. 目标协调 (Eq. 12)
        w_z = np.tanh(self.kw * self.z)
        e1_star = rho * self.r_rho + (1 - w_z) * eta * self.r_eta
        
        # 4. 名义控制器 (Eq. 11)
        e1 = p - p1
        nu1 = v - v1
        u_n = -self.kp * (e1 - e1_star) - self.kv * nu1
        
        # 5. 安全避撞控制器 (Eq. 13)
        d1 = dist1 - self.r
        d2 = dist2 - self.r
        d1_dot = np.dot(g1, v - v1)
        d2_dot = np.dot(g2, v - v2)
        
        u_c = -self.ko * (g1 * d1_dot / max(d1, 0.1) + g2 * d2_dot / max(d2, 0.1))
        
        # 总控制输入 (Eq. 10)
        u_total = u_n + u_c
        
        # 6. 控制映射 (Eq. 4)
        vr = sensor_data['vr']
        theta = sensor_data['theta']
        delta = sensor_data['delta']
        
        vr_safe = np.sign(vr) * max(abs(vr), 0.1) if vr != 0 else 0.1
        
        dp_dvr = np.array([np.cos(theta) - np.sin(theta)*np.tan(delta), 
                           np.sin(theta) + np.cos(theta)*np.tan(delta)])
        dp_ddelta = np.array([-vr_safe*np.sin(theta)*(1/np.cos(delta)**2), 
                               vr_safe*np.cos(theta)*(1/np.cos(delta)**2)])
        dp_dtheta = np.array([-vr_safe*(np.sin(theta) + np.cos(theta)*np.tan(delta)), 
                               vr_safe*(np.cos(theta) - np.sin(theta)*np.tan(delta))])
        
        A = np.column_stack((dp_dvr, dp_ddelta))
        u_residual = u_total - dp_dtheta * (vr_safe / self.L * np.tan(delta))
        
        try:
            a_omega = np.linalg.solve(A, u_residual)
            a_ego, omega_ego = a_omega[0], a_omega[1]
        except np.linalg.LinAlgError:
            a_ego, omega_ego = 0.0, 0.0

        return a_ego, omega_ego, self.mu, self.z, d1, d2

# ==========================================
# 3. 辅助可视化函数
# ==========================================
def draw_car(ax, car):
    car_width = 2.0
    car_length = car.L + 1.2
    back_x = car.x - 0.5 * np.cos(car.theta) + (car_width / 2) * np.sin(car.theta)
    back_y = car.y - 0.5 * np.sin(car.theta) - (car_width / 2) * np.cos(car.theta)
    
    car_rect = patches.Rectangle((back_x, back_y), car_length, car_width, 
                                 angle=np.degrees(car.theta), fill=True, color=car.color, alpha=0.6, edgecolor='black', linewidth=1)
    ax.add_patch(car_rect)
    xh, yh = car.get_front_axle()
    ax.plot(xh, yh, 'ko', markersize=3)
    ax.text(car.x, car.y + 1.5, car.id, fontsize=9, color='black', fontweight='bold')

def draw_environment(ax, lane_width=4.0):
    ax.axhline(0, color='black', linewidth=2)                   
    ax.axhline(lane_width, color='gray', linestyle='--', linewidth=2) 
    ax.axhline(lane_width*2, color='black', linewidth=2)              

# ==========================================
# 4. 主仿真循环
# ==========================================
def main():
    lane_width = 4.0
    L = 2.8 
    
    veh1 = KinematicBicycleModel(id="Veh 1 (Leader)", x=30.0, y=lane_width*1.5, v=15.0, L=L, color='lightblue')
    veh2 = KinematicBicycleModel(id="Veh 2 (Aggressive)", x=0.0, y=lane_width*1.5, v=15.0, L=L, color='royalblue')
    ego = KinematicBicycleModel(id="Ego Vehicle", x=10.0, y=lane_width*0.5, v=15.0, L=L, color='lightgreen')
    
    sensor = EgoSensor(ego)
    controller = EmbodiedController(L)
    
    dt = 0.05
    sim_time = 35.0
    steps = int(sim_time / dt)
    
    # 记录数据
    t_hist, mu_hist, z_hist, d1_hist, d2_hist, ego_y_hist = [], [], [], [], [], []
    
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1) 
    ax_mu_z = plt.subplot(2, 2, 3) 
    ax_dist = plt.subplot(2, 2, 4) 
    
    T = 6.0; omega = 2 * np.pi / T; A_vel = 4.0
    
    for i in range(steps):
        t = i * dt
        
        # 1. 周围车辆动态
        a1, omega1 = 0.0, 0.0
        a2 = A_vel * omega * np.cos(omega * t)
        
        # 稍作调整：让后车在20秒后停止“挑衅”，主动让出空间，测试变道
        if t > 20.0:
            a2 = -1.0 if veh2.v > 10.0 else 0.0
        omega2 = 0.0
        
        # 2. 传感器与控制器
        sensor_data = sensor.read_environment(veh1, veh2)
        a_ego, omega_ego, mu, z, d1, d2 = controller.compute_control(sensor_data, dt)
        
        # 3. 状态更新
        veh1.update(a1, omega1, dt)
        veh2.update(a2, omega2, dt)
        ego.update(a_ego, omega_ego, dt)
        
        # 4. 数据记录
        t_hist.append(t); mu_hist.append(mu); z_hist.append(z)
        d1_hist.append(d1); d2_hist.append(d2); ego_y_hist.append(ego.y)
        
        # 5. 可视化
        if i % 4 == 0:
            ax_anim.cla()
            draw_environment(ax_anim, lane_width)
            draw_car(ax_anim, veh1)
            draw_car(ax_anim, veh2)
            draw_car(ax_anim, ego)
            
            ax_anim.set_xlim(ego.x - 15, ego.x + 45)
            ax_anim.set_ylim(-2, lane_width*2 + 2)
            ax_anim.set_aspect('equal')
            ax_anim.set_title(f"Time: {t:.2f}s | Embodied Control Active")
            
            # mu & z 图表 (复现 Fig. 5 上半部)
            ax_mu_z.cla()
            ax_mu_z.plot(t_hist, mu_hist, 'c-', label="Env Score ($\mu$)")
            ax_mu_z.plot(t_hist, z_hist, 'b-', linewidth=2, label="Opinion State ($z$)")
            ax_mu_z.set_xlim(0, sim_time); ax_mu_z.set_ylim(-0.5, 1.0)
            ax_mu_z.set_title("Decision Dynamics ($\mu$ and $z$)")
            ax_mu_z.legend(loc="upper left"); ax_mu_z.grid(True)
            
            # 安全距离与横向位置 (复现 Fig. 5 下半部与 Fig. 6)
            ax_dist.cla()
            ax_dist.plot(t_hist, ego_y_hist, 'g-', linewidth=2, label="Ego Lateral Pos (y)")
            ax_dist.plot(t_hist, d2_hist, 'r--', alpha=0.6, label="Safe Dist to Veh 2 ($d_2$)")
            ax_dist.axhline(lane_width*1.5, color='gray', linestyle='--')
            ax_dist.set_xlim(0, sim_time); ax_dist.set_ylim(0, 15)
            ax_dist.set_title("Lateral Pos & Safety Distance")
            ax_dist.legend(loc="upper left"); ax_dist.grid(True)
            
            plt.pause(0.01)
            
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()