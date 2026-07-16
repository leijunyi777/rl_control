import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. 车辆动力学模型 & 模拟传感器 (保持不变)
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
        max_delta = np.pi / 4.0 
        self.delta = np.clip(self.delta, -max_delta, max_delta)
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
            'ego_p': self.ego.get_front_axle(),
            'v1_p': veh1.get_front_axle(),
            'v2_p': veh2.get_front_axle()
        }

# ==========================================
# 2. 辅助可视化函数
# ==========================================
def draw_car(ax, car):
    car_width = 2.0
    car_length = car.L + 1.2
    back_x = car.x - 0.5 * np.cos(car.theta) + (car_width / 2) * np.sin(car.theta)
    back_y = car.y - 0.5 * np.sin(car.theta) - (car_width / 2) * np.cos(car.theta)
    
    car_rect = patches.Rectangle(
        (back_x, back_y), car_length, car_width, 
        angle=np.degrees(car.theta), fill=True, color=car.color, alpha=0.6, edgecolor='black', linewidth=1
    )
    ax.add_patch(car_rect)
    xh, yh = car.get_front_axle()
    ax.plot(xh, yh, 'ko', markersize=3)
    ax.text(car.x, car.y + 1.5, car.id, fontsize=9, color='black', fontweight='bold')

def draw_environment(ax, lane_width=4.0):
    ax.axhline(0, color='black', linewidth=2)                   
    ax.axhline(lane_width, color='gray', linestyle='--', linewidth=2) 
    ax.axhline(lane_width*2, color='black', linewidth=2)              

# ==========================================
# 3. 主仿真循环
# ==========================================
def main():
    lane_width = 4.0
    L = 2.8 
    
    # 初始化车辆
    veh1 = KinematicBicycleModel(id="Veh 1 (Leader)", x=30.0, y=lane_width*1.5, v=15.0, L=L, color='lightblue')
    # 初始速度给15.0，让积分常数对齐，防止刚开始出现速度突变
    veh2 = KinematicBicycleModel(id="Veh 2 (Aggressive)", x=0.0, y=lane_width*1.5, v=15.0, L=L, color='royalblue')
    ego = KinematicBicycleModel(id="Ego Vehicle", x=10.0, y=lane_width*0.5, v=15.0, L=L, color='lightgreen')
    
    sensor = EgoSensor(ego)
    
    dt = 0.05
    sim_time = 40.0
    steps = int(sim_time / dt)
    
    # 记录用于画折线图的数据
    t_hist = []
    dist_hist = []      # 前后车相对距离
    rel_vel_hist = []   # 前后车相对速度
    
    # --- 绘图布局 ---
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1) # 上半部分：动画
    ax_dist = plt.subplot(2, 2, 3) # 左下部分：相对距离
    ax_vel = plt.subplot(2, 2, 4)  # 右下部分：相对速度
    
    # 正弦运动参数
    T = 6.0             # 周期(秒)
    omega = 2 * np.pi / T
    A_vel = 4.0         # 速度振幅 (m/s)
    
    for i in range(steps):
        t = i * dt
        
        # --- 行为设定 ---
        # 1. Veh 1 保持匀速
        a1, omega1 = 0.0, 0.0
        
        # 2. Veh 2 完美的正弦波动加速度
        # a(t) = A * omega * cos(omega * t)
        a2 = A_vel * omega * np.cos(omega * t)
        omega2 = 0.0
        
        # 3. 传感器读取
        sensor_data = sensor.read_environment(veh1, veh2)
        
        # 4. 自车保持匀速直行 (预留控制器位置)
        a_ego, omega_ego = 0.0, 0.0
        
        # --- 状态更新 ---
        veh1.update(a1, omega1, dt)
        veh2.update(a2, omega2, dt)
        ego.update(a_ego, omega_ego, dt)
        
        # --- 数据记录 ---
        t_hist.append(t)
        # 车距 = Veh1的x - Veh2的x
        dist_hist.append(ego.x - veh2.x) 
        # 相对速度 = Veh2的v - Veh1的v
        rel_vel_hist.append(ego.v - veh1.v)
        
        # --- 可视化渲染 ---
        if i % 4 == 0:
            # 1. 刷新小车动画
            ax_anim.cla()
            draw_environment(ax_anim, lane_width)
            draw_car(ax_anim, veh1)
            draw_car(ax_anim, veh2)
            draw_car(ax_anim, ego)
            
            ep = sensor_data['ego_p']
            v1p = sensor_data['v1_p']
            v2p = sensor_data['v2_p']
            ax_anim.plot([ep[0], v1p[0]], [ep[1], v1p[1]], 'r--', alpha=0.5)
            ax_anim.plot([ep[0], v2p[0]], [ep[1], v2p[1]], 'r--', alpha=0.5)
            
            ax_anim.set_xlim(ego.x - 15, ego.x + 45)
            ax_anim.set_ylim(-2, lane_width*2 + 2)
            ax_anim.set_aspect('equal')
            ax_anim.set_title(f"Simulation Time: {t:.2f}s | Veh 2 executing sine-wave aggression")
            
            # 2. 刷新相对距离折线图
            ax_dist.cla()
            ax_dist.plot(t_hist, dist_hist, 'b-', linewidth=2)
            ax_dist.axhline(30, color='gray', linestyle='--') # 初始基准距离
            ax_dist.set_xlim(0, sim_time)
            ax_dist.set_ylim(20, 40)
            ax_dist.set_title("Relative Distance ($x_1 - x_2$)")
            ax_dist.set_xlabel("Time (s)")
            ax_dist.set_ylabel("Distance (m)")
            ax_dist.grid(True)
            
            # 3. 刷新相对速度折线图
            ax_vel.cla()
            ax_vel.plot(t_hist, rel_vel_hist, 'r-', linewidth=2)
            ax_vel.axhline(0, color='gray', linestyle='--') # 0相对速度基准
            ax_vel.set_xlim(0, sim_time)
            ax_vel.set_ylim(-6, 6)
            ax_vel.set_title("Relative Velocity ($v_2 - v_1$)")
            ax_vel.set_xlabel("Time (s)")
            ax_vel.set_ylabel("Velocity (m/s)")
            ax_vel.grid(True)
            
            plt.pause(0.01)
            
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()