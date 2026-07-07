import numpy as np
import matplotlib.pyplot as plt
from models import KinematicBicycleModel, EgoVehicleModel
from utils import draw_car, draw_environment

def main():
    lane_width = 4.0
    L = 2.8 
    
    # 1. 初始化车辆
    veh1 = KinematicBicycleModel(id="Veh 1 (Leader)", x=40.0, y=lane_width*1.5, v=15.0, L=L, color='lightblue')
    veh2 = KinematicBicycleModel(id="Veh 2 (Variable)", x=10.0, y=lane_width*1.5, v=15.0, L=L, color='royalblue')
    
    # 车3：自车，带最新的传感器和底层动力学映射函数
    veh3 = EgoVehicleModel(id="Veh 3 (Ego/Sensor)", x=10.0, y=lane_width*0.5, v=15.0, L=L, color='lightgreen')
    
    # 2. 仿真参数
    dt = 0.05
    sim_time = 20.0
    steps = int(sim_time / dt)
    
    # 数据记录队列（关注车3相对于车2的参数）
    t_hist = []
    rel_x_hist = [] # 纵向相对位置 (x2 - x3)
    rel_v_hist = [] # 纵向相对速度 (v2 - v3)
    
    # 3. 初始化绘图窗口
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1) # 上方：动画视图
    ax_pos = plt.subplot(2, 2, 3)  # 左下：相对位置曲线
    ax_vel = plt.subplot(2, 2, 4)  # 右下：相对速度曲线
    
    # 正弦运动参数 (用于车2)
    T = 6.0             
    omega = 2 * np.pi / T
    A_vel = 4.0         
    
    for i in range(steps):
        t = i * dt
        
        # --- 行为计算 ---
        a1, omega1 = 0.0, 0.0
        
        # 车2: 变速运动 (正弦加速度)
        a2 = A_vel * omega * np.cos(omega * t)
        omega2 = 0.0
        
        # --- 车3(自车): 感知与控制 ---
        target_vehicles = {'veh1': veh1, 'veh2': veh2}
        sensor_data = veh3.read_sensor(target_vehicles)
        
        # 【核心更新位置】：适配最新的 compute_control 接口，传入 dt，并接收 3 个返回值
        # 因为在 main1 中我们不强制绘制 mu 的图表，所以用 _ 忽略它
        a3, omega3, _, _, _, _ = veh3.compute_control(sensor_data, dt)
        a3, omega3 = 0.0, 0.0
        
        # --- 状态更新 ---
        veh1.update(a1, omega1, dt)
        veh2.update(a2, omega2, dt)
        veh3.update(a3, omega3, dt)
        
        # --- 记录车3与车2的相对数据 ---
        veh2_data = sensor_data['veh2']
        t_hist.append(t)
        rel_x_hist.append(veh2_data['rel_p'][0]) # 提取纵向相对距离
        rel_v_hist.append(veh2_data['rel_v'][0]) # 提取纵向相对速度
        
        # --- 可视化渲染 ---
        if i % 4 == 0:
            # 1. 刷新动画
            ax_anim.cla()
            draw_environment(ax_anim, lane_width)
            draw_car(ax_anim, veh1)
            draw_car(ax_anim, veh2)
            draw_car(ax_anim, veh3)
            
            p3 = veh3.get_front_axle()
            p2 = veh2.get_front_axle()
            ax_anim.plot([p3[0], p2[0]], [p3[1], p2[1]], 'r--', alpha=0.5)
            
            ax_anim.set_xlim(veh3.x - 15, veh3.x + 45)
            ax_anim.set_ylim(-2, lane_width*2 + 2)
            ax_anim.set_aspect('equal')
            ax_anim.set_title(f"Time: {t:.2f}s | Environment Sandbox (main1 updated)")
            
            # 2. 刷新相对位置曲线
            ax_pos.cla()
            ax_pos.plot(t_hist, rel_x_hist, 'purple', linewidth=2)
            ax_pos.axhline(0, color='gray', linestyle='--')
            ax_pos.set_xlim(0, sim_time)
            ax_pos.set_ylim(-15, 15) 
            ax_pos.set_title("Longitudinal Relative Position ($x_2 - x_3$)")
            ax_pos.set_xlabel("Time (s)")
            ax_pos.set_ylabel("Relative Dist (m)")
            ax_pos.grid(True)
            
            # 3. 刷新相对速度曲线
            ax_vel.cla()
            ax_vel.plot(t_hist, rel_v_hist, 'orange', linewidth=2)
            ax_vel.axhline(0, color='gray', linestyle='--')
            ax_vel.set_xlim(0, sim_time)
            ax_vel.set_ylim(-6, 6)
            ax_vel.set_title("Longitudinal Relative Velocity ($v_2 - v_3$)")
            ax_vel.set_xlabel("Time (s)")
            ax_vel.set_ylabel("Relative Vel (m/s)")
            ax_vel.grid(True)
            
            plt.pause(0.01)
            
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()