import numpy as np
import matplotlib.pyplot as plt
from models import KinematicBicycleModel, EgoVehicleModel
from utils import draw_car, draw_environment

def main():
    lane_width = 4.0
    L = 2.8 
    
    veh1 = KinematicBicycleModel(id="Veh 1 (Leader)", x=30.0, y=lane_width*1.5, v=15.0, L=L, color='lightblue')
    veh2 = KinematicBicycleModel(id="Veh 2 (Aggressive)", x=15.0, y=lane_width*1.5, v=15.0, L=L, color='royalblue')
    
    # 初始化自车
    veh3 = EgoVehicleModel(id="Veh 3 (Ego)", x=20.0, y=lane_width*0.5, v=15.0, L=L, color='lightgreen')
    
    dt = 0.05
    sim_time = 35.0
    steps = int(sim_time / dt)
    
    # 数据队列
    t_hist, mu_hist, z_hist, d1_hist, d2_hist, ego_y_hist = [], [], [], [], [], []
    
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1) 
    ax_mu_z = plt.subplot(2, 2, 3) 
    ax_dist = plt.subplot(2, 2, 4) 
    
    T, omega, A_vel = 6.0, 2 * np.pi / 6.0, 4.0         
    
    for i in range(steps):
        t = i * dt
        
        # 车1匀速，车2进行正弦博弈
        a1, omega1 = 0.0, 0.0
        a2 = A_vel * omega * np.cos(omega * t) 
        
        # 剧情设定：在 t=20s 时，车2决定放弃博弈，减速让出空间
        if t > 20.0:
            a2 = -3 if veh2.v > 12.0 else 0.0
        omega2 = 0.0
        
        # 传感器与终极控制器接入
        sensor_data = veh3.read_sensor({'veh1': veh1, 'veh2': veh2})
        a3, omega3, current_mu, current_z, d1, d2 = veh3.compute_control(sensor_data, dt)
        
        veh1.update(a1, omega1, dt)
        veh2.update(a2, omega2, dt)
        veh3.update(a3, omega3, dt)
        
        # 记录数据
        t_hist.append(t); mu_hist.append(current_mu); z_hist.append(current_z)
        d1_hist.append(d1); d2_hist.append(d2); ego_y_hist.append(veh3.y)
        
        if i % 4 == 0:
            ax_anim.cla()
            draw_environment(ax_anim, lane_width)
            draw_car(ax_anim, veh1); draw_car(ax_anim, veh2); draw_car(ax_anim, veh3)
            ax_anim.set_xlim(veh3.x - 15, veh3.x + 45)
            ax_anim.set_ylim(-2, lane_width*2 + 2)
            ax_anim.set_aspect('equal')
            ax_anim.set_title(f"Time: {t:.2f}s | Full Embodied Opinion Dynamics + Safety Control")
            
            # 左下角：决策演化 (复刻 Fig 5)
            ax_mu_z.cla()
            ax_mu_z.plot(t_hist, mu_hist, 'c-', linewidth=2, label="Env Score ($\mu$)")
            ax_mu_z.plot(t_hist, z_hist, 'b-', linewidth=3, label="Opinion State ($z$)")
            ax_mu_z.axhline(0, color='gray', linestyle='--')
            ax_mu_z.set_xlim(0, sim_time); ax_mu_z.set_ylim(-1.0, 1.5)
            ax_mu_z.set_title("Decision Dynamics ($\mu$ and $z$)")
            ax_mu_z.legend(loc="upper left"); ax_mu_z.grid(True)
            
            # 右下角：安全距离 d1, d2 监控 (复刻 Fig 6)
            ax_dist.cla()
            ax_dist.plot(t_hist, d1_hist, 'purple', linewidth=2, label="Safe Dist to Veh 1 ($d_1$)")
            ax_dist.plot(t_hist, d2_hist, 'red', linewidth=2, label="Safe Dist to Veh 2 ($d_2$)")
            ax_dist.axhline(0, color='black', linestyle='--', linewidth=2, label="Collision Threshold")
            ax_dist.set_xlim(0, sim_time); ax_dist.set_ylim(-2, 35)
            ax_dist.set_title("Safety Distance Monitoring ($d_1, d_2$)")
            ax_dist.legend(loc="upper right"); ax_dist.grid(True)
            
            plt.pause(0.01)
            
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()