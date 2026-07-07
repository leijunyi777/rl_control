import numpy as np
import matplotlib.pyplot as plt
from models import KinematicBicycleModel, EgoVehicleModel
from utils import draw_car, draw_environment

def main():
    lane_width = 4.0
    L = 2.8 
    
    veh1 = KinematicBicycleModel(id="Veh 1 (Leader)", x=25.0, y=lane_width*1.5, v=15.0, L=L, color='lightblue')
    veh2 = KinematicBicycleModel(id="Veh 2 (Variable)", x=10.0, y=lane_width*1.5, v=15.0, L=L, color='royalblue')
    veh3 = EgoVehicleModel(id="Veh 3 (Ego)", x=18.0, y=lane_width*0.5, v=15.0, L=L, color='lightgreen')
    
    dt = 0.05
    sim_time = 20.0
    steps = int(sim_time / dt)
    
    # --- 数据队列 ---
    t_hist, mu_hist, z_hist, gap_21_hist = [], [], [], []
    
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1) 
    ax_mu_z = plt.subplot(2, 2, 3)  # 这里画 mu 和 z 双曲线
    ax_gap = plt.subplot(2, 2, 4)   
    
    T, omega, A_vel = 6.0, 2 * np.pi / 6.0, 4.0         
    r = 0.6
    
    for i in range(steps):
        t = i * dt
        
        a1, omega1 = 0.0, 0.0
        a2 = A_vel * omega * np.cos(omega * t) 
        omega2 = 0.0
        
        sensor_data = veh3.read_sensor({'veh1': veh1, 'veh2': veh2})
        # 接收4个返回值，获取 z
        a3, omega3, current_mu, current_z, _, _ = veh3.compute_control(sensor_data, dt)
        a3, omega3 = 0.0, 0.0
        
        veh1.update(a1, omega1, dt)
        veh2.update(a2, omega2, dt)
        veh3.update(a3, omega3, dt)
        
        t_hist.append(t)
        mu_hist.append(current_mu)
        z_hist.append(current_z)
        
        p1, p2 = veh1.get_front_axle(), veh2.get_front_axle()
        gap_21_hist.append(np.linalg.norm(p1 - p2) - r)
        
        if i % 4 == 0:
            ax_anim.cla()
            draw_environment(ax_anim, lane_width)
            draw_car(ax_anim, veh1); draw_car(ax_anim, veh2); draw_car(ax_anim, veh3)
            ax_anim.set_xlim(veh3.x - 15, veh3.x + 45)
            ax_anim.set_ylim(-2, lane_width*2 + 2)
            ax_anim.set_aspect('equal')
            ax_anim.set_title(f"Time: {t:.2f}s | Opinion Dynamics Active")
            
            # --- 绘制 mu 和 z 的演变曲线 ---
            ax_mu_z.cla()
            ax_mu_z.plot(t_hist, mu_hist, 'c-', linewidth=2, label="Env Score ($\mu$)")
            ax_mu_z.plot(t_hist, z_hist, 'b-', linewidth=3, label="Opinion State ($z$)")
            ax_mu_z.axhline(0, color='gray', linestyle='--')
            ax_mu_z.set_xlim(0, sim_time)
            ax_mu_z.set_ylim(-1.5, 1.5)
            ax_mu_z.set_title("Decision Dynamics ($\mu$ and $z$)")
            ax_mu_z.set_xlabel("Time (s)")
            ax_mu_z.legend(loc="upper left")
            ax_mu_z.grid(True)
            
            ax_gap.cla()
            ax_gap.plot(t_hist, gap_21_hist, 'purple', linewidth=2, label="Gap $\overline{d}_{21}$")
            ax_gap.axhline(10, color='red', linestyle='--') 
            ax_gap.set_xlim(0, sim_time)
            ax_gap.set_ylim(0, 30) 
            ax_gap.set_title("Target Lane Gap ($\overline{d}_{21}$)")
            ax_gap.grid(True)
            
            plt.pause(0.01)
            
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()