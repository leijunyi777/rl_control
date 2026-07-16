import numpy as np
import matplotlib.pyplot as plt
from models import KinematicBicycleModel, EgoVehicleModel
from utils import draw_car, draw_environment

def main():
    lane_width = 4.0
    L = 2.8 
    
    # 1. 初始化车辆
    # 车1：目标车道前车 (初始 x=40)
    veh1 = KinematicBicycleModel(id="Veh 1 (Leader)", x=25.0, y=lane_width*1.5, v=15.0, L=L, color='lightblue')
    # 车2：目标车道后车 (初始 x=10)
    veh2 = KinematicBicycleModel(id="Veh 2 (Variable)", x=10.0, y=lane_width*1.5, v=15.0, L=L, color='royalblue')
    
    # 车3：右侧车道自车
    veh3 = EgoVehicleModel(id="Veh 3 (Ego/Sensor)", x=18.0, y=lane_width*0.5, v=15.0, L=L, color='lightgreen')
    
    # 2. 仿真参数
    dt = 0.05
    sim_time = 20.0
    steps = int(sim_time / dt)
    
    # 数据记录队列
    t_hist = []
    mu_hist = []       # 记录分岔参数 mu
    gap_21_hist = []   # 记录车1与车2之间的有效空隙 (d_bar_21)
    
    # 3. 初始化绘图窗口
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    ax_anim = plt.subplot(2, 1, 1) # 上方：动画视图
    ax_mu = plt.subplot(2, 2, 3)   # 左下：mu 参数曲线
    ax_gap = plt.subplot(2, 2, 4)  # 右下：有效空隙曲线
    
    # 正弦运动参数 (用于车2)
    T = 6.0             
    omega = 2 * np.pi / T
    A_vel = 4.0         
    
    # 论文中的安全裕度
    r = 0.6
    
    for i in range(steps):
        t = i * dt
        
        # --- 行为计算 ---
        a1, omega1 = 0.0, 0.0
        a2 = A_vel * omega * np.cos(omega * t)  # 车2的正弦加速度
        omega2 = 0.0
        
        # --- 车3(自车): 感知与控制 ---
        target_vehicles = {'veh1': veh1, 'veh2': veh2}
        sensor_data = veh3.read_sensor(target_vehicles)
        
        # 获取最新的 mu 值
        a3, omega3, current_mu, _, _, _ = veh3.compute_control(sensor_data, dt)
        a3, omega3 = 0.0, 0.0
        
        # --- 状态更新 ---
        veh1.update(a1, omega1, dt)
        veh2.update(a2, omega2, dt)
        veh3.update(a3, omega3, dt)
        
        # --- 记录核心数据 ---
        t_hist.append(t)
        mu_hist.append(current_mu)
        
        # 计算并记录目标车道车1与车2的有效空隙 d_bar_21 = ||p2 - p1|| - r
        p1 = veh1.get_front_axle()
        p2 = veh2.get_front_axle()
        dist21 = np.linalg.norm(p1 - p2)
        d_bar_21 = dist21 - r
        gap_21_hist.append(d_bar_21)
        
        # --- 可视化渲染 ---
        if i % 4 == 0:
            # 1. 刷新动画
            ax_anim.cla()
            draw_environment(ax_anim, lane_width)
            draw_car(ax_anim, veh1)
            draw_car(ax_anim, veh2)
            draw_car(ax_anim, veh3)
            
            # 画线标示车1和车2之间的 gap (d_bar_21)
            ax_anim.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--', linewidth=2, alpha=0.7)
            ax_anim.text((p1[0]+p2[0])/2, p1[1]+1, 'Gap $\overline{d}_{21}$', color='blue', ha='center')
            
            ax_anim.set_xlim(veh3.x - 15, veh3.x + 45)
            ax_anim.set_ylim(-2, lane_width*2 + 2)
            ax_anim.set_aspect('equal')
            ax_anim.set_title(f"Time: {t:.2f}s | Evaluating Target Lane Gap")
            
            # 2. 刷新 mu 参数曲线
            ax_mu.cla()
            ax_mu.plot(t_hist, mu_hist, 'c-', linewidth=2, label="Env Score ($\mu$)")
            ax_mu.axhline(0, color='gray', linestyle='--')
            ax_mu.set_xlim(0, sim_time)
            # 根据你之前修改的 k_mu，这里假设 mu 的上限在 1.0 左右
            ax_mu.set_ylim(-1.5, 1.5)
            ax_mu.set_title("Bifurcation Parameter ($\mu$)")
            ax_mu.set_xlabel("Time (s)")
            ax_mu.set_ylabel("$\mu$ Value")
            ax_mu.legend(loc="upper right")
            ax_mu.grid(True)
            
            # 3. 刷新目标车道空隙曲线 (d_bar_21)
            ax_gap.cla()
            ax_gap.plot(t_hist, gap_21_hist, 'purple', linewidth=2, label="Gap $\overline{d}_{21}$")
            
            # 画一条基准线 (比如 20m 作为一个安全并道空间的心理预期)
            ax_gap.axhline(10, color='red', linestyle='--', alpha=0.5, label="Tight Gap Threshold") 
            
            ax_gap.set_xlim(0, sim_time)
            # 初始距离大约是 30m，波动范围设定在 20 到 40
            ax_gap.set_ylim(0, 30) 
            ax_gap.set_title("Target Lane Gap ($\overline{d}_{21}$)")
            ax_gap.set_xlabel("Time (s)")
            ax_gap.set_ylabel("Distance (m)")
            ax_gap.legend(loc="upper right")
            ax_gap.grid(True)
            
            plt.pause(0.01)
            
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()