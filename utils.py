import numpy as np
import matplotlib.patches as patches

def draw_car(ax, car):
    """在matplotlib轴上绘制车辆矩形和前轴中心"""
    car_width = 2.0
    car_length = car.L + 1.2
    
    # 计算车身左下角用于绘制矩形
    back_x = car.x - 0.5 * np.cos(car.theta) + (car_width / 2) * np.sin(car.theta)
    back_y = car.y - 0.5 * np.sin(car.theta) - (car_width / 2) * np.cos(car.theta)
    
    car_rect = patches.Rectangle(
        (back_x, back_y), car_length, car_width, 
        angle=np.degrees(car.theta), fill=True, color=car.color, 
        alpha=0.7, edgecolor='black', linewidth=1.5
    )
    ax.add_patch(car_rect)
    
    # 标出前轴中心作为控制参考点
    xh, yh = car.get_front_axle()
    ax.plot(xh, yh, 'ko', markersize=4)
    ax.text(car.x, car.y + 1.8, car.id, fontsize=10, color='black', fontweight='bold')

def draw_environment(ax, lane_width=4.0):
    """绘制两条车道的道路环境"""
    ax.axhline(0, color='black', linewidth=2.5)                   # 道路下边缘
    ax.axhline(lane_width, color='gray', linestyle='--', linewidth=2) # 车道分隔线
    ax.axhline(lane_width*2, color='black', linewidth=2.5)            # 道路上边缘