# =============================================
# 导入必要的库
# =============================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# =============================================
# 1. 物理参数与全局状态变量
# =============================================
d = 4.0          # 阻尼系数（惯性/阻力），越大越难改变观点
alpha = 2.0      # 自我强化系数，越大越容易“钻牛角尖”
b_current = 0.5  # 当前外部偏执（b>0 表示偏向正面）
u_current = 1.5  # 当前注意力值（横轴位置）
x_current = 0.0  # 当前黄点的纵坐标（即上一个稳定点的位置）

# 绘图坐标轴范围
u_min, u_max = 0.0, 3.0
x_min, x_max = -3.0, 3.0

# =============================================
# 2. 核心数学公式与数值迭代器
# =============================================
def dxdt(x, u, b):
    """
    观点变化率公式：x_dot = -d*x + u*tanh(alpha*x) + b（单智能体，无邻居）
    """
    return -d * x + u * np.tanh(alpha * x) + b

def evolve_to_stable(u_val, b_val, x_init, dt=0.01, max_steps=8000):
    """
    从初始观点 x_init 出发，用欧拉法数值积分，直到收敛到稳定点。
    返回值：
        x_final : 最终收敛到的稳定点坐标
        history : 整个迭代过程中 x 的轨迹列表（用于画图）
        steps   : 实际迭代步数
    """
    x = x_init
    history = [x]  # 记录起点
    
    # 【防呆处理】如果正好从绝对0出发，加一个极小的扰动，
    # 模拟现实中的数值噪声，帮助系统决定“向左”还是“向右”破缺
    if abs(x) < 1e-12:
        x = 1e-6
        history = [x]
    
    for step in range(max_steps):
        dx = dxdt(x, u_val, b_val)
        x_new = x + dt * dx
        x = x_new
        history.append(x)
        
        # 收敛判据：速度足够小，说明已经进入稳定点附近
        if abs(dx) < 1e-8:
            break
        # 安全保护：防止数值溢出（如果跑飞了就截断）
        if abs(x) > 10:
            x = np.clip(x, x_min, x_max)
            history.append(x)
            break
            
    return x, history, step + 1

# =============================================
# 3. 全局图形对象（用于动态更新）
# =============================================
fig, ax = plt.subplots(figsize=(9, 7))
plt.subplots_adjust(bottom=0.30)  # 给底部控件留空间

# 存储动态图形元素的引用（方便更新时删除旧图）
scatter_point = None   # 最终的稳定点（黄色大星标）
start_point = None     # 迭代起点（蓝色圆点）
traj_line = None       # 迭代轨迹线（垂直虚线）
traj_gradient = None   # 渐变色轨迹（可选）

# =============================================
# 4. 背景绘图函数（绘制黑色曲线和灰色箭头）
# =============================================
def draw_bifurcation(b_val):
    """
    根据当前的外部偏执 b，重绘整个背景：
    1. 灰色箭头（表示 x 增减的方向）
    2. 黑色实线（平衡点曲线，即 dxdt=0 的位置）
    """
    ax.clear()
    
    # ---- 4a. 绘制方向箭头（向量场） ----
    # 在横轴 u 和纵轴 x 上建立稀疏网格点
    U_arrow, X_arrow = np.meshgrid(np.linspace(u_min, u_max, 12),
                                   np.linspace(x_min, x_max, 12))
    dX = dxdt(X_arrow, U_arrow, b_val)  # 计算每个网格点的 x 变化速度
    dU = np.zeros_like(U_arrow)         # 因为 u 是固定参数，横轴变化率为 0
    
    # 归一化箭头长度，只显示方向（避免长短不一遮挡视线）
    norm = np.sqrt(dU**2 + dX**2)
    norm[norm < 1e-6] = 1.0
    ax.quiver(U_arrow, X_arrow, dU/norm, dX/norm,
              angles='xy', scale_units='xy', scale=0.15,
              color='gray', alpha=0.6, width=0.005)

    # ---- 4b. 绘制平衡点曲线（黑色实线） ----
    # 用精细网格绘制隐函数 dxdt = 0
    U_contour = np.linspace(u_min, u_max, 400)
    X_contour = np.linspace(x_min, x_max, 400)
    U_grid, X_grid = np.meshgrid(U_contour, X_contour)
    Z_grid = dxdt(X_grid, U_grid, b_val)
    # contour 的 levels=[0] 表示只绘制等于 0 的等高线（即平衡点位置）
    ax.contour(U_grid, X_grid, Z_grid, levels=[0], colors='black', linewidths=2.5)

    # ---- 4c. 坐标轴修饰（全英文，符合论文规范） ----
    ax.set_xlabel('Attention (u)', fontsize=13)
    ax.set_ylabel('Opinion (x)', fontsize=13)
    ax.set_title(f'Bifurcation Diagram with External Bias b = {b_val:.2f}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')  # x=0 参考线
    ax.axvline(0, color='black', linewidth=0.5, linestyle=':')  # u=0 参考线
    ax.set_xlim(u_min, u_max)
    ax.set_ylim(x_min, x_max)
    
    # 重绘完成后，重新显示之前保存的稳定点（如果有）
    update_displayed_point(u_current, x_current, history=None)
    plt.draw()

# =============================================
# 5. 更新黄点与轨迹的核心函数
# =============================================
def update_displayed_point(u_val, x_val, history=None):
    """
    在图上更新三个要素：
    1. 起点（蓝色圆点）：history 的第一个值
    2. 轨迹线（青色垂直虚线）：从起点到终点的路径
    3. 终点（黄色大星标）：最终稳定点
    """
    global scatter_point, start_point, traj_line
    
    # ---- 5a. 移除旧的图形元素 ----
    if scatter_point is not None:
        scatter_point.remove()
        scatter_point = None
    if start_point is not None:
        start_point.remove()
        start_point = None
    if traj_line is not None:
        traj_line.remove()
        traj_line = None

    # 如果没有传入轨迹，说明是外部 b 更新导致的重绘，此时只画黄点，不画轨迹
    if history is None:
        # 仅绘制黄点
        x_clipped = np.clip(x_val, x_min, x_max)
        scatter_point = ax.scatter([u_val], [x_clipped],
                                   color='gold', edgecolors='black', 
                                   s=180, marker='*', zorder=5, label='Stable Point')
        ax.text(u_val + 0.05, x_clipped, f'({u_val:.2f}, {x_clipped:.2f})',
                fontsize=9, color='darkred', verticalalignment='center')
        plt.draw()
        return

    # ---- 5b. 有轨迹历史时，绘制完整的迭代路径 ----
    history = np.array(history)
    # 截断历史数据，防止绘制太多点导致卡顿（最多取 500 个均匀点）
    if len(history) > 500:
        indices = np.linspace(0, len(history)-1, 500, dtype=int)
        history = history[indices]
    
    x_start = history[0]
    x_end = history[-1]
    
    # 绘制起点（蓝色实心圆）
    start_point = ax.scatter([u_val], [x_start], 
                             color='deepskyblue', edgecolors='black', 
                             s=100, zorder=4, label='Start Point')
    
    # ---- 5c. 绘制带颜色渐变的轨迹线 ----
    # 由于 u 固定，轨迹是一条竖直线。我们用分段颜色来表示时间推进：
    # 颜色从浅蓝色渐变到深蓝色/紫色
    points = np.array([[u_val, x] for x in history])
    # 创建线段集：每两个相邻点组成一条小线段
    segments = np.array([points[i:i+2] for i in range(len(points)-1)])
    # 创建颜色映射：从 0 到 1 渐变
    lc = LineCollection(segments, cmap='Blues', norm=plt.Normalize(0, 1))
    lc.set_array(np.linspace(0, 1, len(segments)))  # 颜色随时间变深
    lc.set_linewidth(2.5)
    lc.set_alpha(0.8)
    traj_line = ax.add_collection(lc)
    
    # 在轨迹末端（终点附近）画一个小箭头，指示运动方向
    # 如果终点在起点上方，箭头指向上；反之指向下
    if abs(x_end - x_start) > 0.01:
        direction = 1 if x_end > x_start else -1
        arrow_y = x_end - direction * 0.15  # 箭头放在终点稍偏下的位置
        ax.arrow(u_val, arrow_y, 0, direction * 0.25, 
                 head_width=0.08, head_length=0.12, fc='navy', ec='navy', alpha=0.9)
    
    # 绘制终点（黄色大星标）
    x_clipped = np.clip(x_end, x_min, x_max)
    scatter_point = ax.scatter([u_val], [x_clipped],
                               color='gold', edgecolors='black', 
                               s=180, marker='*', zorder=5, label='Stable Point')
    # 显示坐标文本
    ax.text(u_val + 0.05, x_clipped, f'({u_val:.2f}, {x_clipped:.2f})',
            fontsize=9, color='darkred', verticalalignment='center')
    
    # 更新图例（可选）
    ax.legend(loc='upper left', fontsize=8)
    plt.draw()

# =============================================
# 6. 交互回调函数（按钮和文本框的响应）
# =============================================

def on_submit_u_from_zero(text):
    """
    【定位 (从0开始)】按钮的回调：
    无论当前黄点在哪，强制从 x=0 开始迭代，寻找稳定点。
    这模拟了“第一次接触该话题，没有历史包袱”的情形。
    """
    global u_current, x_current
    try:
        u_current = float(text)
        # 从 0 开始迭代
        x_current, history, steps = evolve_to_stable(u_current, b_current, 0.0)
        print(f"Iteration completed in {steps} steps. Final x = {x_current:.5f}")
        update_displayed_point(u_current, x_current, history)
    except ValueError:
        print("Please enter a valid number for u.")

def on_submit_u_continue(text):
    """
    【定位 (继续)】按钮的回调：
    从当前黄点的位置开始迭代（即保留历史记忆）。
    这模拟了“注意力连续变化时，系统沿着分支平滑移动或跳变”的滞后效应。
    """
    global u_current, x_current
    try:
        u_current = float(text)
        # 从当前的 x_current 开始迭代（关键：保留了上一步的终点作为起点）
        x_current, history, steps = evolve_to_stable(u_current, b_current, x_current)
        print(f"Iteration completed in {steps} steps. Final x = {x_current:.5f}")
        update_displayed_point(u_current, x_current, history)
    except ValueError:
        print("Please enter a valid number for u.")

def on_submit_b_update(text):
    """
    【更新 b (继续)】按钮的回调：
    改变外部环境偏执 b，但保留当前观点 x 作为起点继续迭代。
    这完美展示了论文 Fig.8 的“鲁棒性/固执己见”现象：
    即使环境变了，只要起点在上分支，系统可能赖着不下来。
    """
    global b_current, x_current
    try:
        b_current = float(text)
        # 从当前的 x_current 开始迭代（环境变了，但人没变）
        x_current, history, steps = evolve_to_stable(u_current, b_current, x_current)
        print(f"b updated to {b_current:.2f}. Iteration {steps} steps. Final x = {x_current:.5f}")
        # 背景曲线必须重绘（因为 b 变了，黑色实线的形状变了）
        draw_bifurcation(b_current)
        # 注意：draw_bifurcation 内部会调用 update_displayed_point 并传入 history
        update_displayed_point(u_current, x_current, history)
    except ValueError:
        print("Please enter a valid number for b.")

def reset_to_zero():
    """
    【重置 x=0】按钮的回调：
    将当前观点强行置零，清空轨迹线。
    用于模拟“失忆/重置心态”后再做决策。
    """
    global x_current
    x_current = 0.0
    # 传入 None 表示不画轨迹，只画黄点在原点位置
    update_displayed_point(u_current, x_current, history=None)

# =============================================
# 7. 构建交互界面控件（全英文标注）
# =============================================

# ---- 7a. 第一行：u 的控制 ----
# 输入框
ax_u = plt.axes([0.12, 0.20, 0.15, 0.05])
text_box_u = TextBox(ax_u, 'Input u: ', initial=str(u_current))

# 两个并排按钮
ax_btn_zero = plt.axes([0.29, 0.20, 0.13, 0.05])
btn_zero = Button(ax_btn_zero, 'Set (from 0)')
btn_zero.on_clicked(lambda event: on_submit_u_from_zero(text_box_u.text))

ax_btn_cont = plt.axes([0.44, 0.20, 0.13, 0.05])
btn_cont = Button(ax_btn_cont, 'Set (continue)')
btn_cont.on_clicked(lambda event: on_submit_u_continue(text_box_u.text))

# ---- 7b. 第二行：b 的控制 ----
ax_b = plt.axes([0.12, 0.10, 0.15, 0.05])
text_box_b = TextBox(ax_b, 'Input new b: ', initial=str(b_current))

ax_btn_b = plt.axes([0.29, 0.10, 0.13, 0.05])
btn_b = Button(ax_btn_b, 'Update b (continue)')
btn_b.on_clicked(lambda event: on_submit_b_update(text_box_b.text))

# ---- 7c. 重置按钮 ----
ax_btn_reset = plt.axes([0.44, 0.10, 0.13, 0.05])
btn_reset = Button(ax_btn_reset, 'Reset x=0')
btn_reset.on_clicked(lambda event: reset_to_zero())

# =============================================
# 8. 程序初始化（启动时自动运行）
# =============================================
if __name__ == "__main__":
    # 先绘制背景（默认 b=0.5）
    draw_bifurcation(b_current)
    # 从原点开始计算第一个稳定点
    x_current, history, steps = evolve_to_stable(u_current, b_current, 0.0)
    print(f"Initialization: Final x = {x_current:.5f}, Steps = {steps}")
    # 显示黄点和轨迹
    update_displayed_point(u_current, x_current, history)
    # 显示交互窗口
    plt.show()