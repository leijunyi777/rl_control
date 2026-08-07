import numpy as np
import matplotlib.pyplot as plt

# ---------- 参数设置 ----------
u_base = 0.2          # 基础注意力
u_amp = 2.5           # 峰值幅度
sigma_d = 2.0         # 位置容忍度 (m)
sigma_v = 1.5         # 速度容忍度 (m/s)

# ---------- 生成对称网格数据（范围 -5 ~ 5）----------
range_limit = 5.0
num_points = 300      # 网格密度

d_gap_vals = np.linspace(-range_limit, range_limit, num_points)
dv_gap_vals = np.linspace(-range_limit, range_limit, num_points)
D, DV = np.meshgrid(d_gap_vals, dv_gap_vals)

# 计算注意力 u
u = u_base + u_amp * np.exp(- (D**2) / (2 * sigma_d**2) - (DV**2) / (2 * sigma_v**2))

# ---------- 绘制热力图 ----------
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(u,
                extent=[d_gap_vals[0], d_gap_vals[-1], dv_gap_vals[0], dv_gap_vals[-1]],
                origin='lower',
                aspect='equal',
                cmap='coolwarm',
                interpolation='nearest')

plt.colorbar(im, ax=ax, label='Attention $u$')
ax.set_xlabel('$d_{gap}$ (m)', fontsize=12)
ax.set_ylabel('$dv_{gap}$ (m/s)', fontsize=12)
ax.set_title('Attention Heatmap (Click to see values)', fontsize=14)

# 标记完美点 (0,0)
ax.scatter(0, 0, color='black', s=40, marker='x', label='Perfect timing (0,0)')
ax.legend()

# ---------- 交互功能：点击显示数值 ----------
# 用于存放当前显示的标注（每次点击只保留最新一个）
annotation = None
scatter_point = None

def on_click(event):
    global annotation, scatter_point
    
    # 仅当点击在坐标轴内时处理
    if event.inaxes != ax:
        return
    
    # 获取点击的数据坐标
    x_click = event.xdata
    y_click = event.ydata
    
    # 找到最近的网格索引
    idx_x = np.argmin(np.abs(d_gap_vals - x_click))
    idx_y = np.argmin(np.abs(dv_gap_vals - y_click))
    
    # 提取对应的值
    d_val = d_gap_vals[idx_x]
    dv_val = dv_gap_vals[idx_y]
    u_val = u[idx_y, idx_x]   # 注意 u 的索引顺序：行对应 dv，列对应 d
    
    # 清除之前的标注和点
    if annotation is not None:
        annotation.remove()
    if scatter_point is not None:
        scatter_point.remove()
    
    # 在点击位置画一个红色圆圈
    scatter_point = ax.scatter(d_val, dv_val, color='red', s=50, zorder=5, label='Selected')
    
    # 添加文本注释，显示数值
    text_str = f'd_gap={d_val:.2f} m\ndv_gap={dv_val:.2f} m/s\nu={u_val:.3f}'
    annotation = ax.annotate(text_str,
                             xy=(d_val, dv_val),
                             xytext=(10, 10),
                             textcoords='offset points',
                             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.8),
                             fontsize=9,
                             color='black')
    
    # 刷新图形
    fig.canvas.draw_idle()

# 绑定点击事件
fig.canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()
plt.show()