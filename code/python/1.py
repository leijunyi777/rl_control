import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. 参数设置（可调旋钮）
# ============================================
u_base = 0.5      # 基础注意力（平时也有的警惕性）
u_gain = 1.5      # 紧迫感增益（决定上升斜率）

# ============================================
# 2. 生成数据
# ============================================
# 模拟 gap_dot 从 -2.5 到 2.5 m/s（负值代表后车加速靠近）
gap_dot_range = np.linspace(-2.5, 2.5, 1000)

# 向量化计算 u：如果 gap_dot < 0，则取 -gap_dot，否则取 0
u_values = u_base + u_gain * np.maximum(0, -gap_dot_range)

# ============================================
# 3. 绘制图像（全英文）
# ============================================
plt.figure(figsize=(10, 6))

# 主曲线
plt.plot(gap_dot_range, u_values, color='orange', linewidth=3, label=r'$u(\dot{gap})$')

# 标注关键区域背景
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Safe boundary ($\dot{gap}=0$)')
plt.axhline(y=u_base, color='blue', linestyle=':', linewidth=2, alpha=0.6, label=f'Baseline $u_{{base}}$ = {u_base}')

# 危险区域（gap_dot < 0）用浅红色阴影
plt.axvspan(-2.5, 0, alpha=0.15, color='red', label='Danger zone (gap shrinking)')

# 标注文字解释
plt.text(-1.8, 1.8, 'Urgency rises sharply\nas gap shrinks faster', 
         color='darkred', fontsize=11, ha='center', weight='bold')
plt.text(0.5, 0.3, 'Calm / Relaxed\n(Constant attention)', 
         color='darkblue', fontsize=11, ha='center', weight='bold')
plt.text(-1.2, u_base + 0.6, f'Slope = {u_gain}', 
         color='black', fontsize=10, ha='center', rotation=np.degrees(np.arctan(u_gain/1.0))-5)

# 轴标签与标题
plt.xlabel(r'Gap change rate $\dot{gap}(t)$ (m/s)', fontsize=13)
plt.ylabel(r'Attention / Urgency $u(t)$', fontsize=13)
plt.title('Fig: Attention Modulation by Collision Risk (Gap Shrinking Speed)', 
          fontsize=14, weight='bold')

# 图例与网格
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.25)
plt.ylim(0, 4.5)  # 固定纵轴范围，方便观察
plt.xlim(-2.5, 2.5)

# 在曲线上标注几个典型点
plt.plot(0, u_base, 'ko', markersize=8)  # 原点
plt.plot(-1.0, u_base + u_gain * 1.0, 'ro', markersize=8)  # gap_dot = -1 时
plt.plot(-2.0, u_base + u_gain * 2.0, 'ro', markersize=8)  # gap_dot = -2 时
plt.annotate(f'({0}, {u_base})', xy=(0, u_base), xytext=(0.3, u_base-0.2), fontsize=9)
plt.annotate(f'(-1, {u_base+u_gain:.1f})', xy=(-1, u_base+u_gain), xytext=(-1.8, u_base+u_gain+0.3), fontsize=9)
plt.annotate(f'(-2, {u_base+2*u_gain:.1f})', xy=(-2, u_base+2*u_gain), xytext=(-2.8, u_base+2*u_gain+0.3), fontsize=9)

plt.tight_layout()
plt.show()