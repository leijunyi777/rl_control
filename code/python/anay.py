# -*- coding: utf-8 -*-
"""
数据处理程序：读取 main13_random_test_result.csv，
绘制 Reward 折线图（含均值及标准差误差带）、成功率饼图、
平均完成时间柱状图（含误差线）。
图表标题、坐标轴等使用英文，注释使用中文。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持（若需显示中文注释，不影响图表英文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1. 读取 CSV 文件
df = pd.read_csv('main13_random_test_result.csv')

# 2. 计算基本统计量
reward_mean = df['reward'].mean()
reward_std = df['reward'].std()
success_rate = df['success'].mean()          # 成功率为 success=1 的比例
time_mean = df['time'].mean()
time_std = df['time'].std()

print(f"Reward 均值: {reward_mean:.2f}, 标准差: {reward_std:.2f}")
print(f"成功率: {success_rate*100:.2f}%")
print(f"完成时间均值: {time_mean:.2f} 秒, 标准差: {time_std:.2f} 秒")

# ===================== 图1: Reward 折线图 + 均值与误差带 =====================
plt.figure(figsize=(10, 6))

# 绘制每个 run 的 reward 折线（点线图）
plt.plot(df['run'], df['reward'], marker='o', linestyle='-', 
         linewidth=1, markersize=3, color='blue', label='Reward per run')

# 绘制均值水平线
plt.axhline(y=reward_mean, color='red', linestyle='--', 
            label=f'Mean = {reward_mean:.2f}')

# 绘制均值 ± 1 个标准差的填充区域（误差带）
plt.fill_between(df['run'], 
                 reward_mean - reward_std, 
                 reward_mean + reward_std, 
                 color='red', alpha=0.15, label='±1 std')

plt.xlabel('Run Index')
plt.ylabel('Reward')
plt.title('Reward over Runs with Mean and Error Band')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('reward_plot.png', dpi=300)
plt.show()

# ===================== 图2: 成功率（饼图） + 平均完成时间（柱状图含误差线） =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ---- 子图1: 成功率饼图 ----
labels = ['Success', 'Failure']
sizes = [success_rate * 100, (1 - success_rate) * 100]
colors = ['#4CAF50', '#FF5252']
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
        colors=colors, explode=(0.05, 0), shadow=True)
ax1.set_title('Success Rate')

# ---- 子图2: 平均完成时间柱状图（带误差线） ----
ax2.bar(['Time'], [time_mean], yerr=time_std, capsize=10, 
        color='skyblue', edgecolor='black', error_kw={'linewidth': 2})
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Average Completion Time with Error Bar')
# 在柱顶添加均值±标准差文本
ax2.text(0, time_mean + time_std + 0.5, 
         f'Mean = {time_mean:.2f} ± {time_std:.2f}', 
         ha='center', va='bottom', fontsize=10)
ax2.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('stats_plot.png', dpi=300)
plt.show()