# Main11 SAC 训练说明：学习注意力 u(t)

本文档说明 `main11_sac_train.py` 的强化学习结构。它基于 main11 的最新意见动力学，但这次 SAC 不再学习 `b(t)`，而是直接学习注意力/执行信心 `u(t)`。

---

## 1. Main11 使用的意见动力学

核心公式仍然来自论文的单智能体意见动力学：

```text
z_dot = -d * z + u(t) * tanh(alpha * z) + b(t)
```

含义：

```text
z      ego 的并道意愿
d      意见阻尼，防止 z 无限增长或突变
alpha  tanh 的灵敏度
b(t)   客观环境偏见，决定当前空隙是否支持并道
u(t)   主观注意力/执行信心，决定 z 的自我强化有多强
```

在 main11 的普通仿真里，`b(t)` 和 `u(t)` 都由物理状态计算；在本次 SAC 训练里，`b(t)` 仍由公式计算，但 `u(t)` 改由策略网络输出。

---

## 2. 客观偏见 b(t)

`b(t)` 反映目标车道前车和后车之间的空隙是否足够安全：

```text
b(t) = tanh(k_gap * (gap - d_safe) + k_vel * gap_dot)
```

其中：

```text
gap     前车 Veh1 与后车 Veh2 的纵向间隙
gap_dot 间隙变化率，正数表示空隙变大，负数表示空隙变小
d_safe  安全空隙阈值，当前为 15 m
k_gap   距离灵敏度
k_vel   速度趋势灵敏度
```

直观理解：

```text
b > 0  当前空隙客观上支持并道
b < 0  当前空隙偏危险，不支持并道
```

---

## 3. 普通 main11 里的公式 u(t)

main11 的物理公式版本会先计算目标空隙中心：

```text
x_gap = (x_front + x_rear) / 2
v_gap = (v_front + v_rear) / 2
d_gap = x_gap - x_ego
dv_gap = v_ego - v_gap
```

然后用二维高斯/RBF 计算执行信心：

```text
u_formula =
  u_base
+ u_amp * exp(
      - d_gap^2  / (2 * sigma_d^2)
      - dv_gap^2 / (2 * sigma_v^2)
  )
```

它表达的是：

```text
ego 越接近目标空隙中心，u 越大
ego 与空隙平均速度越一致，u 越大
位置差或速度差很大时，u 接近 u_base
```

---

## 4. SAC 这次学习什么

训练脚本为：

```text
main11_sac_train.py
```

SAC 的 action 是一维：

```text
action = u(t)
```

神经网络内部仍输出归一化动作：

```text
raw_action in [-1, 1]
```

代码中映射为真实注意力：

```text
u(t) in [0.0, 3.0]
```

也就是说，本次 SAC 学的是：

```text
在当前相对位置和速度下，应该给意见动力学多大的执行信心 u(t)
```

`b(t)` 不由 SAC 输出，仍然按客观公式计算。

---

## 5. State：8 维相对状态

状态定义不变，仍然是 ego 相对前后车的位置和速度：

```text
state =
[
  ego 相对前车的位置 x, y,
  ego 相对前车的速度 vx, vy,
  ego 相对后车的位置 x, y,
  ego 相对后车的速度 vx, vy
]
```

展开为：

```text
[x_front, y_front, vx_front, vy_front,
 x_rear,  y_rear,  vx_rear,  vy_rear]
```

它告诉策略网络：

```text
ego 当前处在前车和后车的什么相对位置
ego 和它们的相对速度趋势是什么
```

---

## 6. 控制流程

每个仿真步的流程是：

```text
1. 读取 ego、前车、后车状态
2. 构造 8 维 state
3. SAC policy 输出 action
4. action 映射为 u(t)
5. 用目标车道间隙公式计算 b(t)
6. 用 z_dot = -d*z + u*tanh(alpha*z) + b 更新 z
7. 用 z 计算期望并道位置 e31d
8. 控制器计算 u_n + u_c
9. 转换为车辆输入 a 和 omega
10. RK45 推进一步车辆状态
11. 计算 reward 并训练 SAC
```

控制器仍然是论文控制器：

```text
u_total = u_n + u_c
```

其中：

```text
u_n  名义跟踪控制，让 ego 跟随由 z 决定的目标位置
u_c  安全避碰控制，避免离前后车过近
```

---

## 7. Reward 公式

当前奖励机制与最新 SAC 训练保持一致，目标是避免 `progress = 0` 的不动策略：

```text
reward =
  80.0 * delta_progress
+ 0.15 * progress
+ 40.0 * opportunity * max(delta_progress, 0)
- 30.0 * max(-delta_progress, 0)
- 0.25 * opportunity * (1 - progress)
- 0.05 * (1 - progress)
- 0.5 * ||action_t - action_{t-1}||^2
- 2.0 * I[v_y(t) * v_y(t-1) < 0]
- 20.0 * max(0, (safe_margin - d_min) / safe_margin)^2
- 1000 * I[collision]
+ (100 - 2t) * I[success]
```

主要含义：

```text
80.0 * delta_progress
强烈奖励朝目标车道推进。

0.15 * progress
只要已经靠近目标车道，就持续给一点正反馈。

40.0 * opportunity * max(delta_progress, 0)
有并道机会时，如果继续推进，额外奖励。

-30.0 * max(-delta_progress, 0)
如果从目标车道方向退回，明显扣分。

-0.25 * opportunity * (1 - progress)
有机会但还停在原车道附近，会持续扣分。

-0.05 * (1 - progress)
时间惩罚，越晚完成并道，扣分越多。

-0.5 * ||action_t - action_{t-1}||^2
惩罚 u(t) 剧烈跳变，让策略输出更平滑。

-2.0 * I[v_y(t) * v_y(t-1) < 0]
惩罚横向速度反复换向。

-20.0 * safety_penalty
靠前后车太近时提前扣分。

-1000 * I[collision]
碰撞立即给大惩罚。

+(100 - 2t) * I[success]
成功并道给大奖励，越早成功奖励越高。
```

---

## 8. Opportunity 判断

本次训练中，是否存在并道机会由两部分共同判断：

```text
opportunity = 1, if mu > 0.1 or veh12_gap > gap_safe
opportunity = 0, otherwise
```

这样做的原因是：

```text
旧的 mu 有时上升较慢，如果只依赖 mu，策略可能一直学到等待。
加入 veh12_gap > gap_safe 后，只要目标车道空隙已经足够大，就会鼓励 ego 尽快进入。
```

---

## 9. 成功与碰撞条件

成功条件：

```text
progress > 0.95
abs(ego_y - target_lane_y) < 0.2
d_min > 1.5 * r
```

碰撞条件：

```text
min(ego 到前车距离, ego 到后车距离, 前后车间隙) < r
```

其中：

```text
r = 1.5 m
```

---

## 10. 输出文件

训练结束后保存：

```text
main11_sac_u_policy.pth
main11_sac_u_training_result.png
```

回放程序：

```text
main11_sac_replay.py
```

回放默认导出：

```text
main11_sac_u_replay.gif
```

---

## 11. 运行方式

训练：

```bash
python main11_sac_train.py
```

回放：

```bash
python main11_sac_replay.py
```

如果不想导出 GIF，可以在 `main11_sac_replay.py` 顶部设置：

```python
EXPORT_ANIMATION = False
```
