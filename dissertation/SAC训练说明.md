# SAC 强化学习训练说明

本文档用尽量简单的语言说明本项目中的 SAC 训练代码，包括 SAC 的基本原理、它如何作用在本项目的并道控制问题上、RK45 积分求解的微分方程，以及强化学习中的 `state`、`action`、`reward` 是如何定义的。

---

## 1. SAC 是什么

SAC 的全称是 **Soft Actor-Critic**，是一种强化学习算法。

强化学习可以简单理解为：

```text
智能体观察环境 state
        ↓
选择动作 action
        ↓
环境发生变化
        ↓
智能体得到奖励 reward
        ↓
智能体根据奖励调整策略
```

在本项目里，智能体不是直接控制车辆加速度，而是调节论文控制器里的几个关键参数，让车辆更快、更稳、更安全地完成并道。

---

## 2. SAC 的核心思想

SAC 主要包含三类网络：

### 2.1 Actor：策略网络

Actor 负责根据当前状态选择动作：

```text
action = policy(state)
```

在本项目中：

```text
state  = ego 车相对前车和后车的位置、速度
action = [k_mu, k, epsilon]
```

也就是说，Actor 学的是：

```text
当前交通状态下，应该怎样调节决策动态参数
```

### 2.2 Critic：价值网络

Critic 负责评价某个动作好不好。

它学习的是：

```text
Q(state, action)
```

意思是：

```text
在当前 state 下采取 action，未来大概能得到多少奖励
```

SAC 里通常有两个 Critic 网络，代码里对应 `q1` 和 `q2`。这样可以减少价值估计过高的问题。

### 2.3 Entropy：鼓励探索

SAC 和普通 Actor-Critic 一个很大的区别是：它会鼓励策略保持一定随机性。

也就是说，它不会一开始就死盯着某一个动作，而是会多尝试不同动作。

这对本项目很重要，因为一开始智能体并不知道：

```text
k_mu 多大合适？
k 多大合适？
epsilon 多大合适？
```

所以需要探索。

---

## 3. SAC 在本项目里学什么

本项目本身已经有一个基于论文的控制器。

这个控制器包括：

```text
环境评估参数 mu
意见状态 z
目标相对位置 e*
名义跟踪控制 u_n
安全避碰控制 u_c
```

SAC 不直接替代这个控制器，而是学习如何调节其中三个参数：

```text
action = [k_mu, k, epsilon]
```

其中：

```text
k_mu    控制 mu 的衰减速度
k       控制环境评估公式的敏感程度
epsilon 控制 z 的变化快慢
```

直观理解：

```text
k_mu    决定环境评分 mu 多快忘掉过去
k       决定系统对并道机会有多敏感
epsilon 决定并道意愿 z 反应有多快
```

SAC 训练的目标是找到一套策略，使得车辆：

```text
有机会时尽快并道
不要频繁左右犹豫
不要和前后车碰撞
```

---

## 4. RK45 在求解什么

本项目每一步仿真都使用 `solve_ivp` 中的 `RK45` 方法。

RK45 是一种数值积分方法，用来求解常微分方程：

```text
dot x = f(x, t)
```

在本项目中，RK45 同时求解：

```text
前车状态
后车状态
ego 车状态
意见状态 z
环境评分 mu
```

总状态可以写成：

```text
X = [vehicle_1_state,
     vehicle_2_state,
     ego_state,
     z,
     mu]
```

其中每辆车的状态是：

```text
vehicle_state = [x, y, theta, v, delta]
```

含义分别是：

```text
x, y   后轴中心位置
theta  车辆朝向角
v      后轮速度
delta  前轮转角
```

---

## 5. 车辆微分方程

车辆使用自行车模型：

```text
dot x     = v cos(theta)
dot y     = v sin(theta)
dot theta = v tan(delta) / L
dot v     = a
dot delta = omega
```

其中：

```text
a     纵向加速度
omega 转角变化率
L     轴距
```

在代码中，这部分对应：

```python
rear_state_derivative(...)
```

---

## 6. 前轴位置和速度

论文控制器主要控制的是前轴中心点，而车辆状态记录的是后轴中心点。

所以需要先把后轴点转换成前轴点：

```text
p = [x + L cos(theta),
     y + L sin(theta)]
```

前轴速度为：

```text
v_front =
v [cos(theta) - sin(theta) tan(delta),
   sin(theta) + cos(theta) tan(delta)]
```

这些量用于计算 ego 车相对前后车的位置和速度。

---

## 7. 环境评分 mu 的动态

论文中使用 `mu` 判断当前环境是否适合并道。

代码中的形式为：

```text
dot mu =
-k_mu * mu
+ tanh(-k * rho^T g31 * rho^T g32 * (d21 - 2r) * (phi21 + epsilon2))
```

其中：

```text
g31  ego 指向前车方向的单位向量
g32  ego 指向后车方向的单位向量
d21  前车和后车之间的安全间距
phi21 前后车之间距离变化率 / 距离
rho  道路纵向方向
```

直观理解：

```text
如果 ego 位于前后车之间
并且前后车间距足够
并且间距趋势不危险
那么 mu 会变大
```

`mu` 变大后，会推动并道意愿 `z` 增大。

---

## 8. 意见状态 z 的动态

论文中用 `z` 表示 ego 车的并道意愿。

公式为：

```text
dot z = (1 / epsilon) * (-z^2 + mu z)
```

也可以写成：

```text
dot z = (1 / epsilon) * z * (mu - z)
```

含义是：

```text
mu > 0 时，z 倾向于变大，车辆更想并道
mu < 0 时，z 倾向于回到 0，车辆不急着并道
```

`epsilon` 越小，`z` 变化越快。

这也是 SAC 要学习的动作之一。

---

## 9. 目标位置如何由 z 决定

控制器会根据 `z` 计算目标相对位置：

```text
w(z) = tanh(k_w z)
```

```text
e31d = rho * r_rho + eta * (1 - w(z)) * r_eta
```

当 `z` 很小时：

```text
w(z) 约等于 0
ego 保持在原车道
```

当 `z` 变大时：

```text
w(z) 约等于 1
横向偏移逐渐消失
ego 向目标车道并入
```

---

## 10. 控制输入 u

控制输入分成两部分：

```text
u = u_n + u_c
```

其中 `u_n` 是名义跟踪控制：

```text
u_n = -k_p (e31 - e31d) - k_v v31
```

含义是：

```text
让 ego 跟踪由 z 决定的目标位置
```

`u_c` 是安全避碰控制：

```text
u_c = - sum k_o * g3j * phi3j
```

其中：

```text
phi3j = g3j^T v3j / d3j
d3j   = ego 到第 j 辆车的距离 - r
```

含义是：

```text
如果 ego 靠近前车或后车
安全项会变大
把 ego 推离危险区域
```

---

## 11. 从 u 转换到车辆输入

控制器先得到的是前轴点的平面加速度：

```text
u = [u_x, u_y]
```

但真实自行车模型需要的是：

```text
a     纵向加速度
omega 转角变化率
```

因此代码中通过矩阵转换：

```text
[a, omega]^T = A^{-1} (u - B)
```

这样就能把控制器设计出来的 `u` 转换为车辆模型真正能执行的输入。

---

## 12. RL 中的 state 是什么

在 SAC 训练中，每一步的状态 `state` 定义为：

```text
ego 相对前车的位置
ego 相对前车的速度
ego 相对后车的位置
ego 相对后车的速度
```

写成向量是：

```text
state =
[relative_position_to_front,
 relative_velocity_to_front,
 relative_position_to_rear,
 relative_velocity_to_rear]
```

展开后是 8 维：

```text
state =
[x_front, y_front,
 vx_front, vy_front,
 x_rear, y_rear,
 vx_rear, vy_rear]
```

这些量告诉智能体：

```text
ego 现在离前车多远
ego 现在离后车多远
ego 和它们的相对速度是多少
```

---

## 13. RL 中的 action 是什么

动作 `action` 是三个控制参数：

```text
action = [k_mu, k, epsilon]
```

由于神经网络通常输出归一化动作，所以代码中先让 Actor 输出：

```text
[-1, 1] 范围内的动作
```

再映射到真实参数范围：

```text
k_mu    in [0.5, 12.0]
k       in [1.0, 50.0]
epsilon in [0.01, 0.20]
```

智能体每一步都可以根据当前交通状态调节这三个参数。

---

## 14. RL 中的 reward 是什么

奖励函数的目标是让智能体学会：

```text
有机会时尽快并道
不要反复改变方向
不要离前后车太近
不要碰撞
```

当前奖励公式为：

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

其中：

```text
progress       当前变道完成比例
delta_progress 本步变道进展
opportunity    是否存在并道机会
d_min          ego 到前后车的最小距离
safe_margin    安全余量，等于 2.5r
collision      是否碰撞
success        是否成功并道
```

下面对公式中的每一项做更详细的中文注释。

### 14.1 变道进展奖励

```text
+ 80.0 * delta_progress
```

中文注释：

```text
delta_progress = 当前 progress - 上一步 progress
```

含义：

```text
如果 ego 朝目标车道移动，progress 增大，delta_progress > 0，这一项给正奖励。
如果 ego 往原车道退回，progress 减小，delta_progress < 0，这一项变成惩罚。
```

作用：

```text
鼓励车辆持续向目标车道靠近，而不是停在原车道或来回退让。
```

这一项现在是主要的并道驱动力。相比旧版本的 `5.0`，系数提高到 `80.0`，目的是让“确实往目标车道移动”比“停着等待”更有吸引力。

### 14.2 已经进入目标车道方向的持续奖励

```text
+ 0.15 * progress
```

中文注释：

```text
progress 越大，说明 ego 越靠近目标车道。
```

含义：

```text
只要 ego 已经开始向目标车道移动，即使本步 delta_progress 很小，也能获得一点持续正奖励。
```

作用：

```text
防止车辆刚开始并道后又觉得“保持原位更省事”，鼓励它把并道动作继续完成。
```

### 14.3 有机会时推进的额外奖励

```text
+ 40.0 * opportunity * max(delta_progress, 0)
```

中文注释：

```text
opportunity = 1 表示当前存在并道机会
opportunity = 0 表示当前不适合并道
```

在代码中，机会由环境评分和目标车道间隙共同判断：

```text
opportunity = 1, if mu > 0.1 or target_lane_gap > gap_safe
opportunity = 0, otherwise
```

含义：

```text
当环境认为有机会并道时，如果 ego 正在向目标车道推进，就额外加分。
如果 ego 没有推进，max(delta_progress, 0) = 0，这一项不给分。
```

作用：

```text
鼓励“有机会就尽快进入”，不要错过合适的并道窗口。
```

这里使用 `max(delta_progress, 0)`，是为了避免车辆后退时仍然获得机会奖励。相比旧版本的 `2.0`，系数提高到 `40.0`，让“有机会时向目标车道移动”的信号更强。

### 14.4 后退惩罚

```text
- 30.0 * max(-delta_progress, 0)
```

中文注释：

```text
如果 delta_progress < 0，说明 ego 正在从目标车道方向退回原车道。
max(-delta_progress, 0) 会把这种后退量变成正数，再乘以 -30.0 扣分。
```

含义：

```text
车辆不但要动，还要尽量朝正确方向动。
```

作用：

```text
减少并道过程中的反复横向摆动，避免策略学到“先动一下再退回去”的无效动作。
```

### 14.5 有机会但犹豫的惩罚

```text
- 0.25 * opportunity * (1 - progress)
```

中文注释：

```text
1 - progress 表示距离完成并道还差多少。
```

含义：

```text
当 opportunity = 1 时，说明当前适合并道。
如果此时 progress 还很小，说明 ego 还没有进入目标车道，于是每一步扣一点分。
progress 越接近 1，这个惩罚越小。
```

作用：

```text
防止智能体在有机会时一直拖延或原地等待。相比旧版本的 `0.05`，现在提高到 `0.25`，所以 `progress = 0` 且有机会时，每一步都会受到更明显的扣分。
```

### 14.6 时间惩罚

```text
- 0.05 * (1 - progress)
```

中文注释：

```text
每经过一个仿真步，只要还没完成并道，就会扣时间分。
```

含义：

```text
progress 越小，说明越没完成并道，时间惩罚越明显。
progress 越接近 1，说明快完成并道，时间惩罚越弱。
```

作用：

```text
鼓励更快完成并道，而不是拖到仿真结束。即使 `opportunity = 0`，停在原车道也会持续扣分，所以 `progress = 0` 的策略不再容易得到接近 0 或接近 100 的分数。
```

### 14.7 action 平滑惩罚

```text
- 0.5 * ||action_t - action_{t-1}||^2
```

中文注释：

```text
action_t = 当前 SAC 输出的动作
action_{t-1} = 上一步 SAC 输出的动作

本项目中：
action = [k_mu, k, epsilon]
```

含义：

```text
如果当前控制参数和上一时刻差别很大，就扣分。
差别越大，平方惩罚越明显。
```

作用：

```text
避免智能体每一步剧烈改变 k_mu、k、epsilon。
让参数变化更平滑，从而让车辆行为更稳定。
```

### 14.8 横向速度反复换向惩罚

```text
- 2.0 * I[v_y(t) * v_y(t-1) < 0]
```

中文注释：

```text
v_y(t)   = 当前时刻 ego 的横向速度
v_y(t-1) = 上一时刻 ego 的横向速度
I[...]   = 指示函数，条件成立取 1，否则取 0
```

含义：

```text
如果 v_y(t) 和 v_y(t-1) 符号相反，说明横向速度方向发生反转。
例如上一刻向目标车道移动，这一刻又往原车道方向移动。
```

作用：

```text
惩罚左右反复摆动，减少并道过程中的犹豫和振荡。
```

这一项对应“不要反复变速度方向”的要求。

### 14.9 安全距离连续惩罚

```text
- 20.0 * max(0, (safe_margin - d_min) / safe_margin)^2
```

中文注释：

```text
d_min = ego 到前车和后车的最小距离
safe_margin = 2.5 * r
r = 碰撞界限
```

含义：

```text
如果 d_min >= safe_margin：
    ego 距离前后车足够远，不扣分。

如果 d_min < safe_margin：
    ego 已经离某辆车较近，开始扣分。

距离越接近碰撞界限 r，惩罚越大。
```

平方项的作用：

```text
轻微靠近时扣得少；
非常接近时扣得很重。
```

作用：

```text
不等到真正碰撞才惩罚，而是在靠得太近时提前惩罚。
```

### 14.10 碰撞惩罚

```text
- 1000 * I[collision]
```

中文注释：

```text
collision = True 表示发生碰撞
collision = False 表示没有碰撞
```

含义：

```text
一旦发生碰撞，立即扣 1000 分。
```

作用：

```text
明确告诉智能体：碰撞是最严重的失败结果。
```

在代码中，碰撞还会让当前 episode 提前结束。

### 14.11 成功奖励

```text
+ (100 - 2t) * I[success]
```

中文注释：

```text
success = True 表示满足成功并道条件
t = 成功时刻，单位为秒
```

含义：

```text
如果成功并道，则给一次性大奖励。
成功越早，奖励越高。
```

例子：

```text
t = 10s 时：100 - 2 * 10 = 80
t = 20s 时：100 - 2 * 20 = 60
t = 30s 时：100 - 2 * 30 = 40
```

作用：

```text
不仅鼓励成功并道，还鼓励尽快成功。
```

### 14.12 总体效果

这个 reward 设计会把不同策略的分数拉开：

```text
快速、安全、平滑地并道：得分高
慢速但安全地并道：得分中等
有机会却犹豫：持续扣分
横向反复摆动：被换向惩罚扣分
离前后车太近：被安全距离项持续扣分
发生碰撞：大幅负分，并提前结束
```

因此训练曲线不会再轻易全部接近 100，而是能更明显地区分策略优劣。

---

## 15. 成功条件

当前成功条件是：

```text
progress > 0.95
|ego_y - target_lane_y| < 0.2
d_min > 1.5r
```

也就是说：

```text
ego 基本进入目标车道
ego 横向位置足够接近目标车道中心
ego 和前后车保持足够安全距离
```

满足成功条件后，本轮仿真提前结束，并给予成功奖励。

---

## 16. 碰撞条件

碰撞判断为：

```text
d_min < r
```

其中：

```text
d_min = min(ego 到前车距离, ego 到后车距离)
r     = 碰撞界限
```

如果发生碰撞：

```text
reward -= 1000
本轮仿真结束
```

---

## 17. 训练过程

整个训练过程可以理解为：

```text
1. 初始化一轮仿真
2. ego 观察当前 state
3. SAC 的 Actor 输出 action
4. action 被映射成 [k_mu, k, epsilon]
5. 控制器使用这些参数计算 mu、z 和控制输入 u
6. RK45 积分车辆微分方程，推进一个时间步
7. 根据并道进展、安全距离、是否碰撞等计算 reward
8. 把 (state, action, reward, next_state, done) 放入经验池
9. SAC 从经验池采样，更新 Actor 和 Critic
10. 重复以上过程，直到成功、碰撞或仿真时间结束
```

训练很多轮之后，Actor 会逐渐学到：

```text
在什么相对位置和速度下
应该选择什么样的 k_mu、k、epsilon
才能更快、更稳、更安全地完成并道
```

---

## 18. 训练完成后的回放

训练结束后，代码会保存：

```text
sac_policy.pth
```

然后可以运行：

```text
python reply.py
```

`reply.py` 会加载训练好的 policy，并进行一次带图像的仿真回放。

---

## 19. 新手理解总结

可以把整个系统想成三层：

### 第一层：车辆运动层

负责回答：

```text
给定加速度和转角速度，车下一步在哪里？
```

由 RK45 求解自行车模型微分方程。

### 第二层：论文控制器层

负责回答：

```text
当前环境适不适合并道？
ego 应该向哪里移动？
如何避免碰撞？
```

核心变量是：

```text
mu, z, u_n, u_c
```

### 第三层：强化学习层

负责回答：

```text
控制器参数应该怎么调？
```

SAC 学习的是：

```text
state → [k_mu, k, epsilon]
```

最终目标是：

```text
有机会就尽快并道
并道过程尽量平滑
始终避免碰撞
```
