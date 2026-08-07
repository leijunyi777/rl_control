# Lane-Change Simulation Backup

这个文件夹是 `code/python` 主线代码的一份整理版备份。原始源代码没有被修改；这里的脚本经过规范命名，并把项目内部依赖尽量收敛到本文件夹内。

## 依赖关系

备份脚本的项目内依赖已经本地化：

```text
backup/
  model_ode.py                  共享车辆模型、意见动力学、main7 间隙跟随场景、绘图工具
  sim_*.py                      仿真脚本，主要依赖 model_ode.py
  rl_*.py                       SAC 训练/回放脚本，依赖 model_ode.py 和对应的本地训练文件
```

外部 Python 库仍需要：

```text
numpy
scipy
matplotlib
torch   # 仅 RL 训练/回放需要
```

## 版本说明

| 文件 | 来源 | 简要说明 |
| :--- | :--- | :--- |
| `sim_01_main5_basic_collision.py` | `main5.py` | 三车基础 ODE 仿真，包含圆角矩形车辆、距离监控、碰撞阈值 `r=1.5m`。 |
| `sim_02_main6_random_rear.py` | `main6.py` | 后车前 20s 使用随机加减速度，20s 后开始让行。 |
| `sim_03_main7_gap_following.py` | `main7.py` | 仿真时间扩展到 40s；20s 后后车跟随前车并保持 20m 目标间隙；支持 GIF 导出。 |
| `sim_04_main8_multigap_test.py` | `main8_multigap_test.py` | 目标车道扩展为 5 辆车、4 个候选间隙；只测试多间隙 `mu/z/u` 更新，不实际控制 ego 并道。 |
| `sim_05_main9_multigap_control.py` | `main9_multigap_select_control.py` | 在多间隙场景中选择控制间隙并实际控制 ego。 |
| `sim_06_main10_gap_z_nomove.py` | `main10-nomove.py` | 使用新 `b(t), u(t)` 更新 `z_new`；ego 不执行并道控制，只匀速直行；对比旧 `z` 和新 `z`。 |
| `sim_07_main10_gap_z_move.py` | `main10-move.py` | 使用 main10 的新 `z_new` 实际控制 ego 并道。 |
| `sim_08_main11_rbf_u_nomove.py` | `main11-nomove.py` | 使用 main11 最新 RBF 置信度公式计算 `u(t)`；ego 匀速直行，只观察 `z` 更新。 |
| `sim_09_main11_rbf_u_move.py` | `main11-move.py` | 使用 main11 最新 `b(t)` 与 RBF `u(t)` 更新 `z_new` 并实际控制 ego。 |
| `rl_01_main7_sac_train_params.py` | `main_sac_train.py` | SAC 学习 `[k_mu, k, eps]` 三个控制参数，环境基于 main7。 |
| `rl_02_main7_sac_replay.py` | `replay.py` | 加载 main7 SAC 策略并导出回放。 |
| `rl_03_main10_sac_train_b.py` | `main10_sac_train.py` | SAC action 为 `b(t)`，直接注入 main10 新意见动力学。 |
| `rl_04_main10_sac_replay_b.py` | `main10_sac_replay.py` | 加载 main10 的 `b(t)` 策略并回放。 |
| `rl_05_main11_sac_train_u.py` | `main11_sac_train.py` | SAC action 为 `u(t)`，直接注入 main11 最新意见动力学。 |
| `rl_06_main11_sac_replay_u.py` | `main11_sac_replay.py` | 加载 main11 的 `u(t)` 策略并回放。 |

## 核心公式演进

### 旧论文控制链路

```text
mu_dot = -k_mu * mu + tanh(...)
z_dot  = (1 / eps) * z * (mu - z)
u_total = u_n + u_c
```

含义：`mu` 评估环境是否适合并道，`z` 表示并道意愿，`u_n` 跟踪由 `z` 决定的目标位置，`u_c` 负责安全避碰。

### Main10 的新 z 更新

```text
b(t) = tanh(k_gap * (gap - gap_safe) + k_vel * gap_dot)
u(t) = u_base + u_gain * max(0, -gap_dot)
z_dot = -d * z + u(t) * tanh(alpha * z) + b(t)
```

其中 `b(t)` 表示客观并道机会，`u(t)` 表示间隙缩小时的注意力/紧迫程度。

### Main11 的最新 RBF-u 更新

```text
x_gap = (x_front + x_rear) / 2
v_gap = (v_front + v_rear) / 2
d_gap = x_gap - x_ego
dv_gap = v_ego - v_gap

u(t) = u_base + u_amp * exp(
  - d_gap^2  / (2 * sigma_d^2)
  - dv_gap^2 / (2 * sigma_v^2)
)

b(t) = tanh(k_gap * (gap - gap_safe) + k_vel * gap_dot)
z_dot = -d * z + u(t) * tanh(alpha * z) + b(t)
```

直观理解：`b(t)` 判断空隙是否客观安全；`u(t)` 判断 ego 是否已经对齐空隙中心且速度接近，从而决定是否有信心执行并道。

## 运行示例

在 `backup` 文件夹内运行：

```bash
python sim_09_main11_rbf_u_move.py
python rl_05_main11_sac_train_u.py
python rl_06_main11_sac_replay_u.py
```

如果只想检查语法：

```bash
python -m py_compile *.py
```

注意：PowerShell 下 `*.py` 可能不会被 Python 自动展开，可以用：

```powershell
foreach ($file in Get-ChildItem -Filter *.py) { python -m py_compile $file.FullName }
```

## 整理原则

- 不修改 `code/python` 下的源代码。
- 备份文件使用更明确的顺序编号和语义化命名。
- `model_ode.py` 集中保存车辆模型、意见动力学公式、main7 间隙跟随场景和绘图工具。
- main11 文件已独立于 main10，不再继承或读取 main10 参数。
- 已执行逐文件 `py_compile` 检查，backup 内所有 Python 文件语法通过。
