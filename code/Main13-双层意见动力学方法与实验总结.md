# Main13 双层意见动力学控制方法与实验总结  
# Main13 Bilevel Opinion-Dynamics Control: Methodology and Experimental Summary

> 本文档以当前 `main13_common.py`、`main13-move.py`、`main13-nomove.py`、`main13-random-test.py`、`main12_sac_train.py` 和 `main12-value.py` 的最新流程为依据，整理为科研论文方法章节风格的中英文对照说明。  
> This document summarizes the latest Main13 workflow in a research-paper style, based on the current implementations in `main13_common.py`, `main13-move.py`, `main13-nomove.py`, `main13-random-test.py`, `main12_sac_train.py`, and `main12-value.py`.

---

## Abstract / 摘要

**English.**  
This work presents a bilevel decision and control framework for autonomous lane merging in interactive traffic. The core idea is to separate the lane-change problem into two coupled but interpretable layers. The high-level layer decides *which gap* in the target lane should be selected. It observes only the three target-lane vehicles closest to the ego vehicle, forms two candidate gaps, evaluates their relative feasibility through radial-basis-function confidence scores, and evolves a high-level opinion state according to a self-reinforcing opinion-dynamics equation. The sign and magnitude of the high-level opinion determine whether the vehicle should merge into the forward gap, merge into the rear gap, or wait. The low-level layer then performs the actual merging maneuver for the selected gap. It computes an objective gap bias from gap size and gap-change rate, uses either a learned Soft Actor-Critic policy or an analytic Gaussian attention formula to determine the attention intensity, updates a low-level opinion state, converts that opinion into a target point inside the selected gap, and finally generates physical acceleration and steering-rate commands through a front-axle tracking controller with an additional safety-avoidance term. The method is designed so that decision confidence can accumulate smoothly over time, while small but persistent differences between candidate gaps can be amplified into decisive actions without relying on brittle hard thresholds. Experiments are organized in two stages: a single-gap environment derived from Main12 is used to train and evaluate the low-level attention policy, and a multi-vehicle Main13 environment is used to test whether the learned low-level policy and the high-level opinion-dynamics gap selector generalize to randomly changing traffic gaps.

**中文。**  
本文提出一种用于自动驾驶车辆并入目标车道的双层决策与控制框架。核心思想是把并道问题拆分为两个相互耦合但含义清晰的层次。高层负责判断“应该选择目标车道中的哪个空隙”，它只读取距离 ego 车最近的三辆目标车道车辆，由三车形成前后两个候选 gap，通过径向基函数形式的置信度评价两个 gap 的可行性，并利用带自强化注意力项的意见动力学方程更新高层意见状态。高层意见的符号和幅值决定 ego 是向前 gap 并入、向后 gap 并入，还是暂时等待。底层负责执行实际并道动作：它根据所选 gap 的大小和变化速率计算客观偏置，根据 SAC 学习策略或手工高斯注意力公式计算注意力强度，更新底层意见状态，再把意见状态映射为所选 gap 内的目标控制点，最后结合前轴点跟踪控制与安全避障项生成实际加速度和转角变化率。该方法的关键优势是：决策信心可以随时间平滑积累，即使两个候选 gap 的评价差异很小，只要这种差异持续存在，也能够在注意力自更新机制下被放大为明确决策，而不依赖脆弱的硬阈值。实验分为两阶段：先用 Main12 单 gap 环境训练和评价底层注意力策略，再在 Main13 多车随机 gap 环境中验证高层意见动力学 gap 选择器与底层策略的泛化能力。

---

## Notation / 符号定义

**English.**  
The vehicle state is represented by the rear-axle bicycle model state
\[
\mathbf{x}_i=[x_i,y_i,\theta_i,v_i,\delta_i]^\top,
\]
where \((x_i,y_i)\) is the rear-axle center, \(\theta_i\) is the heading angle, \(v_i\) is the rear-wheel longitudinal speed, and \(\delta_i\) is the steering angle. The control task, however, is formulated at the front-axle point. For a vehicle with wheelbase \(L\), the front-axle position and velocity are
\[
\mathbf{p}_i=
\begin{bmatrix}
x_i+L\cos\theta_i\\
y_i+L\sin\theta_i
\end{bmatrix},
\quad
\mathbf{v}_i^f
=v_i
\begin{bmatrix}
\cos\theta_i-\sin\theta_i\tan\delta_i\\
\sin\theta_i+\cos\theta_i\tan\delta_i
\end{bmatrix}.
\]
The target lane center is \(y_T=1.5W\), the original lane center is \(y_O=0.5W\), and the lane width in the current experiments is \(W=4.0\,\mathrm{m}\). The vehicle wheelbase used by the scenario is \(L=2.8\,\mathrm{m}\), while the collision boundary radius used by the controller and visualization is \(r=1.5\,\mathrm{m}\). In Main13, the target lane contains \(N=5\) vehicles, producing \(N-1=4\) physical gaps. The high-level module reads only the nearest three target-lane vehicles, so at each time step it considers two candidate gaps.

**中文。**  
车辆状态采用后轴自行车模型：
\[
\mathbf{x}_i=[x_i,y_i,\theta_i,v_i,\delta_i]^\top,
\]
其中 \((x_i,y_i)\) 为后轴中心坐标，\(\theta_i\) 为航向角，\(v_i\) 为后轮纵向速度，\(\delta_i\) 为前轮转角。但控制器主要控制的是前轴中心点。对于轴距为 \(L\) 的车辆，前轴点位置与速度为
\[
\mathbf{p}_i=
\begin{bmatrix}
x_i+L\cos\theta_i\\
y_i+L\sin\theta_i
\end{bmatrix},
\quad
\mathbf{v}_i^f
=v_i
\begin{bmatrix}
\cos\theta_i-\sin\theta_i\tan\delta_i\\
\sin\theta_i+\cos\theta_i\tan\delta_i
\end{bmatrix}.
\]
当前实验中车道宽度为 \(W=4.0\,\mathrm{m}\)，原车道中心为 \(y_O=0.5W\)，目标车道中心为 \(y_T=1.5W\)。车辆轴距为 \(L=2.8\,\mathrm{m}\)，碰撞边界半径为 \(r=1.5\,\mathrm{m}\)。Main13 的目标车道包含 \(N=5\) 辆车，因此共有 \(N-1=4\) 个物理 gap。高层决策模块每一步只读取距离 ego 最近的三辆目标车道车辆，因此实际参与高层决策的是两个候选 gap。

| Symbol / 符号 | English definition | 中文定义 |
| :--- | :--- | :--- |
| \(\mathbf{x}_i\) | Rear-axle vehicle state | 后轴车辆状态 |
| \(\mathbf{p}_i\) | Front-axle control point | 前轴控制点 |
| \(\mathbf{v}_i^f\) | Front-axle velocity | 前轴速度 |
| \(y(t)\) | High-level opinion for gap direction | 高层 gap 方向意见 |
| \(u_h(t)\) | High-level self-updating attention | 高层自更新注意力 |
| \(z(t)\) | Low-level merge intention opinion | 底层并道意愿意见 |
| \(b(t)\) | Objective low-level gap bias | 底层客观 gap 偏置 |
| \(u(t)\) | Low-level attention intensity | 底层注意力强度 |
| \(C_f,C_r\) | Forward/rear candidate-gap confidence | 前 gap / 后 gap 置信度 |
| \(B=C_f-C_r\) | High-level directional bias | 高层方向偏置 |
| \(d_{\min}\) | Minimum ego-to-target distance | ego 到目标车最小距离 |

---

# 1. Bilevel Control System / 双层控制系统

**English.**  
The Main13 system is organized as a bilevel controller. The high level answers a discrete strategic question: among the locally relevant target-lane gaps, should the ego vehicle aim for the forward gap, the rear gap, or no gap yet? The low level answers a continuous control question: once a candidate gap has been selected, how strongly should the ego vehicle commit to the maneuver, where should the target point be placed, and what acceleration and steering-rate commands should be applied? This decomposition is important because lane merging contains both symbolic decision structure and continuous control structure. A pure continuous controller can move toward a target but may not decide reliably between multiple similar gaps. A pure discrete selector can choose a gap but cannot guarantee smooth entry, safe clearance, and stable steering. The proposed design therefore lets the high level choose the target gap through opinion dynamics, while the low level converts the selected gap into a smooth control target.

**中文。**  
Main13 系统采用双层控制结构。高层回答的是离散策略问题：在局部相关的目标车道空隙中，ego 车应该选择前方 gap、后方 gap，还是暂时等待？底层回答的是连续控制问题：一旦候选 gap 被选中，ego 应该以多强的并道意愿执行、目标控制点应该放在哪里、最终应施加怎样的加速度和转角变化率？这种分解很重要，因为并道问题同时包含符号化决策和连续动力学控制。单纯连续控制器可以跟踪某个目标点，但面对多个相似 gap 时容易犹豫或频繁切换；单纯离散选择器可以选 gap，却不能保证平滑进入、保持安全距离和转向稳定。因此本文让高层通过意见动力学选择目标 gap，再由底层把该 gap 转换为可执行的连续控制目标。

**English.**  
The information flow can be summarized as
\[
\{\mathbf{x}_{1:N},\mathbf{x}_e\}
\rightarrow
\text{nearest three vehicles}
\rightarrow
\{C_f,C_r,B,y,u_h\}
\rightarrow
\text{selected gap}
\rightarrow
\{b(t),u(t),z(t)\}
\rightarrow
\mathbf{p}^{\star}(t)
\rightarrow
\mathbf{u}_{total}
\rightarrow
[a,\omega].
\]
Here \(\mathbf{x}_{1:N}\) are the target-lane vehicle states, \(\mathbf{x}_e\) is the ego state, \(C_f\) and \(C_r\) are the high-level confidences of the forward and rear candidate gaps, \(B\) is the high-level directional bias, \(y\) is the high-level opinion, \(u_h\) is the high-level attention, \(b(t)\) and \(u(t)\) are low-level gap bias and attention, \(z(t)\) is the low-level merge opinion, \(\mathbf{p}^{\star}(t)\) is the desired target point, \(\mathbf{u}_{total}\) is the desired front-axle acceleration vector, and \([a,\omega]\) are the physical bicycle-model inputs. This layered chain is exactly what distinguishes Main13 from the single-gap Main12 environment: Main12 fixes the candidate gap and learns how to enter it, whereas Main13 must first decide which gap is the relevant one.

**中文。**  
整体信息流可写为
\[
\{\mathbf{x}_{1:N},\mathbf{x}_e\}
\rightarrow
\text{最近三辆目标车}
\rightarrow
\{C_f,C_r,B,y,u_h\}
\rightarrow
\text{被选中的 gap}
\rightarrow
\{b(t),u(t),z(t)\}
\rightarrow
\mathbf{p}^{\star}(t)
\rightarrow
\mathbf{u}_{total}
\rightarrow
[a,\omega].
\]
其中 \(\mathbf{x}_{1:N}\) 是目标车道车辆状态，\(\mathbf{x}_e\) 是 ego 状态，\(C_f\) 和 \(C_r\) 是高层对前 gap 与后 gap 的置信度，\(B\) 是高层方向偏置，\(y\) 是高层意见，\(u_h\) 是高层注意力，\(b(t)\) 和 \(u(t)\) 是底层 gap 偏置与注意力，\(z(t)\) 是底层并道意见，\(\mathbf{p}^{\star}(t)\) 是目标控制点，\(\mathbf{u}_{total}\) 是期望前轴加速度向量，\([a,\omega]\) 是最终自行车模型输入。这个链条也是 Main13 和 Main12 的根本区别：Main12 固定一个候选 gap，只学习如何并入；Main13 必须先判断哪个 gap 值得并入。

## 1.1 Opinion Dynamics and Self-Updating Attention / 意见动力学与自更新注意力公式

**English.**  
Both decision layers use the same conceptual opinion-dynamics template:
\[
\dot{o}=-d\,o+u(t)\tanh(\alpha o)+b(t),
\]
where \(o\) is an opinion variable, \(d>0\) is a damping coefficient, \(u(t)\ge 0\) is attention or commitment intensity, \(\alpha>0\) controls the steepness of the saturated self-reinforcement term, and \(b(t)\) is an external environmental bias. The damping term \(-d\,o\) prevents uncontrolled opinion drift. If environmental evidence disappears, the opinion decays back toward neutrality. The nonlinear term \(u(t)\tanh(\alpha o)\) is a bounded self-reinforcement mechanism: once a small opinion is formed, stronger attention makes that opinion easier to maintain and amplify. The bias \(b(t)\) injects objective evidence from the traffic scene. In Main13, this template appears twice. At the high level, \(o=y\) and \(b=B=C_f-C_r\). At the low level, \(o=z\) and \(b=b(t)\) is computed from the selected physical gap.

**中文。**  
两个决策层都使用同一种意见动力学模板：
\[
\dot{o}=-d\,o+u(t)\tanh(\alpha o)+b(t),
\]
其中 \(o\) 是意见变量，\(d>0\) 是阻尼系数，\(u(t)\ge 0\) 是注意力或执行信心强度，\(\alpha>0\) 控制饱和自强化项的陡峭程度，\(b(t)\) 是外部环境偏置。阻尼项 \(-d\,o\) 防止意见无限漂移，当环境证据消失时，意见会回到中性附近。非线性项 \(u(t)\tanh(\alpha o)\) 是有界自强化机制：一旦产生微小意见，较大的注意力会让该意见更容易保持并被放大。偏置项 \(b(t)\) 则把交通环境中的客观证据输入系统。在 Main13 中，该模板出现两次：高层中 \(o=y\)，偏置为 \(B=C_f-C_r\)；底层中 \(o=z\)，偏置为由所选物理 gap 计算得到的 \(b(t)\)。

**English.**  
The high-level attention is not directly assigned by a neural network. It is self-updated by the high-level opinion itself:
\[
\dot{u}_h=\frac{-u_h+S_h(y^2)}{\tau_h},
\]
\[
S_h(y^2)=U_{\max}\frac{(y^2)^n}{K_h^n+(y^2)^n}.
\]
This Hill-type function is symmetric in the sign of \(y\). It does not care whether the vehicle is leaning toward the forward gap or the rear gap; it only cares whether the opinion magnitude \(|y|\) has become sufficiently strong. When \(y\approx 0\), the function output is close to zero, so attention decays and the vehicle remains cautious. When \(|y|\) grows, \(S_h\) increases rapidly, which increases \(u_h\), which then amplifies the self-reinforcement term in the \(y\)-dynamics. This creates a positive feedback loop: persistent evidence produces a small opinion, the opinion raises attention, attention strengthens the opinion, and the decision becomes stable.

**中文。**  
高层注意力不是由神经网络直接给出，而是由高层意见自身进行自更新：
\[
\dot{u}_h=\frac{-u_h+S_h(y^2)}{\tau_h},
\]
\[
S_h(y^2)=U_{\max}\frac{(y^2)^n}{K_h^n+(y^2)^n}.
\]
这个 Hill 函数只与 \(y^2\) 有关，因此对 \(y\) 的正负号对称。它不关心车辆倾向前 gap 还是后 gap，只关心意见强度 \(|y|\) 是否足够明确。当 \(y\approx 0\) 时，函数输出接近 0，注意力衰减，系统保持谨慎等待；当 \(|y|\) 增大时，\(S_h\) 快速上升，使 \(u_h\) 增大，而更大的 \(u_h\) 又会放大 \(y\) 方程中的自强化项。于是系统形成正反馈闭环：持续证据产生微小意见，微小意见提高注意力，注意力进一步强化意见，最终形成稳定决策。

| Term / 项 | Formula / 公式 | English role | 中文作用 |
| :--- | :--- | :--- | :--- |
| Damping / 阻尼 | \(-d\,o\) | Pulls opinion back to neutral and avoids drift. | 拉回中性，避免意见漂移。 |
| Self-reinforcement / 自强化 | \(u\tanh(\alpha o)\) | Amplifies an already formed opinion but remains bounded. | 放大已有意见，同时保持饱和有界。 |
| External bias / 外部偏置 | \(b(t)\) | Injects objective traffic evidence. | 输入交通环境客观证据。 |
| Attention update / 注意力更新 | \(\dot{u}=(-u+S(o^2))/\tau\) | Accumulates execution commitment only when opinion magnitude is large. | 只有意见强度足够大时才积累执行信心。 |

## 1.2 High-Level Bias from Gap Confidence / 高层偏置的计算

**English.**  
At each step, Main13 first identifies the three target-lane vehicles closest to the ego vehicle in longitudinal front-axle coordinate. Let these vehicles be sorted from front to rear as \((i_1,i_2,i_3)\). The forward candidate gap is \((i_1,i_2)\), and the rear candidate gap is \((i_2,i_3)\). For any gap formed by a front vehicle \(F\) and a rear vehicle \(R\), define the gap center and gap velocity as
\[
x_g=\frac{p_F^x+p_R^x}{2},\qquad
v_g=\frac{v_F^x+v_R^x}{2}.
\]
The relative alignment between ego and the gap is
\[
d_g=x_g-p_e^x,\qquad
\Delta v_g=v_e^x-v_g.
\]
Main13 assigns a Gaussian confidence to the gap:
\[
C_g=
\exp\left(
-\frac{d_g^2}{2\sigma_d^2}
-\frac{\Delta v_g^2}{2\sigma_v^2}
\right).
\]
The latest Main13 code uses \(\sigma_d=4.0\) and \(\sigma_v=2.5\). No hard constraint forces \(C_g=0\) when the gap center is behind the ego vehicle; the confidence remains a smooth function of relative position and speed. This is useful for comparing gap candidates continuously and avoiding discontinuities in \(C_f\), \(C_r\), and the high-level opinion.

**中文。**  
每一步中，Main13 首先按照前轴纵向坐标选出距离 ego 最近的三辆目标车道车辆，并按从前到后的顺序记为 \((i_1,i_2,i_3)\)。前方候选 gap 为 \((i_1,i_2)\)，后方候选 gap 为 \((i_2,i_3)\)。对任意由前车 \(F\) 和后车 \(R\) 构成的 gap，定义 gap 中心和平均速度为
\[
x_g=\frac{p_F^x+p_R^x}{2},\qquad
v_g=\frac{v_F^x+v_R^x}{2}.
\]
ego 与该 gap 的对齐误差为
\[
d_g=x_g-p_e^x,\qquad
\Delta v_g=v_e^x-v_g.
\]
Main13 使用高斯形式计算 gap 置信度：
\[
C_g=
\exp\left(
-\frac{d_g^2}{2\sigma_d^2}
-\frac{\Delta v_g^2}{2\sigma_v^2}
\right).
\]
当前最新代码中 \(\sigma_d=4.0\)，\(\sigma_v=2.5\)。最新版本已经删除了“如果 gap 中心在 ego 后方就强制置信度为 0”的硬约束，因此置信度始终是相对位置和相对速度的平滑函数。这有利于连续比较候选 gap，减少 \(C_f\)、\(C_r\) 和高层意见的跳变。

**English.**  
The forward confidence \(C_f\) and rear confidence \(C_r\) are converted into the high-level directional bias through a signed difference:
\[
B=C_f-C_r.
\]
This choice is deliberately simple. A positive value means the forward gap is better aligned with the ego vehicle than the rear gap; a negative value means the rear gap is better aligned; a value near zero indicates ambiguity. The difference also preserves relative preference even when both confidences are small. For example, \(C_f=0.08\) and \(C_r=0.03\) are both low absolute scores, but the forward gap is still more plausible than the rear gap under the current local geometry. The opinion dynamics can use this small persistent difference as a seed. Because \(B\) enters the high-level opinion equation continuously, and because \(u_h\) self-increases when \(|y|\) becomes non-negligible, even a small but consistent advantage can be amplified into a stable decision. A direct hard maximum rule, such as selecting \(\arg\max C_g\) at every step, would react immediately but would also be more sensitive to noise and to changes in which three vehicles are currently closest.

**中文。**  
前 gap 置信度 \(C_f\) 和后 gap 置信度 \(C_r\) 通过有符号差值转换为高层方向偏置：
\[
B=C_f-C_r.
\]
这种设计刻意保持简单。若 \(B>0\)，说明前 gap 相比后 gap 更接近当前 ego 的相对位置与速度条件；若 \(B<0\)，说明后 gap 更优；若 \(B\approx0\)，说明两个候选 gap 难以区分。差值形式还能保留“相对偏好”，即使两个置信度本身都很小。例如 \(C_f=0.08\)、\(C_r=0.03\) 时，二者绝对值都不高，但前 gap 仍比后 gap 更合理。意见动力学可以把这种微小但持续存在的差异作为决策种子。由于 \(B\) 连续进入高层意见方程，而且当 \(|y|\) 变得不再很小时 \(u_h\) 会自增强，所以即使很小的优势也能被放大为稳定决策。相比之下，如果每一步直接使用 \(\arg\max C_g\) 的硬最大选择，反应会更快，但也更容易受到噪声和“最近三车集合切换”的影响。

| Case / 情况 | \(C_f\) | \(C_r\) | \(B=C_f-C_r\) | High-level tendency / 高层倾向 |
| :--- | ---: | ---: | ---: | :--- |
| Forward gap clearly better / 前 gap 明显更好 | 0.85 | 0.10 | 0.75 | FORWARD / 向前 gap |
| Rear gap clearly better / 后 gap 明显更好 | 0.12 | 0.80 | -0.68 | BACKWARD / 向后 gap |
| Both similar / 两者接近 | 0.35 | 0.34 | 0.01 | WAIT or slow accumulation / 等待或缓慢积累 |
| Both weak but different / 都较弱但有差异 | 0.08 | 0.03 | 0.05 | Small seed toward forward / 向前的微弱种子 |

## 1.3 High-Level Opinion Update and Decision Mapping / 高层意见更新与决策映射

**English.**  
The high-level opinion equation implemented in Main13 is
\[
\dot{y}=-d_y y+u_h\tanh(\alpha_y y)+B,
\]
with the current parameters
\[
d_y=2.5,\quad \alpha_y=10.0,\quad \theta_y=0.18.
\]
The high-level attention evolves as
\[
\dot{u}_h=\frac{-u_h+U_{\max}\frac{(y^2)^n}{K_h^n+(y^2)^n}}{\tau_h},
\]
where
\[
\tau_h=1.0,\quad U_{\max}=1.5,\quad K_h=0.2,\quad n=2.
\]
The decision rule is
\[
\text{decision}=
\begin{cases}
\text{FORWARD}, & y>\theta_y,\\
\text{BACKWARD}, & y<-\theta_y,\\
\text{WAIT}, & |y|\le \theta_y.
\end{cases}
\]
If the decision is FORWARD, the selected gap is the front pair \((i_1,i_2)\). If the decision is BACKWARD, the selected gap is the rear pair \((i_2,i_3)\). If the decision is WAIT, no target pair is passed to the low-level merge controller; the low-level opinion decays according to \(\dot{z}=-d_z z\), and the ego control input is not activated.

**中文。**  
Main13 中实现的高层意见方程为
\[
\dot{y}=-d_y y+u_h\tanh(\alpha_y y)+B,
\]
当前参数为
\[
d_y=2.5,\quad \alpha_y=10.0,\quad \theta_y=0.18.
\]
高层注意力更新为
\[
\dot{u}_h=\frac{-u_h+U_{\max}\frac{(y^2)^n}{K_h^n+(y^2)^n}}{\tau_h},
\]
其中
\[
\tau_h=1.0,\quad U_{\max}=1.5,\quad K_h=0.2,\quad n=2.
\]
高层决策映射为
\[
\text{decision}=
\begin{cases}
\text{FORWARD}, & y>\theta_y,\\
\text{BACKWARD}, & y<-\theta_y,\\
\text{WAIT}, & |y|\le \theta_y.
\end{cases}
\]
若决策为 FORWARD，底层选择前 pair \((i_1,i_2)\)；若决策为 BACKWARD，底层选择后 pair \((i_2,i_3)\)；若决策为 WAIT，则不向底层传递目标 pair，底层意见按 \(\dot{z}=-d_z z\) 衰减，同时 ego 实际控制不被激活。

**English.**  
This design has two important behavioral consequences. First, the decision is history-aware. The same instantaneous \(C_f-C_r\) may produce different actions depending on whether previous evidence has already built up \(y\) and \(u_h\). Second, the decision can be cautious near ambiguity but decisive under persistent evidence. When \(B\) fluctuates around zero, the damping term dominates and \(y\) remains inside the waiting band. When \(B\) stays positive or negative for several integration steps, \(y\) gradually exits the band and activates the selected gap. In this sense, high-level opinion dynamics acts as a continuous temporal filter on gap preference. It is not merely a smoother after a discrete decision; it is itself the decision mechanism.

**中文。**  
这种设计会带来两个重要行为特征。第一，决策具有历史记忆。同样的瞬时 \(C_f-C_r\) 值，可能因为之前的证据已经积累出不同的 \(y\) 和 \(u_h\) 而产生不同动作。第二，该机制在模糊区域谨慎，在持续证据下果断。当 \(B\) 围绕 0 波动时，阻尼项占主导，\(y\) 会保持在等待区间内；当 \(B\) 在多个积分步中持续为正或为负时，\(y\) 会逐渐离开等待区间并激活对应 gap。因此，高层意见动力学实际上是对 gap 偏好的连续时间滤波与决策机制，而不是离散决策之后的简单平滑器。

---

# 2. Low-Level Control System / 底层控制系统

**English.**  
The low-level controller receives the selected pair from the high-level module. If no pair is selected, the ego vehicle remains in a passive mode and continues with zero acceleration and zero steering-rate command. If a pair is selected, the low-level system evaluates whether the physical gap is safe and useful, computes a merge opinion \(z(t)\), places a target point inside the gap, and tracks that point. The low-level design is intentionally reusable: it is trained in the simpler Main12 single-gap environment and then reused in Main13 for each selected local pair. This is the central transfer hypothesis of the project: the local geometry of “ego relative to a front vehicle and a rear vehicle” is similar across single-gap and multi-gap environments, even though the high-level context is more complex in the latter.

**中文。**  
底层控制器接收高层选择的目标车辆 pair。如果高层没有选择 pair，ego 保持被动模式，实际输入为零加速度和零转角变化率；如果高层选择了 pair，底层就判断该物理 gap 是否安全可用，计算并道意见 \(z(t)\)，在 gap 内放置目标控制点，并控制车辆跟踪该点。底层设计的一个关键特点是可复用：它在更简单的 Main12 单 gap 环境中训练，然后在 Main13 中对每一个被高层选中的局部 pair 复用。这也是本项目的核心迁移假设：虽然多车环境高层更复杂，但“ego 相对前车和后车”的局部几何结构在单 gap 与多 gap 环境中是相似的。

## 2.1 Low-Level Bias \(b(t)\): Gap Size and Gap-Rate Evaluation / 底层 \(b(t)\)：gap 大小与变化速率评价

**English.**  
For a selected pair consisting of a front vehicle \(F\) and a rear vehicle \(R\), the physical gap and its longitudinal rate are
\[
g(t)=p_F^x-p_R^x,
\qquad
\dot{g}(t)=v_F^x-v_R^x.
\]
The low-level objective bias is
\[
b(t)=\tanh\left(k_g(g(t)-g_{\mathrm{safe}})+k_v\dot{g}(t)\right).
\]
In the current Main13 configuration,
\[
g_{\mathrm{safe}}=5.0\,\mathrm{m},\qquad
k_g=0.2,\qquad
k_v=0.1.
\]
The hyperbolic tangent bounds the bias to \((-1,1)\). A large positive value supports merging into the selected gap. A negative value suppresses the merge opinion. The gap-size term \(g-g_{\mathrm{safe}}\) says whether the gap is larger than the minimum acceptable clearance, while the gap-rate term \(\dot{g}\) describes whether the gap is opening or closing. This is more informative than gap size alone, because a currently large gap can become dangerous if the rear vehicle is closing fast, and a currently marginal gap can become feasible if it is opening.

**中文。**  
对于由前车 \(F\) 和后车 \(R\) 构成的所选 pair，物理 gap 与其纵向变化率定义为
\[
g(t)=p_F^x-p_R^x,
\qquad
\dot{g}(t)=v_F^x-v_R^x.
\]
底层客观偏置为
\[
b(t)=\tanh\left(k_g(g(t)-g_{\mathrm{safe}})+k_v\dot{g}(t)\right).
\]
当前 Main13 参数为
\[
g_{\mathrm{safe}}=5.0\,\mathrm{m},\qquad
k_g=0.2,\qquad
k_v=0.1.
\]
双曲正切函数把偏置限制在 \((-1,1)\)。较大的正值支持并入当前 gap，负值抑制并道意见。gap 大小项 \(g-g_{\mathrm{safe}}\) 表示空隙是否超过最低安全阈值，gap 变化率项 \(\dot{g}\) 表示空隙正在变大还是变小。相比只看 gap 大小，这种评价更合理，因为当前较大的 gap 若正在快速闭合仍可能危险，而当前略小的 gap 若正在打开则可能很快变得可行。

| \(g-g_{\mathrm{safe}}\) | \(\dot{g}\) | Expected \(b(t)\) | English interpretation | 中文解释 |
| :---: | :---: | :---: | :--- | :--- |
| Positive | Positive | Strong positive | Gap is large and opening; merging is encouraged. | gap 足够大且继续变大，强烈鼓励并入。 |
| Positive | Negative | Mild or uncertain | Gap is large now but closing; caution is required. | 当前 gap 大但正在闭合，需要谨慎。 |
| Negative | Positive | Mild or uncertain | Gap is small but improving; wait for more evidence. | 当前 gap 小但正在变大，可继续观察。 |
| Negative | Negative | Strong negative | Gap is small and closing; merging should be suppressed. | gap 小且继续变小，应抑制并道。 |
| Near zero | Near zero | Near zero | Environment gives no strong low-level evidence. | 环境没有明显低层偏置。 |

## 2.2 Why Use Reinforcement Learning for Attention \(u(t)\)? / 为什么用强化学习决定注意力 \(u(t)\)

**English.**  
The analytic attention formula used as a baseline is
\[
u_{\mathrm{RBF}}(t)
=u_{\mathrm{base}}
+u_{\mathrm{amp}}
\exp\left(
-\frac{d_g^2}{2\sigma_d^2}
-\frac{\Delta v_g^2}{2\sigma_v^2}
\right),
\]
where
\[
d_g=x_g-p_e^x,\qquad
\Delta v_g=v_e^x-v_g.
\]
In Main13, the default analytic parameters are \(u_{\mathrm{base}}=0.2\), \(u_{\mathrm{amp}}=2.5\), \(\sigma_d=4.0\), and \(\sigma_v=2.5\). The intuition is clear: when ego is close to the gap center and speed-matched with the gap, attention should be high; otherwise it should fall toward the base level. However, attention is not only a static alignment score. It also determines how aggressively the opinion dynamics self-reinforces. If \(u(t)\) is too low, the vehicle may remain passive even when a feasible gap exists. If \(u(t)\) is too high, the vehicle may commit too early, overshoot, or fight the safety term. A hand-designed Gaussian formula cannot easily account for all interactions among transient gap changes, lateral progress, collision penalties, and smoothness requirements.

**中文。**  
作为基准的手工注意力公式为
\[
u_{\mathrm{RBF}}(t)
=u_{\mathrm{base}}
+u_{\mathrm{amp}}
\exp\left(
-\frac{d_g^2}{2\sigma_d^2}
-\frac{\Delta v_g^2}{2\sigma_v^2}
\right),
\]
其中
\[
d_g=x_g-p_e^x,\qquad
\Delta v_g=v_e^x-v_g.
\]
Main13 当前解析公式参数为 \(u_{\mathrm{base}}=0.2\)、\(u_{\mathrm{amp}}=2.5\)、\(\sigma_d=4.0\)、\(\sigma_v=2.5\)。直观含义很清楚：ego 越接近 gap 中心且速度越匹配，注意力越高；否则注意力回落到基础值。然而注意力不只是一个静态对齐评分，它直接决定意见动力学自强化的强弱。如果 \(u(t)\) 过低，车辆可能在有机会时仍保持原地不动；如果 \(u(t)\) 过高，车辆可能过早承诺、产生超调，或与避障项发生较强冲突。手工高斯公式很难同时处理 gap 瞬态变化、横向进度、碰撞惩罚和平滑性约束之间的耦合。

**English.**  
For this reason, the current framework allows \(u(t)\) to be generated by a Soft Actor-Critic policy trained in Main12. The reinforcement-learning policy does not replace the physical controller. It only replaces the scalar attention input to the opinion dynamics. This is a conservative integration point: the learned component is low-dimensional and interpretable, while the vehicle motion still follows the analytic opinion-control structure. The learned policy can discover when to increase attention earlier than the RBF formula, when to remain cautious despite good instantaneous alignment, and how to avoid action oscillations that reduce reward. Since the state is expressed only by local relative positions and velocities to the front and rear gap vehicles, the same policy can be used in a multi-car scene once the high-level module selects a local pair.

**中文。**  
因此，当前框架允许用 Main12 中训练得到的 Soft Actor-Critic 策略生成 \(u(t)\)。强化学习策略并不替代物理控制器，而只是替代意见动力学中的一个标量注意力输入。这是一个相对保守的集成点：学习部件低维且可解释，车辆运动仍遵循解析的意见动力学控制结构。学习策略可以学到何时应该比 RBF 公式更早提高注意力，何时虽然瞬时对齐较好但仍应谨慎，以及如何减少会降低 reward 的动作振荡。由于 state 只由 ego 相对 gap 前后车的局部位置和速度构成，一旦高层在多车场景中选出一个局部 pair，同一个策略就可以迁移使用。

## 2.3 Soft Actor-Critic Reinforcement Learning / SAC 强化学习简介

**English.**  
Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm that maximizes both expected return and policy entropy. The entropy term encourages exploration and prevents the policy from collapsing too early to a narrow deterministic behavior. In simplified form, SAC learns a stochastic policy \(\pi_\theta(a|s)\) and two action-value critics \(Q_{\phi_1}(s,a)\), \(Q_{\phi_2}(s,a)\). The twin critics reduce overestimation bias by using
\[
Q_{\min}(s,a)=\min(Q_{\phi_1}(s,a),Q_{\phi_2}(s,a)).
\]
The policy objective can be written conceptually as
\[
J_\pi(\theta)
=\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_\theta}
\left[
\alpha_{\mathrm{ent}}\log\pi_\theta(a|s)-Q_{\min}(s,a)
\right],
\]
where \(\alpha_{\mathrm{ent}}\) controls the entropy reward. The critic target is
\[
y_Q=r+\gamma(1-d)
\left[
\min_i Q_{\bar{\phi}_i}(s',a')
-\alpha_{\mathrm{ent}}\log\pi_\theta(a'|s')
\right].
\]
In the implementation, the actor outputs a Gaussian action distribution, the sampled action is squashed by \(\tanh\), and the resulting normalized action lies in \([-1,1]\).

**中文。**  
Soft Actor-Critic（SAC）是一种离策略 actor-critic 强化学习算法，它同时最大化期望回报和策略熵。熵项鼓励探索，防止策略过早收缩为单一确定性动作。简化来说，SAC 学习随机策略 \(\pi_\theta(a|s)\) 和两个动作价值网络 \(Q_{\phi_1}(s,a)\)、\(Q_{\phi_2}(s,a)\)。双 critic 通过
\[
Q_{\min}(s,a)=\min(Q_{\phi_1}(s,a),Q_{\phi_2}(s,a))
\]
降低价值高估。策略目标可概念化写为
\[
J_\pi(\theta)
=\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_\theta}
\left[
\alpha_{\mathrm{ent}}\log\pi_\theta(a|s)-Q_{\min}(s,a)
\right],
\]
其中 \(\alpha_{\mathrm{ent}}\) 控制熵奖励强度。critic 的目标值为
\[
y_Q=r+\gamma(1-d)
\left[
\min_i Q_{\bar{\phi}_i}(s',a')
-\alpha_{\mathrm{ent}}\log\pi_\theta(a'|s')
\right].
\]
在代码实现中，actor 输出高斯动作分布，采样动作经过 \(\tanh\) 压缩，最终归一化动作范围为 \([-1,1]\)。

### 2.3.1 State and Policy Output / State 选择与 Policy 输出

**English.**  
The low-level SAC state is an eight-dimensional local relative state:
\[
s_t=
\left[
\mathbf{p}_e-\mathbf{p}_F,\,
\mathbf{v}_e^f-\mathbf{v}_F^f,\,
\mathbf{p}_e-\mathbf{p}_R,\,
\mathbf{v}_e^f-\mathbf{v}_R^f
\right].
\]
Expanded component-wise,
\[
s_t=[
\Delta x_{eF},\Delta y_{eF},
\Delta v^x_{eF},\Delta v^y_{eF},
\Delta x_{eR},\Delta y_{eR},
\Delta v^x_{eR},\Delta v^y_{eR}
].
\]
The implementation normalizes this vector by
\[
[40,8,20,10,40,8,20,10],
\]
and clips the normalized observation to \([-5,5]\). The policy output is a single scalar normalized action \(a_t^{RL}\in[-1,1]\). It is mapped to physical attention by
\[
u(t)=u_{\min}+\frac{a_t^{RL}+1}{2}(u_{\max}-u_{\min}),
\]
with \(u_{\min}=0.0\) and \(u_{\max}=3.0\). Thus the neural network does not output acceleration or steering; it only chooses how strongly the low-level opinion should self-reinforce.

**中文。**  
底层 SAC 的 state 是八维局部相对状态：
\[
s_t=
\left[
\mathbf{p}_e-\mathbf{p}_F,\,
\mathbf{v}_e^f-\mathbf{v}_F^f,\,
\mathbf{p}_e-\mathbf{p}_R,\,
\mathbf{v}_e^f-\mathbf{v}_R^f
\right].
\]
展开为
\[
s_t=[
\Delta x_{eF},\Delta y_{eF},
\Delta v^x_{eF},\Delta v^y_{eF},
\Delta x_{eR},\Delta y_{eR},
\Delta v^x_{eR},\Delta v^y_{eR}
].
\]
实现中用
\[
[40,8,20,10,40,8,20,10]
\]
对该向量归一化，并把归一化观测裁剪到 \([-5,5]\)。policy 输出一个标量归一化动作 \(a_t^{RL}\in[-1,1]\)，再映射为真实注意力：
\[
u(t)=u_{\min}+\frac{a_t^{RL}+1}{2}(u_{\max}-u_{\min}),
\]
其中 \(u_{\min}=0.0\)、\(u_{\max}=3.0\)。因此神经网络并不直接输出加速度或转角，而只决定底层意见应以多强的注意力进行自强化。

### 2.3.2 Reward Design / Reward 设计

**English.**  
The reward is designed to avoid the undesirable behavior where the ego vehicle stays in the original lane with nearly constant score. It rewards rapid, monotonic lane-change progress, penalizes hesitation when an opportunity exists, discourages oscillatory lateral motion, and strongly penalizes collision. Define the lane progress
\[
P_t=\mathrm{clip}\left(\frac{y_e(t)-y_O}{y_T-y_O},0,1\right),
\]
and the progress increment
\[
\Delta P_t=P_t-P_{t-1}.
\]
Let \(O_t\in\{0,1\}\) indicate a merge opportunity, \(d_{\min}\) be the minimum ego-to-target distance, \(m_s=2.5r\) be the continuous safety margin, \(I_c\) be the collision indicator, and \(I_s\) be the success indicator. The reward used by the Main12 SAC training and inherited by the Main12 evaluation workflow is
\[
\begin{aligned}
R_t=&
80.0\Delta P_t
+0.15P_t
+40.0O_t\max(\Delta P_t,0)\\
&-30.0\max(-\Delta P_t,0)
-0.25O_t(1-P_t)
-0.05(1-P_t)\\
&-0.5\lVert a_t-a_{t-1}\rVert^2
-2.0\mathbf{1}[v_y(t)v_y(t-1)<0]\\
&-20.0\max\left(0,\frac{m_s-d_{\min}}{m_s}\right)^2
-1000\mathbf{1}[I_c]\\
&+(100-2t)\mathbf{1}[I_s].
\end{aligned}
\]
In Main12, success is defined by \(P_t>0.95\), \(|y_e-y_T|<0.2\,\mathrm{m}\), and \(d_{\min}>1.5r\). Collision is triggered when the ego vehicle is closer than \(r\) to a target vehicle; the single-gap environment also monitors the front-rear target gap to avoid hidden target-vehicle overlap. In the Main13 random-test script, the same reward structure is adapted to the multi-car environment by using the best physical gap as the opportunity signal and by measuring \(d_{\min}\) over all target-lane vehicles.

**中文。**  
reward 的设计目标是避免 ego 车停留在原车道却仍获得接近固定高分的坏策略。它奖励快速且单调的横向并道进展，在有机会时惩罚犹豫，抑制横向反复换向，并对碰撞施加强惩罚。定义并道进度
\[
P_t=\mathrm{clip}\left(\frac{y_e(t)-y_O}{y_T-y_O},0,1\right),
\]
以及进度增量
\[
\Delta P_t=P_t-P_{t-1}.
\]
令 \(O_t\in\{0,1\}\) 表示是否存在并道机会，\(d_{\min}\) 为 ego 到目标车的最小距离，\(m_s=2.5r\) 为连续安全距离边界，\(I_c\) 为碰撞指示变量，\(I_s\) 为成功指示变量。Main12 SAC 训练及 Main12 评价流程采用的 reward 为
\[
\begin{aligned}
R_t=&
80.0\Delta P_t
+0.15P_t
+40.0O_t\max(\Delta P_t,0)\\
&-30.0\max(-\Delta P_t,0)
-0.25O_t(1-P_t)
-0.05(1-P_t)\\
&-0.5\lVert a_t-a_{t-1}\rVert^2
-2.0\mathbf{1}[v_y(t)v_y(t-1)<0]\\
&-20.0\max\left(0,\frac{m_s-d_{\min}}{m_s}\right)^2
-1000\mathbf{1}[I_c]\\
&+(100-2t)\mathbf{1}[I_s].
\end{aligned}
\]
Main12 中成功条件为 \(P_t>0.95\)、\(|y_e-y_T|<0.2\,\mathrm{m}\)、且 \(d_{\min}>1.5r\)。当 ego 与目标车辆距离小于 \(r\) 时触发碰撞；单 gap 环境还额外监测前后目标车之间的 gap，避免目标车之间出现隐藏重叠。Main13 random-test 脚本则把同样的 reward 结构扩展到多车环境中：用当前最大物理 gap 判断 opportunity，并在所有目标车中计算 \(d_{\min}\)。

| Reward term / 奖励项 | Formula / 公式 | English purpose | 中文含义 |
| :--- | :--- | :--- | :--- |
| Progress / 进度 | \(80\Delta P_t\) | Strongly rewards motion toward target lane. | 强烈奖励朝目标车道推进。 |
| Lane presence / 当前进度 | \(0.15P_t\) | Gives small dense reward for being closer to target lane. | 已接近目标车道时给小的密集正反馈。 |
| Opportunity progress / 有机会推进 | \(40O_t\max(\Delta P_t,0)\) | Encourages moving when a gap is available. | 有机会时继续并入给予额外奖励。 |
| Reverse progress / 后退惩罚 | \(-30\max(-\Delta P_t,0)\) | Penalizes moving back to original lane. | 惩罚从目标车道方向退回。 |
| Hesitation / 犹豫 | \(-0.25O_t(1-P_t)\) | Penalizes waiting while opportunity exists. | 有机会却不并道时持续扣分。 |
| Time / 时间 | \(-0.05(1-P_t)\) | Encourages earlier completion. | 鼓励更早完成。 |
| Smooth action / 动作平滑 | \(-0.5\lVert a_t-a_{t-1}\rVert^2\) | Reduces attention jumps. | 抑制注意力输出跳变。 |
| Direction flip / 横向反复 | \(-2\mathbf{1}[v_y(t)v_y(t-1)<0]\) | Discourages lateral oscillation. | 惩罚横向速度反复换向。 |
| Safety / 连续安全 | \(-20\max(0,(m_s-d_{\min})/m_s)^2\) | Penalizes near-collision before impact. | 接近危险距离时提前扣分。 |
| Collision / 碰撞 | \(-1000\mathbf{1}[I_c]\) | Makes collision catastrophic. | 碰撞大幅扣分。 |
| Success / 成功 | \((100-2t)\mathbf{1}[I_s]\) | Rewards successful and fast merge. | 成功并道奖励，越快越高。 |

### 2.3.3 Low-Level Opinion \(z(t)\) and Target Point / 底层意见 \(z(t)\) 及其对控制点的影响

**English.**  
After \(b(t)\) and \(u(t)\) are obtained, the low-level opinion is updated by
\[
\dot{z}=-d_z z+u(t)\tanh(\alpha_z z)+b(t),
\]
with
\[
d_z=2.0,\qquad \alpha_z=2.0.
\]
The opinion \(z\) is then converted into a smooth interpolation variable
\[
w=\tanh(k_w z),
\]
where \(k_w=40.0\) in the ego model. When \(z\) is near zero, \(w\approx0\); when \(z\) becomes sufficiently positive, \(w\approx1\). In Main13, the selected gap center is
\[
\mathbf{p}_g=\frac{\mathbf{p}_F+\mathbf{p}_R}{2},
\qquad
\mathbf{v}_g=\frac{\mathbf{v}_F^f+\mathbf{v}_R^f}{2}.
\]
The desired target point is
\[
\mathbf{p}^{\star}
=\mathbf{p}_g+\boldsymbol{\eta}(1-w)r_{\eta},
\qquad
\boldsymbol{\eta}=[0,1]^\top,\quad r_{\eta}=-4.0.
\]
Since the target lane center is \(6\,\mathrm{m}\) and the original lane center is \(2\,\mathrm{m}\), the offset \(r_{\eta}=-4.0\) has a direct interpretation. When \(w=0\), the target point is shifted from the target-lane gap center back toward the original lane center, so the ego does not immediately cut into the lane. When \(w=1\), the offset disappears and the target point becomes the actual selected gap center. Thus \(z(t)\) controls how strongly the ego vehicle commits laterally into the target gap.

**中文。**  
得到 \(b(t)\) 和 \(u(t)\) 后，底层意见按以下公式更新：
\[
\dot{z}=-d_z z+u(t)\tanh(\alpha_z z)+b(t),
\]
其中
\[
d_z=2.0,\qquad \alpha_z=2.0.
\]
意见 \(z\) 随后被转换为平滑插值变量
\[
w=\tanh(k_w z),
\]
ego 模型中 \(k_w=40.0\)。当 \(z\) 接近 0 时，\(w\approx0\)；当 \(z\) 足够大且为正时，\(w\approx1\)。在 Main13 中，所选 gap 的中心与平均速度为
\[
\mathbf{p}_g=\frac{\mathbf{p}_F+\mathbf{p}_R}{2},
\qquad
\mathbf{v}_g=\frac{\mathbf{v}_F^f+\mathbf{v}_R^f}{2}.
\]
期望目标点为
\[
\mathbf{p}^{\star}
=\mathbf{p}_g+\boldsymbol{\eta}(1-w)r_{\eta},
\qquad
\boldsymbol{\eta}=[0,1]^\top,\quad r_{\eta}=-4.0.
\]
由于目标车道中心为 \(6\,\mathrm{m}\)，原车道中心为 \(2\,\mathrm{m}\)，偏移量 \(r_{\eta}=-4.0\) 有很直接的含义。当 \(w=0\) 时，目标点从目标车道 gap 中心向原车道中心偏移，ego 不会立刻切入目标车道；当 \(w=1\) 时，偏移消失，目标点就是所选 gap 中心。因此，\(z(t)\) 控制 ego 横向并入目标 gap 的承诺强度。

---

# 3. Actual Control Input / 实际控制输入

## 3.1 Safety-Avoidance Term \(u_c\) / 避障项 \(u_c\) 设计

**English.**  
The low-level tracking control is augmented by a safety-avoidance term. For each target vehicle \(j\), define
\[
\mathbf{e}_{ej}=\mathbf{p}_e-\mathbf{p}_j,\qquad
d_j=\lVert \mathbf{e}_{ej}\rVert-r,
\qquad
\mathbf{g}_{ej}=\frac{\mathbf{e}_{ej}}{\lVert \mathbf{e}_{ej}\rVert}.
\]
The closing-rate-like scalar is
\[
\phi_j=\frac{\mathbf{g}_{ej}^\top(\mathbf{v}_e^f-\mathbf{v}_j^f)}{d_j}.
\]
The total avoidance control is
\[
\mathbf{u}_c=\sum_j -k_o\mathbf{g}_{ej}\phi_j.
\]
In Main13 this sum is evaluated over all target-lane vehicles when `ENABLE_GLOBAL_UC=True`. The term becomes stronger when the ego vehicle is close to another vehicle and moving in a direction that reduces clearance. Because \(d_j\) appears in the denominator, the repulsive effect increases rapidly near the collision boundary. In the implementation, a signed numerical safeguard prevents division by zero.

**中文。**  
底层跟踪控制会叠加安全避障项。对于每一辆目标车 \(j\)，定义
\[
\mathbf{e}_{ej}=\mathbf{p}_e-\mathbf{p}_j,\qquad
d_j=\lVert \mathbf{e}_{ej}\rVert-r,
\qquad
\mathbf{g}_{ej}=\frac{\mathbf{e}_{ej}}{\lVert \mathbf{e}_{ej}\rVert}.
\]
类似接近率的标量为
\[
\phi_j=\frac{\mathbf{g}_{ej}^\top(\mathbf{v}_e^f-\mathbf{v}_j^f)}{d_j}.
\]
总避障控制为
\[
\mathbf{u}_c=\sum_j -k_o\mathbf{g}_{ej}\phi_j.
\]
当 `ENABLE_GLOBAL_UC=True` 时，Main13 会对所有目标车道车辆求和。该项在 ego 与其他车辆距离较近且相对运动正在缩小间距时变强。由于 \(d_j\) 出现在分母中，靠近碰撞边界时避障作用会迅速增强。代码中使用带符号的小量保护避免除零。

## 3.2 Target Point and Tracking Error \(e_z\) / 总目标点与控制误差设计

**English.**  
The selected target point is computed from the chosen gap center and the low-level opinion-dependent lateral offset:
\[
\mathbf{p}^{\star}(t)
=\frac{\mathbf{p}_F(t)+\mathbf{p}_R(t)}{2}
+\boldsymbol{\eta}(1-\tanh(k_wz(t)))r_{\eta}.
\]
The target velocity is the average velocity of the two vehicles forming the selected gap:
\[
\mathbf{v}^{\star}(t)
=\frac{\mathbf{v}_F^f(t)+\mathbf{v}_R^f(t)}{2}.
\]
The tracking errors are
\[
\mathbf{e}_p=\mathbf{p}_e-\mathbf{p}^{\star},\qquad
\mathbf{e}_v=\mathbf{v}_e^f-\mathbf{v}^{\star}.
\]
The nominal desired front-axle acceleration is
\[
\mathbf{u}_n=-k_p\mathbf{e}_p-k_v\mathbf{e}_v,
\]
where the current ego model uses \(k_p=0.7\) and \(k_v=2.0\). The total desired front-axle acceleration is
\[
\mathbf{u}_{total}=\mathbf{u}_n+\mathbf{u}_c.
\]
If high-level decision is WAIT, the target point is not used for actual motion and the physical input remains zero. If FORWARD or BACKWARD is active, the selected pair determines \(\mathbf{p}^{\star}\) and \(\mathbf{v}^{\star}\), and the ego vehicle tracks the moving gap.

**中文。**  
所选目标点由被选中 gap 的中心和底层意见决定的横向偏移共同构成：
\[
\mathbf{p}^{\star}(t)
=\frac{\mathbf{p}_F(t)+\mathbf{p}_R(t)}{2}
+\boldsymbol{\eta}(1-\tanh(k_wz(t)))r_{\eta}.
\]
目标速度为构成该 gap 的两车前轴速度平均值：
\[
\mathbf{v}^{\star}(t)
=\frac{\mathbf{v}_F^f(t)+\mathbf{v}_R^f(t)}{2}.
\]
跟踪误差定义为
\[
\mathbf{e}_p=\mathbf{p}_e-\mathbf{p}^{\star},\qquad
\mathbf{e}_v=\mathbf{v}_e^f-\mathbf{v}^{\star}.
\]
名义前轴期望加速度为
\[
\mathbf{u}_n=-k_p\mathbf{e}_p-k_v\mathbf{e}_v,
\]
当前 ego 模型中 \(k_p=0.7\)、\(k_v=2.0\)。总期望前轴加速度为
\[
\mathbf{u}_{total}=\mathbf{u}_n+\mathbf{u}_c.
\]
若高层决策为 WAIT，目标点不参与实际控制，车辆物理输入保持为零；若高层为 FORWARD 或 BACKWARD，则所选 pair 决定 \(\mathbf{p}^{\star}\) 和 \(\mathbf{v}^{\star}\)，ego 开始跟踪该运动 gap。

## 3.3 Final PID-Like Physical Input Design / 最终物理控制输入设计

**English.**  
The bicycle model evolves as
\[
\dot{x}=v\cos\theta,\quad
\dot{y}=v\sin\theta,\quad
\dot{\theta}=\frac{v}{L}\tan\delta,\quad
\dot{v}=a,\quad
\dot{\delta}=\omega.
\]
The controller first computes the desired front-axle acceleration \(\mathbf{u}_{total}\). Because the physical inputs are acceleration \(a\) and steering-rate \(\omega\), the implementation solves the local front-axle acceleration relationship
\[
\ddot{\mathbf{p}}_e
=A(\theta,v,\delta)
\begin{bmatrix}
a\\
\omega
\end{bmatrix}
\mathbf{b}(\theta,v,\delta)
\]
for \([a,\omega]^\top\). The matrix \(A\) and vector \(\mathbf{b}\) come from differentiating the front-axle velocity. In compact form,
\[
\begin{bmatrix}
a\\
\omega
\end{bmatrix}
=A^{-1}(\mathbf{u}_{total}-\mathbf{b}).
\]
In Main13, the final commands are clipped to
\[
a\in[-5.0,5.0],\qquad
\omega\in[-0.8,0.8].
\]
The control law is PID-like because it uses proportional position error and derivative velocity error, but it is implemented in the front-axle acceleration space before being converted into bicycle-model inputs. This preserves compatibility with the paper-style point-mass control intuition while still simulating a nonholonomic vehicle.

**中文。**  
自行车模型动力学为
\[
\dot{x}=v\cos\theta,\quad
\dot{y}=v\sin\theta,\quad
\dot{\theta}=\frac{v}{L}\tan\delta,\quad
\dot{v}=a,\quad
\dot{\delta}=\omega.
\]
控制器首先计算期望前轴加速度 \(\mathbf{u}_{total}\)。由于真实物理输入是纵向加速度 \(a\) 和转角变化率 \(\omega\)，代码通过前轴加速度关系
\[
\ddot{\mathbf{p}}_e
=A(\theta,v,\delta)
\begin{bmatrix}
a\\
\omega
\end{bmatrix}
\mathbf{b}(\theta,v,\delta)
\]
反解 \([a,\omega]^\top\)。矩阵 \(A\) 和向量 \(\mathbf{b}\) 来自对前轴速度求导。紧凑写法为
\[
\begin{bmatrix}
a\\
\omega
\end{bmatrix}
=A^{-1}(\mathbf{u}_{total}-\mathbf{b}).
\]
Main13 中最终命令被裁剪为
\[
a\in[-5.0,5.0],\qquad
\omega\in[-0.8,0.8].
\]
该控制律类似 PID，因为它包含位置误差比例项和速度误差微分项，但实际是在前轴加速度空间中构造，再转换为自行车模型输入。这样既保留了论文中点质量控制器的直观形式，又能模拟非完整约束车辆。

---

# 4. Experimental Design / 实验设计

## 4.1 Single-Gap Merging Experiment: Main12 / 单车 gap 并入实验：Main12

**English.**  
The Main12 environment is the training and controlled-evaluation environment for the low-level attention policy. It contains two target-lane vehicles and one ego vehicle. The target lane center is \(y_T=6\,\mathrm{m}\), the original lane center is \(y_O=2\,\mathrm{m}\), and all vehicles start with speed \(15\,\mathrm{m/s}\). The leading target vehicle is initialized at
\[
x_1(0)=30\,\mathrm{m},\quad y_1(0)=6\,\mathrm{m},
\]
and the rear target vehicle is initialized at
\[
x_2(0)=15\,\mathrm{m},\quad y_2(0)=6\,\mathrm{m}.
\]
The ego vehicle is initialized as
\[
x_e(0)=20+\xi,\qquad \xi\sim\mathcal{U}(-5,5),
\quad y_e(0)=2\,\mathrm{m}.
\]
The randomized ego position prevents the learned attention policy from overfitting to a single longitudinal alignment. The selected gap is always the gap between target vehicles 1 and 2; therefore Main12 does not require a high-level gap selector. It isolates the question: given a front-rear pair, can the low-level controller generate a safe and timely merge?

**中文。**  
Main12 是底层注意力策略的训练与受控评价环境。它包含两辆目标车道车辆和一辆 ego 车。目标车道中心为 \(y_T=6\,\mathrm{m}\)，原车道中心为 \(y_O=2\,\mathrm{m}\)，所有车辆初始速度均为 \(15\,\mathrm{m/s}\)。前目标车初始为
\[
x_1(0)=30\,\mathrm{m},\quad y_1(0)=6\,\mathrm{m},
\]
后目标车初始为
\[
x_2(0)=15\,\mathrm{m},\quad y_2(0)=6\,\mathrm{m}.
\]
ego 初始位置为
\[
x_e(0)=20+\xi,\qquad \xi\sim\mathcal{U}(-5,5),
\quad y_e(0)=2\,\mathrm{m}.
\]
随机 ego 初始纵向位置可以防止学习到的注意力策略只适应单一对齐状态。Main12 中候选 gap 始终固定为目标车 1 和目标车 2 之间的 gap，因此不需要高层 gap 选择器。它隔离出一个核心问题：给定一对前后车，底层控制器能否安全且及时地完成并入？

**English.**  
The rear target vehicle has a piecewise longitudinal motion pattern inherited from the Main7 gap-following scenario. Before the yielding stage, for \(t\le20\,\mathrm{s}\), its acceleration is sinusoidal:
\[
a_2(t)=A\omega\cos(\omega t),
\qquad
\omega=\frac{2\pi}{T_p},
\]
where \(A=4.0\) and \(T_p=6.0\,\mathrm{s}\). Assuming \(v_2(0)=v_0\) and \(x_2(0)=x_{20}\), the corresponding ideal longitudinal velocity and position are
\[
v_2(t)=v_0+A\sin(\omega t),
\]
\[
x_2(t)=x_{20}+v_0t+\frac{A}{\omega}(1-\cos(\omega t)).
\]
After \(t>20\,\mathrm{s}\), the rear vehicle switches to a gap-maintaining controller:
\[
a_2(t)=\mathrm{clip}
\left(
0.35(g(t)-20)-1.1(v_2^x(t)-v_1^x(t)),
-5.0,2.0
\right),
\]
where \(g(t)=p_1^x(t)-p_2^x(t)\). This controller makes the rear target vehicle slow down or accelerate so that the gap to the leader converges toward \(20\,\mathrm{m}\). The stage change at \(20\,\mathrm{s}\) creates a controlled opportunity for the ego vehicle to merge.

**中文。**  
后目标车沿用 Main7 gap-following 场景的分段纵向运动。在让行阶段之前，即 \(t\le20\,\mathrm{s}\) 时，其加速度为正弦形式：
\[
a_2(t)=A\omega\cos(\omega t),
\qquad
\omega=\frac{2\pi}{T_p},
\]
其中 \(A=4.0\)、\(T_p=6.0\,\mathrm{s}\)。若 \(v_2(0)=v_0\)、\(x_2(0)=x_{20}\)，则理想纵向速度和位置为
\[
v_2(t)=v_0+A\sin(\omega t),
\]
\[
x_2(t)=x_{20}+v_0t+\frac{A}{\omega}(1-\cos(\omega t)).
\]
当 \(t>20\,\mathrm{s}\) 后，后车切换为保持 gap 的控制器：
\[
a_2(t)=\mathrm{clip}
\left(
0.35(g(t)-20)-1.1(v_2^x(t)-v_1^x(t)),
-5.0,2.0
\right),
\]
其中 \(g(t)=p_1^x(t)-p_2^x(t)\)。该控制器会使后车减速或加速，让它与前车的间距收敛到 \(20\,\mathrm{m}\)。20 秒处的阶段切换为 ego 车创造了受控并道机会。

**English.**  
The single-gap controller uses only the low-level system. The low-level bias \(b(t)\) is computed from the gap \(g(t)\) and \(\dot{g}(t)\); the attention \(u(t)\) is either produced by the SAC policy or by the analytic RBF formula; the opinion \(z(t)\) is updated; the target point is computed; and the ego vehicle follows that point. The Main12 training objective is not to learn the entire vehicle dynamics, but to learn an attention schedule that makes the opinion dynamics commit at useful times while avoiding collision and oscillation.

**中文。**  
单 gap 控制只使用底层系统。底层偏置 \(b(t)\) 由 gap 大小 \(g(t)\) 与变化率 \(\dot{g}(t)\) 计算；注意力 \(u(t)\) 可由 SAC 策略输出，也可由解析 RBF 公式给出；随后更新 \(z(t)\)、计算目标点，并由 ego 车跟踪该点。Main12 的训练目标不是让神经网络学习完整车辆动力学，而是学习一个注意力调度规律，使意见动力学在合适时机产生并道承诺，同时避免碰撞和横向振荡。

**English.**  
The main ablation in the single-gap experiment compares the trained SAC attention against the untrained analytic RBF attention. The evaluation script runs both policies on the same set of randomized initial ego positions and scores them using the same reward function. The expected evidence is not only the mean reward but also the decomposed reward terms: progress reward should increase, hesitation penalty should decrease, collision rate should remain low, and action-smoothness penalties should not become excessive. If the learned policy improves total reward only by becoming more aggressive and increasing safety penalties, the improvement is not considered robust. A useful learned policy should increase successful progress while preserving clearance and smoothness.

**中文。**  
单 gap 实验中的主要消融对比是：训练后的 SAC 注意力策略 vs 未训练的解析 RBF 注意力公式。评价脚本在同一组随机 ego 初始位置上运行两种策略，并使用相同 reward 评分。需要观察的不只是平均 reward，还包括 reward 分项：progress reward 应该提高，hesitation penalty 应该降低，碰撞率应保持较低，动作平滑惩罚不应过大。如果学习策略只是通过过度激进来提高总分并导致安全惩罚增大，则这种提升并不可靠。有效的学习策略应在保持安全距离和平滑性的同时提高成功并道进度。

## 4.2 Multi-Car Gap Merging Experiment: Main13 / 多车 gap 并入实验：Main13

**English.**  
The Main13 experiment extends the single-gap task to a multi-car target lane. The current configuration uses five target-lane vehicles:
\[
N=5,\qquad x_i(0)=48-(i-1)g_0,
\qquad g_0=8.0\,\mathrm{m},
\]
with \(y_i(0)=6\,\mathrm{m}\) and \(v_i(0)=15\,\mathrm{m/s}\). The ego vehicle starts from the original lane with
\[
x_e(0)=30+\xi,\qquad \xi\sim\mathcal{U}(-5,5),
\quad y_e(0)=2\,\mathrm{m}.
\]
Thus both the local relation between ego and the target-lane fleet and the gap evolution are randomized. The simulation time is \(40\,\mathrm{s}\), and the integration step is \(0.05\,\mathrm{s}\). The low-level attention source can be switched between the Main12 SAC policy and the analytic RBF formula by the `USE_RL_U` parameter. In the latest Main13 configuration, `USE_RL_U=True`, so the system uses the trained Main12 attention policy by default.

**中文。**  
Main13 实验把单 gap 任务扩展到多车目标车道。当前配置使用五辆目标车：
\[
N=5,\qquad x_i(0)=48-(i-1)g_0,
\qquad g_0=8.0\,\mathrm{m},
\]
其中 \(y_i(0)=6\,\mathrm{m}\)、\(v_i(0)=15\,\mathrm{m/s}\)。ego 从原车道出发：
\[
x_e(0)=30+\xi,\qquad \xi\sim\mathcal{U}(-5,5),
\quad y_e(0)=2\,\mathrm{m}.
\]
因此，ego 与目标车队的局部关系和 gap 演化都会发生随机变化。仿真总时间为 \(40\,\mathrm{s}\)，积分步长为 \(0.05\,\mathrm{s}\)。底层注意力来源可通过 `USE_RL_U` 在 Main12 SAC 策略与解析 RBF 公式之间切换。最新 Main13 配置中 `USE_RL_U=True`，默认使用 Main12 训练出的注意力策略。

**English.**  
The multi-car gap dynamics are generated by a randomized desired-gap schedule. Every \(T_g=4.0\,\mathrm{s}\), each of the four target-lane gaps may keep the base desired gap or be assigned a scaled desired gap. The multiplier set is
\[
\mathcal{M}=\{0.75,1.0,1.25,1.5\},
\]
and at most two gaps are changed in one period. Therefore the desired gap for gap \(i\) during period \(k\) is
\[
g_{i,\mathrm{des}}^{(k)}
=g_0 m_i^{(k)},\qquad m_i^{(k)}\in\mathcal{M}.
\]
The leading target vehicle has zero acceleration. Each follower vehicle uses a clipped gap-tracking acceleration:
\[
a_{i+1}
=\mathrm{clip}
\left(
k_p^g(g_i-g_{i,\mathrm{des}})
+k_d^g\dot{g}_i,
-a_{\max}^g,a_{\max}^g
\right),
\]
where \(k_p^g=0.55\), \(k_d^g=1.05\), and \(a_{\max}^g=4.0\). This mechanism creates a target lane whose gaps expand, shrink, and reconfigure over time. The ego vehicle must therefore adapt both high-level gap selection and low-level execution.

**中文。**  
多车 gap 动态由随机期望 gap 日程生成。每 \(T_g=4.0\,\mathrm{s}\)，四个目标车道 gap 中最多两个会被重新指定期望间距，其余保持基础期望间距。倍率集合为
\[
\mathcal{M}=\{0.75,1.0,1.25,1.5\},
\]
因此第 \(k\) 个时间段中第 \(i\) 个 gap 的期望值为
\[
g_{i,\mathrm{des}}^{(k)}
=g_0 m_i^{(k)},\qquad m_i^{(k)}\in\mathcal{M}.
\]
目标车队最前车加速度为零。每辆跟随车使用裁剪后的 gap 跟踪加速度：
\[
a_{i+1}
=\mathrm{clip}
\left(
k_p^g(g_i-g_{i,\mathrm{des}})
+k_d^g\dot{g}_i,
-a_{\max}^g,a_{\max}^g
\right),
\]
其中 \(k_p^g=0.55\)、\(k_d^g=1.05\)、\(a_{\max}^g=4.0\)。该机制会产生不断扩大、缩小和重构的目标车道 gap，使 ego 必须同时适应高层 gap 选择和底层并道执行。

**English.**  
At each step, Main13 does not evaluate every possible gap with a global planner. Instead, it selects the nearest three target-lane vehicles by absolute longitudinal distance to the ego front axle and considers only the two gaps formed by them. This locality assumption reduces computational complexity and matches a practical driving intuition: the most relevant merge candidates are usually the gaps close to the ego vehicle. The high-level module then compares the two local gaps through \(C_f\), \(C_r\), and \(B=C_f-C_r\). This is different from a simple maximum-comparison baseline. A direct baseline may compute a score \(S_i\) for each candidate gap, such as \(S_i=b_i(t)\) or \(S_i=C_i\), and select
\[
i^\star=\arg\max_i S_i.
\]
Such a baseline is easy to implement, but it has no internal memory. If two gap scores cross repeatedly due to random gap changes or nearest-three switching, the selected gap may also switch repeatedly. The opinion-dynamics high level instead integrates evidence over time and creates a waiting band, so it should reduce decision chattering.

**中文。**  
每一步中，Main13 并不使用全局规划器评价所有可能 gap，而是按照 ego 前轴纵向距离选出最近三辆目标车道车辆，只考虑这三车形成的两个 gap。这种局部性假设降低了计算复杂度，也符合实际驾驶直觉：最相关的并入候选通常是 ego 附近的 gap。高层模块随后通过 \(C_f\)、\(C_r\) 和 \(B=C_f-C_r\) 比较这两个局部 gap。这不同于简单最大值比较基线。直接基线可以为每个候选 gap 计算评分 \(S_i\)，例如 \(S_i=b_i(t)\) 或 \(S_i=C_i\)，再选择
\[
i^\star=\arg\max_i S_i.
\]
这种基线实现简单，但没有内部记忆。如果两个 gap 的评分由于随机 gap 变化或最近三车集合切换而反复交叉，所选 gap 也会反复跳变。意见动力学高层则通过连续时间积分积累证据，并带有等待区间，因此应能减少决策抖动。

**English.**  
The proposed ablation for validating the high-level mechanism contains four policy variants. First, `Opinion-RL` uses high-level opinion dynamics for gap selection and the trained Main12 SAC policy for low-level attention. Second, `Opinion-RBF` uses high-level opinion dynamics but replaces the learned attention with the analytic RBF formula. Third, `Max-RL` removes the high-level opinion dynamics and directly selects the local gap with the larger instantaneous score, while still using the learned low-level attention. Fourth, `Max-RBF` combines direct maximum gap selection with analytic attention. These variants separate the contributions of the high-level decision mechanism and the low-level learned attention. The main hypothesis is that `Opinion-RL` should produce higher average reward, fewer decision switches, shorter successful merge time, and no increase in collision rate compared with direct maximum selection.

**中文。**  
用于验证高层机制的消融实验可包含四种策略。第一，`Opinion-RL` 使用高层意见动力学选择 gap，并使用 Main12 训练出的 SAC 策略作为底层注意力。第二，`Opinion-RBF` 保留高层意见动力学，但用解析 RBF 公式替代学习注意力。第三，`Max-RL` 移除高层意见动力学，直接选择瞬时评分更大的局部 gap，同时仍使用学习注意力。第四，`Max-RBF` 同时使用直接最大 gap 选择和解析注意力。这四种变体可以区分高层决策机制和底层学习注意力各自的贡献。主要假设是：相比直接最大值选择，`Opinion-RL` 应具有更高平均 reward、更少决策切换、更短成功并道时间，并且不增加碰撞率。

| Variant / 变体 | High-level gap selection | Low-level attention | Purpose / 目的 |
| :--- | :--- | :--- | :--- |
| Opinion-RL | Opinion dynamics \(y,u_h\) | SAC policy \(u(t)\) | Full proposed method / 完整方法 |
| Opinion-RBF | Opinion dynamics \(y,u_h\) | Analytic RBF \(u_{\mathrm{RBF}}\) | Isolate effect of RL attention / 隔离学习注意力作用 |
| Max-RL | Instantaneous \(\arg\max S_i\) | SAC policy \(u(t)\) | Isolate effect of high-level opinion dynamics / 隔离高层意见动力学作用 |
| Max-RBF | Instantaneous \(\arg\max S_i\) | Analytic RBF \(u_{\mathrm{RBF}}\) | Fully hand-designed baseline / 完全手工基线 |

**English.**  
The evaluation metrics should include both task-level and behavior-level quantities. Task-level metrics include average episode reward, success rate, collision rate, final lane-change progress, completion time, and minimum distance. Behavior-level metrics include high-level decision switch count, selected-gap dwell time, mean low-level attention, action smoothness, direction-flip count, and decomposed reward terms. The recently added `main13-random-test.py` script already supports repeated randomized trials, computes total reward using the same reward philosophy, and outputs summary plots including mean reward, standard deviation, success rate, collision rate, final progress, and mean reward-term contributions. To evaluate high-level effectiveness rigorously, the same random seeds should be used across all ablation variants so that each method faces the same ego initial position and gap schedule.

**中文。**  
评价指标应同时包含任务层指标和行为层指标。任务层指标包括平均 episode reward、成功率、碰撞率、最终并道进度、完成时间和最小距离；行为层指标包括高层决策切换次数、所选 gap 停留时间、平均底层注意力、动作平滑性、横向换向次数以及 reward 分项。新建的 `main13-random-test.py` 已支持多次随机试验，会按照相同 reward 思路计算总分，并输出平均 reward、标准差、成功率、碰撞率、最终进度和 reward 分项均值等图表。为了严格验证高层有效性，所有消融方法应使用相同随机种子，使不同方法面对相同 ego 初始位置和 gap 日程。

## 4.3 Current Parameter Summary / 当前参数总结

| Module / 模块 | Parameter / 参数 | Current value / 当前值 | Role / 作用 |
| :--- | :--- | ---: | :--- |
| Simulation | \(T\) | 40.0 s | Total simulation time / 总仿真时间 |
| Simulation | \(\Delta t\) | 0.05 s | Integration interval / 积分间隔 |
| Road | \(W\) | 4.0 m | Lane width / 车道宽度 |
| Vehicle | \(L\) | 2.8 m | Wheelbase / 轴距 |
| Safety | \(r\) | 1.5 m | Collision boundary / 碰撞边界 |
| Main13 fleet | \(N\) | 5 | Number of target-lane vehicles / 目标车道车辆数 |
| Main13 fleet | \(g_0\) | 8.0 m | Base desired gap / 基础期望 gap |
| Gap schedule | \(T_g\) | 4.0 s | Gap adjustment period / gap 调整周期 |
| Gap schedule | \(\mathcal{M}\) | 0.75, 1.0, 1.25, 1.5 | Gap multipliers / gap 倍率 |
| Ego randomization | \(x_e(0)\) | \(30\pm5\) m | Main13 ego initial x / Main13 ego 初始纵向位置 |
| High level | \(d_y\) | 2.5 | High opinion damping / 高层意见阻尼 |
| High level | \(\alpha_y\) | 10.0 | High opinion sensitivity / 高层意见灵敏度 |
| High level | \(U_{\max}\) | 1.5 | Maximum high attention / 高层最大注意力 |
| High level | \(K_h\) | 0.2 | Hill threshold / Hill 阈值 |
| High level | \(n\) | 2 | Hill exponent / Hill 指数 |
| High level | \(\theta_y\) | 0.18 | Decision threshold / 决策阈值 |
| Low level | \(g_{\mathrm{safe}}\) | 5.0 m | Minimum evaluated gap / 底层安全 gap 阈值 |
| Low level | \(k_g\) | 0.2 | Gap-size sensitivity / gap 大小灵敏度 |
| Low level | \(k_v\) | 0.1 | Gap-rate sensitivity / gap 速率灵敏度 |
| Low level | \(d_z\) | 2.0 | Low opinion damping / 底层意见阻尼 |
| Low level | \(\alpha_z\) | 2.0 | Low opinion sensitivity / 底层意见灵敏度 |
| RBF attention | \(u_{\mathrm{base}}\) | 0.2 | Minimum formula attention / 公式注意力基值 |
| RBF attention | \(u_{\mathrm{amp}}\) | 2.5 | Confidence amplitude / 置信度幅值 |
| RBF attention | \(\sigma_d\) | 4.0 | Position bandwidth / 位置尺度 |
| RBF attention | \(\sigma_v\) | 2.5 | Velocity bandwidth / 速度尺度 |
| RL attention | \(u_{\min},u_{\max}\) | 0.0, 3.0 | Learned attention bounds / 学习注意力范围 |
| Physical input | \(a\) | [-5.0, 5.0] | Acceleration clip / 加速度裁剪 |
| Physical input | \(\omega\) | [-0.8, 0.8] | Steering-rate clip / 转角变化率裁剪 |

## 4.4 Expected Results and Interpretation / 预期结果与解释

**English.**  
The expected result is not simply that the ego vehicle reaches the target lane. A scientifically convincing result should show that the controller reaches the target lane for the right reason: it should select a locally feasible gap, maintain sufficient distance, avoid rapid switching between candidate gaps, and complete the lane change with smooth lateral behavior. Therefore, the analysis should not rely on only one score. The reward curve is useful, but it should be interpreted together with success rate, collision rate, minimum distance, decision switching frequency, and decomposed reward terms. If the total reward is high but the safety penalty is also high, the method may be too aggressive. If the collision rate is zero but progress remains near zero, the method is too conservative. If progress is high but direction-flip penalties are frequent, the low-level attention or target-point dynamics may be oscillatory.

**中文。**  
预期结果不应只是 ego 到达目标车道。具有科研说服力的结果应该表明控制器是“以正确原因”完成并道：它应选择局部可行 gap，保持足够距离，避免在候选 gap 之间快速切换，并以平滑横向行为完成并道。因此，分析不能只依赖单一总分。reward 曲线有用，但必须结合成功率、碰撞率、最小距离、决策切换频率和 reward 分项一起解释。如果总 reward 高但安全惩罚也高，说明方法可能过于激进；如果碰撞率为零但 progress 接近零，说明方法过于保守；如果 progress 高但横向换向惩罚频繁，说明底层注意力或目标点动力学可能存在振荡。

**English.**  
The central claim tested by Main13 is that a low-dimensional learned attention policy can transfer from a single-gap setting to a multi-gap setting when embedded inside a structured bilevel controller. The high-level opinion dynamics handles the combinatorial part of choosing a gap, while the low-level SAC policy handles the timing and strength of merge commitment for a selected front-rear pair. This division of labor is useful because it avoids training a large end-to-end policy over all multi-car states from scratch. Instead, the method preserves explicit formulas for confidence, bias, opinion dynamics, target-point construction, and safety avoidance, while using learning only where hand-designed timing is most difficult.

**中文。**  
Main13 要验证的核心观点是：一个低维学习注意力策略，只要嵌入结构化双层控制器中，就可以从单 gap 环境迁移到多 gap 环境。高层意见动力学处理“选择哪个 gap”这一组合决策问题，底层 SAC 策略处理针对某个前后车 pair 的并道承诺时机和强度。这种分工很有价值，因为它避免从零开始训练覆盖所有多车状态的大型端到端策略。相反，该方法保留置信度、偏置、意见动力学、目标点构造和安全避障等显式公式，只在最难手工设计的注意力时序上引入学习。

---

## Conclusion / 结论

**English.**  
The Main13 framework provides an interpretable, modular, and experimentally testable approach to autonomous lane merging in multi-car traffic. The high-level layer uses smooth confidence differences and self-updating opinion dynamics to choose between local candidate gaps, while the low-level layer uses objective gap bias, learned or analytic attention, opinion-driven target-point interpolation, and safety-augmented tracking control to execute the maneuver. The design is especially suitable for ablation studies because each component has a clear mathematical role: \(C_f-C_r\) measures relative gap preference, \(y\) stores high-level directional belief, \(u_h\) controls decision commitment, \(b(t)\) measures selected-gap physical feasibility, \(u(t)\) controls low-level merge urgency, \(z(t)\) maps intention into target-point motion, \(u_c\) handles local collision avoidance, and the final bicycle-model inverse converts desired point acceleration into vehicle inputs. The experimental sequence from Main12 to Main13 directly tests whether a policy trained on a simple local gap can remain useful in a richer multi-gap environment when combined with a structured high-level decision process.

**中文。**  
Main13 框架为多车交通中的自动并道提供了一种可解释、模块化且便于实验验证的方法。高层通过平滑置信度差值和自更新意见动力学在局部候选 gap 中做选择；底层通过客观 gap 偏置、学习或解析注意力、由意见驱动的目标点插值，以及带安全避障的跟踪控制来执行并道。该设计尤其适合消融实验，因为每个部件都有明确数学角色：\(C_f-C_r\) 表示相对 gap 偏好，\(y\) 存储高层方向信念，\(u_h\) 控制决策承诺强度，\(b(t)\) 衡量所选 gap 的物理可行性，\(u(t)\) 控制底层并道紧迫度，\(z(t)\) 把并道意愿映射为目标点运动，\(u_c\) 负责局部避障，最终自行车模型反解则把期望点加速度转换为车辆输入。从 Main12 到 Main13 的实验序列，正是在验证：一个在简单局部 gap 中训练得到的策略，是否能在结构化高层决策机制的帮助下迁移到更复杂的多 gap 多车环境中。
