# 双层意见动力学控制方法与实验总结
# Bilevel Opinion-Dynamics Control: Methodology and Experimental Summary

> 本文档整理一种面向自动驾驶并道任务的双层意见动力学控制框架。为避免方法说明被具体程序文件名限制，全文统一使用“单 gap 环境”和“多 gap 环境”描述实验设置。公式均采用独立 LaTeX 块，以便在 Markdown 或论文写作工具中清晰渲染。
>
> This document summarizes a bilevel opinion-dynamics control framework for autonomous lane merging. To keep the description method-oriented rather than implementation-specific, the experiments are referred to as the single-gap environment and the multi-gap environment. Equations are written as independent LaTeX blocks for clear rendering in Markdown and dissertation writing tools.

---

## Abstract / 摘要

**English.**
This work develops a bilevel decision and control framework for autonomous lane merging in interactive traffic. The proposed method separates the lane-change problem into two coupled levels. The high-level module decides which local gap in the target lane should be selected, while the low-level module converts the selected gap into a continuous control target and physical vehicle inputs. The high-level module observes only the three target-lane vehicles nearest to the ego vehicle, constructs two candidate gaps, evaluates them through smooth confidence functions, and updates a high-level opinion state through a self-reinforcing opinion-dynamics equation. The sign and magnitude of the high-level opinion determine whether the ego vehicle should move toward the forward gap, move toward the rear gap, or wait. The low-level module then evaluates the selected gap using gap size and gap-rate information, determines an attention intensity using either a learned Soft Actor-Critic policy or an analytic radial-basis-function rule, updates a low-level opinion state, and maps that opinion into a target point inside the selected gap. A safety-augmented front-axle tracking controller finally generates acceleration and steering-rate commands.

The central motivation is that lane merging requires both discrete choice and continuous control. A purely continuous controller can follow a target point but may struggle when several nearby gaps are similar. A purely discrete selector can choose a gap but cannot by itself guarantee smooth tracking, sufficient clearance, or stable steering. The proposed decomposition addresses this issue by assigning gap selection to the high-level opinion dynamics and assigning maneuver execution to the low-level controller. Since opinion states integrate evidence over time, small but persistent differences between candidate gaps can be amplified into stable decisions without relying on brittle hard thresholds. The experimental design contains two stages. The single-gap environment is used to study whether the low-level attention policy can learn useful merging commitment. The multi-gap environment is used to test whether the learned low-level policy and the high-level opinion-dynamics selector remain effective when several randomly changing gaps are present.

**中文。**
本文提出一种用于自动驾驶车辆并入目标车道的双层决策与控制框架。该方法把并道问题拆分为两个相互耦合但功能清晰的层次：高层模块负责判断目标车道中哪个局部 gap 更值得选择，底层模块负责把被选中的 gap 转换为连续目标点和实际车辆控制输入。高层模块只读取距离 ego 车最近的三辆目标车道车辆，由三车形成前后两个候选 gap，通过平滑置信度函数评价二者的相对可行性，并利用带自强化项的意见动力学方程更新高层意见状态。高层意见的符号和幅值决定 ego 车是倾向前方 gap、后方 gap，还是暂时等待。底层模块进一步根据所选 gap 的大小和变化速率计算客观偏置，使用 SAC 学习策略或解析径向基函数规则确定注意力强度，更新底层意见状态，并把该意见映射为 gap 内的目标控制点。最后，带安全避障项的前轴点跟踪控制器生成加速度和转角变化率。

该框架的核心动机在于，并道任务同时包含离散选择和连续控制。单纯连续控制器可以跟踪目标点，但在多个相似 gap 同时存在时可能难以稳定选择；单纯离散选择器可以给出 gap 编号，却不能单独保证平滑进入、安全间距和转向稳定。因此，本文用高层意见动力学处理 gap 选择，用底层控制器处理具体并道执行。由于意见状态能够在时间上积累证据，候选 gap 之间即使只有很小但持续存在的差异，也可以被注意力自增强机制放大为稳定决策，而不必依赖容易造成跳变的硬阈值。实验设计分为两个阶段：单 gap 环境用于研究底层注意力策略是否能学到合适的并道承诺时机，多 gap 环境用于验证该底层策略与高层意见动力学选择器在随机变化交通空隙中的泛化能力。

---

## Notation / 符号定义

**English.**
The vehicle state is described by a rear-axle bicycle model. For vehicle \(i\), the state vector is

$$
\mathbf{x}_i =
\begin{bmatrix}
x_i & y_i & \theta_i & v_i & \delta_i
\end{bmatrix}^{\top},
$$

where \((x_i,y_i)\) denotes the rear-axle center, \(\theta_i\) denotes the heading angle, \(v_i\) denotes the longitudinal speed, and \(\delta_i\) denotes the steering angle. The controller is formulated at the front-axle point because this point gives a more direct geometric description of lane-change tracking. For a wheelbase \(L\), the front-axle position and velocity are

$$
\mathbf{p}_i =
\begin{bmatrix}
x_i + L\cos\theta_i \\
y_i + L\sin\theta_i
\end{bmatrix},
$$

$$
\mathbf{v}_i^f =
v_i
\begin{bmatrix}
\cos\theta_i - \sin\theta_i\tan\delta_i \\
\sin\theta_i + \cos\theta_i\tan\delta_i
\end{bmatrix}.
$$

The original lane center is denoted by \(y_O\), the target lane center by \(y_T\), and the lane width by \(W\). The wheelbase is \(L\), and the collision boundary radius used in the safety calculation is \(r\). In the multi-gap environment, the target lane contains \(N\) vehicles and therefore \(N-1\) physical gaps. However, the high-level decision module does not evaluate all gaps globally. It only reads the three target-lane vehicles that are nearest to the ego vehicle in longitudinal front-axle coordinate, and it forms two local candidate gaps from those vehicles.

**中文。**
车辆状态采用后轴自行车模型表示。对第 \(i\) 辆车，其状态向量为

$$
\mathbf{x}_i =
\begin{bmatrix}
x_i & y_i & \theta_i & v_i & \delta_i
\end{bmatrix}^{\top},
$$

其中 \((x_i,y_i)\) 是后轴中心坐标，\(\theta_i\) 是航向角，\(v_i\) 是纵向速度，\(\delta_i\) 是前轮转角。控制器在前轴点上构造，因为前轴点更直接地反映车辆横向进入目标车道时的几何误差。若车辆轴距为 \(L\)，则前轴点的位置和速度分别为

$$
\mathbf{p}_i =
\begin{bmatrix}
x_i + L\cos\theta_i \\
y_i + L\sin\theta_i
\end{bmatrix},
$$

$$
\mathbf{v}_i^f =
v_i
\begin{bmatrix}
\cos\theta_i - \sin\theta_i\tan\delta_i \\
\sin\theta_i + \cos\theta_i\tan\delta_i
\end{bmatrix}.
$$

原车道中心记为 \(y_O\)，目标车道中心记为 \(y_T\)，车道宽度记为 \(W\)。车辆轴距为 \(L\)，安全计算中使用的碰撞边界半径为 \(r\)。在多 gap 环境中，目标车道包含 \(N\) 辆车，因此物理上存在 \(N-1\) 个 gap。但高层决策模块并不进行全局 gap 搜索，而是只读取与 ego 车前轴纵向坐标最接近的三辆目标车道车辆，并由这三辆车形成两个局部候选 gap。

| Symbol / 符号 | English definition | 中文定义 |
| :--- | :--- | :--- |
| \(\mathbf{x}_i\) | Rear-axle vehicle state | 后轴车辆状态 |
| \(\mathbf{p}_i\) | Front-axle control point | 前轴控制点 |
| \(\mathbf{v}_i^f\) | Front-axle velocity | 前轴速度 |
| \(y(t)\) | High-level opinion for gap direction | 高层 gap 方向意见 |
| \(u_h(t)\) | High-level self-updating attention | 高层自更新注意力 |
| \(z(t)\) | Low-level merge-intention opinion | 底层并道意愿意见 |
| \(b(t)\) | Objective low-level gap bias | 底层客观 gap 偏置 |
| \(u(t)\) | Low-level attention intensity | 底层注意力强度 |
| \(C_f,C_r\) | Forward and rear gap confidence | 前 gap 与后 gap 置信度 |
| \(B(t)\) | High-level directional bias | 高层方向偏置 |
| \(d_{\min}\) | Minimum ego-to-target distance | ego 与目标车最小距离 |

---

# 1. Bilevel Control System / 双层控制系统

**English.**
The proposed controller is organized as a bilevel system. The high level answers a strategic question: among the locally relevant gaps in the target lane, should the ego vehicle aim for the forward gap, the rear gap, or neither gap at the present time? The low level answers a continuous control question: once a gap has been selected, how strongly should the vehicle commit to entering it, where should the desired point be placed, and what physical commands should be applied? This division is useful because the two questions have different mathematical characters. Gap selection is a comparison problem with memory and uncertainty, whereas vehicle motion is a tracking problem constrained by safety, steering smoothness, and acceleration limits.

The complete information flow can be summarized as

$$
\{\mathbf{x}_{1:N},\mathbf{x}_e\}
\rightarrow
\text{nearest three target-lane vehicles}
\rightarrow
\{C_f,C_r,B,y,u_h\}
\rightarrow
\text{selected gap}
\rightarrow
\{b(t),u(t),z(t)\}
\rightarrow
\mathbf{p}^{\star}(t)
\rightarrow
\mathbf{u}_{\mathrm{total}}
\rightarrow
\begin{bmatrix} a & \omega \end{bmatrix}.
$$

Here \(\mathbf{x}_{1:N}\) are the target-lane vehicle states, \(\mathbf{x}_e\) is the ego state, \(C_f\) and \(C_r\) are the forward and rear gap confidences, \(B\) is the high-level bias, \(y\) is the high-level opinion, \(u_h\) is the high-level attention, \(b(t)\) and \(u(t)\) are the low-level bias and attention, \(z(t)\) is the low-level opinion, \(\mathbf{p}^{\star}(t)\) is the desired control point, \(\mathbf{u}_{\mathrm{total}}\) is the desired front-axle acceleration vector, and \(a,\omega\) are the final bicycle-model acceleration and steering-rate inputs.

**中文。**
本文控制器采用双层结构。高层回答的是策略问题：在目标车道局部相关的 gap 中，ego 车此刻应该指向前方 gap、后方 gap，还是暂时不选择？底层回答的是连续控制问题：一旦某个 gap 被选中，车辆应以多强的意愿并入，目标控制点应放在哪里，实际车辆应施加什么控制输入？这种分工有明确的数学意义。gap 选择是一个带记忆和不确定性的比较问题，而车辆运动是一个受安全间距、转向平滑性和加速度限制约束的跟踪问题。

整体信息流可以写为

$$
\{\mathbf{x}_{1:N},\mathbf{x}_e\}
\rightarrow
\text{最近三辆目标车道车辆}
\rightarrow
\{C_f,C_r,B,y,u_h\}
\rightarrow
\text{被选中的 gap}
\rightarrow
\{b(t),u(t),z(t)\}
\rightarrow
\mathbf{p}^{\star}(t)
\rightarrow
\mathbf{u}_{\mathrm{total}}
\rightarrow
\begin{bmatrix} a & \omega \end{bmatrix}.
$$

其中 \(\mathbf{x}_{1:N}\) 是目标车道车辆状态，\(\mathbf{x}_e\) 是 ego 车状态，\(C_f\) 和 \(C_r\) 是前后两个候选 gap 的置信度，\(B\) 是高层偏置，\(y\) 是高层意见，\(u_h\) 是高层注意力，\(b(t)\) 和 \(u(t)\) 是底层偏置与注意力，\(z(t)\) 是底层意见，\(\mathbf{p}^{\star}(t)\) 是期望控制点，\(\mathbf{u}_{\mathrm{total}}\) 是期望前轴加速度向量，\(a\) 与 \(\omega\) 是最终输入到自行车模型的加速度和转角变化率。

## 1.1 Opinion Dynamics and Self-Updating Attention / 意见动力学与自更新注意力公式

**English.**
Both levels use the same conceptual opinion-dynamics template. For a generic opinion variable \(o(t)\), the dynamics are

$$
\dot{o}(t) =
-d\,o(t)
+u(t)\tanh\!\left(\alpha o(t)\right)
+b(t).
$$

The damping coefficient \(d>0\) pulls the opinion back toward neutrality and prevents unbounded drift. The attention intensity \(u(t)\geq0\) controls how strongly an already formed opinion reinforces itself. The sensitivity parameter \(\alpha>0\) controls how quickly the nonlinear term saturates. The external bias \(b(t)\) injects objective evidence from the traffic scene. This equation therefore contains three interpretable effects: forgetting, self-reinforcement, and environmental evidence. If \(b(t)\) remains near zero, the damping term makes the opinion decay. If a small but persistent bias exists, the self-reinforcement term can gradually amplify the opinion until a stable decision appears.

**中文。**
两个层次都使用同一种意见动力学模板。对一般意见变量 \(o(t)\)，其动力学为

$$
\dot{o}(t) =
-d\,o(t)
+u(t)\tanh\!\left(\alpha o(t)\right)
+b(t).
$$

其中阻尼系数 \(d>0\) 把意见拉回中性位置，避免意见无限漂移；注意力强度 \(u(t)\geq0\) 决定已有意见被自我强化的程度；灵敏度参数 \(\alpha>0\) 决定非线性项进入饱和区的速度；外部偏置 \(b(t)\) 把交通场景中的客观证据输入系统。因此，该方程同时包含遗忘、自强化和环境证据三种含义。如果 \(b(t)\) 接近零，阻尼项会使意见衰减；如果存在微小但持续的偏置，自强化项会逐渐放大意见，直到形成稳定决策。

For high-level decision making, the opinion variable is \(y(t)\), and the external bias is the confidence difference between the forward and rear candidate gaps:

$$
\dot{y}(t) =
-d_y y(t)
+u_h(t)\tanh\!\left(\alpha_y y(t)\right)
+B(t).
$$

高层决策中，意见变量为 \(y(t)\)，外部偏置为前后候选 gap 的置信度差值：

$$
\dot{y}(t) =
-d_y y(t)
+u_h(t)\tanh\!\left(\alpha_y y(t)\right)
+B(t).
$$

The high-level attention is self-updated rather than directly assigned by a neural network:

$$
\dot{u}_h(t) =
\frac{-u_h(t)+S_h(y(t)^2)}{\tau_h},
$$

$$
S_h(y^2) =
U_{\max}
\frac{(y^2)^n}{K_h^n+(y^2)^n}.
$$

高层注意力不是由神经网络直接指定，而是由高层意见自身更新：

$$
\dot{u}_h(t) =
\frac{-u_h(t)+S_h(y(t)^2)}{\tau_h},
$$

$$
S_h(y^2) =
U_{\max}
\frac{(y^2)^n}{K_h^n+(y^2)^n}.
$$

This Hill-type function depends on \(y^2\), so it is symmetric with respect to the sign of the opinion. It does not care whether the system prefers the forward gap or the rear gap; it only measures whether the preference has become strong enough. When \(y(t)\) is close to zero, attention decays and the vehicle remains cautious. When \(|y(t)|\) grows, \(S_h\) increases, which raises \(u_h(t)\), which then strengthens the self-reinforcing term in the opinion equation. This feedback loop allows a small continuous preference to become a robust gap choice.

这个 Hill 型函数依赖 \(y^2\)，因此对意见符号是对称的。它并不关心系统偏向前 gap 还是后 gap，只关心偏向是否足够强。当 \(y(t)\) 接近零时，注意力衰减，车辆保持谨慎；当 \(|y(t)|\) 增大时，\(S_h\) 上升，进而提高 \(u_h(t)\)，而更大的 \(u_h(t)\) 又会加强意见方程中的自强化项。这个反馈闭环使微小但连续存在的偏好能够逐渐变成稳定 gap 选择。

| Term / 项 | Formula / 公式 | Role / 作用 |
| :--- | :--- | :--- |
| Damping / 阻尼 | \(-d\,o(t)\) | Pulls opinion toward zero; 将意见拉回中性，防止漂移。 |
| Self-reinforcement / 自强化 | \(u(t)\tanh(\alpha o(t))\) | Amplifies an existing opinion while remaining bounded; 有界地放大已有意见。 |
| External bias / 外部偏置 | \(b(t)\) | Injects traffic evidence; 输入交通环境证据。 |
| Attention update / 注意力更新 | \(\dot{u}=(-u+S(o^2))/\tau\) | Builds commitment when opinion magnitude increases; 在意见变强时积累执行信心。 |

## 1.2 High-Level Bias from Gap Confidence / 高层偏置的计算

**English.**
At every time step, the high-level module selects the three target-lane vehicles closest to the ego front axle in longitudinal coordinate. Let these vehicles be ordered from front to rear as \((i_1,i_2,i_3)\). The forward candidate gap is the space between \(i_1\) and \(i_2\), and the rear candidate gap is the space between \(i_2\) and \(i_3\). For any candidate gap formed by a front vehicle \(F\) and a rear vehicle \(R\), the gap center and gap velocity are

$$
x_g(t) =
\frac{p_F^x(t)+p_R^x(t)}{2},
$$

$$
v_g(t) =
\frac{v_F^x(t)+v_R^x(t)}{2}.
$$

The ego-to-gap alignment is defined by the longitudinal position error and the relative speed error:

$$
d_g(t) =
x_g(t)-p_e^x(t),
$$

$$
\Delta v_g(t) =
v_e^x(t)-v_g(t).
$$

The confidence of a gap is then evaluated using a Gaussian radial-basis function:

$$
C_g(t) =
\exp
\left(
-\frac{d_g(t)^2}{2\sigma_d^2}
-\frac{\Delta v_g(t)^2}{2\sigma_v^2}
\right).
$$

This confidence is large when the gap center is close to the ego vehicle and the relative velocity is small. It decreases smoothly as the gap becomes spatially far away or dynamically mismatched. The smooth form is important because high-level decisions should not jump abruptly when the relative order of nearby vehicles changes.

**中文。**
每个时间步，高层模块按照纵向前轴坐标选出距离 ego 车最近的三辆目标车道车辆。将这三辆车按从前到后的顺序记为 \((i_1,i_2,i_3)\)。前方候选 gap 是 \(i_1\) 与 \(i_2\) 之间的空隙，后方候选 gap 是 \(i_2\) 与 \(i_3\) 之间的空隙。对任意由前车 \(F\) 和后车 \(R\) 构成的候选 gap，其中心位置和平均速度定义为

$$
x_g(t) =
\frac{p_F^x(t)+p_R^x(t)}{2},
$$

$$
v_g(t) =
\frac{v_F^x(t)+v_R^x(t)}{2}.
$$

ego 车与该 gap 的对齐程度由纵向位置误差和相对速度误差表示：

$$
d_g(t) =
x_g(t)-p_e^x(t),
$$

$$
\Delta v_g(t) =
v_e^x(t)-v_g(t).
$$

gap 的置信度通过高斯径向基函数计算：

$$
C_g(t) =
\exp
\left(
-\frac{d_g(t)^2}{2\sigma_d^2}
-\frac{\Delta v_g(t)^2}{2\sigma_v^2}
\right).
$$

当 gap 中心接近 ego 车且相对速度较小时，置信度较大；当 gap 在空间上较远或速度匹配较差时，置信度平滑下降。这种平滑形式很重要，因为高层决策不应在邻近车辆集合发生轻微变化时突然跳变。

The forward and rear confidences are converted into a signed directional bias:

$$
B(t) =
C_f(t)-C_r(t).
$$

前 gap 与后 gap 的置信度通过有符号差值转换为方向偏置：

$$
B(t) =
C_f(t)-C_r(t).
$$

This difference is used instead of a hard maximum because it keeps both the direction and the strength of preference. If \(B(t)>0\), the forward gap is more compatible with the current ego state. If \(B(t)<0\), the rear gap is more compatible. If \(B(t)\approx0\), the evidence is ambiguous. Even when both absolute confidences are small, the difference can still encode a weak but meaningful preference. Through the opinion equation, such a weak preference can accumulate over time and become decisive only when it persists. This is the key reason for using \(B=C_f-C_r\): the comparison remains smooth, memory can be added through \(y(t)\), and small advantages do not need to be converted into immediate hard decisions.

这里使用差值而不是硬最大值，是因为差值同时保留了偏好的方向和强度。若 \(B(t)>0\)，说明前 gap 与当前 ego 状态更匹配；若 \(B(t)<0\)，说明后 gap 更匹配；若 \(B(t)\approx0\)，说明证据仍然模糊。即使两个置信度绝对值都较小，二者差值仍可能表示一个微弱但有意义的相对偏好。通过意见动力学方程，这种微弱偏好只有在持续存在时才会积累并逐渐变得明确。这正是使用 \(B=C_f-C_r\) 的原因：比较过程保持平滑，\(y(t)\) 提供时间记忆，小优势不必立刻被转化为硬决策。

| Situation / 情况 | Bias condition / 偏置条件 | Interpretation / 含义 |
| :--- | :--- | :--- |
| Forward gap preferred / 前 gap 更优 | \(B(t)>0\) | Evidence supports moving toward the forward candidate gap; 证据支持向前 gap 并入。 |
| Rear gap preferred / 后 gap 更优 | \(B(t)<0\) | Evidence supports moving toward the rear candidate gap; 证据支持向后 gap 并入。 |
| Ambiguous / 不明确 | \(B(t)\approx0\) | The two candidates are similar, so waiting is reasonable; 两个候选相近，等待更合理。 |

## 1.3 High-Level Opinion Update and Decision Mapping / 高层意见更新与决策映射

**English.**
The high-level opinion \(y(t)\) stores the accumulated directional belief. After computing \(B(t)\) and updating \(u_h(t)\), the system integrates the high-level opinion using an explicit time step \(\Delta t\):

$$
y_{k+1} =
y_k
+\Delta t
\left[
-d_y y_k
+u_{h,k}\tanh(\alpha_y y_k)
+B_k
\right].
$$

The attention is updated as

$$
u_{h,k+1} =
u_{h,k}
+\Delta t
\frac{-u_{h,k}+S_h(y_k^2)}{\tau_h}.
$$

The decision mapping is threshold-based but not memoryless:

$$
\mathrm{decision}_k =
\begin{cases}
\mathrm{forward}, & y_k>\theta_y,\\
\mathrm{rear}, & y_k<-\theta_y,\\
\mathrm{wait}, & |y_k|\leq\theta_y.
\end{cases}
$$

The waiting band \(|y_k|\leq\theta_y\) is essential. It prevents the selected gap from switching whenever \(C_f\) and \(C_r\) are nearly equal. The vehicle therefore acts only after evidence has accumulated enough to move the opinion outside the neutral region. In practice, this mechanism makes the high-level decision smoother than a direct instantaneous comparison.

**中文。**
高层意见 \(y(t)\) 存储已经积累的方向信念。计算 \(B(t)\) 并更新 \(u_h(t)\) 后，系统使用步长 \(\Delta t\) 对高层意见进行离散积分：

$$
y_{k+1} =
y_k
+\Delta t
\left[
-d_y y_k
+u_{h,k}\tanh(\alpha_y y_k)
+B_k
\right].
$$

注意力更新为

$$
u_{h,k+1} =
u_{h,k}
+\Delta t
\frac{-u_{h,k}+S_h(y_k^2)}{\tau_h}.
$$

高层决策映射采用阈值形式，但它不是无记忆的瞬时判断：

$$
\mathrm{decision}_k =
\begin{cases}
\mathrm{forward}, & y_k>\theta_y,\\
\mathrm{rear}, & y_k<-\theta_y,\\
\mathrm{wait}, & |y_k|\leq\theta_y.
\end{cases}
$$

等待区间 \(|y_k|\leq\theta_y\) 非常关键。它避免了 \(C_f\) 与 \(C_r\) 接近时所选 gap 频繁切换，使车辆只有在证据积累到足够程度后才从中性区离开并做出选择。因此，该机制比直接瞬时比较更加平滑，也更符合驾驶中“观察一段时间再行动”的行为逻辑。

# 2. Low-Level Control System / 底层控制系统

**English.**
After the high-level module selects a candidate gap, the low-level module determines how the ego vehicle should enter that gap. The low level does not simply command an immediate lane change. Instead, it first evaluates the physical feasibility of the selected gap, then determines an attention intensity, updates a low-level opinion \(z(t)\), and finally converts \(z(t)\) into a continuous target point. This design makes the maneuver gradual: the target point moves from the original-lane reference toward the selected gap as the low-level opinion grows.

**中文。**
高层模块选出候选 gap 后，底层模块决定 ego 车如何进入该 gap。底层并不是直接命令车辆立刻变道，而是先评价所选 gap 的物理可行性，再确定注意力强度，更新底层意见 \(z(t)\)，最后把 \(z(t)\) 转换为连续目标点。这样的设计使并道动作是渐进的：随着底层意见增强，目标点从原车道参考位置逐渐移动到所选 gap 内。

## 2.1 Low-Level Bias \(b(t)\): Gap Size and Gap-Rate Evaluation / 底层 \(b(t)\)：gap 大小与变化速率评价

**English.**
For a selected gap formed by a front vehicle \(F\) and a rear vehicle \(R\), the physical gap length is

$$
g(t) =
p_F^x(t)-p_R^x(t).
$$

The gap-rate term is

$$
\dot{g}(t) =
v_F^x(t)-v_R^x(t).
$$

The low-level bias evaluates whether the gap is sufficiently large and whether it is opening or closing:

$$
b(t) =
k_g\left[g(t)-g_{\mathrm{safe}}\right]
+k_v\dot{g}(t).
$$

A larger gap makes \(b(t)\) more positive. An opening gap, where \(\dot{g}(t)>0\), also increases \(b(t)\). A small or closing gap decreases \(b(t)\), delaying the growth of the low-level merge intention. This formula is intentionally interpretable: \(g(t)-g_{\mathrm{safe}}\) measures spatial feasibility, while \(\dot{g}(t)\) measures whether the situation is improving or worsening.

**中文。**
对由前车 \(F\) 和后车 \(R\) 构成的已选 gap，其物理长度为

$$
g(t) =
p_F^x(t)-p_R^x(t).
$$

gap 变化率为

$$
\dot{g}(t) =
v_F^x(t)-v_R^x(t).
$$

底层偏置用于评价该 gap 是否足够大，以及 gap 正在变大还是变小：

$$
b(t) =
k_g\left[g(t)-g_{\mathrm{safe}}\right]
+k_v\dot{g}(t).
$$

gap 越大，\(b(t)\) 越偏正；如果 gap 正在扩大，即 \(\dot{g}(t)>0\)，\(b(t)\) 也会增大。相反，小 gap 或正在闭合的 gap 会降低 \(b(t)\)，从而推迟底层并道意愿的增长。该公式的优点是含义直观：\(g(t)-g_{\mathrm{safe}}\) 衡量空间可行性，\(\dot{g}(t)\) 衡量交通形势正在改善还是恶化。

| Gap condition / gap 状态 | Mathematical effect / 数学影响 | Control meaning / 控制含义 |
| :--- | :--- | :--- |
| Large and opening / 大且扩大 | \(g>g_{\mathrm{safe}},\ \dot{g}>0\) | Positive bias, stronger merge tendency; 偏置更正，并道倾向增强。 |
| Large but closing / 大但缩小 | \(g>g_{\mathrm{safe}},\ \dot{g}<0\) | Moderate bias, cautious merge tendency; 偏置受抑制，需要谨慎。 |
| Small and opening / 小但扩大 | \(g<g_{\mathrm{safe}},\ \dot{g}>0\) | Waiting may be appropriate; 可继续观察等待。 |
| Small and closing / 小且缩小 | \(g<g_{\mathrm{safe}},\ \dot{g}<0\) | Negative bias, merge should be suppressed; 偏置更负，应抑制并道。 |

## 2.2 Why Use Reinforcement Learning for Attention \(u(t)\)? / 为什么用强化学习决定注意力 \(u(t)\)

**English.**
The low-level attention \(u(t)\) controls how strongly the low-level opinion reinforces itself. A hand-designed attention rule can be written using a radial-basis function:

$$
u_{\mathrm{RBF}}(t) =
u_{\mathrm{base}}
+u_{\mathrm{amp}}
\exp
\left(
-\frac{d_g(t)^2}{2\sigma_d^2}
-\frac{\Delta v_g(t)^2}{2\sigma_v^2}
\right).
$$

This rule is simple and interpretable. It increases attention when the ego vehicle is close to the target gap and the relative velocity is small. However, the optimal timing of attention is difficult to hand-design because the system includes nonlinear opinion dynamics, moving vehicles, safety avoidance, saturation, and delayed consequences. If attention grows too early, the ego vehicle may commit before the gap is safe. If attention grows too late, the vehicle may miss a useful opportunity and remain conservative. Reinforcement learning is therefore used to learn the attention schedule from reward feedback while keeping the rest of the controller structured and interpretable.

**中文。**
底层注意力 \(u(t)\) 控制底层意见自强化的强度。可以用径向基函数写出一个手工设计的注意力规则：

$$
u_{\mathrm{RBF}}(t) =
u_{\mathrm{base}}
+u_{\mathrm{amp}}
\exp
\left(
-\frac{d_g(t)^2}{2\sigma_d^2}
-\frac{\Delta v_g(t)^2}{2\sigma_v^2}
\right).
$$

这个规则简单且可解释：当 ego 车接近目标 gap 且相对速度较小时，注意力增强。但注意力的最佳时机很难完全手工设计，因为系统同时包含非线性意见动力学、移动车辆、安全避障、输入饱和和延迟后果。如果注意力过早增大，ego 车可能在 gap 尚不安全时过度承诺；如果注意力过晚增大，车辆又可能错过并道机会并表现得过于保守。因此，强化学习被用于从 reward 反馈中学习注意力时序，同时保留其余控制结构的显式公式和可解释性。

## 2.3 Soft Actor-Critic Reinforcement Learning / SAC 强化学习简介

**English.**
Soft Actor-Critic is an off-policy actor-critic algorithm designed for continuous actions. It learns a stochastic policy \(\pi_\phi(a|s)\), two soft Q-functions, and an entropy temperature. The optimization objective balances expected return and policy entropy:

$$
J(\pi) =
\sum_{t}
\mathbb{E}_{(s_t,a_t)\sim\rho_{\pi}}
\left[
r(s_t,a_t)
+\alpha_{\mathrm{SAC}}\mathcal{H}\left(\pi(\cdot|s_t)\right)
\right].
$$

The entropy term encourages exploration, which is useful in this task because a useful merge policy must discover when to commit rather than merely learning to stay still. The learned policy is not asked to replace the model-based controller. It only outputs the attention variable used by the opinion dynamics. This restricted action space makes the learning problem lower-dimensional and improves interpretability.

**中文。**
Soft Actor-Critic 是一种适用于连续动作空间的离策略 actor-critic 算法。它学习随机策略 \(\pi_\phi(a|s)\)、两个 soft Q 函数和熵温度参数。其优化目标在期望回报和策略熵之间进行平衡：

$$
J(\pi) =
\sum_{t}
\mathbb{E}_{(s_t,a_t)\sim\rho_{\pi}}
\left[
r(s_t,a_t)
+\alpha_{\mathrm{SAC}}\mathcal{H}\left(\pi(\cdot|s_t)\right)
\right].
$$

熵项鼓励探索，这对并道任务很重要，因为有用的策略不仅要学会保持安全，还要学会何时主动承诺并道。学习策略并不替代模型控制器，而只是输出意见动力学中的注意力变量。这样的动作空间更低维，也更容易解释。

### 2.3.1 State and Policy Output / State 选择与 Policy 输出

**English.**
The low-level reinforcement-learning state describes the relative geometry and velocity between the ego vehicle and the front and rear vehicles of the selected gap. A typical state is

$$
s_t =
\begin{bmatrix}
\Delta x_F &
\Delta y_F &
\Delta v_F^x &
\Delta v_F^y &
\Delta x_R &
\Delta y_R &
\Delta v_R^x &
\Delta v_R^y
\end{bmatrix}^{\top}.
$$

Here \(F\) and \(R\) denote the front and rear vehicles of the selected gap. The policy output is the low-level attention:

$$
a_t =
u(t),
\qquad
u(t)\in[u_{\min},u_{\max}].
$$

This state choice is deliberately local. It does not require the policy to know every vehicle in the target lane. Instead, the high-level module selects the relevant gap, and the low-level policy only decides the attention intensity for that selected front-rear pair.

**中文。**
底层强化学习状态描述 ego 车与所选 gap 前后两车之间的相对位置和相对速度。典型状态可写为

$$
s_t =
\begin{bmatrix}
\Delta x_F &
\Delta y_F &
\Delta v_F^x &
\Delta v_F^y &
\Delta x_R &
\Delta y_R &
\Delta v_R^x &
\Delta v_R^y
\end{bmatrix}^{\top}.
$$

其中 \(F\) 与 \(R\) 分别表示所选 gap 的前车和后车。策略输出为底层注意力：

$$
a_t =
u(t),
\qquad
u(t)\in[u_{\min},u_{\max}].
$$

这种状态选择刻意保持局部性。策略不需要观察目标车道所有车辆，而是由高层模块先选择相关 gap，再由底层策略只针对该前后车 pair 决定注意力强度。

### 2.3.2 Reward Design / Reward 设计

**English.**
The reward is designed to encourage timely merging, discourage hesitation, penalize oscillatory motion, and strongly penalize collision. A representative per-step reward is

$$
\begin{aligned}
r_t
=&
w_p\,\Delta P_t
+w_o\,O_t\max(\Delta P_t,0)
-w_h\,O_t(1-P_t)
-w_s(1-P_t) \\
&
-w_a\lVert a_t-a_{t-1}\rVert_2^2
-w_f\,\mathbb{I}\!\left[v_y(t)v_y(t-1)<0\right] \\
&
-w_c
\left[
\max\left(0,\frac{d_{\mathrm{safe}}-d_{\min}(t)}{d_{\mathrm{safe}}}\right)
\right]^2
-1000\,\mathbb{I}_{\mathrm{collision}}
+R_{\mathrm{success}}(t)\,\mathbb{I}_{\mathrm{success}} .
\end{aligned}
$$

The lane-change progress is usually defined as a normalized lateral ratio:

$$
P_t =
\mathrm{clip}
\left(
\frac{y_e(t)-y_O}{y_T-y_O},
0,
1
\right),
\qquad
\Delta P_t=P_t-P_{t-1}.
$$

The opportunity term \(O_t\) represents whether the selected gap is currently suitable for merging. The progress terms reward moving toward the target lane, especially when a useful opportunity exists. The hesitation terms penalize remaining far from the target lane, so the policy cannot receive a high score by simply waiting. The action-smoothness and direction-flip terms reduce repeated acceleration or lateral-direction reversals. The safety term is continuous before collision, while the collision term is a large terminal penalty. The success bonus decreases with time, for example

$$
R_{\mathrm{success}}(t) =
100-2t,
$$

so earlier successful merges receive higher reward.

**中文。**
reward 的设计目标是鼓励车辆抓住机会尽快并道，抑制长时间等待，惩罚反复变速度方向的振荡行为，并对碰撞施加强惩罚。一个具有代表性的单步 reward 为

$$
\begin{aligned}
r_t
=&
w_p\,\Delta P_t
+w_o\,O_t\max(\Delta P_t,0)
-w_h\,O_t(1-P_t)
-w_s(1-P_t) \\
&
-w_a\lVert a_t-a_{t-1}\rVert_2^2
-w_f\,\mathbb{I}\!\left[v_y(t)v_y(t-1)<0\right] \\
&
-w_c
\left[
\max\left(0,\frac{d_{\mathrm{safe}}-d_{\min}(t)}{d_{\mathrm{safe}}}\right)
\right]^2
-1000\,\mathbb{I}_{\mathrm{collision}}
+R_{\mathrm{success}}(t)\,\mathbb{I}_{\mathrm{success}} .
\end{aligned}
$$

并道进度通常定义为归一化横向比例：

$$
P_t =
\mathrm{clip}
\left(
\frac{y_e(t)-y_O}{y_T-y_O},
0,
1
\right),
\qquad
\Delta P_t=P_t-P_{t-1}.
$$

其中 \(O_t\) 表示当前所选 gap 是否构成有利并道机会。progress 项奖励车辆向目标车道推进，尤其在机会存在时给予更强鼓励。hesitation 项惩罚车辆长期停留在原车道附近，避免策略通过“不动”获得高分。动作平滑项和横向换向项减少反复加减速或左右方向抖动。安全项在真正碰撞前连续惩罚距离过近，碰撞项则是强终止惩罚。成功奖励随时间下降，例如

$$
R_{\mathrm{success}}(t) =
100-2t,
$$

因此越早完成安全并道，最终得分越高。

| Reward term / 奖励项 | Meaning / 含义 |
| :--- | :--- |
| \(w_p\Delta P_t\) | Rewards direct lane-change progress; 奖励横向并道进度。 |
| \(w_oO_t\max(\Delta P_t,0)\) | Gives extra reward for moving during a valid opportunity; 有机会时前进会得到额外奖励。 |
| \(-w_hO_t(1-P_t)\) | Penalizes hesitation when a gap is available; 有机会却不动会被扣分。 |
| \(-w_s(1-P_t)\) | Applies a general time/inaction penalty; 对长时间未完成并道扣分。 |
| \(-w_a\lVert a_t-a_{t-1}\rVert_2^2\) | Penalizes abrupt action changes; 惩罚动作突变。 |
| \(-w_f\mathbb{I}[v_y(t)v_y(t-1)<0]\) | Penalizes lateral direction flips; 惩罚横向速度频繁换向。 |
| Safety distance penalty / 安全距离惩罚 | Penalizes near-collision continuously; 对接近碰撞进行连续惩罚。 |
| Collision penalty / 碰撞惩罚 | Strong terminal penalty; 发生碰撞时强烈扣分。 |
| Success bonus / 成功奖励 | Rewards early safe completion; 奖励尽早安全完成。 |

### 2.3.3 Low-Level Opinion \(z(t)\) and Target Point / 底层意见 \(z(t)\) 及其对控制点的影响

**English.**
The low-level opinion \(z(t)\) represents the degree of merge commitment for the selected gap. Its dynamics are

$$
\dot{z}(t) =
-d_z z(t)
+u(t)\tanh\!\left(\alpha_z z(t)\right)
+b(t).
$$

The discrete update is

$$
z_{k+1} =
z_k
+\Delta t
\left[
-d_z z_k
+u_k\tanh(\alpha_z z_k)
+b_k
\right].
$$

A low or negative \(z(t)\) keeps the desired target point close to the original lane or a cautious reference. A high positive \(z(t)\) moves the target point toward the selected gap in the target lane. To keep this transition bounded, \(z(t)\) is mapped through a smooth gate:

$$
\lambda_z(t) =
\mathrm{clip}
\left(
\frac{z(t)-z_{\min}}{z_{\max}-z_{\min}},
0,
1
\right).
$$

The target point is then interpolated between a nominal original-lane point \(\mathbf{p}_O^{\star}(t)\) and a selected-gap target point \(\mathbf{p}_G^{\star}(t)\):

$$
\mathbf{p}^{\star}(t) =
\left[1-\lambda_z(t)\right]\mathbf{p}_O^{\star}(t)
+\lambda_z(t)\mathbf{p}_G^{\star}(t).
$$

**中文。**
底层意见 \(z(t)\) 表示 ego 车对所选 gap 的并道承诺程度。其动力学为

$$
\dot{z}(t) =
-d_z z(t)
+u(t)\tanh\!\left(\alpha_z z(t)\right)
+b(t).
$$

离散更新形式为

$$
z_{k+1} =
z_k
+\Delta t
\left[
-d_z z_k
+u_k\tanh(\alpha_z z_k)
+b_k
\right].
$$

当 \(z(t)\) 较低或为负时，期望目标点保持在原车道参考点或谨慎位置附近；当 \(z(t)\) 增大时，目标点逐渐移动到所选 gap 内。为保证过渡有界，\(z(t)\) 先被映射为平滑门控变量：

$$
\lambda_z(t) =
\mathrm{clip}
\left(
\frac{z(t)-z_{\min}}{z_{\max}-z_{\min}},
0,
1
\right).
$$

随后目标点在原车道参考点 \(\mathbf{p}_O^{\star}(t)\) 与所选 gap 目标点 \(\mathbf{p}_G^{\star}(t)\) 之间插值：

$$
\mathbf{p}^{\star}(t) =
\left[1-\lambda_z(t)\right]\mathbf{p}_O^{\star}(t)
+\lambda_z(t)\mathbf{p}_G^{\star}(t).
$$

# 3. Actual Control Input / 实际控制输入

**English.**
The final control layer converts the desired target point into physical vehicle inputs. This layer contains three parts: a safety-avoidance term that repels the ego vehicle from nearby target-lane vehicles, a tracking error that points toward the desired target point, and a bicycle-model inverse that maps the desired front-axle acceleration into longitudinal acceleration and steering-rate commands. Keeping this part model-based is useful because safety and vehicle kinematics should remain explicit rather than being hidden inside a learned black box.

**中文。**
最终控制层把期望目标点转换为实际车辆输入。该层包含三个部分：将 ego 车从附近目标车道车辆旁推开的安全避障项，指向期望目标点的跟踪误差，以及把期望前轴加速度映射为纵向加速度和转角变化率的自行车模型反解。保留这一部分的模型化表达很有价值，因为安全约束和车辆运动学应保持显式，而不是完全隐藏在学习模型中。

## 3.1 Safety-Avoidance Term \(u_c\) / 避障项 \(u_c\) 设计

**English.**
For each surrounding target-lane vehicle \(j\), define the relative vector from that vehicle to the ego front axle as

$$
\mathbf{r}_j(t) =
\mathbf{p}_e(t)-\mathbf{p}_j(t),
\qquad
d_j(t) =
\lVert \mathbf{r}_j(t)\rVert_2.
$$

The safety-avoidance acceleration is constructed as a repulsive field:

$$
\mathbf{u}_c(t) =
\sum_j
k_c
\max\left(0,\frac{d_{\mathrm{safe}}-d_j(t)}{d_{\mathrm{safe}}}\right)^2
\frac{\mathbf{r}_j(t)}{d_j(t)+\epsilon}.
$$

When all vehicles are farther than \(d_{\mathrm{safe}}\), this term is zero. When a vehicle enters the safety region, the repulsive magnitude grows quadratically as distance decreases. The small constant \(\epsilon\) prevents division by zero. This term does not replace collision checking; instead, it provides a continuous pre-collision correction that can steer the ego vehicle away before a hard collision boundary is reached.

**中文。**
对每一辆周围目标车道车辆 \(j\)，定义从该车指向 ego 前轴点的相对向量为

$$
\mathbf{r}_j(t) =
\mathbf{p}_e(t)-\mathbf{p}_j(t),
\qquad
d_j(t) =
\lVert \mathbf{r}_j(t)\rVert_2.
$$

安全避障加速度构造为排斥场：

$$
\mathbf{u}_c(t) =
\sum_j
k_c
\max\left(0,\frac{d_{\mathrm{safe}}-d_j(t)}{d_{\mathrm{safe}}}\right)^2
\frac{\mathbf{r}_j(t)}{d_j(t)+\epsilon}.
$$

当所有车辆距离都大于 \(d_{\mathrm{safe}}\) 时，该项为零；当某辆车进入安全区域后，排斥强度随距离减小而二次增大。小常数 \(\epsilon\) 用于避免除零。该项并不替代碰撞检测，而是在真正到达硬碰撞边界前提供连续修正，使 ego 车提前远离危险区域。

## 3.2 Target Point and Tracking Error \(e_z\) / 总目标点与控制误差设计

**English.**
The selected-gap target point is usually placed near the longitudinal center of the gap and at the target-lane center:

$$
\mathbf{p}_G^{\star}(t) =
\begin{bmatrix}
x_g(t) \\
y_T
\end{bmatrix}.
$$

The original-lane reference point can be placed ahead of the ego vehicle along the original lane:

$$
\mathbf{p}_O^{\star}(t) =
\begin{bmatrix}
p_e^x(t)+\ell_{\mathrm{look}} \\
y_O
\end{bmatrix}.
$$

The opinion-weighted target point is

$$
\mathbf{p}^{\star}(t) =
\left[1-\lambda_z(t)\right]\mathbf{p}_O^{\star}(t)
+\lambda_z(t)\mathbf{p}_G^{\star}(t).
$$

The tracking error is

$$
\mathbf{e}_z(t) =
\mathbf{p}^{\star}(t)-\mathbf{p}_e(t).
$$

This construction gives the low-level opinion a direct geometric meaning. When \(\lambda_z(t)\) is close to zero, the controller behaves like a lane-keeping controller. When \(\lambda_z(t)\) approaches one, the controller behaves like a gap-entry controller.

**中文。**
所选 gap 的目标点通常放在 gap 纵向中心附近，并位于目标车道中心线上：

$$
\mathbf{p}_G^{\star}(t) =
\begin{bmatrix}
x_g(t) \\
y_T
\end{bmatrix}.
$$

原车道参考点可放在 ego 车前方一定前视距离处：

$$
\mathbf{p}_O^{\star}(t) =
\begin{bmatrix}
p_e^x(t)+\ell_{\mathrm{look}} \\
y_O
\end{bmatrix}.
$$

由意见加权得到的总目标点为

$$
\mathbf{p}^{\star}(t) =
\left[1-\lambda_z(t)\right]\mathbf{p}_O^{\star}(t)
+\lambda_z(t)\mathbf{p}_G^{\star}(t).
$$

跟踪误差为

$$
\mathbf{e}_z(t) =
\mathbf{p}^{\star}(t)-\mathbf{p}_e(t).
$$

这种构造使底层意见具有直接几何意义。当 \(\lambda_z(t)\) 接近零时，控制器近似表现为车道保持控制器；当 \(\lambda_z(t)\) 接近一时，控制器逐渐表现为 gap 并入控制器。

## 3.3 Final PID-Like Physical Input Design / 最终物理控制输入设计

**English.**
The desired front-axle acceleration combines proportional position tracking, velocity damping, and safety avoidance:

$$
\mathbf{u}_{\mathrm{total}}(t) =
k_p\mathbf{e}_z(t)
-k_v\mathbf{v}_e^f(t)
+\mathbf{u}_c(t).
$$

This vector is then converted into physical inputs. Let the front-axle velocity direction and its normal direction be

$$
\mathbf{t}_e(t) =
\frac{\mathbf{v}_e^f(t)}
{\lVert\mathbf{v}_e^f(t)\rVert_2+\epsilon},
\qquad
\mathbf{n}_e(t) =
\begin{bmatrix}
-t_e^y(t) \\
t_e^x(t)
\end{bmatrix}.
$$

The longitudinal acceleration can be approximated by projection onto \(\mathbf{t}_e\):

$$
a(t) =
\mathrm{clip}
\left(
\mathbf{u}_{\mathrm{total}}(t)^{\top}\mathbf{t}_e(t),
a_{\min},
a_{\max}
\right).
$$

The steering-rate command is generated from the lateral component:

$$
\omega(t) =
\mathrm{clip}
\left(
k_{\omega}\,
\mathbf{u}_{\mathrm{total}}(t)^{\top}\mathbf{n}_e(t),
\omega_{\min},
\omega_{\max}
\right).
$$

This PID-like structure is not intended to be a full optimal controller. Its purpose is to provide a clear and stable mapping from the opinion-driven target point to feasible vehicle inputs, while input clipping prevents unrealistic acceleration and steering changes.

**中文。**
期望前轴加速度由位置比例跟踪、速度阻尼和安全避障共同组成：

$$
\mathbf{u}_{\mathrm{total}}(t) =
k_p\mathbf{e}_z(t)
-k_v\mathbf{v}_e^f(t)
+\mathbf{u}_c(t).
$$

随后该向量被转换为实际车辆输入。令前轴速度方向及其法向方向为

$$
\mathbf{t}_e(t) =
\frac{\mathbf{v}_e^f(t)}
{\lVert\mathbf{v}_e^f(t)\rVert_2+\epsilon},
\qquad
\mathbf{n}_e(t) =
\begin{bmatrix}
-t_e^y(t) \\
t_e^x(t)
\end{bmatrix}.
$$

纵向加速度可通过在速度方向上的投影近似得到：

$$
a(t) =
\mathrm{clip}
\left(
\mathbf{u}_{\mathrm{total}}(t)^{\top}\mathbf{t}_e(t),
a_{\min},
a_{\max}
\right).
$$

转角变化率由横向分量生成：

$$
\omega(t) =
\mathrm{clip}
\left(
k_{\omega}\,
\mathbf{u}_{\mathrm{total}}(t)^{\top}\mathbf{n}_e(t),
\omega_{\min},
\omega_{\max}
\right).
$$

这种类 PID 结构并不是完整最优控制器，其作用是提供从意见驱动目标点到可执行车辆输入的清晰稳定映射，同时通过输入裁剪避免不现实的加速度和转向变化。

# 4. Experimental Design / 实验设计

**English.**
The experiments are arranged in two stages. The first stage uses a single-gap environment to train and evaluate the low-level attention policy \(u(t)\). This stage fixes the candidate gap, so the learning problem focuses on when the ego vehicle should commit to entering a known front-rear gap. The second stage uses a multi-gap environment to evaluate whether the policy learned in the simpler setting can be reused when high-level gap selection is required. This order separates two scientific questions: whether the low-level SAC policy can outperform a hand-designed attention rule, and whether the high-level opinion-dynamics selector can outperform a memoryless maximum-score selector in a more complex traffic scene.

**中文。**
实验按照两个阶段展开。第一阶段使用单 gap 环境训练和评价底层注意力策略 \(u(t)\)。在该阶段中，候选 gap 是固定的，因此学习问题主要集中在 ego 车应何时增强并道承诺、何时进入一个已知的前后车空隙。第二阶段使用多 gap 环境，检验在简单环境中训练得到的底层策略能否迁移到需要高层 gap 选择的复杂场景中。这样的顺序可以把两个科学问题分开：第一，底层 SAC 策略是否优于手工设计的注意力 \(u(t)\)；第二，在多车随机 gap 中，高层意见动力学选择器是否优于无记忆的最大评分策略。

## 4.1 Single-Gap Merging Experiment / 单 gap 并入实验

**English.**
The single-gap environment contains one front vehicle, one rear vehicle, and one ego vehicle. The front and rear vehicles travel in the target lane, and their longitudinal separation defines the only available merging gap. The ego vehicle starts from the original lane and attempts to merge into this gap. The lane width is \(W=4.0\,\mathrm{m}\), so the original and target lane centers are \(y_O=0.5W=2.0\,\mathrm{m}\) and \(y_T=1.5W=6.0\,\mathrm{m}\), respectively. The initial target-vehicle states are

$$
x_F(0)=30.0\,\mathrm{m},\qquad
x_R(0)=15.0\,\mathrm{m},\qquad
v_F(0)=v_R(0)=15.0\,\mathrm{m/s}.
$$

The ego vehicle has the same initial longitudinal speed but a randomized longitudinal initial position:

$$
x_e(0)=20.0+\xi,\qquad
\xi\sim\mathcal{U}(-5.0,5.0),\qquad
y_e(0)=2.0\,\mathrm{m},\qquad
v_e(0)=15.0\,\mathrm{m/s}.
$$

This randomization makes the training and evaluation less dependent on one special initial alignment. It forces the policy to learn an attention schedule that works when the ego vehicle starts slightly ahead of or behind the nominal gap center.

**中文。**
单 gap 环境包含一辆目标前车、一辆目标后车和一辆 ego 车。前车与后车位于目标车道，二者之间的纵向间距构成唯一可并入 gap。ego 车从原车道出发，并尝试并入该 gap。车道宽度为 \(W=4.0\,\mathrm{m}\)，因此原车道中心为 \(y_O=0.5W=2.0\,\mathrm{m}\)，目标车道中心为 \(y_T=1.5W=6.0\,\mathrm{m}\)。目标车初始状态为

$$
x_F(0)=30.0\,\mathrm{m},\qquad
x_R(0)=15.0\,\mathrm{m},\qquad
v_F(0)=v_R(0)=15.0\,\mathrm{m/s}.
$$

ego 车具有相同初始纵向速度，但纵向初始位置带有随机扰动：

$$
x_e(0)=20.0+\xi,\qquad
\xi\sim\mathcal{U}(-5.0,5.0),\qquad
y_e(0)=2.0\,\mathrm{m},\qquad
v_e(0)=15.0\,\mathrm{m/s}.
$$

该随机化避免训练和评价只依赖某一个特殊的初始对齐位置，使策略必须在 ego 车略微领先或落后于名义 gap 中心时仍能给出合理注意力时序。

**English.**
The rear vehicle is deliberately designed to create a staged merging opportunity. Before \(t_y=20.0\,\mathrm{s}\), the rear vehicle follows a sinusoidal acceleration law. Let

$$
\omega_s=\frac{2\pi}{P_s},\qquad
P_s=6.0\,\mathrm{s},\qquad
A_s=4.0\,\mathrm{m/s}.
$$

For \(0\leq t\leq t_y\), the rear-vehicle acceleration, velocity, and position are

$$
a_R(t)=A_s\omega_s\cos(\omega_s t),
$$

$$
v_R(t)=v_R(0)+A_s\sin(\omega_s t),
$$

$$
x_R(t)=x_R(0)+v_R(0)t+\frac{A_s}{\omega_s}\left[1-\cos(\omega_s t)\right].
$$

Since the front vehicle travels with constant speed \(v_F=15.0\,\mathrm{m/s}\), its longitudinal position is

$$
x_F(t)=x_F(0)+v_F(0)t.
$$

The front-rear gap before yielding is therefore

$$
g(t)=x_F(t)-x_R(t)
=15.0-\frac{A_s}{\omega_s}\left[1-\cos(\omega_s t)\right],
\qquad
0\leq t\leq20.0.
$$

After \(20.0\,\mathrm{s}\), the rear vehicle starts yielding by tracking a desired front-rear gap \(g_{\mathrm{yield}}=20.0\,\mathrm{m}\). The rear-vehicle acceleration becomes

$$
a_R(t)=
\mathrm{clip}
\left(
0.35\left[g(t)-20.0\right]
-1.1\left[v_R(t)-v_F(t)\right],
-5.0,
2.0
\right),
\qquad
t>20.0.
$$

Equivalently, the gap dynamics after yielding are governed by

$$
\dot{g}(t)=v_F(t)-v_R(t),
\qquad
\ddot{g}(t)=-a_R(t),
\qquad
t>20.0.
$$

This piecewise construction has a clear purpose. Before \(20.0\,\mathrm{s}\), the gap is not intentionally opened for the ego vehicle, so the policy should avoid premature aggressive merging. After \(20.0\,\mathrm{s}\), the rear vehicle creates a larger gap and the correct behavior is to increase attention \(u(t)\), allow the opinion \(z(t)\) to grow, and move the target point toward the target lane.

**中文。**
后车运动被设计成分阶段生成并道机会。在 \(t_y=20.0\,\mathrm{s}\) 之前，后车采用正弦加速度。令

$$
\omega_s=\frac{2\pi}{P_s},\qquad
P_s=6.0\,\mathrm{s},\qquad
A_s=4.0\,\mathrm{m/s}.
$$

当 \(0\leq t\leq t_y\) 时，后车加速度、速度和位置为

$$
a_R(t)=A_s\omega_s\cos(\omega_s t),
$$

$$
v_R(t)=v_R(0)+A_s\sin(\omega_s t),
$$

$$
x_R(t)=x_R(0)+v_R(0)t+\frac{A_s}{\omega_s}\left[1-\cos(\omega_s t)\right].
$$

前车保持匀速 \(v_F=15.0\,\mathrm{m/s}\)，因此纵向位置为

$$
x_F(t)=x_F(0)+v_F(0)t.
$$

于是让行前的前后车 gap 可写为

$$
g(t)=x_F(t)-x_R(t)
=15.0-\frac{A_s}{\omega_s}\left[1-\cos(\omega_s t)\right],
\qquad
0\leq t\leq20.0.
$$

在 \(20.0\,\mathrm{s}\) 之后，后车开始让行，并跟踪期望 gap \(g_{\mathrm{yield}}=20.0\,\mathrm{m}\)。后车加速度变为

$$
a_R(t)=
\mathrm{clip}
\left(
0.35\left[g(t)-20.0\right]
-1.1\left[v_R(t)-v_F(t)\right],
-5.0,
2.0
\right),
\qquad
t>20.0.
$$

等价地，让行后的 gap 动态满足

$$
\dot{g}(t)=v_F(t)-v_R(t),
\qquad
\ddot{g}(t)=-a_R(t),
\qquad
t>20.0.
$$

这个分段构造具有明确实验含义。在 \(20.0\,\mathrm{s}\) 之前，目标 gap 并未主动为 ego 车打开，因此策略不应过早激进并道；在 \(20.0\,\mathrm{s}\) 之后，后车开始创造更大 gap，合理策略应提高注意力 \(u(t)\)，使意见 \(z(t)\) 增长，并把目标点推向目标车道。

**English.**
The SAC training problem uses the relative position and velocity between the ego vehicle and the two target vehicles as the state:

$$
s_t=
\begin{bmatrix}
\Delta x_F & \Delta y_F & \Delta v_F^x & \Delta v_F^y &
\Delta x_R & \Delta y_R & \Delta v_R^x & \Delta v_R^y
\end{bmatrix}^{\top}.
$$

The action is one-dimensional and represents the low-level attention:

$$
a_t=u(t),\qquad
u(t)\in[0.0,3.0].
$$

The SAC agent uses a Gaussian policy with two hidden layers of width \(256\), two Q networks, target Q networks, replay-buffer learning, and entropy regularization. The main training parameters are \(200\) episodes, replay-buffer size \(2.0\times10^5\), batch size \(256\), \(1000\) initial random steps, discount factor \(\gamma=0.99\), target-update rate \(\tau=0.005\), and learning rates \(3\times10^{-4}\) for the policy, Q networks, and entropy temperature. The reward combines lane-change progress, opportunity-dependent progress reward, hesitation penalty, time penalty, action-change penalty, lateral direction-flip penalty, safety-distance penalty, collision penalty, and a time-decaying success bonus. The same reward definition is used during value comparison so that the learned and hand-designed policies are judged by an identical metric.

The baseline comparison evaluates the trained SAC attention against the original hand-designed RBF attention:

$$
u_{\mathrm{RBF}}(t)=
u_{\mathrm{base}}+
u_{\mathrm{amp}}
\exp
\left(
-\frac{d_g(t)^2}{2\sigma_d^2}
-\frac{\Delta v_g(t)^2}{2\sigma_v^2}
\right).
$$

The evaluation repeats the simulation \(10\) times with shared random ego initial positions. For each random seed, both policies face the same initial \(x_e(0)\) and the same target-vehicle trajectory. The comparison reports the episode reward of each trial and the mean and standard deviation across trials:

$$
\bar{R}_{\mathrm{SAC}}=\frac{1}{N_{\mathrm{eval}}}\sum_{j=1}^{N_{\mathrm{eval}}}R_{\mathrm{SAC}}^{(j)},
\qquad
\bar{R}_{\mathrm{RBF}}=\frac{1}{N_{\mathrm{eval}}}\sum_{j=1}^{N_{\mathrm{eval}}}R_{\mathrm{RBF}}^{(j)},
\qquad
N_{\mathrm{eval}}=10.
$$

A successful result is not merely a larger total reward. The learned policy should obtain a higher mean reward primarily through larger progress and more successful early merging, while keeping collision rate and safety penalties comparable to or lower than the RBF baseline.

**中文。**
SAC 训练问题的 state 由 ego 车相对于目标前车和目标后车的位置与速度组成：

$$
s_t=
\begin{bmatrix}
\Delta x_F & \Delta y_F & \Delta v_F^x & \Delta v_F^y &
\Delta x_R & \Delta y_R & \Delta v_R^x & \Delta v_R^y
\end{bmatrix}^{\top}.
$$

action 是一维量，表示底层注意力：

$$
a_t=u(t),\qquad
u(t)\in[0.0,3.0].
$$

SAC agent 使用高斯策略网络、两个 Q 网络、目标 Q 网络、经验回放和熵正则化。主要训练参数为：训练 \(200\) 个 episode，经验池容量 \(2.0\times10^5\)，batch size 为 \(256\)，初始随机探索步数为 \(1000\)，折扣因子 \(\gamma=0.99\)，目标网络软更新系数 \(\tau=0.005\)，policy、Q 网络和熵温度学习率均为 \(3\times10^{-4}\)。reward 由并道进度、有机会时的额外进度奖励、犹豫惩罚、时间惩罚、动作变化惩罚、横向换向惩罚、安全距离惩罚、碰撞惩罚和随时间衰减的成功奖励组成。后续价值对比也使用同一 reward 定义，因此学习策略和手工策略由完全一致的指标评价。

对照基线为原始手工设计的 RBF 注意力：

$$
u_{\mathrm{RBF}}(t)=
u_{\mathrm{base}}+
u_{\mathrm{amp}}
\exp
\left(
-\frac{d_g(t)^2}{2\sigma_d^2}
-\frac{\Delta v_g(t)^2}{2\sigma_v^2}
\right).
$$

评价阶段重复 \(10\) 次仿真，并使用共享的随机 ego 初始位置。对每一个随机种子，SAC 策略和 RBF 策略面对完全相同的 \(x_e(0)\) 和目标车轨迹。最终比较每次 episode reward，并统计多次试验的均值与标准差：

$$
\bar{R}_{\mathrm{SAC}}=\frac{1}{N_{\mathrm{eval}}}\sum_{j=1}^{N_{\mathrm{eval}}}R_{\mathrm{SAC}}^{(j)},
\qquad
\bar{R}_{\mathrm{RBF}}=\frac{1}{N_{\mathrm{eval}}}\sum_{j=1}^{N_{\mathrm{eval}}}R_{\mathrm{RBF}}^{(j)},
\qquad
N_{\mathrm{eval}}=10.
$$

理想结果不只是 SAC 总 reward 更高，还应体现为：更大的并道进度、更高的早期成功率，同时碰撞率和安全惩罚不高于手工 RBF 基线。

## 4.2 Multi-Gap Merging Experiment / 多 gap 并入实验

**English.**
The multi-gap environment extends the same merging task to a target lane containing five vehicles and four physical gaps. The target vehicles are initialized with a uniform base spacing:

$$
N=5,\qquad
x_i(0)=48.0-(i-1)g_0,\qquad
g_0=8.0\,\mathrm{m},\qquad
i=1,\ldots,5.
$$

All target vehicles start on the target-lane center \(y_T=6.0\,\mathrm{m}\) with nominal speed \(15.0\,\mathrm{m/s}\). The ego vehicle starts from the original lane with a larger longitudinal randomization range than in the single-gap experiment:

$$
x_e(0)=30.0+\xi,\qquad
\xi\sim\mathcal{U}(-10.0,10.0),\qquad
y_e(0)=2.0\,\mathrm{m},\qquad
v_e(0)=15.0\,\mathrm{m/s}.
$$

The larger random range changes which three vehicles are nearest to the ego vehicle at the beginning of an episode. Therefore, the high-level selector must work under different local traffic configurations rather than repeatedly facing one fixed gap.

**中文。**
多 gap 环境把同一并道任务扩展到包含五辆目标车和四个物理 gap 的目标车道。目标车队以统一基础间距初始化：

$$
N=5,\qquad
x_i(0)=48.0-(i-1)g_0,\qquad
g_0=8.0\,\mathrm{m},\qquad
i=1,\ldots,5.
$$

所有目标车初始位于目标车道中心 \(y_T=6.0\,\mathrm{m}\)，名义速度为 \(15.0\,\mathrm{m/s}\)。ego 车从原车道出发，并且相对于单 gap 实验使用更大的纵向随机范围：

$$
x_e(0)=30.0+\xi,\qquad
\xi\sim\mathcal{U}(-10.0,10.0),\qquad
y_e(0)=2.0\,\mathrm{m},\qquad
v_e(0)=15.0\,\mathrm{m/s}.
$$

更大的随机范围会改变每个 episode 开始时距离 ego 最近的三辆目标车，因此高层选择器必须面对不同的局部交通构型，而不是反复处理同一个固定 gap。

**English.**
The four target-lane gaps are randomly adjusted over time. Every \(T_g=4.0\,\mathrm{s}\), at most two gaps are selected for modification, and each selected gap receives a desired spacing from the multiplier set

$$
\mathcal{M}=\{0.75,1.0,1.25,1.5\}.
$$

For gap \(i\) in schedule period \(k\), the desired gap is

$$
g_{i,\mathrm{des}}^{(k)}=g_0m_i^{(k)},\qquad
m_i^{(k)}\in\mathcal{M}.
$$

The leading target vehicle has zero acceleration. Each following target vehicle tracks the desired gap in front of it through a clipped proportional-derivative rule:

$$
a_{i+1}(t)=
\mathrm{clip}
\left(
0.55\left[g_i(t)-g_{i,\mathrm{des}}^{(k)}\right]
 +1.05\dot{g}_i(t),
-4.0,
4.0
\right).
$$

This mechanism produces a target lane in which gaps open, close, and reconfigure during the \(40.0\,\mathrm{s}\) simulation. The high-level decision module does not evaluate all four gaps with a global planner. Instead, at each step it selects the nearest three target-lane vehicles in longitudinal front-axle coordinate and compares the two local gaps formed by these vehicles. The confidence difference \(B=C_f-C_r\) updates the high-level opinion \(y(t)\), and the sign of \(y(t)\) determines whether the forward local gap, rear local gap, or waiting action is selected.

**中文。**
四个目标车道 gap 会随时间随机调整。每隔 \(T_g=4.0\,\mathrm{s}\)，最多两个 gap 会被选中改变期望间距，被选中的 gap 从倍率集合中抽取一个倍率：

$$
\mathcal{M}=\{0.75,1.0,1.25,1.5\}.
$$

对第 \(k\) 个调度周期中的第 \(i\) 个 gap，其期望间距为

$$
g_{i,\mathrm{des}}^{(k)}=g_0m_i^{(k)},\qquad
m_i^{(k)}\in\mathcal{M}.
$$

目标车队最前车加速度为零。每一辆跟随目标车用裁剪后的 PD 规则跟踪其前方期望 gap：

$$
a_{i+1}(t)=
\mathrm{clip}
\left(
0.55\left[g_i(t)-g_{i,\mathrm{des}}^{(k)}\right]
 +1.05\dot{g}_i(t),
-4.0,
4.0
\right).
$$

该机制使目标车道中的 gap 在 \(40.0\,\mathrm{s}\) 仿真中持续打开、闭合和重构。高层决策模块并不使用全局规划器评价全部四个 gap，而是在每一步根据前轴纵向坐标选择距离 ego 最近的三辆目标车，并比较这三辆车形成的两个局部 gap。置信度差值 \(B=C_f-C_r\) 用于更新高层意见 \(y(t)\)，而 \(y(t)\) 的符号决定选择前方局部 gap、后方局部 gap，还是保持等待。

**English.**
The main purpose of the multi-gap experiment is to test generalization. The low-level policy is not retrained in the five-vehicle environment; it is the attention policy learned in the single-gap environment. This design checks whether a policy trained only to control \(u(t)\) for one front-rear pair can remain useful after a high-level selector provides different front-rear pairs in a richer traffic scene. If the policy generalizes, the controller should still enter a selected gap smoothly, even though the selected gap may switch over time and the target-lane gaps may change randomly.

The high-level ablation compares the opinion-dynamics selector with a simple maximum-score selector. The maximum-score baseline removes the high-level opinion memory and chooses the locally better gap instantaneously:

$$
i_k^{\star}=\arg\max_i S_i(t_k),
$$

where \(S_i(t_k)\) is the instantaneous confidence or gap-evaluation score of candidate gap \(i\). The proposed opinion-dynamics selector instead integrates the confidence difference over time:

$$
\dot{y}(t)=-d_y y(t)+u_h(t)\tanh(\alpha_y y(t))+C_f(t)-C_r(t).
$$

The comparison should be evaluated with repeated randomized trials. In the current test setting, \(N_{\mathrm{test}}=100\) random runs are used. For each method, the total reward, success rate, collision rate, final progress, completion time, minimum distance, and selected-gap switch count should be reported. The main expected advantage of the opinion-based high level is not only a higher mean reward, but also a lower tendency to switch decisions when two local gap scores are close.

**中文。**
多 gap 实验的主要目的在于验证泛化能力。底层策略不会在五车环境中重新训练，而是直接使用单 gap 环境中学到的注意力策略。这样的设计检验了一个只针对单个前后车 pair 学习 \(u(t)\) 的策略，在高层选择器不断提供不同前后车 pair 时是否仍然有效。如果策略具有泛化性，那么即使被选 gap 可能随时间变化、目标车道 gap 也在随机改变，ego 车仍应能够平滑进入所选 gap。

高层消融实验比较意见动力学选择器与简单最大评分选择器。最大评分基线去掉高层意见记忆，并瞬时选择局部评分更高的 gap：

$$
i_k^{\star}=\arg\max_i S_i(t_k),
$$

其中 \(S_i(t_k)\) 表示候选 gap \(i\) 的瞬时置信度或 gap 评价分数。本文方法则通过意见动力学持续积分置信度差值：

$$
\dot{y}(t)=-d_y y(t)+u_h(t)\tanh(\alpha_y y(t))+C_f(t)-C_r(t).
$$

该对比应通过多次随机试验评价。当前测试设置使用 \(N_{\mathrm{test}}=100\) 次随机运行。对每种方法，应报告总 reward、成功率、碰撞率、最终并道进度、完成时间、最小距离和被选 gap 切换次数。意见动力学高层的预期优势不只是更高平均 reward，还包括当两个局部 gap 评分接近时更少发生决策抖动。

## 4.3 Experimental Parameter Summary / 实验参数总结

| Category / 类别 | Symbol / 符号 | Value / 数值 | Description / 说明 |
| :--- | :--- | :--- | :--- |
| Simulation / 仿真 | \(T\) | \(40.0\,\mathrm{s}\) | Episode duration / 单次仿真总时长 |
| Simulation / 仿真 | \(\Delta t\) | \(0.05\,\mathrm{s}\) | Integration interval / 数值积分步长 |
| Road / 道路 | \(W\) | \(4.0\,\mathrm{m}\) | Lane width / 车道宽度 |
| Road / 道路 | \(y_O,y_T\) | \(2.0\,\mathrm{m},6.0\,\mathrm{m}\) | Original and target lane centers / 原车道与目标车道中心 |
| Vehicle / 车辆 | \(L\) | \(2.8\,\mathrm{m}\) | Vehicle wheelbase / 车辆轴距 |
| Safety / 安全 | \(r\) | \(1.5\,\mathrm{m}\) | Collision boundary radius / 碰撞边界半径 |
| Single-gap initial state / 单 gap 初始状态 | \(x_F(0),x_R(0)\) | \(30.0\,\mathrm{m},15.0\,\mathrm{m}\) | Front and rear target-vehicle initial positions / 前后目标车初始位置 |
| Single-gap initial state / 单 gap 初始状态 | \(x_e(0)\) | \(20.0+\mathcal{U}(-5.0,5.0)\,\mathrm{m}\) | Ego randomized initial position / ego 随机初始纵向位置 |
| Single-gap rear motion / 单 gap 后车运动 | \(t_y\) | \(20.0\,\mathrm{s}\) | Yielding starts after this time / 后车开始让行时间 |
| Single-gap rear motion / 单 gap 后车运动 | \(A_s,P_s\) | \(4.0\,\mathrm{m/s},6.0\,\mathrm{s}\) | Sinusoidal velocity amplitude and period before yielding / 让行前正弦速度幅值与周期 |
| Single-gap rear motion / 单 gap 后车运动 | \(g_{\mathrm{yield}}\) | \(20.0\,\mathrm{m}\) | Desired front-rear gap after yielding / 让行后目标 gap |
| Single-gap rear motion / 单 gap 后车运动 | \(a_R\) clip | \([-5.0,2.0]\,\mathrm{m/s^2}\) | Rear-vehicle acceleration bounds after yielding / 后车让行控制加速度范围 |
| Single-gap low level / 单 gap 底层 | \(g_{\mathrm{safe}}\) | \(10.0\,\mathrm{m}\) | Safe-gap threshold for single-gap training / 单 gap 训练中的安全 gap 阈值 |
| SAC training / SAC 训练 | \(N_{\mathrm{ep}}\) | \(200\) | Training episodes / 训练轮次 |
| SAC training / SAC 训练 | \(N_{\mathrm{rand}}\) | \(1000\) steps | Initial random exploration steps / 初始随机探索步数 |
| SAC training / SAC 训练 | \(\gamma,\tau\) | \(0.99,0.005\) | Discount factor and target-network update rate / 折扣因子与目标网络软更新系数 |
| SAC training / SAC 训练 | \(B_{\mathrm{batch}}\) | \(256\) | Batch size / 批量大小 |
| SAC training / SAC 训练 | \(\mathcal{D}\) | \(200000\) | Replay-buffer capacity / 经验池容量 |
| SAC training / SAC 训练 | \(\eta_{\pi},\eta_Q,\eta_{\alpha}\) | \(3\times10^{-4}\) | Policy, Q-network, and entropy-temperature learning rates / 策略、Q 网络和熵温度学习率 |
| SAC training / SAC 训练 | \(H\) | \(256\) | Hidden-layer width / 隐藏层宽度 |
| SAC action / SAC 动作 | \(u_{\min},u_{\max}\) | \(0.0,3.0\) | Learned attention range / 学习注意力范围 |
| Single-gap evaluation / 单 gap 评价 | \(N_{\mathrm{eval}}\) | \(10\) | Repeated value-comparison runs / 重复 reward 对比次数 |
| RBF baseline / RBF 基线 | \(u_{\mathrm{base}},u_{\mathrm{amp}}\) | \(0.2,2.5\) | Hand-designed attention baseline parameters / 手工注意力参数 |
| RBF baseline / RBF 基线 | \(\sigma_d,\sigma_v\) | \(2.0,1.5\) | Single-gap RBF position and velocity bandwidths / 单 gap RBF 位置与速度尺度 |
| Multi-gap fleet / 多 gap 车队 | \(N\) | \(5\) | Number of target-lane vehicles / 目标车数量 |
| Multi-gap fleet / 多 gap 车队 | \(N_g\) | \(4\) | Number of physical gaps / 物理 gap 数量 |
| Multi-gap initial state / 多 gap 初始状态 | \(x_i(0)\) | \(48.0-(i-1)8.0\,\mathrm{m}\) | Target-vehicle initial positions / 目标车初始位置 |
| Multi-gap initial state / 多 gap 初始状态 | \(x_e(0)\) | \(30.0+\mathcal{U}(-10.0,10.0)\,\mathrm{m}\) | Ego randomized initial position / ego 随机初始纵向位置 |
| Multi-gap schedule / 多 gap 调度 | \(g_0\) | \(8.0\,\mathrm{m}\) | Base desired gap / 基础期望 gap |
| Multi-gap schedule / 多 gap 调度 | \(T_g\) | \(4.0\,\mathrm{s}\) | Gap adjustment interval / gap 调整周期 |
| Multi-gap schedule / 多 gap 调度 | \(\mathcal{M}\) | \(\{0.75,1.0,1.25,1.5\}\) | Desired-gap multiplier set / 期望 gap 倍率集合 |
| Multi-gap schedule / 多 gap 调度 | \(N_{\mathrm{chg}}\) | \(\leq2\) gaps/period | Maximum changed gaps per period / 每周期最多调整 gap 数 |
| Multi-gap following / 多 gap 跟驰 | \(k_p^g,k_d^g\) | \(0.55,1.05\) | Gap-tracking gains / gap 跟踪增益 |
| Multi-gap following / 多 gap 跟驰 | \(a_{\max}^g\) | \(4.0\,\mathrm{m/s^2}\) | Target-vehicle acceleration limit / 目标车加速度裁剪 |
| High-level opinion / 高层意见 | \(d_y,\alpha_y\) | \(2.5,10.0\) | High-level damping and sensitivity / 高层阻尼与灵敏度 |
| High-level attention / 高层注意力 | \(\tau_h,U_{\max},K_h,n\) | \(1.0,1.5,0.2,2.0\) | Self-updating attention parameters / 自更新注意力参数 |
| High-level decision / 高层决策 | \(\theta_y\) | \(0.18\) | Waiting-band decision threshold / 等待区间阈值 |
| Multi-gap confidence / 多 gap 置信度 | \(\sigma_d,\sigma_v\) | \(4.0,2.5\) | Confidence bandwidths for local gap comparison / 局部 gap 比较中的位置与速度尺度 |
| Low-level opinion / 底层意见 | \(g_{\mathrm{safe}},k_g,k_v\) | \(5.0\,\mathrm{m},0.2,0.1\) | Gap-bias parameters in multi-gap execution / 多 gap 执行中的 gap 偏置参数 |
| Low-level opinion / 底层意见 | \(d_z,\alpha_z\) | \(2.0,2.0\) | Low-level opinion damping and sensitivity / 底层意见阻尼与灵敏度 |
| Multi-gap evaluation / 多 gap 评价 | \(N_{\mathrm{test}}\) | \(100\) | Random repeated test runs / 随机重复测试次数 |
| Physical input / 物理输入 | \(a\) clip | \([-5.0,5.0]\,\mathrm{m/s^2}\) | Ego acceleration bounds / ego 加速度裁剪 |
| Physical input / 物理输入 | \(\omega\) clip | \([-0.8,0.8]\,\mathrm{rad/s}\) | Ego steering-rate bounds / ego 转角变化率裁剪 |

---

## Conclusion / 结论

**English.**
The bilevel opinion-dynamics framework provides an interpretable and experimentally testable approach to autonomous lane merging. The high-level layer uses smooth confidence differences and self-updating attention to choose between local candidate gaps. The low-level layer uses objective gap bias, learned or analytic attention, opinion-driven target-point interpolation, and safety-augmented tracking control to execute the selected maneuver. Each component has a clear mathematical role: \(C_f-C_r\) measures relative gap preference, \(y(t)\) stores high-level directional belief, \(u_h(t)\) controls decision commitment, \(b(t)\) measures selected-gap feasibility, \(u(t)\) controls low-level merge urgency, \(z(t)\) maps intention into target-point motion, \(\mathbf{u}_c(t)\) provides local safety correction, and the final inverse mapping converts desired front-axle acceleration into feasible vehicle inputs.

The experimental sequence from the single-gap environment to the multi-gap environment tests both specialization and generalization. The single-gap environment isolates the low-level attention problem and makes the learning objective reproducible. The multi-gap environment then evaluates whether the same low-level mechanism remains useful when a high-level selector must choose among randomly changing local gaps. A strong result should therefore show not only high reward, but also low collision rate, reduced decision chattering, acceptable minimum distance, smooth lateral behavior, and consistent success across randomized initial conditions.

**中文。**
双层意见动力学框架为自动并道提供了一种可解释且便于实验验证的方法。高层通过平滑置信度差值和自更新注意力在局部候选 gap 中做选择；底层通过客观 gap 偏置、学习或解析注意力、意见驱动的目标点插值，以及带安全避障的跟踪控制来执行所选动作。每个部件都有明确数学角色：\(C_f-C_r\) 表示相对 gap 偏好，\(y(t)\) 存储高层方向信念，\(u_h(t)\) 控制决策承诺强度，\(b(t)\) 衡量所选 gap 的物理可行性，\(u(t)\) 控制底层并道紧迫度，\(z(t)\) 将意愿映射为目标点运动，\(\mathbf{u}_c(t)\) 提供局部安全修正，最终反解映射则把期望前轴加速度转换为可执行车辆输入。

从单 gap 环境到多 gap 环境的实验序列同时检验了专门化能力和泛化能力。单 gap 环境隔离底层注意力问题，使学习目标更容易复现；多 gap 环境进一步评价同一底层机制在高层选择器面对随机变化局部 gap 时是否仍然有效。因此，理想结果不应只体现为高 reward，还应体现为低碰撞率、较少决策抖动、可接受的最小距离、平滑横向行为，以及在随机初始条件下稳定成功。
