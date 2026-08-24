# 摘要/Abstract

**English.**

In the field of autonomous driving, the task of lane changing for vehicles has always been a fundamental issue. This paper proposes a two-layer decision-making and control framework for autonomous vehicles to merge into the target lane. This method splits the lane changing problem into two inter-coupled but clearly functional layers: the upper-level module is responsible for determining which local gap is more worthy of selection, while the lower-level module is responsible for converting the selected gap into continuous target points and actual vehicle control inputs. The upper-level module only reads the three closest target lane vehicles from the self-vehicle, forms two candidate gaps with these three vehicles, evaluates the relative feasibility of the two gaps through a smooth confidence function, and updates the upper-level opinion state using the opinion dynamics equation with reinforcement terms. The lower-level module further calculates the objective offset based on the size and rate of change of the selected gap, uses the SAC reinforcement learning strategy to determine the attention intensity, updates the lower-level opinion state, and maps this opinion to the target control points within the gap. Finally, the front axle point tracking controller generates vehicle acceleration and angular velocity changes.

Through building simulation experiments in the Python environment, the control method was tested. The experimental results show that in a single-gap environment, the reinforcement learning method of SAC can well learn the strategy and maintain convergence, and the vehicle can successfully complete the task; in a multi-gap environment, the strategies learned in a simple environment have been verified to be applicable in a complex environment, and the excellent performance of the two-layer decision-making system has been verified.


**中文。**

在自动驾驶领域，车辆并道任务一直是一项基本问题。本文提出一种用于自动驾驶车辆并入目标车道的双层决策与控制框架。该方法把并道问题拆分为两个相互耦合但功能清晰的层次：高层模块负责判断目标车道中哪个局部 gap 更值得选择，底层模块负责把被选中的 gap 转换为连续目标点和实际车辆控制输入。高层模块只读取距离自车最近的三辆目标车道车辆，由三车形成前后两个候选 gap，通过平滑置信度函数评价二者的相对可行性，并利用带自强化项的意见动力学方程更新高层意见状态。底层模块进一步根据所选 gap 的大小和变化速率计算客观偏置，使用 SAC 强化学习策略确定注意力强度，更新底层意见状态，并把该意见映射为 gap 内的目标控制点。最后，带安全避障项的前轴点跟踪控制器生成车辆加速度和转角变化率。

本文通过在Python环境里搭建仿真实验，对控制方法进行了测试，实验结果表明：在单gap环境中，SAC的强化学习方法能很好的学习到策略并保持收敛，车辆也能成功的完成任务；在多gap环境中，验证了简单环境学习到的策略在复杂环境下的泛用性，并验证了双层决策系统的优秀使用效果。

# Declaration of originality

I hereby confirm that no portion of the work referred to in the thesis has been submitted in support of an application for another degree or qualification of this or any other university or other institute of learning

本人特此确认，本论文中所述的任何部分内容均未被提交用于申请本校或其他任何大学或学习机构的其他学位或资格。

# Copyright statement

# 简介/Introduction

## 背景与动机/Background and Motivation

### 并道与换道交互中的策略决策方法/Decision-making strategies for merge and lane-change interaction

**English**

Lane changing and lane switching decisions are representative issues in autonomous driving. A lane change alters the available gaps, road right-of-way expectations, and future responses of surrounding vehicles. Therefore, in recent years, research has gradually regarded lane changing as an interactive perception decision task. Game theory models are widely used because they can explicitly describe the strategic relationship between the self-driving vehicle and surrounding vehicles. Lane changing intentions, collision probabilities, and dynamic risks can be incorporated into the payoff function to determine whether a maneuver is beneficial or dangerous [R1, R2]. Physical-inspired models such as molecular interaction potentials further express the attraction and repulsion between vehicles as continuous interaction forces [R3]. Recent studies also consider incomplete information and social driving preferences, as in lane changing conflicts involving human-driven vehicles, surrounding vehicles may exhibit cooperative, aggressive, or uncertain behaviors [R4].

Another important direction is learning-based interactive prediction. Graph neural networks and attention mechanisms represent vehicles as nodes and the influence between vehicles as edges or attention weights. This is highly suitable for lane changing scenarios, as the number and importance of surrounding vehicles will dynamically change. Existing research uses topological graphs combined with driving behavior and traffic context to predict the lane changing intentions of drivers [R5], and some studies use plan-aware graph attention networks to predict the responses of surrounding vehicles to the candidate intentions of the self-driving vehicle, and then the model predicts the control trajectory [R6]. These methods illustrate that interaction modeling should not only predict where other vehicles will go, but also estimate how they will respond to the self-driving vehicle's decision. Cooperative control methods provide a third route. In connected autonomous driving environments, surrounding vehicles can actively create gaps, coordinate lane changing sequences, or maintain fleet stability [R7, R8]. Overall, recent literature provides motivation for the hierarchical framework: this framework requires the combination of strategy interaction modeling, behavior prediction, and safe execution.

**中文**

并道与换道决策是自动驾驶中具有代表性的问题。一次换道会改变周围车辆的可用间隙、路权预期和未来响应。因此，近年研究逐渐将换道视为交互感知决策任务。博弈论模型被广泛使用，是因为它能够显式描述自车与周围车辆之间的策略关系。换道意图、碰撞概率以及动态风险等可以被纳入收益函数，用于判断某个机动是有利还是危险 [R1, R2]。分子相互作用势等物理启发模型进一步将车辆间的吸引与排斥表达为连续交互力 [R3]。近期研究还考虑不完全信息和社会驾驶偏好，因为在人类驾驶车辆参与的换道冲突中，周车可能表现为合作、激进或不确定 [R4]。

另一条重要方向是学习型交互预测。图神经网络和注意力机制将车辆表示为节点，将车辆间影响表示为边或注意力权重。这非常适合换道场景，因为周围车辆的数量和重要性会动态变化。已有研究使用拓扑图结合驾驶行为和交通上下文预测驾驶人换道意图 [R5]，也有研究利用计划感知图注意力网络预测周车对自车候选意图的响应，再由模型预测控制生成可行轨迹 [R6]。这些方法说明，交互建模不应只预测其他车辆会去哪里，还应估计它们会如何响应自车决策。协同控制方法提供了第三条路线。在网联自动驾驶环境中，周围车辆可以主动创造间隙、协调换道顺序或维持车队稳定 [R7, R8]。总体来看，近期文献为层级框架提供了动机：该框架需要结合策略交互建模、行为预测和安全执行。

### 强化学习用于自适应决策动力学/Reinforcement learning for adaptive decision dynamics

**English**

The lane-changing decision-making process exhibits sequentiality, uncertainty, and feedback-driven characteristics. Vehicles need to determine when to wait, when to change lanes, and how to adjust their speed. Traditional fixed-parameter controllers may perform well in static scenarios, but they often lack flexibility when the traffic environment changes or when the system needs to switch between cooperative consensus and competitive divergence. Deep reinforcement learning provides a mechanism for learning decision strategies through repeated interactions. Existing studies have integrated safety rules, future risk assessment, reward shaping, and robust observation modeling into DRL lane-changing strategies [R9-R11]. These studies show that RL should not be used as an unconstrained black box, but should be guided by interpretable state variables, safety-aware reward terms, and robust mechanisms.

At the same time, MARL allows each connected or autonomous vehicle to learn from the behavior of neighboring vehicles while considering group results such as traffic efficiency, comfort, and safety [R12]. Right-of-way coordination and Mix Q-learning further illustrate that the lane-changing task can balance individual benefits with group benefits [R13, R14]. For the project of this paper, these studies support a key idea: RL is not just a direct output of lane-changing actions, but can dynamically adjust the control parameters of the nonlinear decision dynamics model. This enables adaptive changes according to environmental stimuli while still maintaining connection with low-level safety control.


**中文**

换道决策具有序贯性、不确定性和反馈驱动特征。车辆需要决定何时等待、何时并道、如何调整速度。传统固定参数控制器在静态场景中可能表现良好，但当交通环境变化，或系统需要在合作共识与竞争分歧之间切换时，往往缺乏灵活性。深度强化学习提供了一种通过反复交互学习决策策略的机制。已有研究将安全规则、未来风险评估、奖励塑形和鲁棒观测建模融入 DRL 换道策略 [R9-R11]。这些研究表明，RL 不应被作为无约束黑箱使用，而应由可解释状态变量、安全感知奖励项和鲁棒机制引导。

同时 MARL 允许每辆网联或自动驾驶车辆从邻近车辆行为中学习，同时考虑交通效率、舒适性和安全性等群体结果 [R12]。路权协同和 Mix Q-learning 进一步说明，换道任务中可以平衡个体收益与群体收益 [R13, R14]。对于本论文项目而言，这些研究支持一个关键思路：RL 不只是直接输出换道动作，而是可以动态调节非线性决策动力学模型的控制参数。这样就能够根据环境刺激自适应变化，同时仍与低层安全控制保持连接。

### 评价维度：从效率到安全、舒适与适应性/Evaluation dimensions: from efficiency to safety, comfort, and adaptability

**English**

The evaluation of lane-changing decisions has shifted from a single efficiency metric to a multi-dimensional assessment. Average speed, travel time, traffic volume, and lane-changing success rate remain important, but they are not sufficient to support safety-critical autonomous driving systems. A rapid lane change can still be unsafe, uncomfortable, or disrupt the traffic flow. Therefore, recent research has begun to evaluate collision risk, minimum distance, braking feasibility, safety violation rate, acceleration, deceleration, traffic flow stability, and robustness under perception uncertainty [R7, R8, R10, R15]. This project refers to these evaluation perspectives, which emphasize convergence speed, safety violation rate, and task efficiency while also evaluating safety, comfort, and stability.


**中文**

换道决策的评价已经从单一效率指标转向多维评价。平均速度、旅行时间、通行量和换道成功率仍然重要，但它们不足以支撑安全关键的自动驾驶系统。一次快速换道仍可能是不安全、不舒适或扰动交通流的。因此，近期研究开始评价碰撞风险、最小距离、制动可行性、安全违规率、加速度、加加速度、交通流稳定性以及感知不确定性下的鲁棒性 [R7, R8, R10, R15]。本项目参考了这些评价视角，即在强调收敛速度、安全违规率和任务效率的同时，评价安全性、舒适性和稳定性。

## Aims and Objectives / 研究目标与具体目标

**English**

The aim of this project is to develop, integrate, and evaluate a hierarchical decision-making and control framework for interactive autonomous driving scenarios. This framework will use reinforcement learning to adjust the key parameters of the nonlinear decision dynamics/control model, enabling the agent to quickly make decisions in the simulation environment while meeting safety constraints.

This project includes five specific objects. First, establish a nonlinear opinion dynamics model at the policy layer. Second, train the agent using RL methods to dynamically optimize the model parameters. Third, develop a safety perception and execution layer, using control barrier functions or equivalent constraints to convert the high-level policy into collision-free trajectories. Fourth, implement the controller in the Python simulation framework. Fifth, compare with the baseline and evaluate the decision-control effectiveness.

**中文**

本项目的目标是开发、集成并评价一个面向交互式自动驾驶场景的层级决策与控制框架。该框架将使用强化学习调节非线性决策动力学/控制模型的关键参数，使智能体能够在仿真环境中快速形成决策，同时满足安全约束。

本项目包含五个具体目标。第一，建立策略层非线性意见动力学模型。第二，使用RL方法训练智能体，动态优化模型参数。第三，开发安全感知执行层，使用控制屏障函数或等价约束，将高层策略转化为无碰撞轨迹。第四，在Python仿真框架中实现控制器。第五，与基线对比，评价决策控制效果。

# 方法/Methods

**English.**

This controller adopts a two-layer structure. The upper-layer controller completes the discrete control problem of gap selection through the evaluation of different gaps and the update of opinion dynamics; the lower-layer controller, on the other hand, solves the continuous control problem of actual vehicles by using the actual size and change rate of the gap, as well as the attention parameters obtained through RL learning. 

Fig1: Block diagram of the control system structure

**中文。**

本文控制器采用双层结构。高层控制器通过对不同gap的评价以及意见动力学的更新，完成gap选择的离散控制问题；底层控制器则通过gap的实际大小与变化速率，以及通过RL学习得到的注意力参数，完成实际车辆的连续控制问题。

Fig1：控制系统结构框图

## 高层控制系统/High-Level Control System
### 意见动力学与自更新注意力公式/Opinion Dynamics and Self-Updating Attention

**English.**

The same opinion dynamics template [R16] is used at both levels. As shown in the following formula, for the general opinion variable \(o(t)\), its dynamics are where the damping coefficient \(d > 0\) pulls the opinion back to the neutral position to prevent it from drifting infinitely; the attention intensity \(u(t) \geq 0\) determines the degree to which the existing opinion is self-reinforced; the sensitivity parameter \(\alpha > 0\) determines the speed at which the nonlinear term enters the saturation zone; the external bias \(b(t)\) inputs the environmental evaluation into the system. If \(b(t)\) is close to zero, the damping term will cause the opinion to decay; if there is a small but persistent bias, the self-reinforcement term will gradually amplify the opinion until a stable decision is formed.

**中文。**

两个层次都使用同一种意见动力学模板[R16]。如下公式所示，对一般意见变量 \(o(t)\)，其动力学为其中阻尼系数 \(d>0\) 把意见拉回中性位置，避免意见无限漂移；注意力强度 \(u(t)\geq0\) 决定已有意见被自我强化的程度；灵敏度参数 \(\alpha>0\) 决定非线性项进入饱和区的速度；外部偏置 \(b(t)\) 把环境评价输入系统。如果 \(b(t)\) 接近零，阻尼项会使意见衰减；如果存在微小但持续的偏置，自强化项会逐渐放大意见，直到形成稳定决策。

$$
\dot{o}(t) =
-d\,o(t)
+u(t)\tanh\!\left(\alpha o(t)\right)
+b(t).
$$

**English.**

The dynamics of the opinion of top-level decision-makers can be expressed in the following form, where \(y(t)\) represents the opinion of the top-level decision-makers.

**中文。**

高层决策的意见动力学表示为以下形式，其中\(y(t)\)为高层意见

$$
\dot{y}(t) =
-d_y y(t)
+u_h(t)\tanh\!\left(\alpha_y y(t)\right)
+B(t).
$$

**English.**

In high-level decision-making, the external environment bias is determined by the gap confidence level, while attention is updated by the following formula itself. This Hill-type function depends on \(y^2\), so the opinion sign is symmetrical. When \(y(t)\) approaches zero, attention decays and the vehicle remains cautious; when \(|y(t)|\) increases, \(S_h\) rises, thereby increasing \(u_h(t)\), and a larger \(u_h(t)\) further strengthens the self-reinforcing term in the opinion equation. This feedback loop enables small but continuous preferences to gradually transform into stable gap selection.

**中文。**

在高层决策中，外部环境偏置由gap置信度决定，而注意力是由以下公式自身更新，这个 Hill 型函数依赖 \(y^2\)，因此对意见符号是对称的。当 \(y(t)\) 接近零时，注意力衰减，车辆保持谨慎；当 \(|y(t)|\) 增大时，\(S_h\) 上升，进而提高 \(u_h(t)\)，而更大的 \(u_h(t)\) 又会加强意见方程中的自强化项。这个反馈闭环使微小但连续存在的偏好能够逐渐变成稳定 gap 选择。

$$
\dot{u}_h(t) =
\frac{-u_h(t)+S_h(y(t)^2)}{\tau_h},
$$

$$
S_h(y^2) =
U_{\max}
\frac{(y^2)^n}{K_h^n+(y^2)^n}.
$$

### 高层偏置的计算/High-Level Bias from Gap Confidence

**English.**

At every time step, the high-level module selects the three target-lane vehicles closest to the ego front axle in longitudinal coordinate. Let these vehicles be ordered from front to rear as \((i_1,i_2,i_3)\). The forward candidate gap is the space between \(i_1\) and \(i_2\), and the rear candidate gap is the space between \(i_2\) and \(i_3\). For any candidate gap formed by a front vehicle \(F\) and a rear vehicle \(R\), the gap center and gap velocity are

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

**English.**

The ego-to-gap alignment is defined by the longitudinal position error and the relative speed error:

**中文。**

ego 车与该 gap 的对齐程度由纵向位置误差和相对速度误差表示：

$$
d_g(t) =
x_g(t)-p_e^x(t),
$$

$$
\Delta v_g(t) =
v_g(t)-v_e^x(t).
$$

**English.**

The confidence of a gap is then evaluated using a Gaussian radial-basis function:

**中文。**

gap 的置信度通过高斯径向基函数计算：

$$
C_g(t) =
\exp
\left(
-\frac{d_g(t)^2}{2\sigma_d^2}
-\frac{\Delta v_g(t)^2}{2\sigma_v^2}
\right).
$$

**English.**

When the gap center is close to the ego vehicle and the relative speed is low, the confidence level is higher; when the gap is spatially far apart or the speed matching is poor, the confidence level smoothly decreases. The forward and rear confidences are converted into a signed directional bias:

**中文。**

当 gap 中心接近 ego 车且相对速度较小时，置信度较大；当 gap 在空间上较远或速度匹配较差时，置信度平滑下降。前 gap 与后 gap 的置信度通过有符号差值转换为方向偏置：

$$
B(t) =
C_f(t)-C_r(t).
$$

**English.**

The use of differences here is because they retain both the direction and intensity of the preference. If \(B(t) > 0\), it indicates that the previous gap is more in line with the current ego state; if \(B(t) < 0\), it indicates that the subsequent gap is more in line; if \(B(t) \approx 0\), it means the evidence is still ambiguous. Even if the absolute values of the two confidence levels are both small, the difference between them can still represent a weak but meaningful relative preference. Through the opinion dynamics equation, this weak preference will only accumulate and gradually become more clear when it persists. This keeps the comparison process smooth and makes the decision changes more stable.

**中文。**

这里使用差值是因为差值同时保留了偏好的方向和强度。若 \(B(t)>0\)，说明前 gap 与当前 ego 状态更匹配；若 \(B(t)<0\)，说明后 gap 更匹配；若 \(B(t)\approx0\)，说明证据仍然模糊。即使两个置信度绝对值都较小，二者差值仍可能表示一个微弱但有意义的相对偏好。通过意见动力学方程，这种微弱偏好只有在持续存在时才会积累并逐渐变得明确。这让比较过程保持平滑，使决策变化更稳定。

### 高层意见更新与决策映射/High-Level Opinion Update and Decision Mapping

**English.**

The high-level opinion \(y(t)\) stores the accumulated directional belief. After computing \(B(t)\) and updating \(u_h(t)\), the system integrates the high-level opinion using an explicit time step \(\Delta t\), and update the attention:

**中文。**

高层意见 \(y(t)\) 存储已经积累的方向信念。计算 \(B(t)\) 并更新 \(u_h(t)\) 后，系统使用步长 \(\Delta t\) 对高层意见进行离散积分，并更新注意力：

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

$$
u_{h,k+1} =
u_{h,k}
+\Delta t
\frac{-u_{h,k}+S_h(y_k^2)}{\tau_h}.
$$

**English.**

The top-level decision mapping is in the form of a threshold, and the decision is made according to the following formula.:

**中文。**

高层决策映射采用阈值形式，由以下公式做出决策：

$$
\mathrm{decision}_k =
\begin{cases}
\mathrm{forward}, & y_k>\theta_y,\\
\mathrm{rear}, & y_k<-\theta_y,\\
\mathrm{wait}, & |y_k|\leq\theta_y.
\end{cases}
$$

**English.**

The existence of the waiting interval prevents the gap selected from frequently changing when \(C_f\) and \(C_r\) are close, allowing vehicles to leave the neutral zone and make a choice only after sufficient evidence has accumulated. Therefore, this mechanism is smoother than the direct instantaneous comparison. Even if the group selection of the vehicle closest to the ego vehicle changes, the decision will not suddenly change due to a jump in confidence, which is more in line with the actual driving behavior logic.

**中文。**

等待区间的存在避免了 \(C_f\) 与 \(C_r\) 接近时所选 gap 频繁切换，使车辆只有在证据积累到足够程度后才从中性区离开并做出选择。因此，该机制比直接瞬时比较更加平滑。即使离ego车最近的车组选择发生改变，决策也不会因为置信度的跳变而瞬间改变，这也更符合实际驾驶的行为逻辑。

## 底层控制系统/Low-Level Control System

**English.**

After the high-level module selects the candidate gap, the low-level module decides how the ego vehicle should enter that gap. The low-level module first evaluates the physical feasibility of the selected gap, then determines the attention intensity, updates the low-level opinion \(z(t)\), and finally converts \(z(t)\) into continuous target points. This design makes the merging action gradual: as the low-level opinion strengthens, the target points gradually move from the original lane reference position to within the selected gap.

**中文。**
高层模块选出候选 gap 后，底层模块决定 ego 车如何进入该 gap。底层通过先评价所选 gap 的物理可行性，再确定注意力强度，更新底层意见 \(z(t)\)，最后把 \(z(t)\) 转换为连续目标点。这样的设计使并道动作是渐进的：随着底层意见增强，目标点从原车道参考位置逐渐移动到所选 gap 内。

### 底层偏置的计算/Low-Level Bias

**English.**

For a selected gap formed by a front vehicle \(F\) and a rear vehicle \(R\), the physical gap length and  gap-rate term is

**中文。**

对由前车 \(F\) 和后车 \(R\) 构成的已选 gap，其物理长度以及gap变化率为

$$
g(t) =
p_F^x(t)-p_R^x(t).
$$

$$
\dot{g}(t) =
v_F^x(t)-v_R^x(t).
$$

**English.**

The low-level bias evaluates whether the gap is sufficiently large and whether it is opening or closing:

**中文。**

底层偏置用于评价该 gap 是否足够大，以及 gap 正在变大还是变小：

$$
b(t) =
k_g\left[g(t)-g_{\mathrm{safe}}\right]
+k_v\dot{g}(t).
$$

**English.**

A larger gap makes \(b(t)\) more positive. An opening gap, where \(\dot{g}(t)>0\), also increases \(b(t)\). A small or closing gap decreases \(b(t)\), delaying the growth of the low-level merge intention. This formula is intentionally interpretable: \(g(t)-g_{\mathrm{safe}}\) measures spatial feasibility, while \(\dot{g}(t)\) measures whether the situation is improving or worsening.

**中文。**

gap 越大，\(b(t)\) 越偏正；如果 gap 正在扩大，即 \(\dot{g}(t)>0\)，\(b(t)\) 也会增大。相反，小 gap 或正在闭合的 gap 会降低 \(b(t)\)，从而推迟底层并道意愿的增长。该公式的优点是含义直观：\(g(t)-g_{\mathrm{safe}}\) 衡量空间可行性，\(\dot{g}(t)\) 衡量交通形势正在改善还是恶化。

table 1 : The influence of the gap state on the bias/表1：间隙状态对偏置的影响

| Gap condition / gap 状态 | Mathematical effect / 数学影响 | Control meaning / 控制含义 |
| :--- | :--- | :--- |
| Large and opening / 大且扩大 | \(g>g_{\mathrm{safe}},\ \dot{g}>0\) | Positive bias, stronger merge tendency; 偏置更正，并道倾向增强。 |
| Large but closing / 大但缩小 | \(g>g_{\mathrm{safe}},\ \dot{g}<0\) | Moderate bias, cautious merge tendency; 偏置受抑制，需要谨慎。 |
| Small and opening / 小但扩大 | \(g<g_{\mathrm{safe}},\ \dot{g}>0\) | Waiting may be appropriate; 可继续观察等待。 |
| Small and closing / 小且缩小 | \(g<g_{\mathrm{safe}},\ \dot{g}<0\) | Negative bias, merge should be suppressed; 偏置更负，应抑制并道。 |

### 强化学习决定注意力/The attention obtained through reinforcement learning

**English.**

The underlying attention \(u(t)\) regulates the intensity of the self-reinforcement of the underlying opinions. However, the optimal timing of attention is difficult to be fully designed manually because the system simultaneously contains nonlinear opinion dynamics, moving vehicles, safety avoidance, input saturation, and delay consequences. If attention is increased too early, the ego vehicle may overcommit when the gap is not yet safe; if attention is increased too late, the vehicle may miss the merging opportunity and behave too conservatively. Therefore, reinforcement learning is used to learn the temporal sequence of attention from the reward feedback. At the same time, the learned strategy does not replace the model controller, but only outputs the attention variables in the opinion dynamics. Such an action space is lower-dimensional and easier to interpret.

**中文。**

底层注意力 \(u(t)\) 控制底层意见自强化的强度。但注意力的最佳时机很难完全手工设计，因为系统同时包含非线性意见动力学、移动车辆、安全避障、输入饱和和延迟后果。如果注意力过早增大，ego 车可能在 gap 尚不安全时过度承诺；如果注意力过晚增大，车辆又可能错过并道机会并表现得过于保守。因此，强化学习被用于从 reward 反馈中学习注意力时序，同时学习策略并不替代模型控制器，而只是输出意见动力学中的注意力变量。这样的动作空间更低维，也更容易解释。

### SAC 强化学习设计/SAC Reinforcement Learning Design

**English.**

Soft Actor-Critic is a off-policy actor-critic algorithm suitable for continuous action spaces. It learns the random policy \(\pi_\phi(a|s)\), two soft Q functions, and the entropy temperature parameter. Its optimization objective is to balance between the expected return and the policy entropy:
The entropy term encourages exploration, which is crucial for the lane-changing task because a useful policy not only needs to learn to stay safe but also needs to learn when to actively make a lane change.

**中文。**

Soft Actor-Critic 是一种适用于连续动作空间的离策略 actor-critic 算法。它学习随机策略 \(\pi_\phi(a|s)\)、两个 soft Q 函数和熵温度参数。其优化目标在期望回报和策略熵之间进行平衡：
熵项鼓励探索，这对并道任务很重要，因为有用的策略不仅要学会保持安全，还要学会何时主动承诺并道。

$$
J(\pi) =
\sum_{t}
\mathbb{E}_{(s_t,a_t)\sim\rho_{\pi}}
\left[
r(s_t,a_t)
+\alpha_{\mathrm{SAC}}\mathcal{H}\left(\pi(\cdot|s_t)\right)
\right].
$$

**English.**

The low-level reinforcement-learning state describes the relative geometry and velocity between the ego vehicle and the front and rear vehicles of the selected gap. The typical situation is as follows. Here \(F\) and \(R\) denote the front and rear vehicles of the selected gap. 

**中文。**

底层强化学习状态描述 ego 车与所选 gap 前后两车之间的相对位置和相对速度。典型状态如下。其中 \(F\) 与 \(R\) 分别表示所选 gap 的前车和后车。

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

**English.**

The policy output is the low-level attention:

**中文。**

策略输出为底层注意力：

$$
a_t =
u(t),
\qquad
u(t)\in[u_{\min},u_{\max}].
$$

**English.**

The reward is designed to encourage timely merging, discourage hesitation, penalize oscillatory motion, and strongly penalize collision. A representative per-step reward is

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

**English.**

The definition of the lane-changing progress is the normalized ratio:

**中文。**

其中并道进度定义为归一化比例：

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

**English.**

The opportunity term \(O_t\) represents whether the selected gap is currently suitable for merging. The progress terms reward moving toward the target lane, especially when a useful opportunity exists. The hesitation terms penalize remaining far from the target lane, so the policy cannot receive a high score by simply waiting. The action-smoothness and direction-flip terms reduce repeated acceleration or lateral-direction reversals. The safety term is continuous before collision, while the collision term is a large terminal penalty. The success bonus decreases with time, for example

**中文。**

其中 \(O_t\) 表示当前所选 gap 是否构成有利并道机会。progress 项奖励车辆向目标车道推进，尤其在机会存在时给予更强鼓励。hesitation 项惩罚车辆长期停留在原车道附近，避免策略通过“不动”获得高分。动作平滑项和横向换向项减少反复加减速或左右方向抖动。安全项在真正碰撞前连续惩罚距离过近，碰撞项则是强终止惩罚。成功奖励随时间下降，例如

$$
R_{\mathrm{success}}(t) =
100-2t,
$$

table 2: The various meanings of the rewards/表2：奖励的各项含义

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

### 底层意见及其对控制点的影响/Low-Level Opinion and Target Point 

**English.**

The low-level opinion \(z(t)\) represents the degree of merge commitment for the selected gap. Its dynamics and The discrete update are

**中文。**

底层意见 \(z(t)\) 表示 ego 车对所选 gap 的并道承诺程度。其动力学以及离散更新形式为

$$
\dot{z}(t) =
-d_z z(t)
+u(t)\tanh\!\left(\alpha_z z(t)\right)
+b(t).
$$

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

**English.**

A low or negative \(z(t)\) keeps the desired target point close to the original lane or a cautious reference. A high positive \(z(t)\) moves the target point toward the selected gap in the target lane. To keep this transition bounded, \(z(t)\) is mapped through a smooth gate:

**中文。**

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

## Actual Control Input / 实际控制输入

**English.**

The final control layer converts the desired target point into actual vehicle input. This layer consists of three parts: a safety obstacle avoidance item that pushes the ego vehicle away from the nearby target lane vehicles, a tracking error pointing to the desired target point, and a reverse solution of the bicycle model that maps the desired front axle acceleration to longitudinal acceleration and angular rate of change. Through explicit safety constraints and vehicle kinematics expressions, the entire system becomes safer and more interpretable.

**中文。**

最终控制层把期望目标点转换为实际车辆输入。该层包含三个部分：将 ego 车从附近目标车道车辆旁推开的安全避障项，指向期望目标点的跟踪误差，以及把期望前轴加速度映射为纵向加速度和转角变化率的自行车模型反解。通过显式的安全约束和车辆运动学表达，使整个系统更加安全和可解释。

### 避障项设计/Safety-Avoidance Term

**English.**

For each surrounding target-lane vehicle \(j\), define the relative vector from that vehicle to the ego front axle as

**中文。**

对每一辆周围目标车道车辆 \(j\)，定义从该车指向 ego 前轴点的相对向量为

$$
\mathbf{r}_j(t) =
\mathbf{p}_e(t)-\mathbf{p}_j(t),
\qquad
d_j(t) =
\lVert \mathbf{r}_j(t)\rVert_2.
$$

**English.**

The safety-avoidance acceleration is constructed as a repulsive field:

**中文。**

安全避障加速度构造为排斥场：

$$
\mathbf{u}_c(t) =
\sum_j
k_c
\max\left(0,\frac{d_{\mathrm{safe}}-d_j(t)}{d_{\mathrm{safe}}}\right)^2
\frac{\mathbf{r}_j(t)}{d_j(t)}.
$$

**English.**

When all vehicles are farther than \(d_{\mathrm{safe}}\), this term is zero. When a vehicle enters the safety region, the repulsive magnitude grows quadratically as distance decreases. This term does not replace collision checking; instead, it provides a continuous pre-collision correction that can steer the ego vehicle away before a hard collision boundary is reached.

**中文。**

当所有车辆距离都大于 \(d_{\mathrm{safe}}\) 时，该项为零；当某辆车进入安全区域后，排斥强度随距离减小而二次增大。该项并不替代碰撞检测，而是在真正到达硬碰撞边界前提供连续修正，使 ego 车提前远离危险区域。

### 总目标点与控制误差设计/Target Point and Tracking Error

**English.**

The selected-gap target point is usually placed near the longitudinal center of the gap and at the target-lane center. The original-lane reference point can be placed ahead of the ego vehicle along the original lane:

**中文。**

所选 gap 的目标点通常放在 gap 纵向中心附近，并位于目标车道中心线上；原车道参考点可放在 ego 车前方一定前视距离处：

$$
\mathbf{p}_G^{\star}(t) =
\begin{bmatrix}
x_g(t) \\
y_T
\end{bmatrix}.
$$

$$
\mathbf{p}_O^{\star}(t) =
\begin{bmatrix}
p_e^x(t)+\ell_{\mathrm{look}} \\
y_O
\end{bmatrix}.
$$

**English.**

The opinion-weighted target point and The tracking error are:

**中文。**

由意见加权得到的总目标点和跟踪误差为：

$$
\mathbf{p}^{\star}(t) =
\left[1-\lambda_z(t)\right]\mathbf{p}_O^{\star}(t)
+\lambda_z(t)\mathbf{p}_G^{\star}(t).
$$

$$
\mathbf{e}_z(t) =
\mathbf{p}^{\star}(t)-\mathbf{p}_e(t).
$$

**English.**

This construction gives the low-level opinion a direct geometric meaning. When \(\lambda_z(t)\) is close to zero, the controller behaves like a lane-keeping controller. When \(\lambda_z(t)\) approaches one, the controller behaves like a gap-entry controller.

**中文。**

这种构造使底层意见具有直接几何意义。当 \(\lambda_z(t)\) 接近零时，控制器近似表现为车道保持控制器；当 \(\lambda_z(t)\) 接近一时，控制器逐渐表现为 gap 并入控制器。

### 最终物理控制输入设计/Final Physical Input Design

**English.**

The desired front-axle acceleration combines proportional position tracking, velocity damping, and safety avoidance:

**中文。**

期望前轴加速度由位置比例跟踪、速度阻尼和安全避障共同组成：

$$
\mathbf{u}_{\mathrm{total}}(t) =
k_p\mathbf{e}_z(t)
-k_v\mathbf{v}_e^f(t)
+\mathbf{u}_c(t).
$$

**English.**

This vector is then converted into physical inputs. Let the front-axle velocity direction and its normal direction be

**中文。**

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

**English.**

The longitudinal acceleration can be approximated by projection onto \(\mathbf{t}_e\):

**中文。**

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

**English.**

The steering-rate command is generated from the lateral component:

**中文。**

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

**English.**

This kind of PID-like structure function is to provide a clear and stable mapping from the opinion-driven target point to the executable vehicle input, while avoiding unrealistic acceleration and steering changes through input clipping.

**中文。**

这种类 PID 结构作用是提供从意见驱动目标点到可执行车辆输入的清晰稳定映射，同时通过输入裁剪避免不现实的加速度和转向变化。

# 实验设计/Experimental Design

**English.**

The experiment was organized in a way from simple to complex. The single gap environment isolated the underlying issues: testing the learning effect of RL when the target gap has been determined, and whether the controller can learn when to increase attention and commit to merging. The multi-gap environment introduced high-level decisions: when multiple dynamic gaps exist simultaneously, can the system select the appropriate local gap to avoid excessive switching and safely execute merging. The specific parameters of the experiment are listed in the appendix.

**中文。**

实验按照由简单到复杂的方式组织。单 gap 环境隔离底层问题：测试当目标 gap 已经确定时，RL的学习效果以及控制器能否学会何时提高注意力并承诺并道。多 gap 环境引入高层决策：当多个动态 gap 同时存在时，系统能否选择合适的局部 gap，避免过度切换，并安全执行并道。实验具体参数表在附录中。


## 单gap并入实验/Single-Gap Merging Experiment


## 多gap并入实验/Multi-Gap Merging Experiment
# 结果和讨论/Results and discussion
# 结论/Conclusions
# 参考文献/References

R1. Qu, D., Zhang, K., Song, H., Jia, Y., & Dai, S. (2022). Analysis and Modeling of Lane-Changing Game Strategy for Autonomous Driving Vehicles. *IEEE Access, 10*, 69531-69542. https://doi.org/10.1109/access.2022.3187431

R2. Deng, Z., Hu, W. S., Sun, C., Chu, D., Huang, T., Li, W., Yu, C., Pirani, M., Cao, D., & Khajepour, A. (2025). Eliminating Uncertainty of Driver's Social Preferences for Lane Change Decision-Making in Realistic Simulation Environment. *IEEE Transactions on Intelligent Transportation Systems, 26*, 1583-1597. https://doi.org/10.1109/tits.2024.3512784

R3. Zhang, K., Qu, D., Song, H., Wang, T., & Dai, S. (2022). Analysis of Lane-Changing Decision-Making Behavior and Molecular Interaction Potential Modeling for Connected and Automated Vehicles. *Sustainability, 14*, 11049. https://doi.org/10.3390/su141711049

R4. Yang, L., Cao, C., Zhao, Q., Yang, J., & Fan, A. (2026). Lane-Changing Strategy for Autonomous Vehicle With Adaptive Adjustment of Decision-Making Preference Based on Game Theory. *IEEE Transactions on Vehicular Technology, 75*, 130-144. https://doi.org/10.1109/tvt.2025.3592221

R5. Huang, T., Fu, R., Sun, Q., Deng, Z., Liu, Z., Jin, L., & Khajepour, A. (2024). Driver lane change intention prediction based on topological graph constructed by driver behaviors and traffic context for human-machine co-driving system. *Transportation Research Part C: Emerging Technologies, 160*, 104497. https://doi.org/10.1016/j.trc.2024.104497

R6. Yang, K., Li, S., Wang, M., & Tang, X. (2025). Interactive Decision-Making Integrating Graph Neural Networks and Model Predictive Control for Autonomous Driving. *IEEE Transactions on Intelligent Transportation Systems, 26*, 6991-7005. https://doi.org/10.1109/tits.2025.3532936

R7. Jiang, Y., Man, Z., Wang, Y., & Yao, Z. (2024). Cooperative lane-changing for connected autonomous vehicles merging into dedicated lanes in mixed traffic flow. *Expert Systems with Applications, 252*, 124163. https://doi.org/10.1016/j.eswa.2024.124163

R8. Monteiro, F. V., & Ioannou, P. A. (2023). Safe autonomous lane changes and impact on traffic flow in a connected vehicle environment. *Transportation Research Part C: Emerging Technologies, 151*, 104138. https://doi.org/10.1016/j.trc.2023.104138

R9. Lv, K., Pei, X., Chen, C., & Xu, J. (2022). A Safe and Efficient Lane Change Decision-Making Strategy of Autonomous Driving Based on Deep Reinforcement Learning. *Mathematics, 10*, 1551. https://doi.org/10.3390/math10091551

R10. Deng, H., Zhao, Y., Wang, Q., & Nguyen, A.-T. (2023). Deep Reinforcement Learning Based Decision-Making Strategy of Autonomous Vehicle in Highway Uncertain Driving Environments. *Automotive Innovation, 6*, 438-452. https://doi.org/10.1007/s42154-023-00231-6

R11. He, X., Yang, H., Hu, Z., & Lv, C. (2023). Robust Lane Change Decision Making for Autonomous Vehicles: An Observation Adversarial Reinforcement Learning Approach. *IEEE Transactions on Intelligent Vehicles, 8*, 184-193. https://doi.org/10.1109/tiv.2022.3165178

R12. Zhou, W., Chen, D., Yan, J., Li, Z., Yin, H., & Ge, W. (2022). Multi-agent reinforcement learning for cooperative lane changing of connected and autonomous vehicles in mixed traffic. *Autonomous Intelligent Systems, 2*. https://doi.org/10.1007/s43684-022-00023-5

R13. Zhang, J., Chang, C., Zeng, X., & Li, L. (2023). Multi-Agent DRL-Based Lane Change With Right-of-Way Collaboration Awareness. *IEEE Transactions on Intelligent Transportation Systems, 24*, 854-869. https://doi.org/10.1109/tits.2022.3216288

R14. Bi, X., He, M., & Sun, Y. (2025). Mix Q-Learning for Lane Changing: A Collaborative Decision-Making Method in Multi-Agent Deep Reinforcement Learning. *IEEE Transactions on Vehicular Technology, 74*, 8664-8677. https://doi.org/10.1109/tvt.2025.3533006

R15. Li, A., Chavez Armijos, A. S., & Cassandras, C. G. (2025). Robust optimal lane-changing control for Connected Autonomous Vehicles in mixed traffic. *Automatica, 174*, 112169. https://doi.org/10.1016/j.automatica.2025.112169

R16. Bizyaeva, A., Franci, A., & Leonard, N. E. (2023). Nonlinear Opinion Dynamics with Tunable Sensitivity. IEEE Transactions on Automatic Control, 68, 1415-1430. https://doi.org/10.1109/TAC.2022.3159527








