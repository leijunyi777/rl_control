# 标题/Title

**English.**

A Two-Layer Decision-Making System for Vehicle Lane Changing Based on Reinforcement Learning

**中文。**

基于强化学习的车辆并道双层决策系统

# 作者/Author

Junyi Lei/雷君毅 / student ID： 14226506 / Master of Science in Robotics / Year of submission: 2026 

# 摘要/Abstract

**English.**

Lane changing is a fundamental problem in autonomous driving. This dissertation proposes a two-layer decision-making and control framework for autonomous vehicles merging into a target lane. The framework separates strategic decision dynamics from physical execution. The high-level decision-making layer contains two opinion variables: a longitudinal opinion that selects a local gap from nearby target-lane vehicles, and a lateral opinion that determines whether the ego vehicle should commit to the selected gap. The longitudinal module observes the three target-lane vehicles closest to the ego vehicle, forms two candidate gaps, evaluates their relative confidence, and accumulates a temporally smoothed preference through nonlinear opinion dynamics. The lateral module evaluates the selected gap using its size and rate of change, uses a Soft Actor-Critic (SAC) policy to generate the attention intensity, and maps the resulting opinion to continuous target points. The low-level controller then combines safety avoidance, target tracking, and a proportional-integral-derivative (PID)-like physical input law to generate acceleration and steering-rate commands.

Simulation experiments are implemented in Python in two environments. The single-gap environment contains one front vehicle, one rear vehicle, and one merging ego vehicle with a randomized initial longitudinal position; the rear vehicle creates a yielding opportunity after 20 s. The multi-gap environment contains five target-lane vehicles and four dynamically changing gaps over a 40 s horizon. In the single-gap tests, the SAC attention policy achieves an average episodic reward of 233.4 compared with 115.8 for the radial-basis-function (RBF) baseline, completes the task in 6.9 s rather than 29.45 s, and maintains zero collisions. In multi-gap deployment, the single-gap policy maintains a success rate of approximately 97%, and the opinion-based high-level selector improves mean reward by 55.3% over the simple max-score selector.

**中文。**

车辆并道是自动驾驶中的基本问题。本文提出一种用于自动驾驶车辆并入目标车道的双层决策与控制框架。该框架将策略决策动力学与物理执行分离。高层决策层包含两个意见变量：纵向意见用于从目标车道附近车辆中选择局部 gap，横向意见用于判断 ego 车辆是否应当向所选 gap 执行并道。纵向模块观测距离 ego 车辆最近的三辆目标车道车辆，形成两个候选 gap，通过相对置信度评价其可行性，并利用非线性意见动力学累积具有时间平滑性的偏好。横向模块根据所选 gap 的大小和变化速率评价并道条件，使用 Soft Actor-Critic (SAC) 策略生成注意力强度，并把所得意见映射为连续目标点。底层控制器进一步结合安全避障、目标跟踪和类似 proportional-integral-derivative (PID) 的物理输入律，生成加速度和转角变化率命令。

本文在 Python 中构建两个仿真环境进行验证。单 gap 环境包含一辆前车、一辆后车和一辆随机初始纵向位置的 ego 并道车辆，后车在 20 s 后开始形成让行机会。多 gap 环境包含五辆目标车道车辆和四个动态变化的 gap，仿真时长为 40 s。在单 gap 测试中，SAC 注意力策略的平均 episodic reward 为 233.4，高于 radial-basis-function (RBF) 基线的 115.8；任务完成时间为 6.9 s，而 RBF 基线为 29.45 s，并且两者均保持零碰撞。在多 gap 部署中，单 gap 策略保持约 97% 的成功率，基于意见动力学的高层选择器相比简单 max-score 选择器将平均 reward 提高了 55.3%。


# Declaration of originality

I hereby confirm that no portion of the work referred to in the thesis has been submitted in support of an application for another degree or qualification of this or any other university or other institute of learning

本人特此确认，本文中所述的任何部分内容均未被提交用于申请本校或其他任何大学或学习机构的其他学位或资格。

# Copyright statement

# 简介/Introduction

## 背景与动机/Background and Motivation

### 并道与换道交互中的策略决策方法/Decision-making strategies for merge and lane-change interaction

**English**

Lane-changing decisions are representative problems in autonomous driving. A lane change changes the available gaps, right-of-way expectations, and future responses of surrounding vehicles. Recent research has therefore treated lane changing as an interactive decision problem rather than a purely geometric path-planning task. Game-theoretic models are widely used because they explicitly describe the strategic relationship between the ego vehicle and surrounding vehicles. Lane-changing intention, collision probability, and dynamic risk can be incorporated into a payoff function to evaluate whether a manoeuvre is beneficial or dangerous [R1, R2]. Physics-inspired methods, such as molecular interaction potential models, further express attraction and repulsion between vehicles as continuous interaction forces [R3]. Other studies consider incomplete information and social driving preferences, because surrounding human-driven vehicles may behave cooperatively, aggressively, or uncertainly in merging conflicts [R4].

Learning-based interactive prediction provides a second route. Graph neural networks and attention mechanisms represent vehicles as nodes and inter-vehicle influence as edges or attention weights. This representation is suitable for lane changing, where the number and importance of surrounding vehicles vary over time. Existing studies use topological graphs with driving behaviour and traffic context to predict lane-changing intentions [R5], while plan-aware graph attention models predict how surrounding vehicles respond to candidate ego intentions before generating a trajectory [R6]. These studies show that interaction modelling should estimate not only where other vehicles will move, but also how they may respond to the ego decision. Cooperative control methods provide a third route. In connected autonomous driving, surrounding vehicles can create gaps, coordinate lane-changing order, or maintain fleet stability [R7, R8]. Together, these studies motivate a hierarchical framework that combines strategic interaction modelling, adaptive decision dynamics, and safe execution.

**中文**

并道与换道决策是自动驾驶中的代表性问题。一次换道会改变可用间隙、路权预期和周围车辆的未来响应。因此，近年研究逐渐将换道视为交互决策问题，而不是单纯的几何路径规划任务。博弈论模型被广泛使用，是因为它能够显式描述 ego 车辆与周围车辆之间的策略关系。换道意图、碰撞概率和动态风险可以被纳入收益函数，用于评价某个机动是有利还是危险 [R1, R2]。分子相互作用势等物理启发方法进一步将车辆间吸引与排斥表达为连续交互力 [R3]。其他研究还考虑不完全信息和社会驾驶偏好，因为在人类驾驶车辆参与的并道冲突中，周车可能表现为合作、激进或不确定 [R4]。

学习型交互预测提供了第二条路线。图神经网络和注意力机制将车辆表示为节点，将车辆间影响表示为边或注意力权重。该表示适合换道场景，因为周围车辆数量和重要性会随时间变化。已有研究使用拓扑图结合驾驶行为和交通上下文预测驾驶人换道意图 [R5]，也有研究利用计划感知图注意力模型预测周车对 ego 候选意图的响应，再生成轨迹 [R6]。这些研究说明，交互建模不应只估计其他车辆会去哪里，还应估计它们会如何响应 ego 决策。协同控制方法提供了第三条路线。在网联自动驾驶中，周围车辆可以主动创造 gap、协调换道顺序或维持车队稳定 [R7, R8]。总体而言，这些研究为结合策略交互建模、自适应决策动力学和安全执行的层级框架提供了动机。

### 强化学习用于自适应决策动力学/Reinforcement learning for adaptive decision dynamics

**English**

Lane-changing decisions are sequential, uncertain, and feedback driven. A vehicle must decide when to wait, when to merge, and how strongly to commit. Fixed-parameter controllers may work in static cases, but they often lack flexibility when traffic states change or when the system must move between cautious waiting and active merging. Deep reinforcement learning (DRL) provides a mechanism for learning decision strategies through repeated interaction. Existing studies have integrated safety rules, future-risk assessment, reward shaping, and robust observation modelling into DRL lane-changing strategies [R9-R11]. These studies suggest that reinforcement learning (RL) should not be used as an unconstrained black box; instead, it should be constrained by interpretable states, safety-aware rewards, and model-based structure.

Multi-agent reinforcement learning (MARL) further allows connected or autonomous vehicles to learn from neighbouring vehicles while considering system-level outcomes such as traffic efficiency, comfort, and safety [R12]. Right-of-way coordination and Mix Q-learning show that lane changing can balance individual and group benefits [R13, R14]. For this dissertation, these studies support a specific use of RL: the learned policy does not directly output lane-changing motion, but adjusts the attention parameter inside a nonlinear opinion-dynamics model. The decision remains interpretable, while the timing and strength of commitment can adapt to the environment.

**中文**

换道决策具有序贯性、不确定性和反馈驱动特征。车辆需要决定何时等待、何时并道，以及以多强的程度执行并道。固定参数控制器在静态场景中可能有效，但当交通状态变化，或系统需要在谨慎等待和主动并道之间切换时，往往缺乏灵活性。Deep reinforcement learning (DRL) 提供了一种通过反复交互学习决策策略的机制。已有研究将安全规则、未来风险评估、reward shaping 和鲁棒观测建模融入 DRL 换道策略 [R9-R11]。这些研究表明，reinforcement learning (RL) 不应作为无约束黑箱使用，而应由可解释状态、安全感知奖励和模型结构约束。

Multi-agent reinforcement learning (MARL) 进一步允许网联或自动驾驶车辆从邻近车辆行为中学习，同时考虑交通效率、舒适性和安全性等系统级结果 [R12]。路权协同和 Mix Q-learning 表明，换道任务可以平衡个体收益与群体收益 [R13, R14]。对于本文而言，这些研究支持一种具体用法：学习策略不直接输出换道运动，而是调节非线性意见动力学模型中的注意力参数。这样既保持决策可解释，又能让并道承诺的时机和强度根据环境自适应变化。

### 评价维度：从效率到安全、舒适与适应性/Evaluation dimensions: from efficiency to safety, comfort, and adaptability

**English**

The evaluation of lane-changing decisions has shifted from single efficiency metrics to multidimensional assessment. Average speed, travel time, traffic volume, and lane-changing success rate remain important, but they are insufficient for safety-critical autonomous driving. A rapid lane change may still be unsafe, uncomfortable, or disruptive to traffic flow. Recent studies therefore evaluate collision risk, minimum distance, braking feasibility, safety violation rate, acceleration, traffic-flow stability, and robustness under perception uncertainty [R7, R8, R10, R15]. This project follows these evaluation dimensions by measuring task progress, completion time, collision behaviour, minimum distance, reward, and switching stability.

**中文**

换道决策的评价已经从单一效率指标转向多维评价。平均速度、旅行时间、通行量和换道成功率仍然重要，但它们不足以支撑安全关键的自动驾驶系统。一次快速换道仍可能是不安全、不舒适或扰动交通流的。因此，近期研究开始评价碰撞风险、最小距离、制动可行性、安全违规率、加速度、交通流稳定性以及感知不确定性下的鲁棒性 [R7, R8, R10, R15]。本项目参考这些评价维度，测试任务进度、完成时间、碰撞行为、最小距离、reward 和切换稳定性。

## Aims and Objectives / 研究目标与具体目标

**English**

The aim of this project is to develop, integrate, and evaluate a two-layer decision-making and control framework for interactive autonomous lane merging. The framework uses nonlinear opinion dynamics to represent strategic commitment, RL to adapt the lateral attention intensity, and a safety-aware low-level controller to convert opinion states into physical vehicle inputs.

This project has five objectives. First, it formulates a continuous opinion-dynamics model for longitudinal gap selection and lateral merging commitment. Second, it trains a SAC agent to adjust the lateral attention parameter of the opinion model. Third, it designs a low-level execution controller that combines safety avoidance with front-axle target tracking. Fourth, it implements the complete controller in a Python simulation framework. Fifth, it compares the proposed method with RBF and max-score baselines to evaluate efficiency, safety, and generalization.

**中文**

本项目的目标是开发、集成并评价一个面向交互式自动并道的双层决策与控制框架。该框架使用非线性意见动力学表示策略承诺，使用 RL 自适应调节横向注意力强度，并使用安全感知底层控制器将意见状态转化为物理车辆输入。

本项目包含五个具体目标。第一，建立用于纵向 gap 选择和横向并道承诺的连续意见动力学模型。第二，训练 SAC 智能体来调节意见模型中的横向注意力参数。第三，设计结合安全避障与前轴目标跟踪的底层执行控制器。第四，在 Python 仿真框架中实现完整控制器。第五，与 RBF 和 max-score 基线对比，评价效率、安全性和泛化能力。

## 情景与建模/Scenario and Modeling

**English**

The studied scenario is a lane-merging task in which one ego vehicle travels in the original lane and attempts to merge into a target lane containing continuous traffic. The target-lane vehicles are assumed to be observable through their longitudinal positions and velocities. The ego vehicle does not control these vehicles; instead, it observes their motion, selects a feasible local gap, and generates its own control input. In the single-gap environment, the target lane contains one front vehicle and one rear vehicle. In the multi-gap environment, it contains five vehicles, which form four candidate physical gaps. The ego initial longitudinal position is randomized to test whether the controller can tolerate different entry alignments.

The vehicle model is represented by a front-axle kinematic bicycle abstraction. The ego state contains its planar position, heading, and forward speed, while the physical input consists of longitudinal acceleration and steering-rate commands. The front-axle point is used for decision and tracking because it gives a direct geometric reference for entering a target-lane gap. The high-level decision layer determines the desired gap and merge commitment, and the low-level controller converts the corresponding target point into actual acceleration and angular-rate changes under collision-avoidance constraints.

**中文**

本文研究的场景是一辆 ego 车辆在原车道行驶，并尝试并入包含连续交通流的目标车道。假设目标车道车辆的纵向位置和速度可被观测。ego 车辆不控制这些车辆，而是观测其运动，选择可行局部 gap，并生成自身控制输入。在单 gap 环境中，目标车道包含一辆前车和一辆后车。在多 gap 环境中，目标车道包含五辆车，形成四个候选物理 gap。ego 初始纵向位置被随机化，用于测试控制器对不同并入对齐状态的适应性。

车辆模型采用前轴点运动学自行车抽象。ego 状态包含平面位置、航向角和前向速度，物理输入由纵向加速度和转角变化率命令组成。前轴点用于决策和跟踪，是因为它能够为进入目标车道 gap 提供直接几何参考。高层决策层确定期望 gap 和并道承诺，底层控制器在避碰约束下把对应目标点转化为实际加速度和角速度变化。

![Lane-Merging Scenario Illustration](./image/chapter_1_3_lane_change_scene.png)

Fig: Lane-Merging Scenario Illustration

# 方法/Methods

**English.**

The proposed system contains two layers. The high-level decision-making layer includes both longitudinal and lateral opinion dynamics. The longitudinal opinion \(y(t)\) chooses whether the ego vehicle should bias its attention toward a forward or rear candidate gap, and the lateral opinion \(z(t)\) determines how strongly the ego vehicle should move from the original lane toward the selected gap. The low-level control layer does not make a new strategic decision; it converts the target implied by the high-level opinions into safety-aware physical inputs.

**中文。**

本文系统包含两个层次。高层决策层同时包含纵向和横向意见动力学。纵向意见 \(y(t)\) 判断 ego 车辆应更偏向前方候选 gap 还是后方候选 gap；横向意见 \(z(t)\) 判断 ego 车辆应以多强的程度从原车道向所选 gap 移动。底层控制层不再做新的策略决策，而是把高层意见给出的目标转化为安全感知的物理输入。

![Control system structure block diagram](./image/System_block_diagram.png)

Fig1：控制系统结构框图/Control system structure block diagram

## 高层决策/High-Level Decision-Making

### 意见动力学及自我更新/Opinion Dynamics and Self-Update

**English.**

The same continuous opinion-dynamics template [R16] is used for both high-level opinions. Let \(o(t)\in\mathbb{R}\) be a scalar opinion, \(u(t)\geq 0\) be its scalar attention intensity, \(b(t)\in\mathbb{R}\) be an external environmental bias, \(d>0\) be a damping coefficient, and \(\alpha>0\) be a sensitivity coefficient. The opinion evolves as

**中文。**

两个高层意见都使用同一个连续意见动力学模板 [R16]。设 \(o(t)\in\mathbb{R}\) 为标量意见，\(u(t)\geq0\) 为对应的标量注意力强度，\(b(t)\in\mathbb{R}\) 为外部环境偏置，\(d>0\) 为阻尼系数，\(\alpha>0\) 为灵敏度系数。意见演化为

$$
\dot{o}(t)=-d\,o(t)+u(t)\tanh\!\left(\alpha o(t)\right)+b(t).
$$

**English.**

The damping term pulls the opinion toward neutrality, the hyperbolic-tangent term provides bounded self-reinforcement, and the bias term injects environmental evidence. For the longitudinal opinion, the scalar attention \(u_h(t)\) is updated by a Hill-type self-update law,

**中文。**

阻尼项将意见拉向中性位置，双曲正切项提供有界自强化，偏置项注入环境证据。对于纵向意见，标量注意力 \(u_h(t)\) 由 Hill 型自更新律更新：

$$
\dot{u}_h(t)=\frac{-u_h(t)+S_h(y(t)^2)}{\tau_h},\quad
S_h(y^2)=U_{\max}\frac{(y^2)^n}{K_h^n+(y^2)^n}.
$$

### 纵向偏置设计/Longitudinal Direction Bias Design

**English.**

At each time step, the high-level module selects the three target-lane vehicles closest to the ego front axle in the longitudinal coordinate. Let their scalar longitudinal positions be ordered from front to rear as \(x_{i_1}>x_{i_2}>x_{i_3}\). The forward candidate gap is formed by \((i_1,i_2)\), and the rear candidate gap is formed by \((i_2,i_3)\). For a candidate gap \(g\) with front vehicle \(F\) and rear vehicle \(R\), the scalar gap centre \(x_g(t)\) and scalar gap velocity \(v_g(t)\) are

**中文。**

每个时间步，高层模块按照纵向坐标选出距离 ego 前轴点最近的三辆目标车道车辆。将其标量纵向位置按从前到后排序为 \(x_{i_1}>x_{i_2}>x_{i_3}\)。前方候选 gap 由 \((i_1,i_2)\) 构成，后方候选 gap 由 \((i_2,i_3)\) 构成。对于由前车 \(F\) 和后车 \(R\) 构成的候选 gap \(g\)，其标量 gap 中心 \(x_g(t)\) 和标量 gap 速度 \(v_g(t)\) 为

$$
x_g(t)=\frac{x_F(t)+x_R(t)}{2},\qquad
v_g(t)=\frac{v_F(t)+v_R(t)}{2}.
$$

**English.**

Let \(x_e(t)\) and \(v_e(t)\) denote the ego front-axle longitudinal position and speed. The longitudinal alignment error is \(d_g(t)=x_g(t)-x_e(t)\), and the relative speed error is \(\Delta v_g(t)=v_g(t)-v_e(t)\). The confidence of candidate gap \(g\) is evaluated by a radial-basis function,

**中文。**

设 \(x_e(t)\) 和 \(v_e(t)\) 分别为 ego 前轴点纵向位置和速度。纵向对齐误差为 \(d_g(t)=x_g(t)-x_e(t)\)，相对速度误差为 \(\Delta v_g(t)=v_g(t)-v_e(t)\)。候选 gap \(g\) 的置信度由径向基函数评价：

$$
C_g(t)=
\exp\!\left(
-\frac{d_g(t)^2}{2\sigma_d^2}
-\frac{\Delta v_g(t)^2}{2\sigma_v^2}
\right).
$$

**English.**

If \(C_f(t)\) and \(C_r(t)\) are the confidences of the forward and rear candidate gaps, the longitudinal bias is \(B_y(t)=C_f(t)-C_r(t)\). This difference retains both direction and magnitude, so even a small confidence difference can become decisive when it persists through the opinion dynamics.

**中文。**

若 \(C_f(t)\) 和 \(C_r(t)\) 分别表示前方和后方候选 gap 的置信度，则纵向偏置为 \(B_y(t)=C_f(t)-C_r(t)\)。该差值同时保留方向和幅值，因此即使置信度差异较小，只要它持续存在，也能通过意见动力学逐渐形成明确决策。

### 纵向意见与决策/Longitudinal Opinions and Decisions

**English.**

The longitudinal opinion \(y(t)\in\mathbb{R}\) accumulates the directional preference between the two candidate gaps. With the bias \(B_y(t)\) and attention \(u_h(t)\), its continuous dynamics are

**中文。**

纵向意见 \(y(t)\in\mathbb{R}\) 用于累积两个候选 gap 之间的方向偏好。在偏置 \(B_y(t)\) 和注意力 \(u_h(t)\) 作用下，其连续动力学为

$$
\dot{y}(t)=-d_y y(t)+u_h(t)\tanh\!\left(\alpha_y y(t)\right)+B_y(t).
$$

**English.**

The forward gap is selected when \(y(t)>\theta_y\), the rear gap is selected when \(y(t)<-\theta_y\), and no switch is made when \(|y(t)|\leq\theta_y\). This waiting band reduces frequent changes when the two candidate gaps have similar confidence.

**中文。**

当 \(y(t)>\theta_y\) 时选择前方 gap，当 \(y(t)<-\theta_y\) 时选择后方 gap，当 \(|y(t)|\leq\theta_y\) 时不切换。该等待区间能够在两个候选 gap 置信度接近时减少频繁切换。

![Layered opinion decision schematic](./image/layered_opinion_decision_schematic.png)

Fig2：分层意见决策示意图/Layered opinion decision schematic

### 横向偏置设计/Lateral Direction Bias Design

**English.**

After the longitudinal decision chooses a target gap, the lateral opinion evaluates whether the ego vehicle should actively move toward it. For the selected front and rear vehicles, the scalar physical gap length is \(G(t)=x_F(t)-x_R(t)\), and the gap opening rate is \(\dot{G}(t)=v_F(t)-v_R(t)\). The lateral environmental bias is

**中文。**

纵向决策选择目标 gap 后，横向意见评价 ego 车辆是否应主动向该 gap 移动。对于所选前车和后车，标量物理 gap 长度为 \(G(t)=x_F(t)-x_R(t)\)，gap 张开速率为 \(\dot{G}(t)=v_F(t)-v_R(t)\)。横向环境偏置为

$$
B_z(t)=k_g\left[G(t)-G_{\mathrm{safe}}\right]+k_v\dot{G}(t).
$$

### 横向注意力及 SAC 设置/Lateral Attention and SAC Design

**English.**

The lateral attention \(u_z(t)\) controls how strongly the lateral opinion self-reinforces. The Soft Actor-Critic (SAC) policy outputs \(u_z(t)\), while the model-based controller still determines the vehicle motion. Let \(s_t\) be the state, \(a_t\) be the action, \(\rho_\pi\) be the state-action distribution induced by policy \(\pi_\phi(a|s)\), \(r(s_t,a_t)\) be the immediate reward, \(\alpha_{\mathrm{SAC}}\) be the entropy temperature, and \(\mathcal{H}\) be policy entropy. The SAC state \(s_t\in\mathbb{R}^8\) contains relative positions and velocities to the front and rear vehicles of the selected gap, and the action is the scalar attention command \(a_t=u_z(t)\).

**中文。**

横向注意力 \(u_z(t)\) 控制横向意见的自强化强度。Soft Actor-Critic (SAC) 策略输出 \(u_z(t)\)，但车辆运动仍由模型控制器决定。设 \(s_t\) 为状态，\(a_t\) 为动作，\(\rho_\pi\) 为策略 \(\pi_\phi(a|s)\) 诱导的状态-动作分布，\(r(s_t,a_t)\) 为即时奖励，\(\alpha_{\mathrm{SAC}}\) 为熵温度参数，\(\mathcal{H}\) 为策略熵。SAC 状态 \(s_t\in\mathbb{R}^8\) 包含 ego 与所选 gap 前后车的相对位置和速度，动作为标量注意力命令 \(a_t=u_z(t)\)。

$$
s_t=[\Delta x_F,\Delta y_F,\Delta v_{x,F},\Delta v_{y,F},
\Delta x_R,\Delta y_R,\Delta v_{x,R},\Delta v_{y,R}],\qquad
a_t=u_z(t).
$$

**English.**

The reward combines lane-change progress, opportunity usage, action smoothness, safety, and terminal success. The notation \(\mathbb{I}\{\cdot\}\) denotes an indicator function that equals one when its condition is true and zero otherwise:

**中文。**

奖励函数结合并道进度、机会利用、动作平滑性、安全性和终端成功。记号 \(\mathbb{I}\{\cdot\}\) 表示指示函数：当条件成立时取 1，否则取 0。

$$
\begin{aligned}
r_t={}&5\Delta p_t+2O_t\max(\Delta p_t,0)-0.02(1-p_t)
-0.5\|a_t-a_{t-1}\|^2 \\
&-2\mathbb{I}\{v_y(t)v_y(t-\Delta t)<0\}
-20\max\!\left(0,\frac{m_s-d_{\min}}{m_s}\right)^2 \\
&-1000\mathbb{I}\{\mathrm{collision}\}+(100-2t)\mathbb{I}\{\mathrm{success}\}.
\end{aligned}
$$

**English.**

Here, \(p_t\) is the lane-change progress, \(O_t\) indicates whether a useful merging opportunity exists, \(m_s\) is the safety margin, and \(d_{\min}\) is the minimum distance to the neighbouring vehicles. This design rewards active merging progress while penalizing collision risk, oscillatory lateral motion, and unnecessarily long execution.

**中文。**

其中，\(p_t\) 表示并道进度，\(O_t\) 表示是否存在有效并道机会，\(m_s\) 为安全裕度，\(d_{\min}\) 为 ego 与邻近车辆的最小距离。该设计鼓励主动并道进度，同时惩罚碰撞风险、横向速度反复换向和不必要的长时间执行。

### 横向意见与决策/Lateral Opinions and Decisions

**English.**

The lateral opinion \(z(t)\in\mathbb{R}\) accumulates the commitment to move toward the selected gap. Given the learned attention \(u_z(t)\) and the lateral bias \(B_z(t)\), the opinion follows

**中文。**

横向意见 \(z(t)\in\mathbb{R}\) 用于累积向所选 gap 移动的承诺。在学习得到的注意力 \(u_z(t)\) 和横向偏置 \(B_z(t)\) 作用下，其动力学为

$$
\dot{z}(t)=-d_z z(t)+u_z(t)\tanh\!\left(\alpha_z z(t)\right)+B_z(t).
$$

**English.**

The opinion is mapped to a merge ratio \(\lambda_z(t)=\mathrm{clip}((z(t)+1)/2,0,1)\). When \(z(t)\) is low, the desired point remains close to the original lane; as \(z(t)\) increases, it moves toward the selected target-lane gap.

**中文。**

意见被映射为并道比例 \(\lambda_z(t)=\mathrm{clip}((z(t)+1)/2,0,1)\)。当 \(z(t)\) 较低时，期望点接近原车道；随着 \(z(t)\) 增大，期望点逐渐移向目标车道中所选 gap。

## 底层控制设计/Low-Level Control Design

### 安全规避条款/Safety-Avoidance Term

**English.**

The low-level controller uses the high-level target while adding a safety-avoidance term inspired by dissipative barrier feedback for embodied opinion-dynamics lane-merging control [R17]. Let \(p_e(t)\in\mathbb{R}^2\) be the ego front-axle position, \(p_j(t)\in\mathbb{R}^2\) be the position of neighbouring vehicle \(j\), \(r_j(t)=p_e(t)-p_j(t)\) be the relative position vector, and \(d_j(t)=\|r_j(t)\|\) be the Euclidean distance. Let \(R_c\) be the activation distance and \(k_c\) be the avoidance gain. The avoidance vector is

**中文。**

底层控制器使用高层目标，同时添加安全避障项，该项受到 embodied opinion-dynamics 并道控制中 dissipative barrier feedback 思想的启发 [R17]。设 \(p_e(t)\in\mathbb{R}^2\) 为 ego 前轴点位置，\(p_j(t)\in\mathbb{R}^2\) 为邻近车辆 \(j\) 的位置，\(r_j(t)=p_e(t)-p_j(t)\) 为相对位置向量，\(d_j(t)=\|r_j(t)\|\) 为欧氏距离。设 \(R_c\) 为避障激活距离，\(k_c\) 为避障增益，则避障向量为

$$
u_c(t)=\sum_j
\mathbb{I}\{d_j(t)<R_c\}\,
k_c\left(\frac{1}{d_j(t)}-\frac{1}{R_c}\right)
\frac{r_j(t)}{d_j(t)^3}.
$$

### 目标点与跟踪误差/Target Point and Tracking Error

**English.**

Let \(L_p\) be the preview distance, \(y_O\) be the original-lane centre, \(y_T\) be the target-lane centre, \(p_O(t)=[x_e(t)+L_p,\;y_O]^\top\) be the original-lane preview point, and \(p_G(t)=[x_g(t),\;y_T]^\top\) be the selected gap centre in the target lane. The opinion-weighted target and tracking error are

**中文。**

设 \(L_p\) 为预瞄距离，\(y_O\) 为原车道中心，\(y_T\) 为目标车道中心，\(p_O(t)=[x_e(t)+L_p,\;y_O]^\top\) 为原车道预瞄点，\(p_G(t)=[x_g(t),\;y_T]^\top\) 为目标车道中所选 gap 中心。意见加权目标点和跟踪误差为

$$
p_{\mathrm{tar}}(t)=(1-\lambda_z(t))p_O(t)+\lambda_z(t)p_G(t)+u_c(t),
\qquad
e_p(t)=p_{\mathrm{tar}}(t)-p_e(t).
$$

### 最终物理输入设计/Final Physical Input Design

**English.**

Let \(\psi(t)\) be the ego heading. The longitudinal and lateral unit vectors are \(e_\parallel(t)=[\cos\psi(t),\sin\psi(t)]^\top\) and \(e_\perp(t)=[-\sin\psi(t),\cos\psi(t)]^\top\). Let \(v_{\mathrm{tar}}(t)\) be the target velocity, \(k_a\), \(k_x\), and \(k_\omega\) be tracking gains, \(a_{\min}\) and \(a_{\max}\) be acceleration limits, and \(\omega_{\min}\) and \(\omega_{\max}\) be steering-rate limits. The commanded acceleration \(a(t)\) and steering-rate command \(\omega(t)\) are

**中文。**

设 \(\psi(t)\) 为 ego 航向角。纵向和横向单位向量分别为 \(e_\parallel(t)=[\cos\psi(t),\sin\psi(t)]^\top\) 和 \(e_\perp(t)=[-\sin\psi(t),\cos\psi(t)]^\top\)。设 \(v_{\mathrm{tar}}(t)\) 为目标速度，\(k_a\)、\(k_x\) 和 \(k_\omega\) 为跟踪增益，\(a_{\min}\) 和 \(a_{\max}\) 为加速度限幅，\(\omega_{\min}\) 和 \(\omega_{\max}\) 为转角变化率限幅。加速度命令 \(a(t)\) 和转角变化率命令 \(\omega(t)\) 为

$$
\begin{aligned}
a(t)&=\mathrm{clip}\!\left(k_a(v_{\mathrm{tar}}(t)-v_e(t))+k_x e_p(t)^\top e_\parallel(t),a_{\min},a_{\max}\right),\\
\omega(t)&=\mathrm{clip}\!\left(k_\omega e_p(t)^\top e_\perp(t),\omega_{\min},\omega_{\max}\right).
\end{aligned}
$$

**English.**

This final stage is separated from the high-level opinions. The opinion variables decide which gap to use and how strongly to merge, while the low-level controller executes the motion subject to tracking and safety constraints.

**中文。**

最终控制阶段与高层意见相分离。意见变量决定使用哪个 gap 以及以多强程度并道，底层控制器则在跟踪和安全约束下执行实际运动。

# 实验设计/Experimental Design

**English.**

The experiments are organized from simple to complex. The single-gap environment isolates the lateral attention and merge-commitment problem: it tests whether RL can learn when to increase attention and commit to merging when the target gap is fixed. The multi-gap environment introduces longitudinal high-level decisions: when several dynamic gaps exist simultaneously, it tests whether the system can select an appropriate local gap, avoid excessive switching, and safely execute merging. The experimental parameters are listed in the appendix.

**中文。**

实验按照由简单到复杂的方式组织。单 gap 环境隔离横向注意力与并道承诺问题：测试当目标 gap 已经确定时，RL 能否学会何时提高注意力并承诺并道。多 gap 环境引入纵向高层决策：当多个动态 gap 同时存在时，系统能否选择合适的局部 gap，避免过度切换，并安全执行并道。实验具体参数表在附录中。


## 单 gap 并入实验/Single-Gap Merging Experiment

**English.**

The single-gap environment contains one front vehicle, one rear vehicle, and one ego vehicle. The front and rear vehicles travel in the target lane, and their longitudinal separation defines the only available merging gap. The ego vehicle starts from the original lane and attempts to merge into this gap. The lane width is \(W=4.0\,\mathrm{m}\), so the original and target lane centers are \(y_O=0.5W=2.0\,\mathrm{m}\) and \(y_T=1.5W=6.0\,\mathrm{m}\), respectively. The initial target-vehicle states are

**中文。**

单 gap 环境包含一辆目标前车、一辆目标后车和一辆 ego 车。前车与后车位于目标车道，二者之间的纵向间距构成唯一可并入 gap。ego 车从原车道出发，并尝试并入该 gap。车道宽度为 \(W=4.0\,\mathrm{m}\)，因此原车道中心为 \(y_O=0.5W=2.0\,\mathrm{m}\)，目标车道中心为 \(y_T=1.5W=6.0\,\mathrm{m}\)。目标车初始状态为

$$
x_F(0)=30.0\,\mathrm{m},\qquad
x_R(0)=15.0\,\mathrm{m},\qquad
v_F(0)=v_R(0)=15.0\,\mathrm{m/s}.
$$

**English.**

The ego vehicle has the same initial longitudinal speed but a randomized longitudinal initial position. This randomization makes the training and evaluation less dependent on one special initial alignment. It forces the policy to learn an attention schedule that works when the ego vehicle starts slightly ahead of or behind the gap center.

**中文。**

ego 车具有相同初始纵向速度，但纵向初始位置带有随机扰动，该随机化避免训练和评价只依赖某一个特殊的初始对齐位置，使策略必须在 ego 车领先或落后于 gap 中心时仍能给出合理注意力时序。

$$
x_e(0)=20.0+\xi,\qquad
\xi\sim\mathcal{U}(-5.0,5.0),\qquad
y_e(0)=2.0\,\mathrm{m},\qquad
v_e(0)=15.0\,\mathrm{m/s}.
$$

**English.**

The rear vehicle is deliberately designed to create a staged merging opportunity. Before \(t_y=20.0\,\mathrm{s}\), it follows a sinusoidal law with period \(P_s=6.0\,\mathrm{s}\), velocity-amplitude parameter \(A_s=4.0\,\mathrm{m/s}\), angular frequency \(\omega_s=2\pi/P_s\), and front-rear gap \(g(t)=x_F(t)-x_R(t)\). Let

**中文。**

后车运动被设计成分阶段生成并道机会。在 \(t_y=20.0\,\mathrm{s}\) 之前，后车采用正弦规律，其中周期为 \(P_s=6.0\,\mathrm{s}\)，速度幅值参数为 \(A_s=4.0\,\mathrm{m/s}\)，角频率为 \(\omega_s=2\pi/P_s\)，前后车 gap 为 \(g(t)=x_F(t)-x_R(t)\)。令

$$
\begin{aligned}
\omega_s&=\frac{2\pi}{P_s},\qquad P_s=6.0\,\mathrm{s},\qquad A_s=4.0\,\mathrm{m/s},\\
a_R(t)&=A_s\omega_s\cos(\omega_s t),\\
v_R(t)&=v_R(0)+A_s\sin(\omega_s t),\\
x_R(t)&=x_R(0)+v_R(0)t+\frac{A_s}{\omega_s}\left[1-\cos(\omega_s t)\right],\\
g(t)&=x_F(t)-x_R(t)=15.0-\frac{A_s}{\omega_s}\left[1-\cos(\omega_s t)\right],\quad 0\leq t\leq20.0.
\end{aligned}
$$

**English.**

For \(0\leq t\leq t_y\), the rear-vehicle acceleration, velocity, and position are

**中文。**

当 \(0\leq t\leq t_y\) 时，后车加速度、速度和位置为


**English.**

Since the front vehicle travels with constant speed \(v_F=15.0\,\mathrm{m/s}\), the front-rear gap before yielding is therefore

**中文。**

前车保持匀速 \(v_F=15.0\,\mathrm{m/s}\),于是让行前的前后车 gap 可写为


**English.**

After \(20.0\,\mathrm{s}\), the rear vehicle starts yielding by tracking a desired front-rear gap \(g_{\mathrm{yield}}=20.0\,\mathrm{m}\). The acceleration law uses proportional and damping gains \(0.35\) and \(1.1\), with acceleration clipped to \([-5.0,2.0]\,\mathrm{m/s^2}\). The rear-vehicle acceleration becomes:

**中文。**

在 \(20.0\,\mathrm{s}\) 之后，后车开始让行，并跟踪期望 gap \(g_{\mathrm{yield}}=20.0\,\mathrm{m}\)。该加速度律使用比例与阻尼增益 \(0.35\) 和 \(1.1\)，并将加速度限制在 \([-5.0,2.0]\,\mathrm{m/s^2}\)。后车加速度变为

$$
\begin{aligned}
a_R(t)&=\mathrm{clip}\left(0.35\left[g(t)-20.0\right]-1.1\left[v_R(t)-v_F(t)\right],-5.0,2.0\right),\quad t>20.0,\\
\dot{g}(t)&=v_F(t)-v_R(t),\qquad \ddot{g}(t)=-a_R(t),\qquad t>20.0.
\end{aligned}
$$

**English.**

Equivalently, the gap dynamics after yielding are governed by

**中文。**

等价地，让行后的 gap 动态满足


**English.**

This piecewise construction has a clear purpose. Before \(20.0\,\mathrm{s}\), the gap is not intentionally opened for the ego vehicle, so the policy should avoid premature aggressive merging. After \(20.0\,\mathrm{s}\), the rear vehicle creates a larger gap and the correct behavior is to increase attention \(u(t)\), allow the opinion \(z(t)\) to grow, and move the target point toward the target lane.

**中文。**

这个分段构造具有明确实验含义。在 \(20.0\,\mathrm{s}\) 之前，目标 gap 并未主动为 ego 车打开，因此策略不应过早激进并道；在 \(20.0\,\mathrm{s}\) 之后，后车开始创造更大 gap，合理策略应提高注意力 \(u(t)\)，使意见 \(z(t)\) 增长，并把目标点推向目标车道。

**English.**

The SAC agent uses a Gaussian policy with two hidden layers of width \(256\), two Q networks, target Q networks, replay-buffer learning, and entropy regularization. The main training parameters are \(200\) episodes, replay-buffer size \(2.0\times10^5\), batch size \(256\), \(1000\) initial random steps, discount factor \(\gamma=0.99\), target-update rate \(\tau=0.005\), and learning rates \(3\times10^{-4}\) for the policy, Q networks, and entropy temperature. The subsequent value comparison uses the same reward as defined above so that the learned and hand-designed policies are judged by an identical metric. In the radial-basis-function baseline, \(u_{\mathrm{base}}\) is the base attention, \(u_{\mathrm{amp}}\) is the attention amplitude, and \(\sigma_d\) and \(\sigma_v\) are the same position and velocity bandwidths used for gap confidence. The baseline comparison evaluates the trained SAC attention against the original hand-designed RBF attention:

**中文。**

SAC agent 使用高斯策略网络、两个 Q 网络、目标 Q 网络、经验回放和熵正则化。主要训练参数为：训练 \(200\) 个 episode，经验池容量 \(2.0\times10^5\)，batch size 为 \(256\)，初始随机探索步数为 \(1000\)，折扣因子 \(\gamma=0.99\)，目标网络软更新系数 \(\tau=0.005\)，policy、Q 网络和熵温度学习率均为 \(3\times10^{-4}\)。后续价值对比也由前文定义的同一 reward 执行，因此学习策略和手工策略由完全一致的指标评价。在 radial-basis-function 基线中，\(u_{\mathrm{base}}\) 为基础注意力，\(u_{\mathrm{amp}}\) 为注意力幅值，\(\sigma_d\) 与 \(\sigma_v\) 为 gap 置信度中使用的位置和速度带宽。对照基线为原始手工设计的 RBF 注意力：

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

**English.**

The evaluation repeats the simulation \(100\) times with shared random ego initial positions. For each random seed, both policies face the same initial \(x_e(0)\) and the same target-vehicle trajectory. The comparison reports each episode reward and summarizes the mean and standard deviation across trials.

**中文。**

评价阶段重复 \(100\) 次仿真，并使用共享的随机 ego 初始位置。对每一个随机种子，SAC 策略和 RBF 策略面对完全相同的 \(x_e(0)\) 和目标车轨迹。最终比较每次 episode reward，并统计多次试验的均值与标准差：


## 多 gap 并入实验/Multi-Gap Merging Experiment

![multi_gap_environment_schematic](./image/multi_gap_environment_schematic.png)

Fig3：多间隙环境示意图/multi gap environment schematic

**English.**

The multi-gap environment extends the same merging task to a target lane containing five vehicles and four physical gaps. The target vehicles are initialized with a uniform base spacing:

**中文。**

多 gap 环境把同一并道任务扩展到包含五辆目标车和四个物理 gap 的目标车道。目标车队以统一基础间距初始化：

$$
N=5,\qquad
x_i(0)=48.0-(i-1)g_0,\qquad
g_0=8.0\,\mathrm{m},\qquad
i=1,\ldots,5.
$$

**English.**

All target vehicles start on the target-lane center \(y_T=6.0\,\mathrm{m}\) with nominal speed \(15.0\,\mathrm{m/s}\). The ego vehicle starts from the original lane with a larger longitudinal randomization range than in the single-gap experiment. The larger random range changes which three vehicles are nearest to the ego vehicle at the beginning of an episode. Therefore, the high-level selector must work under different local traffic configurations rather than repeatedly facing one fixed gap.


**中文。**

所有目标车初始位于目标车道中心 \(y_T=6.0\,\mathrm{m}\)，名义速度为 \(15.0\,\mathrm{m/s}\)。ego 车从原车道出发，并且相对于单 gap 实验使用更大的纵向随机范围。更大的随机范围会改变每个 episode 开始时距离 ego 最近的三辆目标车，因此高层选择器必须面对不同的局部交通构型，而不是反复处理同一个固定 gap。

$$
x_e(0)=30.0+\xi,\qquad
\xi\sim\mathcal{U}(-10.0,15.0),\qquad
y_e(0)=2.0\,\mathrm{m},\qquad
v_e(0)=15.0\,\mathrm{m/s}.
$$

**English.**

The four target-lane gaps are randomly adjusted over time. Every \(T_g=4.0\,\mathrm{s}\), at most two gaps are selected for modification, and each selected gap receives a desired spacing from the multiplier set \(\mathcal{M}=\{0.75,1.0,1.25,1.5\}\).

**中文。**

四个目标车道 gap 会随时间随机调整。每隔 \(T_g=4.0\,\mathrm{s}\)，最多两个 gap 会被选中改变期望间距，被选中的 gap 从倍率集合中抽取一个倍率：


**English.**

The leading target vehicle has zero acceleration. For each following vehicle \(i+1\), let \(g_i(t)\) be its current gap to vehicle \(i\), \(g_{i,\mathrm{des}}^{(k)}\) be the desired gap during adjustment interval \(k\), and \(\dot{g}_i(t)\) be the gap-rate term. With proportional and damping gains \(0.55\) and \(1.05\), each following target vehicle tracks the desired gap through a clipped proportional-derivative rule:

**中文。**

目标车队最前车加速度为零。对于每一辆跟随车 \(i+1\)，设 \(g_i(t)\) 为它与前车 \(i\) 的当前 gap，\(g_{i,\mathrm{des}}^{(k)}\) 为第 \(k\) 个调整周期内的期望 gap，\(\dot{g}_i(t)\) 为 gap 变化率。结合比例与阻尼增益 \(0.55\) 和 \(1.05\)，每一辆跟随目标车用裁剪后的 PD 规则跟踪其前方期望 gap：

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

**English.**

This mechanism produces a target lane in which gaps open, close, and reconfigure during the \(40.0\,\mathrm{s}\) simulation. The high-level decision module selects the nearest three target-lane vehicles in longitudinal front-axle coordinate and compares the two local gaps formed by these vehicles. The multi-gap experiment is designed to evaluate generalization. The underlying strategy uses the attention strategy learned in the single-gap environment. This design tests whether a strategy that only learns \(u(t)\) for a single pair of preceding and following vehicles remains effective when the high-level selector continuously provides different pairs of preceding and following vehicles. If the strategy is generalizable, then even if the gap selection and gap environment change, the ego vehicle should still be able to smoothly enter the selected gap.

**中文。**

该机制使目标车道中的 gap 在 \(40.0\,\mathrm{s}\) 仿真中持续打开、闭合和重构。高层决策模块在每一步根据前轴纵向坐标选择距离 ego 最近的三辆目标车，并比较这三辆车形成的两个局部 gap。多 gap 实验的主要目的在于验证泛化能力。横向注意力策略使用单 gap 环境中学到的注意力策略。这样的设计检验了一个只针对单个前后车 pair 学习 \(u(t)\) 的策略，在高层选择器不断提供不同前后车 pair 时是否仍然有效。如果策略具有泛化性，那么即使 gap 选择和 gap 环境在变化，ego 车仍应能够平滑进入所选 gap。

**English.**

The high-level ablation compares the opinion-dynamics selector with a simple maximum-score selector. The maximum-score baseline removes the high-level opinion memory and instantaneously chooses the candidate with the larger score.

**中文。**

高层消融实验比较意见动力学选择器与简单最大评分选择器。最大评分基线去掉高层意见记忆，并瞬时选择局部评分更高的 gap：


**English.**

Here \(S_i(t_k)\) is the instantaneous confidence or gap-evaluation score of candidate gap \(i\). The proposed opinion-dynamics selector instead integrates the confidence difference \(C_f(t)-C_r(t)\) through the high-level opinion dynamics defined above.

**中文。**

其中 \(S_i(t_k)\) 表示候选 gap \(i\) 的瞬时置信度或 gap 评价分数。本文方法则通过意见动力学持续积分置信度差值：


**English.**

This comparison was evaluated through multiple random ablation experiments. The current test setup uses \(N_{\mathrm{test}} = 100\) random runs. For each method, the total reward and the number of times the selected gap was switched were compared.

**中文。**

该对比通过多次随机消融试验评价。当前测试设置使用 \(N_{\mathrm{test}}=100\) 次随机运行。对每种方法，对比总 reward 和被选 gap 切换次数。

# 结果和讨论/Results and discussion

## 单 gap SAC 训练/Single-gap SAC Training

![fig1_single_gap_training](./image/fig1_single_gap_training.png)

Fig4：单间隙训练 reward/Single-gap training reward
   
**English.**

The training results show that in the initial 50 episodes, the algorithm conducted extensive exploration, resulting in significant fluctuations in the reward. The algorithm converged after 50 episodes. The SAC agent converged within 50 episodes, achieving a stable episodic reward of 230\(\pm\)10 (Fig. 4). The Q1-loss decreased from an initial peak of 134 to below 20, indicating accurate value function    approximation. The entropy coefficient α decayed monotonically from 0.96 to 0.12, confirming a smooth transition from exploration to exploitation. The policy consistently reached the target gap (progress ≥ 0.95) with a 97.5% success rate (195/200) and zero collisions. The average control command (mean_u) stabilized around 1.9, suggesting an efficient, non-conservative driving policy. These results demonstrate that SAC effectively learns a robust and safe gap-acceptance strategy.

**中文。**

训练结果显示，在初始的50个 episodes 里，算法进行了广泛的探索，导致 reward 波动剧烈，在50个 episodes 后完成收敛，获得稳定的回合奖励为230±10。Q1损失从初始峰值134下降至低于20，表明价值函数逼近准确。熵系数α从0.96单调衰减至0.12，证实了从探索到利用的平滑过渡。策略持续达到目标间隙（进展≥0.95），成功率达到97.5%（195/200），且无碰撞发生。平均控制指令（mean_u）稳定在约1.9，表明驾驶策略高效且非保守。这些结果表明，SAC 能够有效学习出一种稳健且安全的间隙接受策略。

## 与基线 RBF 控制器的对比/Comparison with Baseline RBF Controller

![fig2_single_gap_policy_vs_rbf](./image/fig2_single_gap_policy_vs_rbf.png)

Fig5：SAC 与 RBFreward 对比/Comparison between SAC and RBF reward

**English.**

The SAC policy achieves a higher average episodic reward than the RBF baseline, with 233.4 versus 115.8. Both policies maintain 100% success and zero collisions, but SAC completes the task in 6.9 s, whereas RBF requires 29.45 s. This indicates that the hand-designed RBF rule is safe but conservative, while SAC learns a more efficient yet still collision-free strategy.

In terms of safety indicators, the RBF baseline maintained a larger minimum distance ($d_{\min} \approx 9.7$ m), reflecting its inherent conservatism. The SAC attention policy is closer to the obstacle ($d_{\min} \approx 2.6$ m), but always remains outside the collision radius, showing that the learned attention can balance time efficiency and collision safety through the reward structure.

**中文。**

与手工设计的基线相对比，定量结果如表所示。SAC 注意力策略获得了显著更高的平均 episodic reward（$\bar{R}_{\mathrm{SAC}} = 233.4 $），而 RBF 基线仅为（$\bar{R}_{\mathrm{RBF}} = 115.8 $），平均提升了约 117.6 分。尽管两种策略均实现了 100% 的成功率且无碰撞，最显著的差异体现在任务执行效率上。SAC 注意力策略平均仅需6.9秒即可完成换道间隙选择。相比之下，RBF 控制器每轮测试均消耗 29.45 秒，这非常接近预设的仿真时间上限。这表明，尽管 RBF 手工规则保证了安全，但策略较为保守，未能充分利用车辆的纵向运动能力。而 SAC 智能体则学会了积极但安全的注意力输出，将任务完成时间缩短了超过 75%。在安全性指标方面，RBF 基线保持了更大的最小距离（$d_{\min} \approx 9.7$ m），反映了其固有的保守性。SAC 注意力策略虽然更接近障碍物（$d_{\min} \approx 2.6$ m），但始终保持在碰撞半径之外，说明学习得到的注意力能够通过奖励结构平衡时间效率与避碰安全性。

Table 3: Performance comparison between SAC and RBF controllers

| Metric | SAC Policy | RBF Baseline | Improvement |
| :--- | :--- | :--- | :--- |
| **Episodic Reward** ($\bar{R}$) | **$233.4 \pm 4.2$** | $115.8 \pm 2.1$ | **+101.5%** |
| **Task Time (s)** | **$6.9 \pm 0.2$** | $29.45 \pm 2.4$ | **-76.6%** |
| **Success Rate** | 100% | 100% | — |
| **Collision Rate** | 0% | 0% | — |
| **Min. Distance (m)** | $2.62 \pm 0.23$ | $9.70 \pm 0.01$ | — |

## 多间隙泛化/Multi-Gap

![fig3_multi_gap_transfer](./image/fig3_multi_gap_transfer.png)

Fig6：多 gap 泛化实验结果/The results of the multiple gap generalization experiments

**English.**

The lateral SAC attention policy was trained in a single-gap environment and then directly deployed to a dynamic multi-gap scenario. The experimental results show that it has good generalization ability in terms of safety and task completion: the success rate remains at a high level (≈0.97). Even if the high-level selector switches, in the vast majority of cases, the lateral attention policy never violates the safety boundary. These data confirm that the learned attention policy generalizes beyond the training environment.

Although the reward remained consistent with that in the single-gap experiment in most cases, in some difficult cases, the strategy might deteriorate and lead to failure. Analysis of the experimental data of failures and low rewards, the failures mainly arise from two factors: the repeated decision-making caused by the complex environment, resulting in a long decision-making time exceeding the maximum simulation time and leading to task failure; and the decision being at a local optimal solution, resulting in overly cautious actions and excessive time spent, thereby causing a serious deduction in the step term. The reasons for the decline in generalization performance may be that too little environmental information was input during training, and only the instantaneous state was focused on while ignoring the changes on the time scale.

**中文。**

将横向 SAC 注意力策略在单间隙环境中进行训练，随后被直接部署到动态多间隙场景中。实验结果表明其在安全性和任务完成方面具有良好的泛化能力：成功率保持较高水平（≈0.97）。即使高层选择器切换，在绝大多数情况下，横向注意力策略也从未违反安全边界。这些数据证实，SAC 策略已具备一种可泛化的能力。

虽然 reward 在大部分实验里保持与单 gap 一致，但在部分特殊情况下，策略可能发生退化并导致失败，通过对失败和低 reward 的实验数据分析，其具体原因主要分为两个方面：复杂环境带来的决策反复变动，导致决策时间过长，超过最大仿真时间导致任务失败，以及决策处于局部最优解，导致动作过于谨慎和并入时间过长，进而在步长项上扣分严重。产生这些泛化性能下降的原因可能是在训练中输入环境信息太少，且只关注瞬时状态而忽略在时间尺度上的变化。

## 高层决策消融实验/High-level decision-making Ablation experiment

![fig4_multi_gap_ablation](./image/fig4_multi_gap_ablation.png)

Fig7：opinion 和 max 高层策略 reward 对比/Comparison of high-level strategy rewards between Opinion and Max

**English.**

The opinion-dynamics selector achieves a higher mean reward than the max-score selector (+55.3%) and switches less often (0.61 versus 0.76). Although max-score sometimes finishes faster, its lower reward indicates larger safety penalties and less smooth control. The opinion-based selector therefore provides more temporally consistent guidance and more stable behaviour, despite a slightly lower success rate caused by timeout cases.

**中文。**

通过对比了两种高层决策策略：Opinion 策略（基于注意力机制给出的推荐间隙）与 Max 策略（选择最大化某一指标的间隙）。Opinion 策略取得了显著更高的平均奖励（+55.3%），表明其选择的间隙在 安全性、平稳性和整体收益上优于 Max 策略。尽管 Max 策略在某些情况下完成时间更短（平均 8.2 s 对比 12.6 s），但其低奖励主要源于较大的安全惩罚和不够平滑的控制，反映出所选间隙可能过于激进或不利于横向注意力策略平稳执行。在切换次数上，Opinion 策略平均切换次数更少（0.61 vs 0.76），说明其提供的间隙建议具有更好的时间一致性，减少了因频繁切换目标导致的底层控制抖动，有利于提升乘坐舒适性和算法稳定性。Opinion 策略的成功率（97%）略低于 Max（100%），主要原因是超过最大仿真时间导致的超时失败。综合来看，Opinion 策略在总奖励和切换稳定性上均优于 Max 策略，验证了基于注意力的高层决策能更有效地引导横向注意力策略，产生更平稳的换道行为。

Table 4: Comparison of high-level strategies between Opinion and Max
| Metric | Opinion Strategy | Max Strategy | Improvement |
| :--- | :--- | :--- | :--- |
| **Mean Episode Reward** (\(\bar{R}\)) | 196.8 | 126.7 | **+55.3%** |
| **Mean Switch Count** | 0.61 | 0.76 | **-19.7%** |
| **Mean Completion Time (s)** | 12.6 | 8.2 | — |
| **Success Rate** | **97%** | 100% | — |

# 结论/Conclusions

**English.**

The proposed two-layer framework provides an interpretable method for autonomous lane merging. The high-level decision-making layer uses longitudinal opinion dynamics to select a local candidate gap and lateral opinion dynamics to decide the strength of merging commitment. The low-level control layer then converts this high-level decision into collision-aware target tracking and physical vehicle inputs. This revised structure separates strategic decision formation from execution, while preserving a continuous opinion-dynamics formulation throughout the system.

Experiments show that the SAC-trained lateral attention policy in the single-gap environment converges to a stable behaviour and outperforms the hand-designed RBF attention baseline, mainly by reducing completion time while remaining collision-free. In the multi-gap environment, the same policy maintains a high success rate, indicating that the learned attention has partial generalization beyond the training scenario. The remaining failures mainly occur in difficult cases with repeated gap switching or excessive waiting, which suggests that future work should include temporal state information and stronger communication between longitudinal selection and lateral commitment. The ablation study further shows that the opinion-based high-level selector achieves higher reward and fewer switches than the max-score selector, confirming the value of temporally smoothed opinion dynamics for interactive lane-changing decisions.

**中文。**

本文提出的双层框架为自动并道提供了一种可解释方法。高层决策层使用纵向意见动力学选择局部候选 gap，并使用横向意见动力学决定并道承诺强度。底层控制层进一步将该高层决策转化为带碰撞规避的目标跟踪和物理车辆输入。该结构将策略决策形成与执行分离，同时在整个系统中保持连续意见动力学形式。

实验表明，单 gap 环境中训练得到的 SAC 横向注意力策略能够收敛到稳定行为，并优于手工设计的 RBF 注意力基线，主要体现在保持无碰撞的同时缩短完成时间。在多 gap 环境中，同一策略保持较高成功率，说明学习得到的注意力在训练场景之外具有一定泛化能力。剩余失败主要出现在 gap 反复切换或过度等待的困难样本中，这说明后续工作应加入时间状态信息，并增强纵向选择和横向承诺之间的信息连接。消融实验进一步表明，基于意见动力学的高层选择器相比 max-score 选择器获得更高 reward 和更少切换次数，验证了时间平滑意见动力学对交互式换道决策的价值。

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

R17. Tang, Z., & Xing, Y. (2026). Embodied Opinion Dynamics for Safety-Critical Motion Control in Dynamic Environments. arXiv preprint arXiv:2606.13465. https://doi.org/10.48550/arXiv.2606.13465

# 附录/Appendices
## project outline
## Risk Assessment
## Experimental Parameter Summary / 实验参数总结

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
| Lateral opinion / 横向意见 | \(g_{\mathrm{safe}},k_g,k_v\) | \(5.0\,\mathrm{m},0.2,0.1\) | Gap-bias parameters in multi-gap execution / 多 gap 执行中的 gap 偏置参数 |
| Lateral opinion / 横向意见 | \(d_z,\alpha_z\) | \(2.0,2.0\) | Lateral opinion damping and sensitivity / 横向意见阻尼与灵敏度 |
| Multi-gap evaluation / 多 gap 评价 | \(N_{\mathrm{test}}\) | \(100\) | Random repeated test runs / 随机重复测试次数 |
| Physical input / 物理输入 | \(a\) clip | \([-5.0,5.0]\,\mathrm{m/s^2}\) | Ego acceleration bounds / ego 加速度裁剪 |
| Physical input / 物理输入 | \(\omega\) clip | \([-0.8,0.8]\,\mathrm{rad/s}\) | Ego steering-rate bounds / ego 转角变化率裁剪 |


