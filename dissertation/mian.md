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

## 2. Aims and Objectives / 研究目标与具体目标

**English**

The aim of this project is to develop, integrate, and evaluate a hierarchical decision-making and control framework for interactive autonomous driving scenarios. This framework will use reinforcement learning to adjust the key parameters of the nonlinear decision dynamics/control model, enabling the agent to quickly make decisions in the simulation environment while meeting safety constraints.

This project includes five specific objects. First, establish a nonlinear opinion dynamics model at the policy layer. Second, train the agent using RL methods to dynamically optimize the model parameters. Third, develop a safety perception and execution layer, using control barrier functions or equivalent constraints to convert the high-level policy into collision-free trajectories. Fourth, implement the controller in the Python simulation framework. Fifth, compare with the baseline and evaluate the decision-control effectiveness.

**中文**

本项目的目标是开发、集成并评价一个面向交互式自动驾驶场景的层级决策与控制框架。该框架将使用强化学习调节非线性决策动力学/控制模型的关键参数，使智能体能够在仿真环境中快速形成决策，同时满足安全约束。

本项目包含五个具体目标。第一，建立策略层非线性意见动力学模型。第二，使用RL方法训练智能体，动态优化模型参数。第三，开发安全感知执行层，使用控制屏障函数或等价约束，将高层策略转化为无碰撞轨迹。第四，在Python仿真框架中实现控制器。第五，与基线对比，评价决策控制效果。


# 方法/Methods
## 高层控制系统/High-Level Control System
### 意见动力学与自更新注意力公式/Opinion Dynamics and Self-Updating Attention
### 高层偏置的计算/High-Level Bias from Gap Confidence
### 高层意见更新与决策映射/High-Level Opinion Update and Decision Mapping
## 底层控制系统/Low-Level Control System
### 底层偏置的计算/Low-Level Bias
### 强化学习决定注意力/The attention obtained through reinforcement learning
### SAC 强化学习简介/Soft Actor-Critic Reinforcement Learning
### 底层意见及其对控制点的影响/Low-Level Opinion and Target Point 
## Actual Control Input / 实际控制输入
### 避障项设计/Safety-Avoidance Term
### 总目标点与控制误差设计/Target Point and Tracking Error
### 最终物理控制输入设计/Final Physical Input Design
# 实验设计/Experimental Design
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








