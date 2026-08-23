# Interaction Modeling for Autonomous Merge/Lane-Change Decision-Making / 自动驾驶并道与换道决策中的交互建模

## 1. Background and Motivation / 背景与动机

### 1.1 Decision-making strategies for merge and lane-change interaction / 并道与换道交互中的策略决策方法

**English**

Merge and lane-change decision-making is a representative multi-agent interaction problem in autonomous driving. A lane-change action changes available gaps, right-of-way expectations, and future responses of neighbouring vehicles. Therefore, recent studies regard lane changing as an interaction-aware decision task rather than simple rule-based gap selection. Game-theoretic models are widely used because they explicitly describe the strategic relationship between the ego vehicle and surrounding vehicles. Lane-changing intention, collision probability, dynamic risk, safety, efficiency, comfort, and driver preference can be integrated into a payoff function to determine whether a manoeuvre is beneficial or risky [R1, R2]. Physical-inspired models, such as molecular interaction potential fields, further express attraction and repulsion among vehicles as continuous interaction forces [R3]. Recent work also considers incomplete information and social driving preference, because human-driven vehicles may be cooperative, aggressive, or uncertain during lane-changing conflicts [R4].

Another important direction is learning-based interaction prediction. Graph neural networks and attention mechanisms represent vehicles as nodes and their mutual influence as edges or attention weights. This is suitable for lane-change scenarios because the number and relevance of surrounding vehicles change dynamically. Topological graphs have been used to predict driver lane-change intention from driver behaviour and traffic context [R5], while plan-informed graph attention networks can predict how surrounding vehicles may respond to the ego vehicle's candidate intentions before model predictive control generates a feasible trajectory [R6]. These methods show that interaction modeling should predict not only where other vehicles will go, but also how they react to the ego decision. Cooperative control methods provide a third route. In connected and automated vehicle environments, surrounding vehicles can actively create gaps, coordinate lane-change order, or maintain platoon stability [R7, R8]. Overall, the literature motivates a hierarchical framework that combines strategic interaction modeling, prediction, and safe execution.

**中文**

并道与换道决策是自动驾驶中具有代表性的多智能体交互问题。一次换道会改变周围车辆的可用间隙、路权预期和未来响应。因此，近年研究逐渐将换道视为交互感知决策任务，而不是简单的规则型间隙选择。博弈论模型被广泛使用，是因为它能够显式描述自车与周围车辆之间的策略关系。换道意图、碰撞概率、动态风险、安全性、效率、舒适性和驾驶偏好可以被纳入收益函数，用于判断某个机动是有利还是危险 [R1, R2]。分子相互作用势等物理启发模型进一步将车辆间的吸引与排斥表达为连续交互力 [R3]。近期研究还考虑不完全信息和社会驾驶偏好，因为在人类驾驶车辆参与的换道冲突中，周车可能表现为合作、激进或不确定 [R4]。

另一条重要方向是学习型交互预测。图神经网络和注意力机制将车辆表示为节点，将车辆间影响表示为边或注意力权重。这非常适合换道场景，因为周围车辆的数量和重要性会动态变化。已有研究使用拓扑图结合驾驶行为和交通上下文预测驾驶人换道意图 [R5]，也有研究利用计划感知图注意力网络预测周车对自车候选意图的响应，再由模型预测控制生成可行轨迹 [R6]。这些方法说明，交互建模不应只预测其他车辆会去哪里，还应估计它们会如何响应自车决策。协同控制方法提供了第三条路线。在网联自动驾驶环境中，周围车辆可以主动创造间隙、协调换道顺序或维持车队稳定 [R7, R8]。总体来看，近期文献为层级框架提供了动机：该框架需要结合策略交互建模、行为预测和安全执行。

### 1.2 Reinforcement learning for adaptive decision dynamics / 强化学习用于自适应决策动力学

**English**

Reinforcement learning is especially relevant to this project because lane-change decision-making is sequential, uncertain, and feedback-driven. A vehicle must decide when to wait, when to merge, how to adjust speed, and how to update its decision after other agents respond. Traditional fixed-parameter controllers may perform well in static scenarios, but they often lack flexibility when traffic changes or when the system must shift between cooperative consensus and competitive dissensus. Deep reinforcement learning provides a mechanism for learning decision policies through repeated interaction. Existing studies have integrated safety rules, future risk assessment, reward shaping, and robust observation modeling into DRL lane-change policies [R9-R11]. These studies suggest that RL should not be used as an unconstrained black box; it should be guided by interpretable state variables, safety-aware reward terms, and robustness mechanisms.

This point is important for the present project because the proposed framework does not treat reinforcement learning as a direct replacement for nonlinear control. Instead, RL is positioned as an adaptive parameter-optimisation layer. The nonlinear decision-dynamics model still provides structure, such as opinion evolution, consensus formation, dissensus, and role allocation, while the RL agent learns how to tune the governing parameters when the interaction context changes. This separation keeps the mathematical model interpretable while still allowing the system to improve from simulation feedback.

Multi-agent reinforcement learning is closely aligned with the project's aim. In mixed traffic, vehicles are not passive obstacles but interacting decision makers. MARL allows each connected or autonomous vehicle to learn from neighbouring vehicles while also considering collective outcomes such as traffic efficiency, comfort, and safety [R12]. Right-of-way collaboration and Mix Q-learning further show how individual and group benefits can be balanced in lane-changing tasks [R13, R14]. For this dissertation, these studies support using RL not merely to output a lane-change action, but to tune parameters of a nonlinear decision-dynamics model. In this way, high-level strategy, such as consensus, dissensus, role allocation, or gap selection, can adapt while remaining connected to safe control.

**中文**

强化学习与本项目高度相关，因为换道决策具有序贯性、不确定性和反馈驱动特征。车辆需要决定何时等待、何时并道、如何调整速度，以及在其他智能体响应后如何更新决策。传统固定参数控制器在静态场景中可能表现良好，但当交通环境变化，或系统需要在合作共识与竞争分歧之间切换时，往往缺乏灵活性。深度强化学习提供了一种通过反复交互学习决策策略的机制。已有研究将安全规则、未来风险评估、奖励塑形和鲁棒观测建模融入 DRL 换道策略 [R9-R11]。这些研究表明，RL 不应被作为无约束黑箱使用，而应由可解释状态变量、安全感知奖励项和鲁棒机制引导。

这一点对本项目很重要，因为所提出的框架并不是用强化学习直接替代非线性控制。相反，RL 被定位为自适应参数优化层。非线性决策动力学模型仍然提供结构，例如意见演化、共识形成、分歧和角色分配；RL 智能体则学习在交互环境变化时如何调节关键控制参数。这种分离方式既保留了数学模型的可解释性，又允许系统从仿真反馈中改进。

多智能体强化学习与本项目 aim 更加贴合。在混合交通中，车辆不是被动障碍物，而是相互影响的决策主体。MARL 允许每辆网联或自动驾驶车辆从邻近车辆行为中学习，同时考虑交通效率、舒适性和安全性等群体结果 [R12]。路权协同和 Mix Q-learning 进一步说明，换道任务中可以平衡个体收益与群体收益 [R13, R14]。对于本论文项目而言，这些研究支持一个关键思路：RL 不只是直接输出换道动作，而是可以动态调节非线性决策动力学模型的控制参数。这样，高层策略行为，例如共识、分歧、角色分配或间隙选择，就能够根据环境刺激自适应变化，同时仍与低层安全控制保持连接。

### 1.3 Evaluation dimensions: from efficiency to safety, comfort, and adaptability / 评价维度：从效率到安全、舒适与适应性

**English**

Evaluation in lane-change decision-making has shifted from single efficiency metrics to multi-dimensional assessment. Average speed, travel time, throughput, and lane-change success rate remain important, but they are insufficient for safety-critical autonomous systems. A fast lane change may still be unsafe, uncomfortable, or disruptive to traffic flow. Recent studies therefore evaluate collision risk, minimum distance, braking feasibility, safety violations, acceleration, jerk, traffic-flow stability, and robustness under perception uncertainty [R7, R8, R10, R15]. This evaluation view matches the project outline, which emphasises convergence speed, safety violations, and task efficiency. For a simulation-based dissertation, the key motivation is to benchmark an RL-optimised hierarchical framework against fixed-parameter baselines by efficiency, safety, comfort, adaptability, and stability. This also keeps the evaluation aligned with the aim of linking strategic decision dynamics to safe physical execution in final assessment.

**中文**

换道决策的评价已经从单一效率指标转向多维评价。平均速度、旅行时间、通行量和换道成功率仍然重要，但它们不足以支撑安全关键的自动驾驶系统。一次快速换道仍可能是不安全、不舒适或扰动交通流的。因此，近期研究开始评价碰撞风险、最小距离、制动可行性、安全违规率、加速度、加加速度、交通流稳定性以及感知不确定性下的鲁棒性 [R7, R8, R10, R15]。这种评价视角与 project outline 一致，即强调收敛速度、安全违规率和任务效率。对于基于仿真的 dissertation 而言，关键动机是将 RL 优化的层级框架与固定参数基线对比，同时评价效率、安全性、舒适性、适应性和稳定性。这也使评价体系与项目 aim 保持一致，即把策略层决策动力学与安全物理执行连接起来。

## 2. Aims and Objectives / 研究目标与具体目标

**English**

The aim of this project is to develop, integrate, and evaluate a hierarchical decision-making and control framework for interactive autonomous driving, using merge/lane-change behaviour as the representative multi-agent task. The framework will use reinforcement learning to tune key parameters of a nonlinear decision-dynamics/control model, enabling distributed agents to reach rapid consensus or dissensus while respecting safety constraints in simulation. Here, consensus may mean cooperative gap creation, yielding, or platoon coordination, while dissensus may mean competitive role separation or conflict resolution.

The project has five objectives. First, it will formulate a strategic nonlinear opinion-dynamics model for consensus, dissensus, role assignment, and lane-change intention. Second, it will train an RL agent, such as SAC or PPO, to optimise the governing parameters so that cooperative or competitive behaviour adapts to changing traffic. Third, it will develop a safety-aware execution layer, using control barrier functions or equivalent constraints, to convert high-level strategies into collision-free kinodynamic trajectories. Fourth, it will integrate the RL-driven strategic layer and low-level safety controller in a Python-based multi-agent simulator. Fifth, it will benchmark the framework against fixed-parameter baselines using convergence speed, task efficiency, safety violation rate, motion smoothness, and robustness.

These objectives define the expected contribution. The project will not simply compare one lane-change policy with another; it will test whether learned parameter adaptation can improve the decision dynamics of a multi-agent system while preserving safety constraints. If successful, the framework can show how strategic-level learning and execution-level safety control can be connected in a transparent simulation pipeline.

**中文**

本项目的 aim 是开发、集成并评价一个面向交互式自动驾驶场景的层级决策与控制框架，其中以并道/换道行为作为代表性多智能体任务。该框架将使用强化学习调节非线性决策动力学/控制模型的关键参数，使分布式智能体能够在仿真环境中快速形成共识或分歧，同时满足安全约束。在这一语境下，共识可以对应于间隙创造、让行或编队协调中的合作一致；分歧则可以表示竞争性角色分离或冲突解决。

本项目包含五个具体 objectives。第一，建立策略层非线性意见动力学模型，用于描述共识、分歧、角色分配和换道意图。第二，使用RL方法训练智能体，动态优化模型参数，使合作或竞争行为适应交通变化。第三，开发安全感知执行层，使用控制屏障函数或等价约束，将高层策略转化为无碰撞轨迹。第四，在 Python 多智能体仿真框架中集成 RL 策略层与低层安全控制器。第五，与固定参数基线对比，评价收敛速度、任务效率、安全违规率、运动平顺性和鲁棒性。

这些 objectives 也界定了项目的预期贡献。本项目不是简单比较两个换道策略，而是检验学习型参数自适应是否能够改善多智能体系统的决策动力学，同时保持安全约束。如果验证成功，该框架可以说明策略层学习与执行层安全控制如何在透明的仿真流程中连接起来。

## References / 参考文献

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
