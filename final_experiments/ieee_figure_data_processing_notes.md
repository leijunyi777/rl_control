# IEEE figure data processing notes

# IEEE 图像数据处理说明

## Source data

## 数据来源

The figures are generated from four CSV files in `D:\workspace\rl_control\final_experiments`.

本文中的实验图由 `D:\workspace\rl_control\final_experiments` 文件夹中的四个 CSV 文件生成。

| Research question | 研究问题 | CSV file | Sample size | Repeated unit |
|---|---|---:|---:|---|
| Single-gap SAC learning curve | 单 gap 环境中的 SAC 学习曲线 | `single_gap_sac_train.csv` | 100 episodes | episode |
| Single-gap SAC vs. RBF reference | 单 gap 环境中 SAC 策略与 RBF 参考策略对比 | `single_gap_compare.csv` | 100 paired runs | shared ego initial position |
| Multi-gap transfer evaluation | 多 gap 环境中的单 gap 策略泛化评价 | `multi_gap_eval.csv` | 100 runs | stochastic multi-gap rollout |
| Multi-gap high-level ablation | 多 gap 环境中高层决策模块消融对比 | `multi_gap_opinion_vs_max.csv` | 100 paired runs | shared gap seed and ego seed |

No missing values were detected in these four files. The plotting script does not delete, filter, normalize, or rescale any observation. The only derived curve is the 25-episode trailing rolling mean in the single-gap training panel. This curve is included because it was requested as a trend guide; the raw episode reward is plotted simultaneously and remains the primary evidence.

在这四个文件中没有检测到缺失值。绘图脚本不会删除、筛选、归一化或重新缩放任何观测数据。唯一额外生成的曲线是单 gap 训练图中的 25 episode 后向滑动平均 reward 曲线。该曲线仅作为趋势辅助显示，原始 episode reward 仍然同时绘制，并作为主要证据。

## Column audit

## 数据列检查

`single_gap_sac_train.csv` contains `episode`, `reward`, `progress`, `collision`, `success`, `mean_u`, `ego_x0`, `q1_loss`, `q2_loss`, `policy_loss`, `alpha`, and `total_steps`. The reward range is approximately -342.54 to 239.94, with 95 successful episodes and no collision episodes in the current file.

`single_gap_sac_train.csv` 包含 `episode`、`reward`、`progress`、`collision`、`success`、`mean_u`、`ego_x0`、`q1_loss`、`q2_loss`、`policy_loss`、`alpha` 和 `total_steps` 等列。当前文件中的 reward 范围约为 -342.54 到 239.94，共有 95 个成功 episode，且没有碰撞 episode。

`single_gap_compare.csv` contains paired SAC and RBF results for the same ego initial positions. The key columns are `sac_reward`, `rbf_reward`, `reward_diff`, `sac_progress`, `rbf_progress`, `sac_success`, `rbf_success`, `sac_collision`, `rbf_collision`, `sac_time`, `rbf_time`, `sac_min_distance`, and `rbf_min_distance`. In the current file, both methods have 100% success and 0% collision, while SAC reaches success much earlier and has higher reward.

`single_gap_compare.csv` 包含相同 ego 初始位置下 SAC 策略与 RBF 参考策略的配对结果。关键列包括 `sac_reward`、`rbf_reward`、`reward_diff`、`sac_progress`、`rbf_progress`、`sac_success`、`rbf_success`、`sac_collision`、`rbf_collision`、`sac_time`、`rbf_time`、`sac_min_distance` 和 `rbf_min_distance`。在当前文件中，两种方法的成功率均为 100%，碰撞率均为 0%，但 SAC 策略达到成功所需时间明显更短，并获得更高 reward。

`multi_gap_eval.csv` contains `run`, `low_level`, `seed`, `gap_seed`, `ego_seed`, `ego_x0`, `decision_method`, `reward`, `progress`, `success`, `collision`, `time`, `steps`, `min_distance`, and `switch_count`. The current file uses `low_level = SAC` and `decision_method = opinion` for all rows.

`multi_gap_eval.csv` 包含 `run`、`low_level`、`seed`、`gap_seed`、`ego_seed`、`ego_x0`、`decision_method`、`reward`、`progress`、`success`、`collision`、`time`、`steps`、`min_distance` 和 `switch_count` 等列。当前文件中所有行均使用 `low_level = SAC` 和 `decision_method = opinion`，即底层控制采用单 gap 训练得到的 SAC 策略，高层决策采用意见动力学模块。

`multi_gap_opinion_vs_max.csv` contains paired high-level decision results, including `opinion_reward`, `max_reward`, `reward_diff`, `opinion_success`, `max_success`, `opinion_collision`, `max_collision`, `opinion_time`, `max_time`, and switch-count columns. The pairing unit is one shared `(gap_seed, ego_seed)` pair per run.

`multi_gap_opinion_vs_max.csv` 包含高层意见动力学方法与简单 `max()` 方法的配对消融结果，主要列包括 `opinion_reward`、`max_reward`、`reward_diff`、`opinion_success`、`max_success`、`opinion_collision`、`max_collision`、`opinion_time`、`max_time` 以及切换次数相关列。该对比的配对单位是每一次 run 中共享的一组 `(gap_seed, ego_seed)`，因此两种方法面对的是相同的随机交通场景。

## Visual encoding

## 图形编码方式

The script uses Matplotlib and a color-blind-aware Okabe-Ito subset. Color is paired with marker shape and line style so that the figures remain readable in grayscale. Bar-only summaries are avoided where possible; paired raw observations are shown as points connected by light gray lines. Mean and 95% confidence intervals are overlaid for group-level summaries.

绘图脚本使用 Matplotlib，并采用对色觉缺陷相对友好的 Okabe-Ito 配色子集。图中不仅依赖颜色区分类别，还结合了点形状和线型，因此在灰度打印时仍具有较好的可读性。在可行情况下，脚本避免只使用柱状图汇总结果，而是显示所有原始重复实验点；对于配对实验，则使用浅灰色连线表示同一场景下两种方法之间的对应关系。组级统计结果通过均值和 95% 置信区间叠加显示。

The confidence interval is computed as:

置信区间计算公式为：

$$
\bar{x} \pm 1.96 \frac{s}{\sqrt{n}},
$$

where \( \bar{x} \) is the sample mean, \(s\) is the sample standard deviation, and \(n\) is the number of finite observations. This is a descriptive normal-approximation interval, not a formal hypothesis test.

其中，\( \bar{x} \) 表示样本均值，\(s\) 表示样本标准差，\(n\) 表示有限有效观测值数量。该区间是描述性正态近似置信区间，并不是正式的假设检验结果。

For the single-gap SAC learning curve, the original episode reward is plotted together with a 25-episode rolling mean. This encoding highlights the large reward variation during early exploration and the later stabilization near a local optimum, while still preserving all raw episode-level observations.

对于单 gap SAC 学习曲线，图中同时绘制原始 episode reward 和 25 episode 滑动平均曲线。这样的编码方式用于突出训练前期探索阶段 reward 波动较大、后期在局部最优附近逐渐稳定的过程，同时仍保留所有 episode 级原始观测。

For the multi-gap transfer figure, each run is shown as an independent raw point rather than being connected as a time series. This avoids implying temporal continuity between repeated stochastic rollouts. The y-axis uses a symmetric logarithmic scale to compress the negative reward region while keeping all negative and positive observations visible.

对于多 gap 泛化评价图，每次 run 被显示为独立原始点，而不是用折线连接成时间序列。这可以避免误导读者认为相邻 run 之间具有时间连续关系。该图的 y 轴采用对称对数坐标，用于压缩负 reward 区域，同时仍然保留全部正负 reward 观测值。

## Current descriptive results

## 当前描述性结果

The current single-gap comparison supports the claim that the SAC policy outperforms the RBF reference under the recorded evaluation conditions. The mean SAC reward is about 235.29, while the mean RBF reward is about 115.91. The mean paired reward difference, SAC minus RBF, is about 119.38.

当前单 gap 对比结果支持“SAC 策略在记录的评价条件下优于 RBF 参考策略”这一结论。SAC 策略的平均 reward 约为 235.29，而 RBF 参考策略的平均 reward 约为 115.91。配对 reward 差值，即 SAC 减去 RBF，平均约为 119.38。

The current multi-gap transfer file indicates high transfer success under the recorded setup, with mean reward about 215.35, success rate about 97%, and collision rate about 1%.

当前多 gap 泛化评价文件表明，在记录的实验设置下，单 gap 训练得到的 SAC 策略在多 gap 环境中具有较高的迁移成功率。其平均 reward 约为 215.35，成功率约为 97%，碰撞率约为 1%。

The current high-level ablation file should be interpreted carefully. The mean reward difference, opinion minus max, is positive at about 70.10, but the variability is large. The opinion module has lower success rate and a nonzero collision rate in the current CSV, while the max baseline has 100% success and no collisions in this file. Therefore, the present data do not cleanly support a simple statement that the opinion module is uniformly better. The figure is designed to show this honestly rather than hiding the negative paired runs.

当前高层决策消融结果需要谨慎解释。`opinion - max` 的平均 reward 差值为正，约为 70.10，但结果方差较大。在当前 CSV 文件中，意见动力学高层模块的成功率低于 `max` 基线，并且存在非零碰撞率；而 `max` 基线在该文件中为 100% 成功且没有碰撞。因此，当前数据并不能直接、干净地支持“意见动力学模块在所有情况下都优于 max 方法”这一简单表述。图形设计保留了这些不利样本，而不是通过筛选或隐藏负向配对结果来强化结论。

## Reproducibility

## 可复现性

Run the plotting script from the final experiment directory:

在最终实验文件夹中运行绘图脚本：

```powershell
cd D:\workspace\rl_control\final_experiments
python plot_ieee_analysis.py
```

The script writes PNG files to:

脚本会将 PNG 图像写入以下文件夹：

```text
D:\workspace\rl_control\final_experiments\figures_ieee
```

It also writes `figure_summary_statistics.csv`, which records the descriptive statistics used for captions and results reporting.

脚本还会生成 `figure_summary_statistics.csv`，其中记录了图注和实验结果汇报中可使用的描述性统计量。
