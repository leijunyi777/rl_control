# IEEE figure data processing notes

## Source data

The figures are generated from four CSV files in `D:\workspace\rl_control\final_experiments`:

| Research question | CSV file | Sample size | Repeated unit |
|---|---:|---:|---|
| Single-gap SAC learning curve | `single_gap_sac_train.csv` | 100 episodes | episode |
| Single-gap SAC vs. RBF reference | `single_gap_compare.csv` | 100 paired runs | shared ego initial position |
| Multi-gap transfer evaluation | `multi_gap_eval.csv` | 100 runs | stochastic multi-gap rollout |
| Multi-gap high-level ablation | `multi_gap_opinion_vs_max.csv` | 100 paired runs | shared gap seed and ego seed |

No missing values were detected in these four files. The plotting script does not delete, filter, normalize, or rescale any observation. The only derived curve is the 25-episode trailing rolling mean in the single-gap training panel. This curve is included because it was requested as a trend guide; the raw episode reward is plotted simultaneously and remains the primary evidence.

## Column audit

`single_gap_sac_train.csv` contains `episode`, `reward`, `progress`, `collision`, `success`, `mean_u`, `ego_x0`, `q1_loss`, `q2_loss`, `policy_loss`, `alpha`, and `total_steps`. The reward range is approximately -342.54 to 239.94, with 95 successful episodes and no collision episodes in the current file.

`single_gap_compare.csv` contains paired SAC and RBF results for the same ego initial positions. The key columns are `sac_reward`, `rbf_reward`, `reward_diff`, `sac_progress`, `rbf_progress`, `sac_success`, `rbf_success`, `sac_collision`, `rbf_collision`, `sac_time`, `rbf_time`, `sac_min_distance`, and `rbf_min_distance`. In the current file, both methods have 100% success and 0% collision, while SAC reaches success much earlier and has higher reward.

`multi_gap_eval.csv` contains `run`, `low_level`, `seed`, `gap_seed`, `ego_seed`, `ego_x0`, `decision_method`, `reward`, `progress`, `success`, `collision`, `time`, `steps`, `min_distance`, and `switch_count`. The current file uses `low_level = SAC` and `decision_method = opinion` for all rows.

`multi_gap_opinion_vs_max.csv` contains paired high-level decision results, including `opinion_reward`, `max_reward`, `reward_diff`, `opinion_success`, `max_success`, `opinion_collision`, `max_collision`, `opinion_time`, `max_time`, and switch-count columns. The pairing unit is one shared `(gap_seed, ego_seed)` pair per run.

## Visual encoding

The script uses Matplotlib and a color-blind-aware Okabe-Ito subset. Color is paired with marker shape and line style so that the figures remain readable in grayscale. Bar-only summaries are avoided where possible; paired raw observations are shown as points connected by light gray lines. Mean and 95% confidence intervals are overlaid for group-level summaries. The confidence interval is computed as:

$$
\bar{x} \pm 1.96 \frac{s}{\sqrt{n}},
$$

where \( \bar{x} \) is the sample mean, \(s\) is the sample standard deviation, and \(n\) is the number of finite observations. This is a descriptive normal-approximation interval, not a formal hypothesis test.

## Current descriptive results

The current single-gap comparison supports the claim that the SAC policy outperforms the RBF reference under the recorded evaluation conditions. The mean SAC reward is about 235.29, while the mean RBF reward is about 115.91. The mean paired reward difference, SAC minus RBF, is about 119.38.

The current multi-gap transfer file indicates high transfer success under the recorded setup, with mean reward about 215.35, success rate about 97%, and collision rate about 1%.

The current high-level ablation file should be interpreted carefully. The mean reward difference, opinion minus max, is positive at about 20.10, but the variability is large. The opinion module has lower success rate and a nonzero collision rate in the current CSV, while the max baseline has 100% success and no collisions in this file. Therefore, the present data do not cleanly support a simple statement that the opinion module is uniformly better. The figure is designed to show this honestly rather than hiding the negative paired runs.

## Reproducibility

Run the plotting script from the final experiment directory:

```powershell
cd D:\workspace\rl_control\final_experiments
python plot_ieee_analysis.py
```

The script writes PNG files to:

```text
D:\workspace\rl_control\final_experiments\figures_ieee
```

It also writes `figure_summary_statistics.csv`, which records the descriptive statistics used for captions and results reporting.
