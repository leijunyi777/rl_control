"""生成 IEEE 风格实验分析图。

数据来源为当前目录中的 4 个 CSV：
1. single_gap_sac_train.csv
2. single_gap_compare.csv
3. multi_gap_eval.csv
4. multi_gap_opinion_vs_max.csv

处理原则：
- 不删除、不筛选、不归一化原始数据。
- 单 gap 训练图同时显示原始 reward 和 25 episode rolling mean；rolling mean
  仅用于趋势辅助，原始数据完整保留。
- 对比图显示全部重复实验点，并叠加均值和 95% CI。
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = {
    "train": BASE_DIR / "single_gap_sac_train.csv",
    "single_compare": BASE_DIR / "single_gap_compare.csv",
    "multi_eval": BASE_DIR / "multi_gap_eval.csv",
    "multi_compare": BASE_DIR / "multi_gap_opinion_vs_max.csv",
}

OUTPUT_DIR = BASE_DIR / "figures_ieee"
ROLLING_WINDOW = 25

# Okabe-Ito / grayscale-readable subset.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
BLACK = "#000000"
GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"


def configure_style() -> None:
    """设置适合 IEEE 双栏/单栏论文的 Matplotlib 风格。"""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV，不修改任何字段。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(rows: list[dict[str, str]], column: str) -> list[float]:
    """将指定列转换为 float；空值保留为 nan，不删除。"""
    values = []
    for row in rows:
        text = row.get(column, "")
        values.append(float(text) if text != "" else math.nan)
    return values


def finite(values: list[float]) -> list[float]:
    """仅用于统计量计算；绘图原始序列仍保留位置。"""
    return [value for value in values if math.isfinite(value)]


def mean_ci95(values: list[float]) -> tuple[float, float]:
    """返回均值和正态近似 95% CI 半宽。"""
    clean = finite(values)
    if not clean:
        return math.nan, math.nan
    mean = statistics.fmean(clean)
    if len(clean) < 2:
        return mean, 0.0
    sd = statistics.stdev(clean)
    return mean, 1.96 * sd / math.sqrt(len(clean))


def rolling_mean(values: list[float], window: int) -> tuple[list[int], list[float]]:
    """计算 trailing rolling mean；不替代原始 reward 曲线。"""
    x_values = []
    y_values = []
    for index in range(window - 1, len(values)):
        block = finite(values[index - window + 1 : index + 1])
        if len(block) == window:
            x_values.append(index + 1)
            y_values.append(statistics.fmean(block))
    return x_values, y_values


def draw_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def draw_mean_ci(ax, x: float, values: list[float], color: str, label: str | None = None) -> None:
    """叠加均值和 95% CI。"""
    mean, ci = mean_ci95(values)
    ax.errorbar(
        [x],
        [mean],
        yerr=[ci],
        fmt="D",
        color=color,
        markerfacecolor="white",
        markeredgewidth=1.2,
        capsize=3,
        label=label,
        zorder=5,
    )


def plot_training(ax, rows: list[dict[str, str]]) -> None:
    episodes = as_float(rows, "episode")
    rewards = as_float(rows, "reward")
    ax.plot(episodes, rewards, color=LIGHT_GRAY, linewidth=0.9, marker="o", markersize=2.2, label="Episode reward")
    roll_x, roll_y = rolling_mean(rewards, ROLLING_WINDOW)
    ax.plot(roll_x, roll_y, color=BLUE, linewidth=1.8, label=f"{ROLLING_WINDOW}-episode rolling mean")
    ax.axhline(0.0, color=BLACK, linewidth=0.7, linestyle=":")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Single-gap SAC learning")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, color="#E6E6E6", linewidth=0.5)
    draw_panel_label(ax, "A")


def plot_single_compare(ax, rows: list[dict[str, str]]) -> None:
    sac = as_float(rows, "sac_reward")
    rbf = as_float(rows, "rbf_reward")
    for left, right in zip(rbf, sac):
        ax.plot([0, 1], [left, right], color=LIGHT_GRAY, linewidth=0.7, zorder=1)
    ax.scatter([0] * len(rbf), rbf, color=VERMILLION, marker="s", s=16, alpha=0.75, label="RBF raw runs", zorder=3)
    ax.scatter([1] * len(sac), sac, color=BLUE, marker="o", s=16, alpha=0.75, label="SAC raw runs", zorder=3)
    draw_mean_ci(ax, 0, rbf, VERMILLION, "Mean +/- 95% CI")
    draw_mean_ci(ax, 1, sac, BLUE, None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["RBF ref.", "SAC policy"])
    ax.set_ylabel("Reward")
    ax.set_title("Single-gap reward comparison")
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    draw_panel_label(ax, "B")


def plot_multi_generalization(ax, rows: list[dict[str, str]]) -> None:
    runs = as_float(rows, "run")
    rewards = as_float(rows, "reward")
    success = as_float(rows, "success")
    mean_reward, ci_reward = mean_ci95(rewards)
    ax.scatter(runs, rewards, color=BLUE, marker="o", s=16, alpha=0.75, label="Raw runs")
    ax.axhline(mean_reward, color=BLACK, linewidth=1.1, linestyle="--", label="Mean reward")
    ax.fill_between([min(runs), max(runs)], mean_reward - ci_reward, mean_reward + ci_reward, color=BLUE, alpha=0.12, label="95% CI")
    ax.set_xlabel("Run")
    ax.set_ylabel("Reward")
    ax.set_title("Multi-gap transfer evaluation")
    ax.grid(True, color="#E6E6E6", linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")
    success_rate = 100.0 * statistics.fmean(finite(success))
    ax.text(
        0.98,
        1.03,
        f"Success: {success_rate:.1f}%",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=7,
        clip_on=False,
    )
    draw_panel_label(ax, "C")


def plot_multi_ablation(ax, rows: list[dict[str, str]]) -> None:
    opinion = as_float(rows, "opinion_reward")
    max_method = as_float(rows, "max_reward")
    for left, right in zip(max_method, opinion):
        ax.plot([0, 1], [left, right], color=LIGHT_GRAY, linewidth=0.7, zorder=1)
    ax.scatter([0] * len(max_method), max_method, color=VERMILLION, marker="s", s=16, alpha=0.72, label="max raw runs", zorder=3)
    ax.scatter([1] * len(opinion), opinion, color=GREEN, marker="^", s=18, alpha=0.72, label="opinion raw runs", zorder=3)
    draw_mean_ci(ax, 0, max_method, VERMILLION, "Mean +/- 95% CI")
    draw_mean_ci(ax, 1, opinion, GREEN, None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["max baseline", "opinion module"])
    ax.set_ylabel("Reward")
    ax.set_title("High-level decision ablation")
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.5)
    ax.legend(frameon=False, loc="lower left")
    draw_panel_label(ax, "D")


def save_figure(fig, path: Path) -> None:
    """导出 PNG，不使用 bbox_inches='tight'，避免改变物理尺寸。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    print(f"Saved {path}")


def write_summary_csv(data: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    """输出图中使用的描述性统计，便于写论文 caption。"""
    records = []
    specs = [
        ("single_train", "reward", data["train"], "reward"),
        ("single_compare", "SAC reward", data["single_compare"], "sac_reward"),
        ("single_compare", "RBF reward", data["single_compare"], "rbf_reward"),
        ("single_compare", "SAC - RBF reward", data["single_compare"], "reward_diff"),
        ("multi_eval", "reward", data["multi_eval"], "reward"),
        ("multi_eval", "success", data["multi_eval"], "success"),
        ("multi_eval", "collision", data["multi_eval"], "collision"),
        ("multi_compare", "opinion reward", data["multi_compare"], "opinion_reward"),
        ("multi_compare", "max reward", data["multi_compare"], "max_reward"),
        ("multi_compare", "opinion - max reward", data["multi_compare"], "reward_diff"),
        ("multi_compare", "opinion success", data["multi_compare"], "opinion_success"),
        ("multi_compare", "max success", data["multi_compare"], "max_success"),
        ("multi_compare", "opinion collision", data["multi_compare"], "opinion_collision"),
        ("multi_compare", "max collision", data["multi_compare"], "max_collision"),
    ]
    for dataset, metric, rows, column in specs:
        values = as_float(rows, column)
        clean = finite(values)
        mean, ci = mean_ci95(values)
        records.append(
            {
                "dataset": dataset,
                "metric": metric,
                "n": len(clean),
                "mean": mean,
                "ci95_half_width": ci,
                "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
                "min": min(clean) if clean else math.nan,
                "max": max(clean) if clean else math.nan,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "figure_summary_statistics.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {out}")


def make_figures(output_dir: Path) -> None:
    configure_style()
    data = {name: read_csv(path) for name, path in DATA_FILES.items()}

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2), layout="constrained")
    plot_training(axes[0, 0], data["train"])
    plot_single_compare(axes[0, 1], data["single_compare"])
    plot_multi_generalization(axes[1, 0], data["multi_eval"])
    plot_multi_ablation(axes[1, 1], data["multi_compare"])
    save_figure(fig, output_dir / "ieee_experiment_summary.png")
    plt.close(fig)

    single_specs = [
        ("fig1_single_gap_training.png", plot_training, data["train"], (3.5, 2.4)),
        ("fig2_single_gap_policy_vs_rbf.png", plot_single_compare, data["single_compare"], (3.5, 2.4)),
        ("fig3_multi_gap_transfer.png", plot_multi_generalization, data["multi_eval"], (3.5, 2.4)),
        ("fig4_multi_gap_ablation.png", plot_multi_ablation, data["multi_compare"], (3.5, 2.4)),
    ]
    for filename, func, rows, size in single_specs:
        fig, ax = plt.subplots(figsize=size, layout="constrained")
        func(ax, rows)
        save_figure(fig, output_dir / filename)
        plt.close(fig)

    write_summary_csv(data, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IEEE-style analysis figures from final experiment CSV files.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="directory for PNG outputs")
    args = parser.parse_args()
    make_figures(Path(args.output_dir))


if __name__ == "__main__":
    main()
