"""绘制论文方法与实验环境示意图。

输出两张 PNG：
1. layered_opinion_decision_schematic.png
   展示双层/多层意见动力学如何在前后 gap 中进行决策。
2. multi_gap_environment_schematic.png
   展示当前多 gap 仿真环境，包括 5 辆目标车、4 个 gap 和 ego 随机初始区域。

本脚本仅用于论文示意图绘制，不读取或修改仿真数据。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# =========================
# 与当前 final_experiments 环境一致的关键参数
# =========================
LANE_WIDTH = 4.0
VEHICLE_L = 2.8
CAR_DRAW_LENGTH = 3.0
CAR_DRAW_WIDTH = 1.2
COLLISION_R = 1.5

TARGET_SPEED = 15.0
NUM_TARGET_VEHICLES = 5
BASE_GAP = 8.0
GAP_MULTIPLIERS = (0.75, 1.0, 1.25, 1.5)

MULTI_EGO_X_MIN = 20.0
MULTI_EGO_X_MAX = 45.0
MULTI_EGO_X_BASE = 0.5 * (MULTI_EGO_X_MIN + MULTI_EGO_X_MAX)
MULTI_EGO_X_RANDOM_RANGE = 0.5 * (MULTI_EGO_X_MAX - MULTI_EGO_X_MIN)

TARGET_LANE_Y = LANE_WIDTH * 1.5
EGO_LANE_Y = LANE_WIDTH * 0.5

OUTPUT_DIR = Path(__file__).resolve().parent / "figures_ieee"


# Okabe-Ito / Wong colors, chosen for color-blind and grayscale readability.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
BLACK = "#000000"
GRAY = "#777777"
LIGHT_GRAY = "#E6E6E6"
LANE_FILL = "#F7F7F7"
GAP_FILL = "#F0F6FF"
EGO_ZONE_FILL = "#DCEBFF"


def configure_style() -> None:
    """设置论文图风格。"""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_road(ax, x_min: float, x_max: float) -> None:
    """绘制两车道道路背景。"""
    ax.add_patch(Rectangle((x_min, 0.0), x_max - x_min, 2.0 * LANE_WIDTH, facecolor=LANE_FILL, edgecolor="none", zorder=0))
    ax.axhline(0.0, color=BLACK, linewidth=1.0)
    ax.axhline(LANE_WIDTH, color=GRAY, linestyle=(0, (5, 4)), linewidth=0.9)
    ax.axhline(2.0 * LANE_WIDTH, color=BLACK, linewidth=1.0)
    ax.text(x_min + 0.5, TARGET_LANE_Y + 0.95, "Target lane", color=BLACK, fontsize=8, va="center")
    ax.text(x_min + 0.5, EGO_LANE_Y - 0.95, "Ego lane", color=BLACK, fontsize=8, va="center")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.7, 2.0 * LANE_WIDTH + 0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitudinal position x (m)")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_car(ax, x: float, y: float, label: str, color: str, alpha: float = 1.0) -> None:
    """绘制圆角矩形车辆。"""
    patch = FancyBboxPatch(
        (x - CAR_DRAW_LENGTH / 2.0, y - CAR_DRAW_WIDTH / 2.0),
        CAR_DRAW_LENGTH,
        CAR_DRAW_WIDTH,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        facecolor=color,
        edgecolor=BLACK,
        linewidth=0.8,
        alpha=alpha,
        zorder=4,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=7, zorder=5)


def draw_gap_span(ax, x_left: float, x_right: float, y: float, label: str, color: str) -> None:
    """在两车之间绘制 gap 范围。"""
    x0 = x_left + CAR_DRAW_LENGTH / 2.0
    x1 = x_right - CAR_DRAW_LENGTH / 2.0
    ax.add_patch(Rectangle((x0, y - 0.52), x1 - x0, 1.04, facecolor=GAP_FILL, edgecolor=color, linewidth=0.8, alpha=0.75, zorder=1))
    arrow = FancyArrowPatch((x0, y), (x1, y), arrowstyle="<->", mutation_scale=9, color=color, linewidth=1.1, zorder=6)
    ax.add_patch(arrow)
    ax.text((x0 + x1) / 2.0, y + 0.75, label, ha="center", va="bottom", color=color, fontsize=8, zorder=7)


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str, label: str) -> None:
    """绘制从 ego 指向候选 gap 的箭头。"""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        connectionstyle="arc3,rad=0.18",
        color=color,
        linewidth=1.3,
        zorder=8,
    )
    ax.add_patch(arrow)
    if label:
        mid_x = 0.5 * (start[0] + end[0])
        mid_y = 0.5 * (start[1] + end[1])
        ax.text(mid_x, mid_y, label, color=color, fontsize=7, ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2}, zorder=9)


def draw_box(ax, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    """绘制方法流程框。"""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="white",
        edgecolor=color,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2.0, xy[1] + height / 2.0, text, ha="center", va="center", fontsize=7, color=BLACK, zorder=3)


def connect_boxes(ax, start: tuple[float, float], end: tuple[float, float], color: str = GRAY) -> None:
    """绘制流程框之间的箭头。"""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color=color, linewidth=0.9, zorder=1))


def plot_layered_opinion_decision(output_dir: Path) -> None:
    """图 1：多层意见动力学决策示意。"""
    configure_style()
    fig = plt.figure(figsize=(7.16, 3.25), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax_scene = fig.add_subplot(gs[0, 0])
    ax_flow = fig.add_subplot(gs[0, 1])

    # 三辆目标车形成两个候选 gap；ego 从邻车道同时评估前后 gap。
    target_x = [20.0, 32.0, 44.0]
    ego_x = 32.0
    draw_road(ax_scene, 12.0, 52.0)
    draw_gap_span(ax_scene, target_x[0], target_x[1], TARGET_LANE_Y, r"rear gap $G_r$", VERMILLION)
    draw_gap_span(ax_scene, target_x[1], target_x[2], TARGET_LANE_Y, r"front gap $G_f$", BLUE)
    draw_car(ax_scene, target_x[0], TARGET_LANE_Y, "V1", "#88CCEE")
    draw_car(ax_scene, target_x[1], TARGET_LANE_Y, "V2", "#44AA99")
    draw_car(ax_scene, target_x[2], TARGET_LANE_Y, "V3", "#88CCEE")
    draw_car(ax_scene, ego_x, EGO_LANE_Y, "Ego", "#B7E4A8")

    rear_center = ((target_x[0] + CAR_DRAW_LENGTH / 2.0 + target_x[1] - CAR_DRAW_LENGTH / 2.0) / 2.0, TARGET_LANE_Y - 0.6)
    front_center = ((target_x[1] + CAR_DRAW_LENGTH / 2.0 + target_x[2] - CAR_DRAW_LENGTH / 2.0) / 2.0, TARGET_LANE_Y - 0.6)
    draw_arrow(ax_scene, (ego_x - 0.55, EGO_LANE_Y + 0.7), rear_center, VERMILLION, "")
    draw_arrow(ax_scene, (ego_x + 0.55, EGO_LANE_Y + 0.7), front_center, BLUE, "")
    ax_scene.set_title("Candidate gaps observed by ego vehicle")

    # 右侧展示高层与底层意见动力学的逻辑。
    ax_flow.set_axis_off()
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    draw_box(ax_flow, (0.08, 0.78), 0.84, 0.14, "Relative state\n(position, velocity)", BLUE)
    draw_box(ax_flow, (0.08, 0.58), 0.84, 0.14, "High-level opinion\nfront vs. rear gap bias", PURPLE)
    draw_box(ax_flow, (0.08, 0.38), 0.84, 0.14, r"Decision state\n$\dot y=-d_hy+u_h\tanh(\alpha_h y)+b_h$", PURPLE)
    draw_box(ax_flow, (0.08, 0.18), 0.84, 0.14, "Selected target gap\nFRONT / REAR / WAIT", GREEN)
    draw_box(ax_flow, (0.08, 0.00), 0.84, 0.12, r"Low-level control\n$u(t)\rightarrow \dot z \rightarrow p_d$", VERMILLION)
    connect_boxes(ax_flow, (0.50, 0.78), (0.50, 0.72))
    connect_boxes(ax_flow, (0.50, 0.58), (0.50, 0.52))
    connect_boxes(ax_flow, (0.50, 0.38), (0.50, 0.32))
    connect_boxes(ax_flow, (0.50, 0.18), (0.50, 0.12))
    ax_flow.set_title("Layered opinion-dynamics decision")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "layered_opinion_decision_schematic.png"
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f"Saved {png_path}")


def plot_multi_gap_environment(output_dir: Path) -> None:
    """图 2：当前多 gap 实验环境示意。"""
    configure_style()
    fig, ax = plt.subplots(figsize=(7.16, 2.75), layout="constrained")

    # 对应 multi_gap_env.py: x = 48 - index * BASE_GAP。
    target_x = [48.0 - i * BASE_GAP for i in range(NUM_TARGET_VEHICLES)]
    x_min = min(target_x) - 8.0
    x_max = max(target_x) + 8.0
    draw_road(ax, x_min, x_max)

    # ego 随机初始区域：x in [20, 40] m。
    ax.add_patch(
        Rectangle(
            (MULTI_EGO_X_MIN, EGO_LANE_Y - 0.85),
            MULTI_EGO_X_MAX - MULTI_EGO_X_MIN,
            1.7,
            facecolor=EGO_ZONE_FILL,
            edgecolor=BLUE,
            linewidth=1.0,
            hatch="///",
            alpha=0.55,
            zorder=1,
        )
    )
    ax.text(
        MULTI_EGO_X_BASE,
        EGO_LANE_Y - 1.25,
        rf"random ego initial region: $x_0 \in [{MULTI_EGO_X_MIN:.0f},{MULTI_EGO_X_MAX:.0f}]$ m",
        ha="center",
        va="top",
        color=BLUE,
        fontsize=8,
    )

    colors = ["#88CCEE", "#44AA99", "#DDCC77", "#CC6677", "#AA4499"]
    sorted_x = sorted(target_x)
    for i in range(len(sorted_x) - 1):
        draw_gap_span(ax, sorted_x[i], sorted_x[i + 1], TARGET_LANE_Y, rf"$G_{i+1}$", GRAY)
    for index, x in enumerate(target_x):
        draw_car(ax, x, TARGET_LANE_Y, f"V{index + 1}", colors[index % len(colors)])

    draw_car(ax, MULTI_EGO_X_BASE, EGO_LANE_Y, "Ego", "#B7E4A8", alpha=0.95)
    ax.set_title("Multi-gap experimental environment")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "multi_gap_environment_schematic.png"
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f"Saved {png_path}")


def main() -> None:
    plot_layered_opinion_decision(OUTPUT_DIR)
    plot_multi_gap_environment(OUTPUT_DIR)


if __name__ == "__main__":
    main()
