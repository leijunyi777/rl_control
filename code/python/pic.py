import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =========================
# 可调参数
# =========================
LANE_WIDTH = 4.0
CAR_LENGTH = 4.5
CAR_WIDTH = 2.0

EGO_X = 20.0
EGO_Y = LANE_WIDTH * 0.5
TARGET_LANE_Y = LANE_WIDTH * 1.5

OUTPUT_PNG = "chapter_1_3_lane_change_scene.png"
OUTPUT_SVG = "chapter_1_3_lane_change_scene.svg"


def draw_rounded_car(ax, x, y, length, width, color, alpha=0.85, zorder=5):
    """绘制圆角矩形车辆，车辆中心位于 (x, y)。"""
    lower_left_x = x - length / 2
    lower_left_y = y - width / 2

    car = patches.FancyBboxPatch(
        (lower_left_x, lower_left_y),
        length,
        width,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        linewidth=1.2,
        edgecolor="black",
        facecolor=color,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(car)

    rear_x = x - length * 0.30
    front_x = x + length * 0.30
    ax.plot([rear_x, front_x], [y, y], color="black", linewidth=1.0, alpha=0.6, zorder=zorder + 1)
    ax.plot(rear_x, y, "ko", markersize=2.8, zorder=zorder + 2)
    ax.plot(front_x, y, "ko", markersize=2.8, zorder=zorder + 2)


def draw_road(ax):
    """绘制两车道道路环境。"""
    ax.axhline(0, color="black", linewidth=2.0)
    ax.axhline(LANE_WIDTH, color="gray", linestyle="--", linewidth=1.8)
    ax.axhline(LANE_WIDTH * 2, color="black", linewidth=2.0)

    # 车道说明放到左侧外部，避免与车辆重叠
    ax.text(
        -8.5,
        LANE_WIDTH * 0.5,
        "Original lane",
        ha="left",
        va="center",
        fontsize=9,
        color="dimgray",
    )
    ax.text(
        -8.5,
        LANE_WIDTH * 1.5,
        "Target lane",
        ha="left",
        va="center",
        fontsize=9,
        color="dimgray",
    )


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)

    draw_road(ax)

    # 目标车道连续车流
    traffic_x = [4, 14, 24, 34, 44]
    for i, x in enumerate(traffic_x):
        color = "lightblue" if i % 2 == 0 else "royalblue"
        draw_rounded_car(
            ax,
            x=x,
            y=TARGET_LANE_Y,
            length=CAR_LENGTH,
            width=CAR_WIDTH,
            color=color,
            alpha=0.80,
        )

        # 车辆编号放到车辆下方，不压在车身上
        ax.text(
            x,
            TARGET_LANE_Y - CAR_WIDTH * 0.85,
            f"V{i + 1}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    # ego 车辆
    draw_rounded_car(
        ax,
        x=EGO_X,
        y=EGO_Y,
        length=CAR_LENGTH,
        width=CAR_WIDTH,
        color="lightgreen",
        alpha=0.90,
        zorder=8,
    )

   
    # ego 指向目标车道的并道意图箭头
    ax.annotate(
        "",
        xy=(EGO_X + 6.0, TARGET_LANE_Y - 2.25),
        xytext=(EGO_X + 2.4, EGO_Y + 0.35),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=2.0,
            color="darkgreen",
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=10,
    )

    # 并道意图说明放到箭头下方，不与车辆重叠
    ax.text(
        EGO_X + 5.8,
        EGO_Y + 0.25,
        "Merging intention",
        ha="left",
        va="center",
        fontsize=9,
        color="darkgreen",
    )

    # 车流方向箭头和文字放到目标车道上方
    ax.annotate(
        "",
        xy=(49.0, TARGET_LANE_Y + 2.65),
        xytext=(40.0, TARGET_LANE_Y + 2.65),
        arrowprops=dict(arrowstyle="->", linewidth=1.8, color="black"),
    )
    ax.text(
        39.5,
        TARGET_LANE_Y + 2.65,
        "Traffic flow",
        ha="right",
        va="center",
        fontsize=9,
        color="black",
    )

    # 坐标轴与排版
    ax.set_xlim(-9, 52)
    ax.set_ylim(-1.0, LANE_WIDTH * 2 + 1.0)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Longitudinal position x (m)")
    ax.set_ylabel("Lateral position y (m)")

    # 不添加顶部标题，避免正文中重复解释情景
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUTPUT_PNG, dpi=600)
    fig.savefig(OUTPUT_SVG)
    plt.show()


if __name__ == "__main__":
    main()