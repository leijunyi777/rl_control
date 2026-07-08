import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as transforms


def draw_car(ax, car, wheelbase=None):
    """将车辆绘制为圆角矩形，并显示前后轴连线。"""
    visual_length = float(wheelbase if wheelbase is not None else car.L)
    visual_width = 0.6 * visual_length
    corner_radius = 0.18 * visual_length

    rear = np.array([car.x, car.y])
    heading = np.array([np.cos(car.theta), np.sin(car.theta)])
    normal = np.array([-np.sin(car.theta), np.cos(car.theta)])
    center = rear + 0.5 * visual_length * heading
    lower_left = center - 0.5 * visual_length * heading - 0.5 * visual_width * normal

    car_patch = patches.FancyBboxPatch(
        (lower_left[0], lower_left[1]),
        visual_length,
        visual_width,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        facecolor=car.color,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.75,
    )
    car_patch.set_transform(
        transforms.Affine2D().rotate_around(lower_left[0], lower_left[1], car.theta) + ax.transData
    )
    ax.add_patch(car_patch)

    front = rear + visual_length * heading
    ax.plot(rear[0], rear[1], "ko", markersize=3)
    ax.plot(front[0], front[1], "ko", markersize=3)
    ax.plot([rear[0], front[0]], [rear[1], front[1]], "k-", linewidth=1, alpha=0.6)
    ax.text(car.x, car.y + 1.0, car.id, fontsize=9, color="black", fontweight="bold")


def draw_environment(ax, lane_width=4.0):
    """绘制双车道道路环境。"""
    ax.axhline(0, color="black", linewidth=2.5)
    ax.axhline(lane_width, color="gray", linestyle="--", linewidth=2)
    ax.axhline(lane_width * 2, color="black", linewidth=2.5)
