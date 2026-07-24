"""Run a visual demo of the ball avoidance environment."""

from __future__ import annotations

import argparse

try:
    from .ball_env import BallAvoidanceEnv
except ImportError:
    from ball_env import BallAvoidanceEnv


def simulate(env: BallAvoidanceEnv, max_steps: int):
    env.reset()
    frames = [env.pos]
    infos = []
    for _ in range(max_steps):
        _, _, done, info = env.step()
        frames.append(env.pos)
        infos.append(info)
        if done:
            break
    return frames, infos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2500)
    args = parser.parse_args()

    env = BallAvoidanceEnv()
    _, infos = simulate(env, args.max_steps)
    final_info = infos[-1] if infos else {"stop_reason": "not started", "min_clearance": 0.0}
    print(
        "Finished: "
        f"reason={final_info['stop_reason']}, "
        f"time={final_info['time']:.2f}s, "
        f"min_clearance={final_info['min_clearance']:.2f}"
    )

    if args.text_only:
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib is not installed; use --text-only or install matplotlib for plots.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if args.no_animation:
        env.render(ax)
        plt.show()
        return

    replay_env = BallAvoidanceEnv()
    replay_env.reset()

    def update(_frame_index):
        if not replay_env.done:
            replay_env.step()
        replay_env.render(ax)
        return ax.patches + ax.lines

    anim = FuncAnimation(
        fig,
        update,
        frames=max(len(env.path), 1),
        interval=30,
        blit=False,
        repeat=False,
    )
    fig._ball_env_animation = anim
    plt.show()


if __name__ == "__main__":
    main()
