# ball_env

This folder contains a standalone 2D ball obstacle-avoidance environment.

- 9 obstacle balls have radii in `[2, 5]`.
- Each obstacle ball moves on a larger circular orbit around a nearby center.
- Obstacle orbits are checked analytically so obstacle balls cannot collide.
- The default layout guarantees that the straight line from start to goal is blocked.
- The controlled ball has radius `1`.
- The target attracts the controlled ball as an acceleration.
- Obstacle avoidance uses the `example.md` safety-feedback structure:
  `u = u_n + u_c`, `u_c = -sum(k_o * g_j * dot_d_j / d_j)`.
- The environment stops on collision, goal arrival, or timeout.

Run:

```powershell
cd D:\workspace\rl_control\code
python -m ball_env.run_demo --text-only
```

For animation or a static final plot, install `matplotlib`, then run:

```powershell
python -m ball_env.run_demo
python -m ball_env.run_demo --no-animation
```

Useful tuning knobs are in `EnvConfig` inside `ball_env.py`, especially
`k_goal`, `k_damping`, `k_o`, `tangential_gain`, `influence_distance`,
`max_accel`, and `max_speed`.
