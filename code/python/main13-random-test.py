import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import main13_common as m13
from models_ode import front_velocity


NUM_TEST_RUNS = 10
ENABLE_EGO_CONTROL = True
USE_RL_U = m13.USE_RL_U
TEST_RANDOM_SEED = None
PRINT_EACH_RUN = True

RESULT_FIG_PATH = "main13_random_test_result.png"
RESULT_CSV_PATH = "main13_random_test_result.csv"


def lane_progress(ego_state):
    original_lane_y = m13.LANE_WIDTH * 0.5
    target_lane_y = m13.LANE_WIDTH * 1.5
    return float(np.clip((ego_state[1] - original_lane_y) / (target_lane_y - original_lane_y), 0.0, 1.0))


def normalized_u(u_t):
    width = max(m13.RL_U_HIGH - m13.RL_U_LOW, 1e-6)
    return float(np.clip(2.0 * (u_t - m13.RL_U_LOW) / width - 1.0, -1.0, 1.0))


def compute_step_reward(diag, previous_progress, previous_u_norm, previous_lateral_velocity, collision_radius, t):
    current_progress = lane_progress(diag["ego_state"])
    progress_delta = current_progress - previous_progress
    best_gap = float(np.max(diag["gap_values"])) if len(diag["gap_values"]) > 0 else 0.0
    opportunity = 1.0 if best_gap > m13.GAP_SAFE else 0.0

    current_u_norm = normalized_u(float(diag["u_t"]))
    current_lateral_velocity = front_velocity(diag["ego_state"], m13.VEHICLE_L)[1]
    lateral_direction_flip = (
        previous_lateral_velocity * current_lateral_velocity < 0.0
        and abs(previous_lateral_velocity) > 1e-3
        and abs(current_lateral_velocity) > 1e-3
    )

    safe_margin = 2.5 * collision_radius
    progress_reward = 80.0 * progress_delta
    lane_progress_reward = 0.15 * current_progress
    opportunity_reward = 40.0 * opportunity * max(progress_delta, 0.0)
    reverse_progress_penalty = -30.0 * max(-progress_delta, 0.0)
    hesitation_penalty = -0.25 * opportunity * (1.0 - current_progress)
    time_penalty = -0.05 * (1.0 - current_progress)
    action_smooth_penalty = -0.5 * (current_u_norm - previous_u_norm) ** 2
    direction_flip_penalty = -2.0 if lateral_direction_flip else 0.0
    safety_penalty = -20.0 * max(0.0, (safe_margin - diag["min_distance"]) / safe_margin) ** 2
    collision_penalty = -1000.0 if diag["collision"] else 0.0
    success_bonus = (100.0 - 2.0 * t) if diag["success"] else 0.0

    terms = {
        "progress": progress_reward,
        "lane_progress": lane_progress_reward,
        "opportunity": opportunity_reward,
        "reverse_progress": reverse_progress_penalty,
        "hesitation": hesitation_penalty,
        "time": time_penalty,
        "action_smooth": action_smooth_penalty,
        "direction_flip": direction_flip_penalty,
        "safety": safety_penalty,
        "collision": collision_penalty,
        "success": success_bonus,
    }
    return float(sum(terms.values())), terms, current_progress, current_u_norm, current_lateral_velocity


def run_one_episode(run_index, base_rng):
    if TEST_RANDOM_SEED is None:
        gap_seed = None
        ego_seed = None
    else:
        gap_seed = int(base_rng.integers(0, 2**31 - 1))
        ego_seed = int(base_rng.integers(0, 2**31 - 1))

    old_gap_seed = m13.GAP_RANDOM_SEED
    old_ego_seed = m13.EGO_RANDOM_SEED
    old_use_rl_u = m13.USE_RL_U
    m13.GAP_RANDOM_SEED = gap_seed
    m13.EGO_RANDOM_SEED = ego_seed
    m13.USE_RL_U = USE_RL_U

    try:
        dynamics = m13.build_scene(enable_ego_control=ENABLE_EGO_CONTROL, use_rl_u=USE_RL_U)
    finally:
        m13.GAP_RANDOM_SEED = old_gap_seed
        m13.EGO_RANDOM_SEED = old_ego_seed
        m13.USE_RL_U = old_use_rl_u

    state = dynamics.pack_state()
    initial_ego_x = float(dynamics.ego.x)
    total_reward = 0.0
    term_totals = None
    final_diag = dynamics.diagnostics(state, 0.0)
    previous_progress = lane_progress(final_diag["ego_state"])
    previous_u_norm = normalized_u(float(final_diag["u_t"]))
    previous_lateral_velocity = front_velocity(final_diag["ego_state"], m13.VEHICLE_L)[1]
    steps = 0

    for step_index in range(int(m13.SIM_TIME / m13.DT)):
        t0 = step_index * m13.DT
        t1 = t0 + m13.DT
        sol = solve_ivp(
            dynamics.rhs,
            (t0, t1),
            state,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            max_step=m13.DT / 5.0,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        state = sol.y[:, -1]
        dynamics.apply_state(state)
        state = np.concatenate(
            [vehicle.get_state() for vehicle in dynamics.target_vehicles]
            + [dynamics.ego.get_state(), state[-3:]]
        )
        final_diag = dynamics.diagnostics(state, t1)
        reward, terms, previous_progress, previous_u_norm, previous_lateral_velocity = compute_step_reward(
            final_diag,
            previous_progress,
            previous_u_norm,
            previous_lateral_velocity,
            dynamics.ego.r,
            t1,
        )
        total_reward += reward
        steps += 1

        if term_totals is None:
            term_totals = {key: 0.0 for key in terms}
        for key, value in terms.items():
            term_totals[key] += value

        if final_diag["collision"] and m13.STOP_ON_COLLISION:
            break
        if final_diag["success"] and m13.STOP_ON_SUCCESS:
            break

    if term_totals is None:
        term_totals = {}

    return {
        "run": run_index,
        "reward": float(total_reward),
        "progress": float(previous_progress),
        "success": bool(final_diag["success"]),
        "collision": bool(final_diag["collision"]),
        "time": float(steps * m13.DT),
        "steps": int(steps),
        "ego_x0": initial_ego_x,
        "min_distance": float(final_diag["min_distance"]),
        "final_decision": final_diag["decision"],
        "gap_seed": -1 if gap_seed is None else gap_seed,
        "ego_seed": -1 if ego_seed is None else ego_seed,
        "terms": term_totals,
    }


def save_results_csv(results, path):
    term_names = sorted(results[0]["terms"].keys()) if results else []
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "run",
            "reward",
            "progress",
            "success",
            "collision",
            "time",
            "steps",
            "ego_x0",
            "min_distance",
            "final_decision",
            "gap_seed",
            "ego_seed",
            *[f"term_{name}" for name in term_names],
        ])
        for result in results:
            writer.writerow([
                result["run"],
                result["reward"],
                result["progress"],
                int(result["success"]),
                int(result["collision"]),
                result["time"],
                result["steps"],
                result["ego_x0"],
                result["min_distance"],
                result["final_decision"],
                result["gap_seed"],
                result["ego_seed"],
                *[result["terms"].get(name, 0.0) for name in term_names],
            ])


def plot_results(results, fig_path):
    rewards = np.array([result["reward"] for result in results], dtype=float)
    progress = np.array([result["progress"] for result in results], dtype=float)
    times = np.array([result["time"] for result in results], dtype=float)
    success = np.array([result["success"] for result in results], dtype=float)
    collision = np.array([result["collision"] for result in results], dtype=float)
    min_distances = np.array([result["min_distance"] for result in results], dtype=float)
    run_ids = np.arange(1, len(results) + 1)

    term_names = sorted(results[0]["terms"].keys()) if results else []
    mean_terms = np.array([
        np.mean([result["terms"].get(name, 0.0) for result in results])
        for name in term_names
    ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(run_ids, rewards, "o-", color="tab:blue", label="Episode reward")
    axes[0, 0].axhline(np.mean(rewards), color="black", linestyle="--", label=f"Mean={np.mean(rewards):.2f}")
    axes[0, 0].fill_between(run_ids, np.mean(rewards) - np.std(rewards), np.mean(rewards) + np.std(rewards), color="tab:blue", alpha=0.12, label="Mean +/- std")
    axes[0, 0].set_title("Reward per Random Run")
    axes[0, 0].set_xlabel("Run")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(run_ids, progress, "o-", color="tab:green", label="Final progress")
    axes[0, 1].plot(run_ids, times / max(m13.SIM_TIME, 1e-6), "s--", color="tab:orange", label="Time ratio")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_title("Progress and Finish Time")
    axes[0, 1].set_xlabel("Run")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].bar(["Success rate", "Collision rate"], [np.mean(success), np.mean(collision)], color=["tab:purple", "tab:red"])
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("Outcome Rates")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].grid(True, axis="y")

    if len(term_names) > 0:
        colors = ["tab:blue" if value >= 0.0 else "tab:red" for value in mean_terms]
        axes[1, 1].barh(term_names, mean_terms, color=colors)
    axes[1, 1].set_title("Mean Reward Terms")
    axes[1, 1].set_xlabel("Mean contribution")
    axes[1, 1].grid(True, axis="x")

    fig.suptitle(
        f"Main13 Random Test | runs={len(results)} | mean={np.mean(rewards):.2f} | "
        f"std={np.std(rewards):.2f} | success={np.mean(success):.2%} | collision={np.mean(collision):.2%}"
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    print(f"Saved result figure to {fig_path}")

    print("\nMain13 random test summary")
    print(f"Runs: {len(results)}")
    print(f"Reward mean/std/min/max: {np.mean(rewards):.3f} / {np.std(rewards):.3f} / {np.min(rewards):.3f} / {np.max(rewards):.3f}")
    print(f"Progress mean/std: {np.mean(progress):.3f} / {np.std(progress):.3f}")
    print(f"Success rate: {np.mean(success):.2%}")
    print(f"Collision rate: {np.mean(collision):.2%}")
    print(f"Mean finish time: {np.mean(times):.3f}s")
    print(f"Minimum distance mean/min: {np.mean(min_distances):.3f}m / {np.min(min_distances):.3f}m")


def main():
    base_rng = np.random.default_rng(TEST_RANDOM_SEED)
    results = []
    print(f"Main13 random test starts | runs={NUM_TEST_RUNS} | ego_control={ENABLE_EGO_CONTROL} | use_rl_u={USE_RL_U}")
    print(f"Config: vehicles={m13.NUM_TARGET_VEHICLES}, base_gap={m13.BASE_GAP}, gap_safe={m13.GAP_SAFE}, ego_x={m13.EGO_X_BASE}+/-{m13.EGO_X_RANDOM_RANGE}")

    for run_index in range(1, NUM_TEST_RUNS + 1):
        result = run_one_episode(run_index, base_rng)
        results.append(result)
        if PRINT_EACH_RUN:
            print(
                f"Run {run_index:03d} | reward={result['reward']:9.3f} | progress={result['progress']:.3f} | "
                f"success={result['success']} | collision={result['collision']} | time={result['time']:.2f}s | "
                f"ego_x0={result['ego_x0']:.3f} | min_dist={result['min_distance']:.3f}m"
            )

    if not results:
        print("No runs were executed.")
        return

    save_results_csv(results, RESULT_CSV_PATH)
    print(f"Saved result data to {RESULT_CSV_PATH}")
    plot_results(results, RESULT_FIG_PATH)
    plt.show()


if __name__ == "__main__":
    main()
