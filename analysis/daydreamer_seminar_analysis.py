#!/usr/bin/env python3
"""Seminar analysis plots for DayDreamer side/top server runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logdir"
OUTDIR = ROOT / "analysis" / "seminar_plots"
HARDSHIFT_RUN = "run_gruenau_server_hardshift_side2top_01"
TOP_BASELINE_RUN = "run_gruenau_server_top_01"
HARDSHIFT_SHIFT_STEP = 1_000_000
EPISODE_LENGTH = 10_000
TOP_COMPARE_HORIZON = 500_000

RUNS = {
    "Side Brain": "run_gruenau_server_side_01",
    "Top Brain": "run_gruenau_server_top_01",
}

EVAL_CONDITIONS = [
    ("Side->Side (In)", "run_gruenau_server_side_01", "eval_episodes"),
    ("Side->Top (Cross)", "run_gruenau_server_side_01", "cross_eval_top_world_episodes"),
    ("Top->Top (In)", "run_gruenau_server_top_01", "eval_episodes"),
    ("Top->Side (Cross)", "run_gruenau_server_top_01", "cross_eval_side_world_episodes"),
]

COLORS = {
    "Side Brain": "#1f77b4",
    "Top Brain": "#d62728",
}


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_training_scores(run_name: str) -> Tuple[np.ndarray, np.ndarray]:
    scores_path = LOGDIR / run_name / "scores.jsonl"
    rows = read_jsonl(scores_path)
    steps, scores = [], []
    for row in rows:
        if "episode/score" in row:
            steps.append(float(row["step"]))
            scores.append(float(row["episode/score"]))
    if scores:
        return np.array(steps), np.array(scores)

    # Some runs had eval-only writes to scores.jsonl; recover training scores
    # from metrics.jsonl in that case.
    metrics_path = LOGDIR / run_name / "metrics.jsonl"
    rows = read_jsonl(metrics_path)
    for row in rows:
        if "episode/score" in row:
            steps.append(float(row["step"]))
            scores.append(float(row["episode/score"]))
    if not scores:
        raise RuntimeError(f"No training scores found in {run_name}")
    return np.array(steps), np.array(scores)


def load_eval_returns(run_name: str, episode_dir: str) -> np.ndarray:
    folder = LOGDIR / run_name / episode_dir
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No eval episodes found in {folder}")
    returns = []
    for fp in files:
        with np.load(fp) as data:
            if "reward" not in data:
                raise RuntimeError(f"Missing 'reward' key in {fp}")
            returns.append(float(np.sum(data["reward"])))
    return np.array(returns)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.mean(values), dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    out[window - 1 :] = np.convolve(values, kernel, mode="valid")
    return out


def cumulative_mean(values: np.ndarray) -> np.ndarray:
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(training: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, (steps, scores) in training.items():
        color = COLORS[label]
        x = steps / 1e6
        smooth = rolling_mean(scores, window=10)
        run_best = np.maximum.accumulate(scores)

        axes[0].plot(x, scores, color=color, alpha=0.28, linewidth=1.4)
        axes[0].plot(x, smooth, color=color, linewidth=2.6, label=f"{label} (10-ep MA)")
        axes[1].plot(x, run_best, color=color, linewidth=2.2, label=label)
        axes[1].plot(x, cumulative_mean(scores), color=color, linewidth=1.6, linestyle="--", alpha=0.8)

    axes[0].set_title("Episode Return Over Training")
    axes[0].set_xlabel("Environment steps (millions)")
    axes[0].set_ylabel("Episode return")
    axes[0].legend()

    axes[1].set_title("Best Return and Cumulative Mean")
    axes[1].set_xlabel("Environment steps (millions)")
    axes[1].set_ylabel("Episode return")
    axes[1].legend()

    fig.suptitle("Learning Performance: Side vs Top Training", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "training_learning_curves.png", dpi=220)
    plt.close(fig)


def plot_eval_boxplot(eval_returns: Dict[str, np.ndarray]) -> None:
    labels = list(eval_returns.keys())
    values = [eval_returns[label] for label in labels]
    colors = ["#4c78a8", "#9ecae9", "#e15759", "#fcbba1"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    box = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    rng = np.random.default_rng(7)
    for i, vals in enumerate(values, start=1):
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, s=24, c="black", alpha=0.7)

    ax.set_title("Evaluation Performance (10 Episodes per Condition)")
    ax.set_ylabel("Episode return (sum of rewards)")
    ax.tick_params(axis="x", rotation=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "eval_returns_boxplot.png", dpi=220)
    plt.close(fig)


def plot_transfer_heatmap(eval_returns: Dict[str, np.ndarray]) -> None:
    side_side = float(np.mean(eval_returns["Side->Side (In)"]))
    side_top = float(np.mean(eval_returns["Side->Top (Cross)"]))
    top_side = float(np.mean(eval_returns["Top->Side (Cross)"]))
    top_top = float(np.mean(eval_returns["Top->Top (In)"]))

    matrix = np.array([[side_side, side_top], [top_side, top_top]])
    rows = ["Side brain", "Top brain"]
    cols = ["Side world", "Top world"]

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(matrix, cmap="YlGnBu")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean episode return", rotation=90)

    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    ax.set_title("Transfer Matrix: Mean Eval Return")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text_color = "white" if matrix[i, j] > matrix.mean() else "black"
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=12)

    fig.tight_layout()
    fig.savefig(OUTDIR / "eval_transfer_heatmap.png", dpi=220)
    plt.close(fig)


def plot_relative_cross_drop(eval_returns: Dict[str, np.ndarray]) -> None:
    side_in = float(np.mean(eval_returns["Side->Side (In)"]))
    side_cross = float(np.mean(eval_returns["Side->Top (Cross)"]))
    top_in = float(np.mean(eval_returns["Top->Top (In)"]))
    top_cross = float(np.mean(eval_returns["Top->Side (Cross)"]))

    ratios = np.array([100.0 * side_cross / side_in, 100.0 * top_cross / top_in])
    drops = 100.0 - ratios
    labels = ["Side Brain", "Top Brain"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.bar(labels, ratios, color=["#1f77b4", "#d62728"], alpha=0.85)
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_ylim(0, max(110.0, float(np.max(ratios) + 10)))
    ax.set_ylabel("Cross-domain / in-domain return (%)")
    ax.set_title("Generalization Retention Across Viewpoints")

    for bar, ratio, drop in zip(bars, ratios, drops):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            ratio + 1.5,
            f"{ratio:.1f}%\n(-{drop:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(OUTDIR / "eval_cross_domain_retention.png", dpi=220)
    plt.close(fig)


def build_training_summary(training: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[dict]:
    rows = []
    for label, (steps, scores) in training.items():
        final_n = min(10, len(scores))
        auc_norm = float(np.trapz(scores, steps) / (steps[-1] - steps[0]))
        rows.append(
            {
                "run": label,
                "episodes": int(len(scores)),
                "start_score": float(scores[0]),
                "final_score": float(scores[-1]),
                "best_score": float(np.max(scores)),
                "mean_score_all": float(np.mean(scores)),
                "mean_score_final10": float(np.mean(scores[-final_n:])),
                "std_score_final10": float(np.std(scores[-final_n:], ddof=1)),
                "auc_normalized_score": auc_norm,
            }
        )
    return rows


def build_eval_summary(eval_returns: Dict[str, np.ndarray]) -> List[dict]:
    rows = []
    for label, returns in eval_returns.items():
        rows.append(
            {
                "condition": label,
                "episodes": int(len(returns)),
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns, ddof=1)),
                "median_return": float(np.median(returns)),
                "min_return": float(np.min(returns)),
                "max_return": float(np.max(returns)),
            }
        )
    return rows


def build_transfer_summary(eval_returns: Dict[str, np.ndarray]) -> List[dict]:
    side_in = float(np.mean(eval_returns["Side->Side (In)"]))
    side_cross = float(np.mean(eval_returns["Side->Top (Cross)"]))
    top_in = float(np.mean(eval_returns["Top->Top (In)"]))
    top_cross = float(np.mean(eval_returns["Top->Side (Cross)"]))

    return [
        {
            "brain": "Side Brain",
            "in_domain_mean": side_in,
            "cross_domain_mean": side_cross,
            "retention_ratio_pct": 100.0 * side_cross / side_in,
            "drop_pct": 100.0 * (1.0 - side_cross / side_in),
        },
        {
            "brain": "Top Brain",
            "in_domain_mean": top_in,
            "cross_domain_mean": top_cross,
            "retention_ratio_pct": 100.0 * top_cross / top_in,
            "drop_pct": 100.0 * (1.0 - top_cross / top_in),
        },
    ]


def load_last_replay_stats(run_name: str) -> Tuple[float, float]:
    metrics_rows = read_jsonl(LOGDIR / run_name / "metrics.jsonl")
    last = None
    for row in metrics_rows:
        if "replay/replay_steps" in row and "replay/replay_trajs" in row:
            last = row
    if last is None:
        return np.nan, np.nan
    return float(last["replay/replay_steps"]), float(last["replay/replay_trajs"])


def hardshift_split(
    steps: np.ndarray,
    scores: np.ndarray,
    shift_step: int = HARDSHIFT_SHIFT_STEP,
    episode_length: int = EPISODE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre_mask = steps <= (shift_step + episode_length)
    post_mask = steps > (shift_step + episode_length)
    return steps[pre_mask], scores[pre_mask], steps[post_mask], scores[post_mask]


def plot_hardshift_timeline(
    hard_steps: np.ndarray,
    hard_scores: np.ndarray,
    shift_step: int = HARDSHIFT_SHIFT_STEP,
) -> None:
    pre_steps, pre_scores, post_steps, post_scores = hardshift_split(hard_steps, hard_scores, shift_step)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12.5, 5.0))

    ax.plot(hard_steps / 1e6, hard_scores, color="#777777", alpha=0.25, linewidth=1.2, label="Raw score")
    ax.plot(pre_steps / 1e6, rolling_mean(pre_scores, window=8), color="#1f77b4", linewidth=2.6, label="Stage1 side (8-ep MA)")
    ax.plot(post_steps / 1e6, rolling_mean(post_scores, window=8), color="#ff7f0e", linewidth=2.6, label="Stage2 top (8-ep MA)")
    ax.axvline(shift_step / 1e6, color="black", linestyle="--", linewidth=1.4, label="Hard shift @1.0M")

    ax.set_title("Hard Shift Timeline: Side -> Top in One Run")
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Episode return")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTDIR / "hardshift_timeline.png", dpi=220)
    plt.close(fig)


def plot_hardshift_vs_top_500k(
    hard_steps: np.ndarray,
    hard_scores: np.ndarray,
    top_steps: np.ndarray,
    top_scores: np.ndarray,
) -> None:
    _, _, post_steps, post_scores = hardshift_split(hard_steps, hard_scores)
    top_mask = top_steps <= TOP_COMPARE_HORIZON
    top_steps = top_steps[top_mask]
    top_scores = top_scores[top_mask]

    hard_rel = post_steps - post_steps[0]
    top_rel = top_steps - top_steps[0]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.plot(hard_rel / 1e6, post_scores, color="#ff7f0e", alpha=0.2, linewidth=1.2)
    ax.plot(top_rel / 1e6, top_scores, color="#d62728", alpha=0.2, linewidth=1.2)
    ax.plot(hard_rel / 1e6, rolling_mean(post_scores, window=8), color="#ff7f0e", linewidth=2.8, label="Hard-shift Stage2 (8-ep MA)")
    ax.plot(top_rel / 1e6, rolling_mean(top_scores, window=8), color="#d62728", linewidth=2.8, label="Top-only baseline (8-ep MA)")
    ax.set_title("Post-Shift Adaptation vs Top-Only Baseline (first 500k)")
    ax.set_xlabel("Relative environment steps from segment start (millions)")
    ax.set_ylabel("Episode return")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "hardshift_vs_top500k.png", dpi=220)
    plt.close(fig)


def plot_hardshift_summary_bars(summary_row: dict) -> None:
    labels = [
        "Pre-shift\nlast10 mean",
        "Post-shift\nfirst10 mean",
        "Post-shift\nlast10 mean",
        "Top baseline\nlast10 <=500k",
    ]
    values = [
        float(summary_row["pre_shift_last10_mean"]),
        float(summary_row["post_shift_first10_mean"]),
        float(summary_row["post_shift_last10_mean"]),
        float(summary_row["top_baseline_last10_mean_500k"]),
    ]
    colors = ["#1f77b4", "#ffbb78", "#ff7f0e", "#d62728"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_title("Hard Shift Summary: Collapse and Partial Recovery")
    ax.set_ylabel("Episode return")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 20, f"{val:.0f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTDIR / "hardshift_summary_bars.png", dpi=220)
    plt.close(fig)


def build_hardshift_summary(
    hard_steps: np.ndarray,
    hard_scores: np.ndarray,
    top_steps: np.ndarray,
    top_scores: np.ndarray,
) -> List[dict]:
    _, _, post_steps, post_scores = hardshift_split(hard_steps, hard_scores)
    pre_steps, pre_scores, _, _ = hardshift_split(hard_steps, hard_scores)

    top_mask = top_steps <= TOP_COMPARE_HORIZON
    top_scores_500k = top_scores[top_mask]

    pre_last10 = pre_scores[-10:]
    post_first10 = post_scores[:10]
    post_last10 = post_scores[-10:]
    top_last10 = top_scores_500k[-10:]
    replay_steps, replay_trajs = load_last_replay_stats(HARDSHIFT_RUN)
    replay_capacity_eps = replay_trajs if np.isfinite(replay_trajs) else np.nan
    top_fraction_end = (
        min(1.0, len(post_scores) / replay_capacity_eps) if np.isfinite(replay_capacity_eps) and replay_capacity_eps > 0 else np.nan
    )
    old_fraction_end = (1.0 - top_fraction_end) if np.isfinite(top_fraction_end) else np.nan

    row = {
        "shift_step": HARDSHIFT_SHIFT_STEP,
        "pre_shift_episodes": int(len(pre_scores)),
        "post_shift_episodes": int(len(post_scores)),
        "pre_shift_last10_mean": float(np.mean(pre_last10)),
        "post_shift_first10_mean": float(np.mean(post_first10)),
        "post_shift_last10_mean": float(np.mean(post_last10)),
        "post_shift_mean_500k": float(np.mean(post_scores)),
        "post_shift_best_500k": float(np.max(post_scores)),
        "top_baseline_mean_500k": float(np.mean(top_scores_500k)),
        "top_baseline_best_500k": float(np.max(top_scores_500k)),
        "top_baseline_last10_mean_500k": float(np.mean(top_last10)),
        "recovery_vs_pre_last10_pct": 100.0 * float(np.mean(post_last10)) / float(np.mean(pre_last10)),
        "recovery_vs_top_last10_pct": 100.0 * float(np.mean(post_last10)) / float(np.mean(top_last10)),
        "initial_drop_vs_pre_last10_pct": 100.0 * (1.0 - float(np.mean(post_first10)) / float(np.mean(pre_last10))),
        "post_shift_start_step": float(post_steps[0]),
        "post_shift_end_step": float(post_steps[-1]),
        "replay_steps_end": replay_steps,
        "replay_trajs_end": replay_trajs,
        "estimated_top_data_fraction_end_pct": 100.0 * top_fraction_end if np.isfinite(top_fraction_end) else np.nan,
        "estimated_old_side_data_fraction_end_pct": 100.0 * old_fraction_end if np.isfinite(old_fraction_end) else np.nan,
    }
    return [row]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    training = {label: load_training_scores(run) for label, run in RUNS.items()}
    eval_returns = {label: load_eval_returns(run, ep_dir) for label, run, ep_dir in EVAL_CONDITIONS}

    plot_training_curves(training)
    plot_eval_boxplot(eval_returns)
    plot_transfer_heatmap(eval_returns)
    plot_relative_cross_drop(eval_returns)

    hard_steps, hard_scores = load_training_scores(HARDSHIFT_RUN)
    top_steps, top_scores = load_training_scores(TOP_BASELINE_RUN)
    plot_hardshift_timeline(hard_steps, hard_scores)
    plot_hardshift_vs_top_500k(hard_steps, hard_scores, top_steps, top_scores)

    training_summary = build_training_summary(training)
    eval_summary = build_eval_summary(eval_returns)
    transfer_summary = build_transfer_summary(eval_returns)
    hardshift_summary = build_hardshift_summary(hard_steps, hard_scores, top_steps, top_scores)
    plot_hardshift_summary_bars(hardshift_summary[0])

    write_csv(training_summary, OUTDIR / "training_summary.csv")
    write_csv(eval_summary, OUTDIR / "eval_summary.csv")
    write_csv(transfer_summary, OUTDIR / "transfer_summary.csv")
    write_csv(hardshift_summary, OUTDIR / "hardshift_summary.csv")

    print(f"Wrote plots and tables to: {OUTDIR}")
    for row in transfer_summary:
        print(
            f"{row['brain']}: in={row['in_domain_mean']:.1f}, cross={row['cross_domain_mean']:.1f}, "
            f"retention={row['retention_ratio_pct']:.1f}%"
        )
    hs = hardshift_summary[0]
    print(
        "Hard shift: pre-last10={:.1f}, post-first10={:.1f}, post-last10={:.1f}, top-last10@500k={:.1f}".format(
            hs["pre_shift_last10_mean"],
            hs["post_shift_first10_mean"],
            hs["post_shift_last10_mean"],
            hs["top_baseline_last10_mean_500k"],
        )
    )


if __name__ == "__main__":
    main()
