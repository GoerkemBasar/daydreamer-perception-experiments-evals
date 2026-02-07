#!/usr/bin/env python3
"""Generate seminar analysis plots for DayDreamer side/top server runs."""

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
    rows = read_jsonl(LOGDIR / run_name / "scores.jsonl")
    steps, scores = [], []
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


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    training = {label: load_training_scores(run) for label, run in RUNS.items()}
    eval_returns = {label: load_eval_returns(run, ep_dir) for label, run, ep_dir in EVAL_CONDITIONS}

    plot_training_curves(training)
    plot_eval_boxplot(eval_returns)
    plot_transfer_heatmap(eval_returns)
    plot_relative_cross_drop(eval_returns)

    training_summary = build_training_summary(training)
    eval_summary = build_eval_summary(eval_returns)
    transfer_summary = build_transfer_summary(eval_returns)

    write_csv(training_summary, OUTDIR / "training_summary.csv")
    write_csv(eval_summary, OUTDIR / "eval_summary.csv")
    write_csv(transfer_summary, OUTDIR / "transfer_summary.csv")

    print(f"Wrote plots and tables to: {OUTDIR}")
    for row in transfer_summary:
        print(
            f"{row['brain']}: in={row['in_domain_mean']:.1f}, cross={row['cross_domain_mean']:.1f}, "
            f"retention={row['retention_ratio_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()

