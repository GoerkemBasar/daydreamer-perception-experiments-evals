#!/usr/bin/env python3
"""Generate slide-ready camera pose graphics from custom walker camera setup."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "analysis" / "seminar_plots"


CAMERAS = {
    "Side Camera": {
        "task": "dmc_custom_walker_side",
        "mode": "trackcom",
        "target": "torso",
        "pos": np.array([0.0, -3.0, 1.0]),
        "xyaxes": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "color": "#1f77b4",
    },
    "Top Camera": {
        "task": "dmc_custom_walker_top",
        "mode": "trackcom",
        "target": "torso",
        "pos": np.array([0.0, 0.0, 4.0]),
        "xyaxes": np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0]),
        "color": "#d62728",
    },
}

VIEW_IMAGES = {
    "Side Camera": ROOT / "side.png",
    "Top Camera": ROOT / "top.png",
}


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec.copy()
    return vec / norm


def camera_vectors(cfg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    x_axis = normalize(cfg["xyaxes"][:3])
    y_axis = normalize(cfg["xyaxes"][3:])
    z_axis = normalize(np.cross(x_axis, y_axis))
    forward = -z_axis
    target = np.zeros(3, dtype=float)  # torso-centered interpretation for trackcom.
    to_target = normalize(target - cfg["pos"])
    align = float(np.dot(forward, to_target))
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "forward": forward,
        "to_target": to_target,
        "align": np.array([align]),
    }


def to_str(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in vec.tolist()) + "]"


def export_camera_csv(vectors: Dict[str, Dict[str, np.ndarray]]) -> None:
    rows = []
    for name, cfg in CAMERAS.items():
        v = vectors[name]
        rows.append(
            {
                "camera_name": name,
                "task": cfg["task"],
                "mode": cfg["mode"],
                "target": cfg["target"],
                "pos": to_str(cfg["pos"]),
                "x_axis_from_xyaxes": to_str(v["x_axis"]),
                "y_axis_from_xyaxes": to_str(v["y_axis"]),
                "z_axis_cross_x_y": to_str(v["z_axis"]),
                "forward_view_dir_-z": to_str(v["forward"]),
                "unit_vector_to_torso": to_str(v["to_target"]),
                "forward_dot_to_target": f"{v['align'][0]:.4f}",
            }
        )
    path = OUTDIR / "camera_vectors.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_camera_markdown(vectors: Dict[str, Dict[str, np.ndarray]]) -> None:
    lines = [
        "# Camera Vectors Used in Experiments",
        "",
        "Source: embodied/envs/dmc.py (custom walker camera injection).",
        "",
    ]
    for name, cfg in CAMERAS.items():
        v = vectors[name]
        lines.extend(
            [
                f"## {name} ({cfg['task']})",
                f"- pos: `{to_str(cfg['pos'])}`",
                f"- x axis (xyaxes[0:3]): `{to_str(v['x_axis'])}`",
                f"- y axis (xyaxes[3:6]): `{to_str(v['y_axis'])}`",
                f"- z axis (x cross y): `{to_str(v['z_axis'])}`",
                f"- forward view direction (-z): `{to_str(v['forward'])}`",
                f"- unit vector camera->torso: `{to_str(v['to_target'])}`",
                f"- forward dot camera->torso: `{v['align'][0]:.4f}`",
                "",
            ]
        )
    (OUTDIR / "camera_vectors.md").write_text("\n".join(lines), encoding="utf-8")


def set_axes_equal(ax: plt.Axes) -> None:
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    x_range = abs(xlim[1] - xlim[0])
    y_range = abs(ylim[1] - ylim[0])
    z_range = abs(zlim[1] - zlim[0])
    max_range = max(x_range, y_range, z_range) / 2.0
    x_mid = np.mean(xlim)
    y_mid = np.mean(ylim)
    z_mid = np.mean(zlim)
    ax.set_xlim3d(x_mid - max_range, x_mid + max_range)
    ax.set_ylim3d(y_mid - max_range, y_mid + max_range)
    ax.set_zlim3d(z_mid - max_range, z_mid + max_range)


def plot_combined_world(vectors: Dict[str, Dict[str, np.ndarray]]) -> None:
    fig = plt.figure(figsize=(10.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    torso = np.zeros(3, dtype=float)
    ax.scatter([torso[0]], [torso[1]], [torso[2]], c="black", s=80, label="Torso COM (target)")

    for name, cfg in CAMERAS.items():
        color = cfg["color"]
        pos = cfg["pos"]
        v = vectors[name]

        ax.scatter([pos[0]], [pos[1]], [pos[2]], c=color, s=70, label=name)
        ax.plot([pos[0], torso[0]], [pos[1], torso[1]], [pos[2], torso[2]], color=color, linestyle="--", linewidth=1.4)

        forward = v["forward"] * 1.5
        ax.quiver(pos[0], pos[1], pos[2], forward[0], forward[1], forward[2], color=color, linewidth=2.1)
        ax.text(pos[0], pos[1], pos[2] + 0.18, name, color=color, fontsize=10)

    # World axes at origin.
    ax.quiver(0, 0, 0, 1.3, 0, 0, color="gray", linewidth=1.2)
    ax.quiver(0, 0, 0, 0, 1.3, 0, color="gray", linewidth=1.2)
    ax.quiver(0, 0, 0, 0, 0, 1.3, color="gray", linewidth=1.2)
    ax.text(1.35, 0, 0, "X", color="gray")
    ax.text(0, 1.35, 0, "Y", color="gray")
    ax.text(0, 0, 1.35, "Z", color="gray")

    ax.set_title("Custom Walker Camera Positions and Forward Directions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=23, azim=-50)
    ax.grid(True, alpha=0.3)
    set_axes_equal(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTDIR / "camera_pose_world_3d.png", dpi=240)
    plt.close(fig)


def plot_per_camera_frames(vectors: Dict[str, Dict[str, np.ndarray]]) -> None:
    fig = plt.figure(figsize=(13.0, 5.8))
    names = list(CAMERAS.keys())

    for i, name in enumerate(names, start=1):
        cfg = CAMERAS[name]
        v = vectors[name]
        pos = cfg["pos"]
        color = cfg["color"]

        ax = fig.add_subplot(1, 2, i, projection="3d")
        torso = np.zeros(3, dtype=float)

        ax.scatter([torso[0]], [torso[1]], [torso[2]], c="black", s=70)
        ax.scatter([pos[0]], [pos[1]], [pos[2]], c=color, s=65)
        ax.plot([pos[0], torso[0]], [pos[1], torso[1]], [pos[2], torso[2]], color=color, linestyle="--", linewidth=1.3)

        # Camera frame axes.
        ax.quiver(pos[0], pos[1], pos[2], *(v["x_axis"] * 0.95), color="#ff7f0e", linewidth=2.0)
        ax.quiver(pos[0], pos[1], pos[2], *(v["y_axis"] * 0.95), color="#2ca02c", linewidth=2.0)
        ax.quiver(pos[0], pos[1], pos[2], *(v["z_axis"] * 0.95), color="#9467bd", linewidth=2.0)
        ax.quiver(pos[0], pos[1], pos[2], *(v["forward"] * 1.25), color=color, linewidth=2.4)

        ax.text(*(pos + v["x_axis"] * 1.05), "x_cam", color="#ff7f0e", fontsize=9)
        ax.text(*(pos + v["y_axis"] * 1.05), "y_cam", color="#2ca02c", fontsize=9)
        ax.text(*(pos + v["z_axis"] * 1.05), "z_cam", color="#9467bd", fontsize=9)
        ax.text(*(pos + v["forward"] * 1.35), "forward", color=color, fontsize=9)

        ax.set_title(f"{name}\npos={to_str(pos)}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=22, azim=-48)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-4.0, 2.5)
        ax.set_ylim(-4.0, 4.0)
        ax.set_zlim(-1.0, 5.0)
        set_axes_equal(ax)

    fig.suptitle("Camera Frame Axes Derived from xyaxes (Experiments)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "camera_pose_frames.png", dpi=240)
    plt.close(fig)


def _pose_overlay_text(name: str, cfg: Dict[str, np.ndarray], vecs: Dict[str, np.ndarray]) -> str:
    pos = cfg["pos"]
    forward = vecs["forward"]
    return (
        f"{name}\n"
        f"task: {cfg['task']}\n"
        f"pos = [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]\n"
        f"forward = [{forward[0]:.1f}, {forward[1]:.1f}, {forward[2]:.1f}]"
    )


def _draw_xy_inset(ax: plt.Axes, cfg: Dict[str, np.ndarray], vecs: Dict[str, np.ndarray]) -> None:
    inset = ax.inset_axes([0.03, 0.03, 0.27, 0.27])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.88))

    torso_xy = np.array([0.0, 0.0])
    cam_xy = cfg["pos"][:2]
    f_xy = vecs["forward"][:2]

    inset.scatter([torso_xy[0]], [torso_xy[1]], c="black", s=26)
    inset.scatter([cam_xy[0]], [cam_xy[1]], c=cfg["color"], s=26)
    inset.plot([cam_xy[0], torso_xy[0]], [cam_xy[1], torso_xy[1]], linestyle="--", color=cfg["color"], linewidth=1.0)

    if np.linalg.norm(f_xy) > 1e-8:
        f_xy = normalize(f_xy) * 0.9
        inset.arrow(
            cam_xy[0], cam_xy[1], f_xy[0], f_xy[1],
            width=0.03, head_width=0.18, head_length=0.18, color=cfg["color"], length_includes_head=True
        )
        inset.text(cam_xy[0] + f_xy[0], cam_xy[1] + f_xy[1], "f_xy", fontsize=7, color=cfg["color"])
    else:
        inset.text(cam_xy[0] - 0.35, cam_xy[1] - 0.35, "forward along -Z", fontsize=7, color=cfg["color"])

    inset.set_title("XY pose", fontsize=8)
    inset.set_xlim(-3.8, 2.0)
    inset.set_ylim(-3.8, 3.8)
    inset.tick_params(labelsize=6)
    inset.grid(True, alpha=0.35)


def plot_integrated_view_images(vectors: Dict[str, Dict[str, np.ndarray]]) -> None:
    entries = [("Side Camera", VIEW_IMAGES["Side Camera"]), ("Top Camera", VIEW_IMAGES["Top Camera"])]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2))
    for ax, (name, image_path) in zip(axes, entries):
        cfg = CAMERAS[name]
        vecs = vectors[name]
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"{name} Observation", fontsize=13, color=cfg["color"])

        # Colored frame around each observation.
        h, w = img.shape[0], img.shape[1]
        frame = plt.Rectangle((0, 0), w, h, fill=False, linewidth=8, edgecolor=cfg["color"])
        ax.add_patch(frame)

        # Pose info box.
        ax.text(
            0.97,
            0.05,
            _pose_overlay_text(name, cfg, vecs),
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="bottom",
            color="white",
            bbox=dict(facecolor="black", alpha=0.65, boxstyle="round,pad=0.35"),
        )

        # Attention marker for camera pose annotation.
        ax.annotate(
            "Camera pose used in training/eval",
            xy=(0.76, 0.15),
            xytext=(0.54, 0.32),
            xycoords="axes fraction",
            textcoords="axes fraction",
            color=cfg["color"],
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=1.8),
            bbox=dict(facecolor="white", edgecolor=cfg["color"], alpha=0.78, boxstyle="round,pad=0.25"),
        )

        _draw_xy_inset(ax, cfg, vecs)

    fig.suptitle("Experiment Views with Their Respective Camera Pose", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTDIR / "side_top_integrated_camera_pose.png", dpi=240)
    plt.close(fig)

    # Single-image versions for slide flexibility.
    for name, image_path in entries:
        cfg = CAMERAS[name]
        vecs = vectors[name]
        img = plt.imread(image_path)
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"{name} ({cfg['task']})", fontsize=14, color=cfg["color"])
        h, w = img.shape[0], img.shape[1]
        frame = plt.Rectangle((0, 0), w, h, fill=False, linewidth=8, edgecolor=cfg["color"])
        ax.add_patch(frame)
        ax.text(
            0.98,
            0.05,
            _pose_overlay_text(name, cfg, vecs),
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            color="white",
            bbox=dict(facecolor="black", alpha=0.68, boxstyle="round,pad=0.35"),
        )
        _draw_xy_inset(ax, cfg, vecs)
        fig.tight_layout()
        out_name = "side_integrated_camera_pose.png" if name == "Side Camera" else "top_integrated_camera_pose.png"
        fig.savefig(OUTDIR / out_name, dpi=240)
        plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    vectors = {name: camera_vectors(cfg) for name, cfg in CAMERAS.items()}
    export_camera_csv(vectors)
    export_camera_markdown(vectors)
    plot_combined_world(vectors)
    plot_per_camera_frames(vectors)
    plot_integrated_view_images(vectors)
    print(f"Wrote camera graphics and vectors to: {OUTDIR}")


if __name__ == "__main__":
    main()
