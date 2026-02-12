"""Headless matplotlib-based 3D pose visualization.

Replaces poseviz (which requires mayavi/VTK/OpenGL) for environments
without a display server, such as Gradio / HuggingFace Spaces.
"""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def render_pose_result(image_rgb, boxes, poses3d, poses2d, joint_edges):
    """Render 2D detections + 3D poses into a single matplotlib figure.

    Args:
        image_rgb: uint8 RGB image [H, W, 3].
        boxes: [N, 4+] detection boxes (x, y, w, h, ...).
        poses3d: [N, J, 3] world-space 3D joint positions in mm.
        poses2d: [N, J, 2] image-space 2D joint positions.
        joint_edges: [E, 2] pairs of joint indices defining the skeleton.

    Returns:
        uint8 RGB numpy array of the rendered figure.
    """
    fig = plt.figure(figsize=(12, 5.5), dpi=100)

    # -- Left panel: image with 2D overlay --
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(image_rgb)
    image_ax.set_axis_off()

    if len(boxes) > 0:
        for box in boxes[:, :4]:
            x, y, w, h = box
            image_ax.add_patch(
                Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
            )

    if len(poses2d) > 0:
        for pose2d in poses2d:
            for i_start, i_end in joint_edges:
                image_ax.plot(
                    [pose2d[i_start, 0], pose2d[i_end, 0]],
                    [pose2d[i_start, 1], pose2d[i_end, 1]],
                    linewidth=1.5,
                    color="cyan",
                    marker="o",
                    markersize=2,
                )

    # -- Right panel: 3D skeleton --
    pose_ax = fig.add_subplot(1, 2, 2, projection="3d")
    pose_ax.view_init(elev=5, azim=-85)

    if len(poses3d) > 0:
        # Matplotlib uses Z-up; metrabs uses Y-up.  Rotate 90 deg around X.
        poses3d_plot = poses3d.copy()
        poses3d_plot[..., 1], poses3d_plot[..., 2] = (
            poses3d_plot[..., 2],
            -poses3d_plot[..., 1],
        )

        all_pts = poses3d_plot.reshape(-1, 3)
        center = all_pts.mean(axis=0)
        span = max(np.ptp(all_pts, axis=0).max() / 2, 500)

        pose_ax.set_xlim3d(center[0] - span, center[0] + span)
        pose_ax.set_ylim3d(center[1] - span, center[1] + span)
        pose_ax.set_zlim3d(center[2] - span, center[2] + span)
        pose_ax.set_box_aspect((1, 1, 1))

        colors = plt.cm.tab10.colors
        for idx, pose3d in enumerate(poses3d_plot):
            c = colors[idx % len(colors)]
            for i_start, i_end in joint_edges:
                pose_ax.plot(
                    [pose3d[i_start, 0], pose3d[i_end, 0]],
                    [pose3d[i_start, 1], pose3d[i_end, 1]],
                    [pose3d[i_start, 2], pose3d[i_end, 2]],
                    linewidth=2,
                    color=c,
                    marker="o",
                    markersize=3,
                )
    else:
        pose_ax.text(
            0.5, 0.5, 0.5, "No poses detected", transform=pose_ax.transAxes, ha="center"
        )

    pose_ax.set_xlabel("X")
    pose_ax.set_ylabel("Y")
    pose_ax.set_zlabel("Z")

    fig.tight_layout()
    result = _fig_to_rgb(fig)
    plt.close(fig)
    return result


def _fig_to_rgb(fig):
    """Convert a matplotlib figure to an RGB numpy array without saving to disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    from PIL import Image

    img = np.array(Image.open(buf).convert("RGB"))
    buf.close()
    return img
