"""Gradio app for MeTRAbs 3D human pose estimation.

Works in headless environments (no display server needed) by using
matplotlib instead of poseviz/mayavi for visualization.

Usage:
    pip install gradio
    python app.py
"""

import os
import os.path as osp
import tempfile

# Set DATA_ROOT before importing posepile (which requires it)
if "DATA_ROOT" not in os.environ:
    os.environ["DATA_ROOT"] = ""

import cameralib
import cv2
import gradio as gr
import numpy as np
import torch

from metrabs_pytorch.scripts.run_video import (
    MODEL_DIR,
    _ensure_model_downloaded,
    load_multiperson_model,
)
from metrabs_pytorch.scripts.viz_matplotlib import render_pose_result
from metrabs_pytorch.util import get_config

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model reference (loaded once)
_model = None


def _load_model():
    """Load the multi-person pose estimation model (cached globally)."""
    global _model
    if _model is not None:
        return _model

    model_dir = osp.abspath(MODEL_DIR)
    _ensure_model_downloaded(model_dir)
    device = torch.device(DEVICE)
    config_path = osp.join(model_dir, "config.yaml")
    cfg = get_config(config_path)

    _model = load_multiperson_model(model_dir=model_dir, cfg=cfg, device=device)
    return _model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
SKELETON = "bml_movi_87"


def predict_image(image_rgb: np.ndarray, det_threshold: float, num_aug: int):
    """Run pose estimation on a single image and return a visualization."""
    if image_rgb is None:
        return None

    model = _load_model()
    device = torch.device(DEVICE)

    joint_edges = model.per_skeleton_joint_edges[SKELETON].cpu().numpy()

    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).to(device)
    camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=image_tensor.shape[1:])

    with torch.inference_mode(), torch.device(device):
        pred = model.detect_poses(
            image_tensor,
            detector_threshold=det_threshold,
            suppress_implausible_poses=False,
            max_detections=10,
            intrinsic_matrix=camera.intrinsic_matrix,
            skeleton=SKELETON,
            num_aug=num_aug,
        )

    boxes = pred["boxes"].cpu().numpy()
    poses3d = pred["poses3d"].cpu().numpy()
    poses2d = pred["poses2d"].cpu().numpy()

    viz = render_pose_result(image_rgb, boxes, poses3d, poses2d, joint_edges)
    return viz


def predict_video(video_path: str, det_threshold: float, num_aug: int):
    """Run pose estimation on each frame of a video and return an annotated video."""
    if video_path is None:
        return None

    model = _load_model()
    device = torch.device(DEVICE)
    joint_edges = model.per_skeleton_joint_edges[SKELETON].cpu().numpy()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if not cap.isOpened():
        raise gr.Error(f"Could not open video: {video_path}")

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = None
    frame_idx = 0
    max_frames = 300  # limit to avoid timeouts

    with torch.inference_mode(), torch.device(device):
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_idx >= max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(device)
            camera = cameralib.Camera.from_fov(
                fov_degrees=55, imshape=image_tensor.shape[1:]
            )

            pred = model.detect_poses(
                image_tensor,
                detector_threshold=det_threshold,
                suppress_implausible_poses=False,
                max_detections=10,
                intrinsic_matrix=camera.intrinsic_matrix,
                skeleton=SKELETON,
                num_aug=num_aug,
            )

            boxes = pred["boxes"].cpu().numpy()
            poses3d = pred["poses3d"].cpu().numpy()
            poses2d = pred["poses2d"].cpu().numpy()

            viz_frame = render_pose_result(frame_rgb, boxes, poses3d, poses2d, joint_edges)

            if writer is None:
                h, w = viz_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            writer.write(cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR))
            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    return out_path if frame_idx > 0 else None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_demo():
    with gr.Blocks(title="MeTRAbs - 3D Human Pose Estimation") as demo:
        gr.Markdown(
            "# MeTRAbs: Metric-Scale Truncation-Robust 3D Human Pose Estimation\n"
            "Upload an image or video to estimate 3D human poses.\n"
            "Left panel shows 2D detections, right panel shows 3D skeleton."
        )

        with gr.Row():
            det_threshold = gr.Slider(
                0.0, 1.0, value=0.3, step=0.05, label="Detector Threshold"
            )
            num_aug = gr.Slider(1, 20, value=5, step=1, label="Num Test-Time Augmentations")

        with gr.Tabs():
            with gr.TabItem("Image"):
                with gr.Row():
                    img_input = gr.Image(type="numpy", label="Input Image")
                    img_output = gr.Image(type="numpy", label="Pose Estimation Result")
                img_btn = gr.Button("Estimate Poses", variant="primary")
                img_btn.click(
                    predict_image, [img_input, det_threshold, num_aug], img_output
                )

            with gr.TabItem("Video"):
                with gr.Row():
                    vid_input = gr.Video(label="Input Video")
                    vid_output = gr.Video(label="Pose Estimation Result")
                vid_btn = gr.Button("Estimate Poses", variant="primary")
                vid_btn.click(
                    predict_video, [vid_input, det_threshold, num_aug], vid_output
                )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
