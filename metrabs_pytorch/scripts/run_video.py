import argparse
import os
import os.path as osp
import platform
from contextlib import nullcontext

# Set DATA_ROOT before importing posepile (which requires it)
# Most of the code expects DATA_ROOT to exist, even if empty
if "DATA_ROOT" not in os.environ:
    os.environ["DATA_ROOT"] = ""

import cameralib
import cv2
import numpy as np
import posepile.joint_info
import poseviz
import simplepyutils as spu
import torch

import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config

curr_dir = osp.dirname(osp.abspath(__file__))
# check if torch model exists otherwise download it
MODEL_DIR = curr_dir + "/../../metrabs_eff2l_384px_800k_28ds_pytorch"
URL = "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2l_384px_800k_28ds_pytorch.tar.gz"

if not osp.exists(f"{MODEL_DIR}/ckpt.pt"):
    import shutil
    import tarfile
    import tempfile
    import urllib.request

    os.makedirs(MODEL_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = osp.join(tmp, "model.tar.gz")
        print(f"Downloading model archive to {archive_path} ...")
        urllib.request.urlretrieve(URL, archive_path)

        print("Extracting archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            # Extract everything to temp dir
            tar.extractall(path=tmp)

            # Move contents of the top-level folder to MODEL_DIR
            top_folder = osp.join(tmp, tar.getmembers()[0].name.split(os.sep)[0])
            for f in os.listdir(top_folder):
                shutil.move(osp.join(top_folder, f), MODEL_DIR)

        print(f"Model ready in {MODEL_DIR}")


def run_metrabs_video(
    video_path: str | int,
    skeleton: str = "bml_movi_87",
    device: str = "cuda",
    detector_threshold: float = 0.01,
    max_detections: int = 1,
    num_aug: int = 5,
    visualize: bool = True,
    max_frames: int | None = None,
    model_dir: str = MODEL_DIR,
):
    """
    Run Metrabs multiperson pose estimation on a video or webcam.

    video_path:
      - str: path to video file
      - int: webcam index (e.g. 0)
    """

    device = torch.device(device)

    # Convert model_dir to absolute path
    model_dir = osp.abspath(model_dir)

    # Load config + model ONCE
    # Convert to absolute path for get_config
    config_path = osp.join(model_dir, "config.yaml")
    cfg = get_config(config_path)
    mp_model = load_multiperson_model(
        model_dir=model_dir,
        cfg=cfg,
        device=device,
    )

    joint_names = mp_model.per_skeleton_joint_names[skeleton]
    joint_edges = mp_model.per_skeleton_joint_edges[skeleton].cpu().numpy()

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video source: {video_path}"

    # Handle visualization - poseviz has issues on macOS due to Linux-specific dependencies
    if visualize:
        if platform.system() == "Darwin":  # macOS
            print(
                "Warning: poseviz visualization is not fully supported on macOS "
                "(requires Linux-specific libraries).\n"
                "Disabling visualization. Use --no-viz to suppress this warning."
            )
            visualize = False
            viz_ctx = nullcontext()
        else:
            try:
                viz_ctx = poseviz.PoseViz(joint_names, joint_edges, paused=False)
            except (OSError, ImportError) as e:
                print(
                    f"Warning: Could not initialize poseviz visualization: {e}\n"
                    "Continuing without visualization. Use --no-viz to suppress this warning."
                )
                visualize = False
                viz_ctx = nullcontext()
    else:
        viz_ctx = nullcontext()

    frame_idx = 0

    with torch.inference_mode(), torch.device(device), viz_ctx as viz:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if max_frames is not None and frame_idx >= max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Torch image [3, H, W]
            image = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(device)

            camera = cameralib.Camera.from_fov(
                fov_degrees=55,
                imshape=image.shape[1:],
            )

            pred = mp_model.detect_poses(
                image,
                detector_threshold=detector_threshold,
                suppress_implausible_poses=False,
                max_detections=max_detections,
                intrinsic_matrix=camera.intrinsic_matrix,
                skeleton=skeleton,
                num_aug=num_aug,
            )
            if visualize:
                viz.update(
                    frame=frame_rgb,
                    boxes=pred["boxes"].cpu().numpy(),
                    poses2d=pred["poses2d"].cpu().numpy(),
                    poses3d=pred["poses3d"].cpu().numpy(),
                    camera=camera,
                )

            yield {
                "frame_idx": frame_idx,
                "boxes": pred["boxes"].cpu().numpy(),
                "poses2d": pred["poses2d"].cpu().numpy(),
                "poses3d": pred["poses3d"].cpu().numpy(),
            }

            frame_idx += 1

    cap.release()


def load_multiperson_model(model_dir: str, cfg, device: torch.device):
    model_pytorch = load_crop_model(model_dir, cfg, device)
    skeleton_infos = spu.load_pickle(f"{model_dir}/skeleton_infos.pkl")
    joint_transform_matrix = np.load(f"{model_dir}/joint_transform_matrix.npy")

    with torch.device(device):
        return multiperson_model.Pose3dEstimator(
            model_pytorch.to(device), skeleton_infos, joint_transform_matrix
        )


def load_crop_model(model_dir: str, cfg, device: torch.device):
    ji_np = np.load(f"{model_dir}/joint_info.npz")
    ji = posepile.joint_info.JointInfo(ji_np["joint_names"], ji_np["joint_edges"])
    backbone_raw = getattr(effnet_pt, f"efficientnet_v2_{cfg.efficientnet_size}")()
    preproc_layer = effnet_pt.PreprocLayer()
    backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features)
    model = metrabs_pt.Metrabs(backbone, ji)
    model.eval()

    inp = torch.zeros((1, 3, cfg.proc_side, cfg.proc_side), dtype=torch.float32)
    intr = torch.eye(3, dtype=torch.float32)[np.newaxis]

    model((inp, intr))
    model.load_state_dict(torch.load(f"{model_dir}/ckpt.pt"))
    return model


def cli_video():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--device", default=None, help="Device to use, e.g., cuda or cpu"
    )
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--max-frames", type=int)

    args = parser.parse_args()

    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video

    # Auto-select device if not specified
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"No device specified. Using: {device}")
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

    results = list(
        run_metrabs_video(
            video_path=video_source,
            device=device,
            visualize=not args.no_viz,
            max_frames=args.max_frames,
            model_dir=args.model_dir,
        )
    )

    print(f"Processed {len(results)} frames")
    print(results[0].keys())


if __name__ == "__main__":
    cli_video()
