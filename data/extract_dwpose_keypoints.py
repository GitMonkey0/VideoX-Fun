import os
import cv2
import glob
import numpy as np
import torch
from multiprocessing import Pool, set_start_method
from DWPose.ControlNet_v1_1_nightly.annotator.dwpose import DWposeDetector


def pose_from_frame(detector, frame):
    """
    复刻 DWposeDetector.__call__ 中“构建pose字典”的部分，但不 draw。
    返回的坐标是归一化到 [0,1] 的（与 draw_pose 兼容）。
    """
    H, W, _ = frame.shape
    with torch.no_grad():
        candidate, subset = detector.pose_estimation(frame)

        # normalize
        candidate = candidate.copy()
        subset = subset.copy()
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)

        # body
        nums, keys, locs = candidate.shape
        body = candidate[:, :18].copy().reshape(nums * 18, locs)

        score = subset[:, :18].copy()
        for i in range(len(score)):
            for j in range(len(score[i])):
                score[i][j] = int(18 * i + j) if score[i][j] > 0.3 else -1

        # invisible -> -1
        un_visible = subset < 0.3
        candidate[un_visible] = -1

        faces = candidate[:, 24:92].copy()
        hands = candidate[:, 92:113].copy()
        hands = np.vstack([hands, candidate[:, 113:].copy()])

        pose = {
            "bodies": {"candidate": body.astype(np.float32), "subset": score.astype(np.float32)},
            "hands": hands.astype(np.float32),
            "faces": faces.astype(np.float32),
        }
        return pose, (H, W)


def init_worker(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        detector = DWposeDetector()
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _pose, _shape = pose_from_frame(detector, dummy)
        print(f"GPU {gpu_id} warmed up successfully")
        return True
    except Exception as e:
        print(f"GPU {gpu_id} warmup failed: {e}")
        return False


def process_video_extract(args):
    video_path, out_npz_path, gpu_id = args
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)

        detector = DWposeDetector()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"GPU {gpu_id}: cannot open {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-6:
            fps = 25.0

        poses = []
        H = W = None
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pose, (h, w) = pose_from_frame(detector, frame)
            if H is None:
                H, W = h, w
            poses.append(pose)
            n += 1

        cap.release()

        # 用 object 数组保存每帧dict（np.savez 需要）
        poses = np.array(poses, dtype=object)
        np.savez_compressed(out_npz_path, poses=poses, fps=np.float32(fps), H=np.int32(H), W=np.int32(W))

        print(f"GPU {gpu_id}: {os.path.basename(video_path)} ({n} frames) -> {os.path.basename(out_npz_path)}")
        return True
    except Exception as e:
        print(f"GPU {gpu_id} error on {video_path}: {e}")
        return False


def batch_extract(input_dir, output_dir, num_gpus=8):
    set_start_method("spawn", force=True)

    print("Warming up GPUs...")
    with Pool(processes=num_gpus) as pool:
        warm = pool.map(init_worker, list(range(num_gpus)))

    working_gpus = [i for i, ok in enumerate(warm) if ok]
    if not working_gpus:
        raise RuntimeError("No GPU warmed up successfully.")

    print(f"Working GPUs: {working_gpus}")

    video_files = glob.glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True)
    args = []
    for i, vp in enumerate(video_files):
        rel = os.path.relpath(vp, input_dir)
        out_npz = os.path.join(output_dir, os.path.splitext(rel)[0] + ".npz")
        args.append((vp, out_npz, working_gpus[i % len(working_gpus)]))

    with Pool(processes=len(working_gpus)) as pool:
        results = pool.map(process_video_extract, args)

    print(f"Completed: {sum(results)}/{len(video_files)}")


if __name__ == "__main__":
    batch_extract(
        "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/test/raw_videos",
        "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/test/dwpose_keypoints",
        num_gpus=8,
    )