import os
import cv2
import glob
import numpy as np
from multiprocessing import Pool, set_start_method
from DWPose.ControlNet_v1_1_nightly.annotator.dwpose import draw_pose


def render_one_npz(args):
    npz_path, out_mp4_path = args
    try:
        os.makedirs(os.path.dirname(out_mp4_path), exist_ok=True)

        data = np.load(npz_path, allow_pickle=True)
        poses = data["poses"]              # object array, each item is dict
        fps = float(data["fps"])
        H = int(data["H"])
        W = int(data["W"])

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_mp4_path, fourcc, fps, (W, H))

        for pose in poses:
            # pose is dict with numpy arrays already
            canvas = draw_pose(pose, H, W)
            out.write(canvas)

        out.release()
        print(f"Rendered: {os.path.basename(npz_path)} -> {os.path.basename(out_mp4_path)} ({len(poses)} frames)")
        return True
    except Exception as e:
        print(f"Render error on {npz_path}: {e}")
        return False


def batch_render(keypoints_dir, output_dir, num_workers=8):
    set_start_method("spawn", force=True)

    npz_files = glob.glob(os.path.join(keypoints_dir, "**", "*.npz"), recursive=True)
    args = []
    for p in npz_files:
        rel = os.path.relpath(p, keypoints_dir)
        out_mp4 = os.path.join(output_dir, os.path.splitext(rel)[0] + ".mp4")
        args.append((p, out_mp4))

    with Pool(processes=num_workers) as pool:
        results = pool.map(render_one_npz, args)

    print(f"Completed: {sum(results)}/{len(npz_files)}")


if __name__ == "__main__":
    batch_render(
        "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/test/dwpose_keypoints",
        "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/test/dwpose_videos",
        num_workers=16,
    )