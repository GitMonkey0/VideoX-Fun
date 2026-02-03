import os
import glob
import json
import cv2
import numpy as np

from comfyui.annotator.dwpose_utils.wholebody import Keypoint
from comfyui.annotator.dwpose_utils.util import draw_handpose


def convert_mediapipe_to_keypoints(mp_frame_data):
    """将MediaPipe JSON格式转换为VideoX-Fun格式"""
    converted_hands = []

    multi_hand_landmarks = mp_frame_data.get("multi_hand_landmarks", [])
    multi_handedness = mp_frame_data.get("multi_handedness", [])

    for hand_idx, landmarks_3d in enumerate(multi_hand_landmarks):
        handedness_label = "Unknown"
        if hand_idx < len(multi_handedness):
            handedness_label = multi_handedness[hand_idx].get("label", "Unknown")

        # 这里只取 x,y（假设 landmarks_3d 的每个点是 [x,y,z] 或 [x,y]）
        landmarks_2d = [{"x": p[0], "y": p[1]} for p in landmarks_3d]

        converted_hands.append(
            {"handedness": handedness_label, "landmarks": landmarks_2d}
        )

    return converted_hands


def generate_control_video(converted_results_list, output_path, width, height, fps=30):
    """从转换后的结果生成控制视频"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # AVI
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_data in converted_results_list:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for hand in frame_data:
            landmarks = hand.get("landmarks", [])
            if not landmarks:
                continue

            kp_list = [
                Keypoint(x=lm["x"], y=lm["y"], score=1.0, id=i)
                for i, lm in enumerate(landmarks)
            ]
            canvas = draw_handpose(canvas, kp_list)

        out.write(canvas)

    out.release()


def process_one_json(json_path, output_video_path, width=512, height=512, fps=30):
    with open(json_path, "r") as f:
        mediapipe_data = json.load(f)

    converted_data = [convert_mediapipe_to_keypoints(frame) for frame in mediapipe_data]
    generate_control_video(converted_data, output_video_path, width, height, fps)


def batch_process(input_dir, output_dir, width=512, height=512, fps=30, exts=(".json",)):
    os.makedirs(output_dir, exist_ok=True)

    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    files = sorted(files)

    if not files:
        print(f"No files found in {input_dir} with exts={exts}")
        return

    for in_path in files:
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(output_dir, f"{base}.mp4")

        try:
            process_one_json(in_path, out_path, width, height, fps)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Failed: {in_path} -> {e}")


if __name__ == "__main__":
    batch_process(
        input_dir="/mnt/bn/douyin-ai4se-general-wl/lht/data/interhand/landmark_keypoints",
        output_dir="/mnt/bn/douyin-ai4se-general-wl/lht/data/interhand/landmark_control_videos",
        width=512,
        height=512,
        fps=30,
    )