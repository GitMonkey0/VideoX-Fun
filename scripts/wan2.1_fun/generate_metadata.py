# generate_metadata.py
# 生成 videox_fun 训练用 metadata.json（json list）
# 每条包含：file_path、control_file_path、text、type、hl_path
#
# 依据：hl 目录下 .npz 文件数量为准；按同名基准去 raw_videos / dwpose_videos 下找对应视频文件

import os
import json
import glob

HL_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/train/hl"
RAW_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/train/raw_videos"
CONTROL_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/train/dwpose_videos"
OUT_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/train/videx_fun"
OUT_PATH = os.path.join(OUT_DIR, "metadata.json")

VIDEO_EXTS = [".mp4", ".webm", ".mov", ".mkv", ".avi"]


def find_video_by_stem(directory: str, stem: str):
    for ext in VIDEO_EXTS:
        p = os.path.join(directory, stem + ext)
        if os.path.isfile(p):
            return p
        # 有些数据可能是大小写扩展名
        p2 = os.path.join(directory, stem + ext.upper())
        if os.path.isfile(p2):
            return p2
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hl_files = sorted(glob.glob(os.path.join(HL_DIR, "*.npz")))
    metadata = []
    miss_raw, miss_control = 0, 0

    for hl_path in hl_files:
        stem = os.path.splitext(os.path.basename(hl_path))[0]

        raw_path = find_video_by_stem(RAW_DIR, stem)
        control_path = find_video_by_stem(CONTROL_DIR, stem)

        if raw_path is None:
            miss_raw += 1
            continue
        if control_path is None:
            miss_control += 1
            continue

        metadata.append(
            {
                "file_path": raw_path,
                "control_file_path": control_path,
                "text": "",
                "type": "video",
                "hl_path": hl_path,
            }
        )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"HL npz files: {len(hl_files)}")
    print(f"Written items: {len(metadata)}")
    print(f"Skipped (missing raw video): {miss_raw}")
    print(f"Skipped (missing control video): {miss_control}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()