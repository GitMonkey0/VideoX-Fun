# generate_metadata.py
# 生成 videox_fun 训练用 metadata.json（json list）
# 每条包含：file_path、control_file_path、text、type、hl_path
# 依据：hl npz、raw videos、control videos 三者同名(stem)交集；缺一不写入

import os
import json
import glob

HL_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/train/hl"
RAW_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/train/raw_videos"
CONTROL_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/train/dwpose_videos"

OUT_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/train/videx_fun"
OUT_PATH = os.path.join(OUT_DIR, "metadata.json")

VIDEO_EXTS = [".mp4", ".webm", ".mov", ".mkv", ".avi"]


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def build_hl_map(hl_dir: str):
    m = {}
    for p in glob.glob(os.path.join(hl_dir, "*.npz")):
        m[stem(p)] = p
    return m


def build_video_map(video_dir: str):
    """
    扫描目录下所有视频扩展名文件，返回 stem->path。
    若同 stem 有多个扩展名，保留第一个（按 ext 列表优先级）。
    """
    m = {}
    # 按优先级匹配：先收集，后按 VIDEO_EXTS 顺序覆盖策略
    candidates = {}
    for ext in VIDEO_EXTS + [e.upper() for e in VIDEO_EXTS]:
        for p in glob.glob(os.path.join(video_dir, "*" + ext)):
            s = stem(p)
            candidates.setdefault(s, []).append(p)

    # 选择优先级最高的那个
    for s, paths in candidates.items():
        # 按 VIDEO_EXTS/upper 的顺序挑
        def keyfn(p):
            ext = os.path.splitext(p)[1]
            try:
                return (VIDEO_EXTS + [e.upper() for e in VIDEO_EXTS]).index(ext)
            except ValueError:
                return 10**9

        m[s] = sorted(paths, key=keyfn)[0]
    return m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hl_map = build_hl_map(HL_DIR)
    raw_map = build_video_map(RAW_DIR)
    control_map = build_video_map(CONTROL_DIR)

    stems_hl = set(hl_map.keys())
    stems_raw = set(raw_map.keys())
    stems_control = set(control_map.keys())

    common_stems = sorted(stems_hl & stems_raw & stems_control)

    metadata = []
    for s in common_stems:
        metadata.append(
            {
                "file_path": raw_map[s],
                "control_file_path": control_map[s],
                "text": "",
                "type": "video",
                "hl_path": hl_map[s],
            }
        )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 统计
    miss_raw = len(stems_hl - stems_raw)
    miss_control = len(stems_hl - stems_control)

    print(f"HL npz stems: {len(stems_hl)}")
    print(f"RAW video stems: {len(stems_raw)}")
    print(f"CONTROL video stems: {len(stems_control)}")
    print(f"Intersection (written items): {len(common_stems)}")
    print(f"HL missing RAW (by stem): {miss_raw}")
    print(f"HL missing CONTROL (by stem): {miss_control}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()