#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from glob import glob
from statistics import mean
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = "/mnt/bn/douyin-ai4se-general-wl/lht/data/interhand"
KPT_DIR = os.path.join(ROOT, "landmark_keypoints")  # 注意改成你新结果目录
RAW_VIDEO_DIR = os.path.join(ROOT, "raw_videos")
CONTROL_VIDEO_DIR = os.path.join(ROOT, "landmark_control_videos")  # 如果有新可视化目录就改这里

# 测试集最小帧数要求
MIN_TEST_FRAMES = 81
MIN_TRAIN_FRAMES = 21
MAX_TRAIN_FRAMES = 127

def analyze_one_json(path):
    """
    分析单个关键点 json，返回:
      (path, frames, missing_frames, missing_ratio, both_hands_frames, both_hands_ratio)

    说明：
    - 新版结构：顶层是 dict，帧在 data["frames"] 里
    - 每帧结构示例：
        {
          "frame_index": ...,
          "timestamp_ms": ...,
          "hands": [
            {
              "hand_index": 0,
              "handedness": {...},
              "landmarks": [...],
              "world_landmarks": [...]
            },
            ...
          ]
        }
    - missing_frames: hands 为空的帧数
    - both_hands_frames: 一帧中 "有至少两只手" 的帧数
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        # 出错则当作空
        return path, 0, 0, 0.0, 0, 0.0

    frames = data.get("frames", [])
    total_frames = len(frames)
    if total_frames == 0:
        return path, 0, 0, 0.0, 0, 0.0

    missing = 0
    both_hands = 0

    for frame in frames:
        hands = frame.get("hands", [])
        # 没有任何手
        if not hands:
            missing += 1
            continue

        # 统计双手：这里简单地视 “hands 数量 >= 2” 为双手都存在
        # 如需严格区分左右，可以用 handedness 信息再判断
        if len(hands) >= 2:
            both_hands += 1

    missing_ratio = missing / total_frames
    both_hands_ratio = both_hands / total_frames

    return (
        path,
        total_frames,
        missing,
        missing_ratio,
        both_hands,
        both_hands_ratio,
    )


def text_hist(values, bins, title):
    print(f"\n{title}")
    if not values:
        print("  (无数据)")
        return

    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
        else:
            if v == bins[-1]:
                counts[-1] += 1

    max_count = max(counts) if counts else 1
    for i in range(len(counts)):
        left, right = bins[i], bins[i + 1]
        bar_len = int(counts[i] / max_count * 40) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  [{left:.2f}, {right:.2f}) count={counts[i]:5d} {bar}")


def main():
    json_paths = sorted(glob(os.path.join(KPT_DIR, "*.json")))
    n_files = len(json_paths)
    print(f"共发现 {n_files} 个关键点 json 文件")

    if n_files == 0:
        return

    # per_file_stats: 每条记录为
    # (path, frames, missing_frames, missing_ratio, both_hands_frames, both_hands_ratio)
    per_file_stats = []
    all_frames = []
    all_missing_ratio = []
    all_both_hands_ratio = []

    max_workers = min(16, os.cpu_count() or 4)
    print(f"使用 {max_workers} 个进程并行分析...")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_one_json, p): p for p in json_paths}
        for i, fut in enumerate(as_completed(futures), 1):
            (
                path,
                frames,
                missing,
                mr,
                both_hands_frames,
                both_hands_ratio,
            ) = fut.result()

            per_file_stats.append(
                (path, frames, missing, mr, both_hands_frames, both_hands_ratio)
            )
            all_frames.append(frames)
            all_missing_ratio.append(mr)
            all_both_hands_ratio.append(both_hands_ratio)

            if i % 500 == 0 or i == n_files:
                print(f"  已完成 {i}/{n_files} ({i / n_files * 100:.1f}%)")

    # 去掉 frames == 0 的全空文件
    per_file_stats = [s for s in per_file_stats if s[1] > 0]
    all_frames = [s[1] for s in per_file_stats]
    all_missing_ratio = [s[3] for s in per_file_stats]
    all_both_hands_ratio = [s[5] for s in per_file_stats]

    if not per_file_stats:
        print("\n无有效样本（所有 json 的 frames 都为 0）")
        return

    print("\n=== 全局统计 ===")
    print(f"  文件数                 : {len(all_frames)}")
    print(f"  最小帧数               : {min(all_frames)}")
    print(f"  最大帧数               : {max(all_frames)}")
    print(f"  平均帧数               : {mean(all_frames):.2f}")

    print("\n缺失比例分布（无手帧 / 总帧）：")
    print(f"  最小缺失比例           : {min(all_missing_ratio):.3f}")
    print(f"  最大缺失比例           : {max(all_missing_ratio):.3f}")
    print(f"  平均缺失比例           : {mean(all_missing_ratio):.3f}")

    print("\n双手同时存在比例（双手帧 / 总帧）：")
    print(f"  最小双手比例           : {min(all_both_hands_ratio):.3f}")
    print(f"  最大双手比例           : {max(all_both_hands_ratio):.3f}")
    print(f"  平均双手比例           : {mean(all_both_hands_ratio):.3f}")

    missing_bins = [i / 10 for i in range(11)]  # 0.0～1.0
    text_hist(all_missing_ratio, missing_bins, "缺失比例直方图（文字版）")
    text_hist(all_both_hands_ratio, missing_bins, "双手比例直方图（文字版）")

    max_f = max(all_frames) or 1
    step = max(1, max_f // 10)
    frame_bins = list(range(0, max_f + step, step))
    text_hist(all_frames, frame_bins, "帧数直方图（文字版）")

    # =========================
    # 选取“优先双手都在 + 帧数多 + 缺失少”的前 100 条作为测试集
    # 增加：只从帧数 >= MIN_TEST_FRAMES 的样本里选
    # 排序 key: (-both_hands_ratio, -frames, missing_ratio)
    # =========================
    print("\n=== 选取双手比例高、帧数多且缺失少的前 100 条作为候选测试集 ===")
    print(f"测试集最小帧数要求：{MIN_TEST_FRAMES}")

    # 只保留帧数>0的有效样本
    valid_stats = per_file_stats[:]

    # 测试集候选：帧数 >= MIN_TEST_FRAMES
    valid_stats_for_test = [s for s in valid_stats if s[1] >= MIN_TEST_FRAMES]
    print(f"满足帧数 >= {MIN_TEST_FRAMES} 的候选样本数：{len(valid_stats_for_test)}")

    test_stats = []
    if valid_stats_for_test:
        valid_stats_for_test.sort(
            key=lambda x: (-x[5], -x[1], x[3])
        )  # x[5]=both_hands_ratio, x[1]=frames, x[3]=missing_ratio

        top_k = 100
        test_stats = valid_stats_for_test[:top_k]

        print(f"实际可选文件数：{len(valid_stats_for_test)}")
        print(f"取前 {len(test_stats)} 条（按双手比例降序 + 帧数降序 + 缺失比例升序排序）:\n")

        print(
            f"{'rank':>4s}  {'file':60s}  {'frames':>7s}  "
            f"{'missing':>7s}  {'miss_r':>7s}  {'both2':>7s}  {'both_r':>7s}"
        )
        print("-" * 120)
        for idx, (path, frames, missing, mr, both2, br) in enumerate(test_stats, 1):
            print(
                f"{idx:4d}  {os.path.basename(path):60s}  "
                f"{frames:7d}  {missing:7d}  {mr:7.3f}  {both2:7d}  {br:7.3f}"
            )

        # =============== 对测试集做一个简单统计 ===============
        test_frames = [s[1] for s in test_stats]
        test_missing_ratio = [s[3] for s in test_stats]
        test_both_ratio = [s[5] for s in test_stats]

        print("\n=== 测试集统计（Top 100） ===")
        print(f"  样本数                 : {len(test_stats)}")
        print(
            f"  帧数 min/mean/max      : {min(test_frames)}, "
            f"{mean(test_frames):.2f}, {max(test_frames)}"
        )
        print(
            f"  缺失比例 min/mean/max  : {min(test_missing_ratio):.3f}, "
            f"{mean(test_missing_ratio):.3f}, {max(test_missing_ratio):.3f}"
        )
        print(
            f"  双手比例 min/mean/max  : {min(test_both_ratio):.3f}, "
            f"{mean(test_both_ratio):.3f}, {max(test_both_ratio):.3f}"
        )

        text_hist(test_missing_ratio, missing_bins, "测试集 缺失比例直方图（文字版）")
        text_hist(test_both_ratio, missing_bins, "测试集 双手比例直方图（文字版）")
        text_hist(test_frames, frame_bins, "测试集 帧数直方图（文字版）")
    else:
        print("没有满足最小帧数要求的样本，测试集将为空")

    # =========================
    # 训练集：除去测试集后，依然按 (-both_hands_ratio, -frames, missing_ratio) 排序，取前 2000
    # =========================
    print("\n=== 在剩余样本中选取双手比例高、帧数多且缺失少的前 2000 条作为训练集 ===")

    test_paths = set(s[0] for s in test_stats)
    # 从全部有效样本中去掉测试集
    # remaining_stats = [s for s in valid_stats if s[0] not in test_paths]
    remaining_stats = [
        s for s in valid_stats
        if (s[0] not in test_paths)
        and (MIN_TRAIN_FRAMES <= s[1] <= MAX_TRAIN_FRAMES)
    ]

    # 对剩余样本排序
    remaining_stats.sort(
        key=lambda x: (-x[5], -x[1], x[3])
    )

    train_k = 2000
    train_stats = remaining_stats[:train_k]

    print(f"剩余可选文件数：{len(remaining_stats)}")
    print(f"训练集取前 {len(train_stats)} 条（与测试集不重叠）")

    # =============== 对训练集做统计 ===============
    if train_stats:
        train_frames = [s[1] for s in train_stats]
        train_missing_ratio = [s[3] for s in train_stats]
        train_both_ratio = [s[5] for s in train_stats]

        print("\n=== 训练集统计（Top 2000） ===")
        print(f"  样本数                 : {len(train_stats)}")
        print(
            f"  帧数 min/mean/max      : {min(train_frames)}, "
            f"{mean(train_frames):.2f}, {max(train_frames)}"
        )
        print(
            f"  缺失比例 min/mean/max  : {min(train_missing_ratio):.3f}, "
            f"{mean(train_missing_ratio):.3f}, {max(train_missing_ratio):.3f}"
        )
        print(
            f"  双手比例 min/mean/max  : {min(train_both_ratio):.3f}, "
            f"{mean(train_both_ratio):.3f}, {max(train_both_ratio):.3f}"
        )

        text_hist(
            train_missing_ratio, missing_bins, "训练集 缺失比例直方图（文字版）"
        )
        text_hist(train_both_ratio, missing_bins, "训练集 双手比例直方图（文字版）")
        text_hist(train_frames, frame_bins, "训练集 帧数直方图（文字版）")
    else:
        print("  没有可用的训练集样本（remaining_stats 为空）")

    # =========================
    # 写出 JSON 列表：
    # [
    #   {
    #     "file_path": "<raw_video_path>",
    #     "control_file_path": "<control_video_path>",
    #     "text": "",
    #     "type": "video"
    #   },
    #   ...
    # ]
    # =========================
    def build_item(json_path: str):
        base = os.path.splitext(os.path.basename(json_path))[0]  # 去掉 .json

        # file_path: 去掉末尾的 _landmarks
        if base.endswith("_landmarks"):
            raw_base = base[: -len("_landmarks")]
        else:
            raw_base = base

        # control_file_path: 末尾 _landmarks -> _skeleton
        if base.endswith("_landmarks"):
            ctrl_base = base[: -len("_landmarks")] + "_skeleton"
        else:
            ctrl_base = base + "_skeleton"

        video_name = raw_base + ".mp4"
        control_video_name = ctrl_base + ".mp4"

        file_path = os.path.join(RAW_VIDEO_DIR, video_name)
        control_file_path = os.path.join(CONTROL_VIDEO_DIR, control_video_name)

        return {
            "file_path": file_path,
            "control_file_path": control_file_path,
            "text": "",
            "type": "video",
        }

    test_items = [build_item(p) for (p, _, _, _, _, _) in test_stats]
    train_items = [build_item(p) for (p, _, _, _, _, _) in train_stats]

    test_json_path = os.path.join(ROOT, "test_top100.json")
    train_json_path = os.path.join(ROOT, "train_top2000.json")

    with open(test_json_path, "w") as f:
        json.dump(test_items, f, ensure_ascii=False, indent=2)
    with open(train_json_path, "w") as f:
        json.dump(train_items, f, ensure_ascii=False, indent=2)

    print(f"\n已写出测试集 JSON: {test_json_path}")
    print(f"已写出训练集 JSON: {train_json_path}")


if __name__ == "__main__":
    main()

