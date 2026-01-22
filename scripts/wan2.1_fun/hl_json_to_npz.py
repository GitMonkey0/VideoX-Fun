#!/usr/bin/env python
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

MEDIAPIPE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def build_direction_table() -> np.ndarray:
    directions = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x == 0 and y == 0 and z == 0:
                    continue
                directions.append([x, y, z])
    directions = np.array(directions, dtype=np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-8)
    return directions


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def extract_hand_vectors(hand_landmarks: List[List[float]], normalize: bool = True) -> np.ndarray:
    points = np.asarray(hand_landmarks, dtype=np.float32)
    vectors = []
    for parent, child in MEDIAPIPE_EDGES:
        vectors.append(points[child] - points[parent])
    vectors = np.stack(vectors, axis=0)
    if normalize:
        vectors = normalize_vectors(vectors)
    return vectors


def assign_hl_ids(
    vectors: np.ndarray,
    direction_table: np.ndarray,
    dir_mode: str = "center",
) -> Tuple[np.ndarray, np.ndarray]:
    if vectors.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=-1)
    valid = norms > 1e-8
    vectors_unit = np.zeros_like(vectors, dtype=np.float32)
    vectors_unit[valid] = vectors[valid] / norms[valid, None]

    scores = vectors_unit @ direction_table.T
    ids = np.argmax(scores, axis=-1).astype(np.int64)

    if dir_mode == "raw":
        dirs = vectors_unit
    else:
        dirs = direction_table[ids]
        dirs[~valid] = 0

    ids[~valid] = 0
    return ids, dirs.astype(np.float32)


def read_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def frame_count(records: List[Dict]) -> int:
    if not records:
        return 0
    max_index = max(item.get("frame_index", i) for i, item in enumerate(records))
    return max_index + 1


def index_by_label(handedness_list: List[Dict]) -> Dict[str, int]:
    mapping = {}
    for idx, hand in enumerate(handedness_list or []):
        label = hand.get("label")
        if label and label not in mapping:
            mapping[label] = idx
    return mapping


def build_hl_arrays(
    records: List[Dict],
    use_world: bool = True,
    two_hands: bool = False,
    hand_order: List[str] = None,
    dir_mode: str = "center",
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if hand_order is None:
        hand_order = ["Right", "Left"]

    direction_table = build_direction_table()
    num_frames = frame_count(records)
    joints_per_hand = len(MEDIAPIPE_EDGES)
    total_joints = joints_per_hand * (2 if two_hands else 1)

    hl_ids = np.zeros((num_frames, total_joints), dtype=np.int64)
    hl_dirs = np.zeros((num_frames, total_joints, 3), dtype=np.float32)

    for item in records:
        frame_idx = item.get("frame_index")
        if frame_idx is None or frame_idx >= num_frames:
            continue

        world = item.get("multi_hand_world_landmarks") or []
        image = item.get("multi_hand_landmarks") or []
        handedness = item.get("multi_handedness") or []
        hand_map = index_by_label(handedness)

        hand_sources = world if use_world and len(world) > 0 else image

        hand_vectors = {}
        for label, idx in hand_map.items():
            if idx < len(hand_sources):
                hand_vectors[label] = extract_hand_vectors(hand_sources[idx], normalize=normalize)

        if not hand_vectors and len(hand_sources) > 0:
            hand_vectors[handedness[0].get("label", "Right") if handedness else "Right"] = extract_hand_vectors(
                hand_sources[0], normalize=normalize
            )

        if two_hands:
            frame_ids = []
            frame_dirs = []
            for label in hand_order[:2]:
                vectors = hand_vectors.get(label)
                if vectors is None:
                    frame_ids.append(np.zeros((joints_per_hand,), dtype=np.int64))
                    frame_dirs.append(np.zeros((joints_per_hand, 3), dtype=np.float32))
                else:
                    ids, dirs = assign_hl_ids(vectors, direction_table, dir_mode=dir_mode)
                    frame_ids.append(ids)
                    frame_dirs.append(dirs)
            hl_ids[frame_idx] = np.concatenate(frame_ids, axis=0)
            hl_dirs[frame_idx] = np.concatenate(frame_dirs, axis=0)
        else:
            vectors = None
            for label in hand_order:
                if label in hand_vectors:
                    vectors = hand_vectors[label]
                    break
            if vectors is None and hand_vectors:
                vectors = list(hand_vectors.values())[0]
            if vectors is not None:
                ids, dirs = assign_hl_ids(vectors, direction_table, dir_mode=dir_mode)
                hl_ids[frame_idx] = ids
                hl_dirs[frame_idx] = dirs

    return hl_ids, hl_dirs


def resolve_output_path(input_path: str, output_path: str) -> str:
    if output_path is None:
        return os.path.splitext(input_path)[0] + ".npz"
    if os.path.isdir(output_path):
        base = os.path.splitext(os.path.basename(input_path))[0] + ".npz"
        return os.path.join(output_path, base)
    return output_path


def convert_file(
    input_path: str,
    output_path: str,
    use_world: bool,
    two_hands: bool,
    hand_order: List[str],
    dir_mode: str,
    normalize: bool,
) -> str:
    records = read_json(input_path)
    hl_ids, hl_dirs = build_hl_arrays(
        records,
        use_world=use_world,
        two_hands=two_hands,
        hand_order=hand_order,
        dir_mode=dir_mode,
        normalize=normalize,
    )
    np.savez_compressed(output_path, hl_ids=hl_ids, hl_dirs=hl_dirs)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert hand landmark JSON to HL npz format.")
    parser.add_argument("--input", required=True, help="Input JSON file or directory.")
    parser.add_argument("--output", default=None, help="Output file or directory.")
    parser.add_argument("--use_world", action="store_true", help="Use multi_hand_world_landmarks when available.")
    parser.add_argument("--two_hands", action="store_true", help="Output 40 joints (Right+Left).")
    parser.add_argument("--hand_order", default="Right,Left", help="Comma-separated hand order.")
    parser.add_argument("--dir_mode", choices=["center", "raw"], default="center", help="Use center direction or raw unit vector.")
    parser.add_argument("--no_normalize", action="store_true", help="Do not normalize regional vectors.")
    args = parser.parse_args()

    input_path = args.input 
    output_path = args.output
    hand_order = [h.strip() for h in args.hand_order.split(",") if h.strip()]
    normalize = not args.no_normalize

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for name in sorted(os.listdir(input_path)):
            if not name.endswith(".json"):
                continue
            in_file = os.path.join(input_path, name)
            out_file = resolve_output_path(in_file, output_path)
            convert_file(in_file, out_file, args.use_world, args.two_hands, hand_order, args.dir_mode, normalize)
    else:
        out_file = resolve_output_path(input_path, output_path)
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        convert_file(input_path, out_file, args.use_world, args.two_hands, hand_order, args.dir_mode, normalize)


if __name__ == "__main__":
    main()
