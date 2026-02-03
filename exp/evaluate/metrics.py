import os
import json
import argparse
import numpy as np
from glob import glob
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

def map_pred_to_gt_name(pred_fname: str) -> str:
    """
    000000_Capture1__ROM09_...__cam400372_h512_w336_nf125_fps30.json
    -> Capture1__ROM09_...__cam400372_landmarks.json
    """
    name = pred_fname
    parts = name.split("_", 1)
    name_core = parts[1] if len(parts) == 2 else name  # 去掉前缀 000000_

    # 去掉后面 _h512... 等等，只保留到 camXXXXXX 这部分
    if "_h" in name_core:
        name_core = name_core[:name_core.index("_h")]
    else:
        # 如果未来格式变了，没有 _h，就先去掉尾部的 .json
        if name_core.endswith(".json"):
            name_core = name_core[:-5]

    # 新的 GT 命名规则：加上 _landmarks.json
    gt_name = name_core + "_landmarks.json"
    return gt_name
    
def load_frames(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "frames" in data:
        return data["frames"]

    raise ValueError(f"Unknown json format: {path}")

# -----------------------------
# 2) MediaPipe Hands skeleton
# -----------------------------
ROOT_IDX = 0
FINGERTIPS = [4, 8, 12, 16, 20]

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17)                # palm (optional but common)
]

def extract_hands_from_frame(frame: dict) -> Dict[str, np.ndarray]:
    """
    返回 dict: key->(21,3) world landmarks
    - 旧格式：frame["multi_hand_world_landmarks"] + frame["multi_handedness"]
    - 新格式：frame["hands"][i]["world_landmarks"] + ["handedness"]["category_name"]
    """
    out: Dict[str, np.ndarray] = {}

    for i, hand in enumerate(frame["hands"]):
        wl = hand.get("world_landmarks", None)
        if not wl or len(wl) != 21:
            continue

        # wl: list of {"x","y","z"} -> (21,3)
        arr = np.asarray([[p["x"], p["y"], p["z"]] for p in wl], dtype=np.float32)
        if arr.shape != (21, 3):
            continue

        handed = hand.get("handedness", None) or {}
        label = handed.get("category_name", None)  # "Left"/"Right"
        if label in ("Left", "Right"):
            key = label
        else:
            key = f"hand{i}"

        out[key] = arr
    return out

def match_hands(pred_frame, gt_frame):
    ph = extract_hands_from_frame(pred_frame)
    gh = extract_hands_from_frame(gt_frame)

    pred_list = list(ph.items())
    gt_list = list(gh.items())
    if not pred_list or not gt_list:
        return []

    used = set()
    pairs = []
    for pk, ppts in pred_list:
        pw = ppts[ROOT_IDX]
        best_j, best_d = None, 1e9
        for j, (gk, gpts) in enumerate(gt_list):
            if j in used:
                continue
            d = float(np.linalg.norm(pw - gpts[ROOT_IDX]))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None:
            used.add(best_j)
            gk, gpts = gt_list[best_j]
            pairs.append((ppts, gpts, f"{pk}->{gk}"))
    return pairs


# -----------------------------
# 4) 几何指标：MPJPE / RA / PA / PCK / tips / bone
# -----------------------------
def per_joint_l2(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1)  # (21,)


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(per_joint_l2(pred, gt).mean())


def root_align(pred: np.ndarray, gt: np.ndarray, root_idx: int = ROOT_IDX) -> Tuple[np.ndarray, np.ndarray]:
    return pred - pred[root_idx:root_idx+1], gt - gt[root_idx:root_idx+1]


def kabsch_align(pred: np.ndarray, gt: np.ndarray, with_scale: bool = False) -> np.ndarray:
    """
    对 pred 做 (R,t) 或 (s,R,t) 对齐到 gt
    pred, gt: (N,3)
    """
    X = pred.astype(np.float64)
    Y = gt.astype(np.float64)

    muX = X.mean(axis=0, keepdims=True)
    muY = Y.mean(axis=0, keepdims=True)
    X0 = X - muX
    Y0 = Y - muY

    H = X0.T @ Y0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    if with_scale:
        varX = (X0 ** 2).sum()
        scale = (S.sum() / varX) if varX > 1e-12 else 1.0
    else:
        scale = 1.0

    X_aligned = scale * (X0 @ R) + muY
    return X_aligned.astype(np.float32)


def compute_pa_mpjpe(pred: np.ndarray, gt: np.ndarray, with_scale: bool = False) -> float:
    pred_aligned = kabsch_align(pred, gt, with_scale=with_scale)
    return compute_mpjpe(pred_aligned, gt)


def compute_pck(pred: np.ndarray, gt: np.ndarray, thr_m: float) -> float:
    d = per_joint_l2(pred, gt)
    return float((d < thr_m).mean())


def compute_tips_epe(pred: np.ndarray, gt: np.ndarray) -> float:
    d = per_joint_l2(pred[FINGERTIPS], gt[FINGERTIPS])
    return float(d.mean())


def bone_length_relative_error(pred: np.ndarray, gt: np.ndarray) -> float:
    eps = 1e-8
    errs = []
    for a, b in HAND_BONES:
        lp = float(np.linalg.norm(pred[a] - pred[b]))
        lg = float(np.linalg.norm(gt[a] - gt[b]))
        errs.append(abs(lp - lg) / (lg + eps))
    return float(np.mean(errs)) if errs else None


# -----------------------------
# 5) 动态指标：TE / vel / acc / jerk（支持 timestamp）
# -----------------------------
def _safe_dt_seconds(ts_ms: np.ndarray, fps_fallback: float) -> np.ndarray:
    """
    ts_ms: (T,) int/float
    返回 dt: (T-1,) seconds
    - 用 timestamp 差分
    - 异常/非正/过小 采用 fallback = 1/fps
    """
    if ts_ms is None or len(ts_ms) < 2:
        return np.full((0,), 1.0 / fps_fallback, dtype=np.float32)

    dt = np.diff(ts_ms.astype(np.float64)) / 1000.0
    fallback = 1.0 / fps_fallback
    # 过小或非正都替换
    dt = np.where(dt <= 1e-6, fallback, dt)
    # 极端大也可以裁一下（可选）
    dt = np.clip(dt, 1e-4, 1.0)
    return dt.astype(np.float32)


def trajectory_error_wrist(pred_seq: np.ndarray, gt_seq: np.ndarray) -> Optional[float]:
    """
    pred_seq, gt_seq: (T,3)
    """
    if pred_seq.shape[0] == 0:
        return None
    return float(np.linalg.norm(pred_seq - gt_seq, axis=-1).mean())


def dynamic_error(pred_seq: np.ndarray, gt_seq: np.ndarray, ts_ms: np.ndarray,
                  order: int, fps_fallback: float) -> Optional[float]:
    """
    pred_seq, gt_seq: (T,J,3)
    order: 1 vel, 2 acc, 3 jerk
    使用非均匀 dt：差分 / dt
    """
    T = pred_seq.shape[0]
    if T <= order:
        return None

    dt = _safe_dt_seconds(ts_ms, fps_fallback=fps_fallback)  # (T-1,)
    # 构造每阶 dt
    P = pred_seq
    G = gt_seq
    dt_k = dt

    for k in range(order):
        # 一阶差分
        P = P[1:] - P[:-1]
        G = G[1:] - G[:-1]
        # 除以 dt（广播到 (T-1,1,1)）
        P = P / dt_k[:, None, None]
        G = G / dt_k[:, None, None]

        # 下一阶的 dt 长度会减少 1
        if k < order - 1:
            dt_k = dt_k[1:]  # (T-2), (T-3)...

    # L2 error over (T-order, J)
    d = np.linalg.norm(P - G, axis=-1)
    return float(d.mean())


# -----------------------------
# 6) 视频级汇总：公共前缀 + 多手匹配
# -----------------------------
@dataclass
class VideoMetrics:
    mpjpe: Optional[float] = None
    ra_mpjpe: Optional[float] = None
    pa_mpjpe: Optional[float] = None
    pck_1cm: Optional[float] = None
    pck_2cm: Optional[float] = None
    pck_5cm: Optional[float] = None
    tips_epe: Optional[float] = None
    bone_len_rel_err: Optional[float] = None
    wrist_te: Optional[float] = None
    vel_err: Optional[float] = None
    acc_err: Optional[float] = None
    jerk_err: Optional[float] = None
    valid_frames: int = 0
    valid_pairs: int = 0


def compute_video_metrics(pred_frames: List[dict],
                          gt_frames: List[dict],
                          fps_fallback: float = 30.0,
                          pa_with_scale: bool = False,
                          pck_thresholds_m: Tuple[float, float, float] = (0.01, 0.02, 0.05),
                          dynamic_mode: str = "per_hand_then_avg") -> VideoMetrics:
    """
    dynamic_mode:
      - "per_hand_then_avg": 按 hand key（Left/Right/hand0/hand1）分别组成连续序列计算动态指标，再平均（推荐）
      - "concat_all_pairs": 直接把所有(pair,frame)拼成长序列再算动态（更简单但语义略弱）
    """
    n = min(len(pred_frames), len(gt_frames))
    if n == 0:
        return VideoMetrics()

    # 逐帧几何聚合
    mpjpes, ra_mpjpes, pa_mpjpes = [], [], []
    tips, bone_errs = [], []
    pck_map = {thr: [] for thr in pck_thresholds_m}

    valid_frames = 0
    valid_pairs = 0

    # 为动态指标准备：按 hand_key 收集时间序列（只收集“同时存在且匹配”的帧）
    # seq_map[key] = list of (pred_pts, gt_pts, timestamp_ms)
    seq_map: Dict[str, List[Tuple[np.ndarray, np.ndarray, int]]] = {}

    for i in range(n):
        pf = pred_frames[i]
        gf = gt_frames[i]
        pairs = match_hands(pf, gf)
        if not pairs:
            continue

        valid_frames += 1
        ts = int(gf.get("timestamp_ms", pf.get("timestamp_ms", i)))  # timestamp 兜底

        for pred_pts, gt_pts, key in pairs:
            valid_pairs += 1

            mpjpes.append(per_joint_l2(pred_pts, gt_pts))

            p_ra, g_ra = root_align(pred_pts, gt_pts, ROOT_IDX)
            ra_mpjpes.append(per_joint_l2(p_ra, g_ra))

            p_pa = kabsch_align(pred_pts, gt_pts, with_scale=pa_with_scale)
            pa_mpjpes.append(per_joint_l2(p_pa, gt_pts))

            tips.append(compute_tips_epe(pred_pts, gt_pts))
            bone_errs.append(bone_length_relative_error(pred_pts, gt_pts))

            for thr in pck_thresholds_m:
                pck_map[thr].append(compute_pck(pred_pts, gt_pts, thr))

            seq_map.setdefault(key, []).append((pred_pts, gt_pts, ts))

    if valid_pairs == 0:
        return VideoMetrics(valid_frames=valid_frames, valid_pairs=valid_pairs)

    vm = VideoMetrics(
        mpjpe=float(np.concatenate(mpjpes, axis=0).mean()),
        ra_mpjpe=float(np.concatenate(ra_mpjpes, axis=0).mean()),
        pa_mpjpe=float(np.concatenate(pa_mpjpes, axis=0).mean()),
        pck_1cm=float(np.mean(pck_map.get(0.01, []))) if pck_map.get(0.01) else None,
        pck_2cm=float(np.mean(pck_map.get(0.02, []))) if pck_map.get(0.02) else None,
        pck_5cm=float(np.mean(pck_map.get(0.05, []))) if pck_map.get(0.05) else None,
        tips_epe=float(np.mean(tips)) if tips else None,
        bone_len_rel_err=float(np.mean(bone_errs)) if bone_errs else None,
        valid_frames=valid_frames,
        valid_pairs=valid_pairs,
    )

    # ---- 动态指标 ----
    wrist_tes, vel_errs, acc_errs, jerk_errs = [], [], [], []

    if dynamic_mode == "concat_all_pairs":
        # 拼在一起（不推荐但简单）
        all_pred, all_gt, all_ts = [], [], []
        for key, items in seq_map.items():
            for p, g, ts in items:
                all_pred.append(p)
                all_gt.append(g)
                all_ts.append(ts)
        if len(all_pred) >= 2:
            P = np.stack(all_pred, axis=0)
            G = np.stack(all_gt, axis=0)
            TS = np.asarray(all_ts, dtype=np.int64)
            wrist_tes.append(trajectory_error_wrist(P[:, ROOT_IDX], G[:, ROOT_IDX]))
            vel_errs.append(dynamic_error(P, G, TS, order=1, fps_fallback=fps_fallback))
            acc_errs.append(dynamic_error(P, G, TS, order=2, fps_fallback=fps_fallback))
            jerk_errs.append(dynamic_error(P, G, TS, order=3, fps_fallback=fps_fallback))

    else:
        # per_hand_then_avg：每个 key 单独形成序列计算
        for key, items in seq_map.items():
            if len(items) < 2:
                continue
            P = np.stack([x[0] for x in items], axis=0)  # (T,21,3)
            G = np.stack([x[1] for x in items], axis=0)
            TS = np.asarray([x[2] for x in items], dtype=np.int64)

            wrist_tes.append(trajectory_error_wrist(P[:, ROOT_IDX], G[:, ROOT_IDX]))
            vel_errs.append(dynamic_error(P, G, TS, order=1, fps_fallback=fps_fallback))
            acc_errs.append(dynamic_error(P, G, TS, order=2, fps_fallback=fps_fallback))
            jerk_errs.append(dynamic_error(P, G, TS, order=3, fps_fallback=fps_fallback))

    def _mean_ignore_none(xs):
        xs2 = [x for x in xs if x is not None]
        return float(np.mean(xs2)) if xs2 else None

    vm.wrist_te = _mean_ignore_none(wrist_tes)
    vm.vel_err = _mean_ignore_none(vel_errs)
    vm.acc_err = _mean_ignore_none(acc_errs)
    vm.jerk_err = _mean_ignore_none(jerk_errs)

    return vm


# -----------------------------
# 7) 批量跑 + 汇总
# -----------------------------
def mean_ignore_none(xs: List[Optional[float]]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None]
    return float(np.mean(xs2)) if xs2 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--gt_root", type=str, required=True)
    parser.add_argument("--fps_fallback", type=float, default=30.0)
    parser.add_argument("--pa_with_scale", action="store_true", help="PA-MPJPE 是否允许 scale（默认不允许）")
    parser.add_argument("--dynamic_mode", type=str, default="per_hand_then_avg",
                        choices=["per_hand_then_avg", "concat_all_pairs"])
    parser.add_argument("--save_csv", type=str, default=None, help="可选：保存 per-video 指标到 CSV")
    args = parser.parse_args()

    gt_files = glob(os.path.join(args.gt_root, "*.json"))
    gt_name_to_path = {os.path.basename(p): p for p in gt_files}

    pred_files = sorted(glob(os.path.join(args.pred_root, "*.json")))

    per_video_rows = []
    missing_gt = []

    for pred_path in pred_files:
        pred_name = os.path.basename(pred_path)
        gt_name = map_pred_to_gt_name(pred_name)

        if gt_name not in gt_name_to_path:
            missing_gt.append((pred_name, gt_name))
            continue

        gt_path = gt_name_to_path[gt_name]
        pred_frames = load_frames(pred_path)
        gt_frames = load_frames(gt_path)

        vm = compute_video_metrics(
            pred_frames, gt_frames,
            fps_fallback=args.fps_fallback,
            pa_with_scale=args.pa_with_scale,
            dynamic_mode=args.dynamic_mode
        )

        row = {"pred_file": pred_name, "gt_file": gt_name}
        row.update(asdict(vm))
        per_video_rows.append(row)

    # ---- 打印 per-video ----
    print("==== Per-video metrics (world landmarks, meters) ====")
    for r in per_video_rows:
        print(f"\n{r['pred_file']}  <->  {r['gt_file']}")
        print(f"  valid_frames={r['valid_frames']} valid_pairs={r['valid_pairs']}")
        print(f"  MPJPE(m)            : {r['mpjpe']}")
        print(f"  RA-MPJPE(m)         : {r['ra_mpjpe']}")
        print(f"  PA-MPJPE(m)         : {r['pa_mpjpe']}")
        print(f"  PCK@1cm             : {r['pck_1cm']}")
        print(f"  PCK@2cm             : {r['pck_2cm']}")
        print(f"  PCK@5cm             : {r['pck_5cm']}")
        print(f"  Tips EPE(m)         : {r['tips_epe']}")
        print(f"  BoneLenRelErr       : {r['bone_len_rel_err']}")
        print(f"  Wrist TE(m)         : {r['wrist_te']}")
        print(f"  VelErr(m/s)         : {r['vel_err']}")
        print(f"  AccErr(m/s^2)       : {r['acc_err']}")
        print(f"  JerkErr(m/s^3)      : {r['jerk_err']}")

    # ---- overall 汇总（对 per-video 平均）----
    mpjpe_all = [r["mpjpe"] for r in per_video_rows]
    ra_all = [r["ra_mpjpe"] for r in per_video_rows]
    pa_all = [r["pa_mpjpe"] for r in per_video_rows]
    pck1_all = [r["pck_1cm"] for r in per_video_rows]
    pck2_all = [r["pck_2cm"] for r in per_video_rows]
    pck5_all = [r["pck_5cm"] for r in per_video_rows]
    tips_all = [r["tips_epe"] for r in per_video_rows]
    bone_all = [r["bone_len_rel_err"] for r in per_video_rows]
    te_all = [r["wrist_te"] for r in per_video_rows]
    vel_all = [r["vel_err"] for r in per_video_rows]
    acc_all = [r["acc_err"] for r in per_video_rows]
    jerk_all = [r["jerk_err"] for r in per_video_rows]

    print("\n==== Overall (mean over videos, ignore None) ====")
    print(f"Mean MPJPE(m)        : {mean_ignore_none(mpjpe_all)}")
    print(f"Mean RA-MPJPE(m)     : {mean_ignore_none(ra_all)}")
    print(f"Mean PA-MPJPE(m)     : {mean_ignore_none(pa_all)}")
    print(f"Mean PCK@1cm         : {mean_ignore_none(pck1_all)}")
    print(f"Mean PCK@2cm         : {mean_ignore_none(pck2_all)}")
    print(f"Mean PCK@5cm         : {mean_ignore_none(pck5_all)}")
    print(f"Mean Tips EPE(m)     : {mean_ignore_none(tips_all)}")
    print(f"Mean BoneLenRelErr   : {mean_ignore_none(bone_all)}")
    print(f"Mean Wrist TE(m)     : {mean_ignore_none(te_all)}")
    print(f"Mean VelErr(m/s)     : {mean_ignore_none(vel_all)}")
    print(f"Mean AccErr(m/s^2)   : {mean_ignore_none(acc_all)}")
    print(f"Mean JerkErr(m/s^3)  : {mean_ignore_none(jerk_all)}")

    if missing_gt:
        print("\n[WARN] 以下预测文件映射后找不到对应 GT：")
        for pred, gt in missing_gt:
            print(f"  pred: {pred} -> expected gt: {gt}")

    # ---- 可选保存 CSV（不依赖 pandas）----
    if args.save_csv:
        import csv
        keys = list(per_video_rows[0].keys()) if per_video_rows else []
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_video_rows:
                w.writerow(r)
        print(f"\nSaved CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()