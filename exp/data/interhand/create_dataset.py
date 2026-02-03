#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

IMGNUM = re.compile(r"^image(\d+)\.jpg$", re.IGNORECASE)

def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", s)

def make_unique(dst: Path, stem: str, ext=".mp4") -> Path:
    out = dst / f"{stem}{ext}"
    if not out.exists():
        return out
    i = 1
    while True:
        cand = dst / f"{stem}__{i:03d}{ext}"
        if not cand.exists():
            return cand
        i += 1

def run_ffmpeg(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed\nCMD: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")

def find_cam_dirs_structured(src: Path):
    cam_dirs = []
    for cap in sorted(src.glob("Capture*")):
        if not cap.is_dir():
            continue
        for cam in cap.glob("*/*"):  # CaptureX/SEQ/camXXXX
            if cam.is_dir() and cam.name.startswith("cam"):
                cam_dirs.append(cam)
    return cam_dirs

def list_jpgs(cam_dir: Path):
    return [p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]

def detect_image2_pattern(cam_dir: Path, imgs):
    """
    如果文件名形如 image00001.jpg 且从最小到最大连续无缺帧，则返回：
    (start_number, ndigits)
    否则返回 None
    """
    nums = []
    ndigits = None
    for p in imgs:
        m = IMGNUM.match(p.name)
        if not m:
            return None
        n_str = m.group(1)
        if ndigits is None:
            ndigits = len(n_str)
        elif len(n_str) != ndigits:
            return None
        nums.append(int(n_str))

    if not nums:
        return None

    nums.sort()
    start = nums[0]
    end = nums[-1]
    if len(nums) != (end - start + 1):
        return None  # 有缺帧/重复，不能用 pattern

    return start, ndigits

def write_concat_list(imgs, list_path: Path):
    lines = []
    for im in sorted(imgs, key=lambda p: p.name):
        s = str(im.resolve())
        s = s.replace("'", r"'\''")
        lines.append(f"file '{s}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def images_to_video(cam_dir: Path, out_mp4: Path, fps: float, crf: int, preset: str, ffmpeg_threads: int):
    imgs = list_jpgs(cam_dir)
    if not imgs:
        return False

    # 1) 尽量用 image2 pattern（更快）
    pat = detect_image2_pattern(cam_dir, imgs)
    if pat is not None:
        start_number, ndigits = pat
        pattern = f"image%0{ndigits}d.jpg"
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error", "-stats",
            "-threads", str(ffmpeg_threads),
            "-framerate", str(fps),
            "-start_number", str(start_number),
            "-i", str(cam_dir / pattern),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-preset", preset,
            str(out_mp4),
        ]
        run_ffmpeg(cmd)
        return True

    # 2) fallback：concat demuxer（慢一些，但兼容乱序/缺帧）
    with tempfile.TemporaryDirectory(prefix="ffconcat_") as td:
        list_path = Path(td) / "list.txt"
        # 按 imagexxxxx 的数字自然排序（否则容易乱）
        def sort_key(p: Path):
            m = IMGNUM.match(p.name)
            if m:
                return (0, int(m.group(1)))
            return (1, p.name.lower())
        imgs_sorted = sorted(imgs, key=sort_key)

        write_concat_list(imgs_sorted, list_path)
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error", "-stats",
            "-threads", str(ffmpeg_threads),
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            # 不再用 fps filter，减轻处理；用输出 -r 保持帧率
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-r", str(fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-preset", preset,
            str(out_mp4),
        ]
        run_ffmpeg(cmd)
        return True

def worker(cam_dir_str: str, src_str: str, dst_str: str, fps: float, crf: int, preset: str, ffmpeg_threads: int, overwrite: bool):
    cam_dir = Path(cam_dir_str)
    src = Path(src_str)
    dst = Path(dst_str)

    rel = cam_dir.relative_to(src)
    stem = sanitize("__".join(rel.parts))
    out_mp4 = make_unique(dst, stem, ".mp4")

    if (not overwrite) and out_mp4.exists():
        return ("skip", cam_dir_str, str(out_mp4), "exists")

    try:
        done = images_to_video(cam_dir, out_mp4, fps, crf, preset, ffmpeg_threads)
        if done:
            return ("ok", cam_dir_str, str(out_mp4), "")
        else:
            return ("skip", cam_dir_str, str(out_mp4), "no_images")
    except Exception as e:
        return ("fail", cam_dir_str, str(out_mp4), str(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--fps", type=float, default=5.0)

    # 性能相关
    ap.add_argument("--workers", type=int, default=0, help="并行进程数，0=自动")
    ap.add_argument("--ffmpeg-threads", type=int, default=1, help="每个ffmpeg进程使用的线程数")
    ap.add_argument("--preset", default="veryfast", help="x264 preset: ultrafast/superfast/veryfast/faster/fast/medium...")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print("finding cam dirs by structure...", flush=True)
    cam_dirs = find_cam_dirs_structured(src)
    print(f"Found {len(cam_dirs)} camera dirs", flush=True)

    cpu = os.cpu_count() or 8
    workers = args.workers if args.workers and args.workers > 0 else min(16, cpu)  # 你也可改成 cpu//2
    print(f"Using workers={workers}, ffmpeg_threads={args.ffmpeg_threads}, preset={args.preset}, crf={args.crf}", flush=True)

    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                worker,
                str(d), str(src), str(dst),
                args.fps, args.crf, args.preset, args.ffmpeg_threads,
                args.overwrite
            )
            for d in cam_dirs
        ]
        for fu in as_completed(futs):
            status, cam_dir_str, out_mp4_str, msg = fu.result()
            if status == "ok":
                ok += 1
                # 只打印少量也行；这里保持你原来风格
                print(f"[OK] {Path(out_mp4_str).name}", flush=True)
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"[FAIL] {cam_dir_str}\n{msg}\n", flush=True)

    print(f"Done ok={ok} skip={skip} fail={fail} out={dst}", flush=True)

if __name__ == "__main__":
    main()