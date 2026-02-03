import os
import sys
import json
import traceback
from pathlib import Path

import torch
import cv2
from PIL import Image
from omegaconf import OmegaConf

from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer as HFAutoTokenizer

# --------------------------
# 让脚本能从任意位置运行：把项目根目录加进 sys.path
# --------------------------
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKLWan,
    CLIPModel,
    WanT5EncoderModel,
    WanTransformer3DModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanFunControlPipeline
from videox_fun.utils.utils import (
    filter_kwargs,
    get_image_latent,
    get_video_to_video_latent,
    save_videos_grid,
)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)


# --------------------------
# 工具函数：安全读取视频元信息 + 首帧
# --------------------------
def read_video_meta_and_first_frame(video_path: str):
    """
    返回: (fps_int, total_frames_int, first_frame_pil)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(round(fps)) if fps and fps > 1e-3 else 16

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"cannot read first frame: {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        first_frame = Image.fromarray(frame)
    finally:
        cap.release()

    # 有些 mp4 的 CAP_PROP_FRAME_COUNT 不准，做一次保守兜底
    if total_frames <= 0:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return fps, 1, first_frame
        try:
            n = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                n += 1
            total_frames = max(1, n)
        finally:
            cap.release()

    return fps, total_frames, first_frame


def align_num_frames(num_frames: int, temporal_compression_ratio: int):
    """
    Wan/MagVAE 典型要求：num_frames = k * ratio + 1。
    这里做“向下对齐”，确保 <= 原始帧数，避免采样越界。
    """
    if num_frames <= 1:
        return 1
    aligned = ((num_frames - 1) // temporal_compression_ratio) * temporal_compression_ratio + 1
    return max(1, int(aligned))


def round_to_nearest_multiple(x: int, base: int = 16, min_value: int | None = None):
    """
    四舍五入到最接近的 base 倍数；如 min_value 给定，保证结果 >= min_value。
    注意：最接近可能会上调导致比原图大；后续 resize 会做插值。
    """
    if x <= 0:
        y = base
    else:
        y = int(base * round(x / base))
        y = max(base, y)
    if min_value is not None:
        y = max(int(min_value), y)
    return y


def choose_sample_size_from_first_frame(first_frame: Image.Image, base: int = 16):
    """
    按首帧分辨率，H/W 四舍五入到最接近的 16 倍数。
    返回 sample_size=[H,W]（与你的 pipeline 约定一致）。
    """
    w, h = first_frame.size
    h2 = round_to_nearest_multiple(h, base=base, min_value=base)
    w2 = round_to_nearest_multiple(w, base=base, min_value=base)
    return [h2, w2]


# --------------------------
# 主函数
# --------------------------
def main():
    # ====== 你需要改的路径/参数（集中放这里）======
    test_json_path = "/mnt/bn/douyin-ai4se-general-wl/lht/data/interhand/videox_fun/test_top100.json"
    config_path = "config/wan2.1/wan_civitai.yaml"
    model_name = "/mnt/bn/douyin-ai4se-general-wl/lht/ckpt/Wan2.1-Fun-V1.1-1.3B-Control"
    # transformer_path = None
    transformer_path = "/mnt/bn/douyin-ai4se-general-wl/lht/workspace/VideoX-Fun/output_dir_20260202_153428/checkpoint-625/diffusion_pytorch_model.safetensors"

    # output_dir = "/mnt/bn/douyin-ai4se-general-wl/lht/workspace/VideoX-Fun/exp/outputs/train_2e_5/interhand_fun_control_top100"
    output_dir = "/mnt/bn/douyin-ai4se-general-wl/lht/workspace/VideoX-Fun/exp/outputs/output_dir_20260202_153428/interhand_fun_control_top100"
    tmp_dir = os.path.join(output_dir, "_tmp_first_frames")

    # 推理超参
    prompt_default = ""
    negative_prompt = ""
    guidance_scale = 6.0
    num_inference_steps = 10
    seed_base = 43

    # sampler
    sampler_name = "Flow"  # "Flow" / "Flow_Unipc" / "Flow_DPM++"
    shift = 3  # 仅 Unipc / DPM++ 有效；Flow 会忽略也没事

    # 多卡（按你的原脚本）
    ulysses_degree = 4
    ring_degree = 2
    fsdp_dit = False
    fsdp_text_encoder = True

    # 显存模式
    GPU_memory_mode = "model_full_load"  # 或 model_cpu_offload / sequential_cpu_offload / *_and_qfloat8
    weight_dtype = torch.bfloat16

    # teacache/cfg_skip
    enable_teacache = True
    teacache_threshold = 0.10
    num_skip_start_steps = 5
    teacache_offload = False
    cfg_skip_ratio = 0.0

    # 分辨率按首帧对齐到 16 倍数
    spatial_align_base = 16

    # ====== 读评测集 ======
    with open(test_json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    assert isinstance(samples, list), "test json must be a list[dict]"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # ====== 初始化分布式设备 ======
    device = set_multi_gpus_devices(ulysses_degree, ring_degree)

    is_distributed = (ulysses_degree * ring_degree) > 1
    if is_distributed:
        import torch.distributed as dist

        rank = dist.get_rank()
    else:
        rank = 0

    # ====== 加载配置/模型（一次加载，循环复用）======
    config = OmegaConf.load(config_path)

    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(
            model_name,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = HFAutoTokenizer.from_pretrained(
        os.path.join(model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder")),
    ).to(weight_dtype).eval()

    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]

    # 兼容仓库脚本：Unipc/DPM++ 时把 shift 写到 scheduler kwargs
    if sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, OmegaConf.to_container(config["scheduler_kwargs"])))

    if transformer_path is not None:
        if rank == 0:
            print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    pipeline = WanFunControlPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )

    # ====== multi-gpu shard（可选）======
    if ulysses_degree > 1 or ring_degree > 1:
        from functools import partial

        transformer.enable_multi_gpus_inference()
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            if rank == 0:
                print("[INFO] Add FSDP DIT")
        if fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            if rank == 0:
                print("[INFO] Add FSDP TEXT ENCODER")

    # ====== 显存模式 ======
    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    # ====== TeaCache / cfg_skip ======
    if enable_teacache:
        coefficients = get_teacache_coefficients(model_name)
        if coefficients is not None:
            pipeline.transformer.enable_teacache(
                coefficients,
                num_inference_steps,
                teacache_threshold,
                num_skip_start_steps=num_skip_start_steps,
                offload=teacache_offload,
            )
    if cfg_skip_ratio and cfg_skip_ratio > 0:
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

    # ====== 主循环 ======
    temporal_ratio = int(getattr(vae.config, "temporal_compression_ratio", 4))

    failed = []
    for idx, it in enumerate(samples):
        try:
            file_path = it["file_path"]
            control_path = it["control_file_path"]
            prompt = it.get("text") or prompt_default

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"file_path not found: {file_path}")
            if not os.path.exists(control_path):
                raise FileNotFoundError(f"control_file_path not found: {control_path}")

            # 读取 file_path 的首帧作为 start_image/clip_image（你原逻辑）
            fps_f, total_f, first_frame = read_video_meta_and_first_frame(file_path)

            # 分辨率：按首帧取最近的 16 倍数
            sample_size = choose_sample_size_from_first_frame(first_frame, base=spatial_align_base)
            height, width = sample_size[0], sample_size[1]

            # ---- 关键：先读 control，再决定 num_frames（以 control 为基准）----
            fps_c, total_c, _ = read_video_meta_and_first_frame(control_path)

            control_video, _, _, _ = get_video_to_video_latent(
                control_path,
                video_length=total_c,  # 先尽量读满
                sample_size=sample_size,
                fps=None,              # 关键：不要抽帧
                ref_image=None,
            )

            if control_video is None or control_video.ndim < 5:
                raise RuntimeError(
                    f"control_video invalid shape: {None if control_video is None else control_video.shape}"
                )

            T_control = int(control_video.shape[2])
            if T_control <= 0:
                raise RuntimeError("control_video has zero frames")

            num_frames = align_num_frames(T_control, temporal_ratio)
            num_frames = min(num_frames, T_control)  # 防御性写法
            control_video = control_video[:, :, :num_frames]

            # start_image / clip_image
            start_image = get_image_latent(first_frame, sample_size=sample_size)
            clip_image = first_frame.resize((width, height), Image.BICUBIC)

            generator = torch.Generator(device=device).manual_seed(seed_base + idx)

            with torch.no_grad():
                out = pipeline(
                    prompt,
                    num_frames=num_frames,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    control_video=control_video,
                    control_camera_video=None,
                    ref_image=None,
                    start_image=start_image,
                    clip_image=clip_image,
                    shift=shift,
                ).videos  # [1,3,T,H,W] in [0,1]

            # 只在 rank0 保存
            if rank == 0:
                stem = Path(file_path).stem
                save_name = f"{idx:06d}_{stem}_h{height}_w{width}_nf{num_frames}_fps{fps_c}.mp4"
                save_path = os.path.join(output_dir, save_name)
                save_videos_grid(out, save_path, fps=int(fps_c) if fps_c else 16)
                print(f"[OK] {idx+1}/{len(samples)} saved: {save_path}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            if rank == 0:
                print(f"[FAIL] {idx+1}/{len(samples)} file={it.get('file_path')} err={e}")
                traceback.print_exc()
            failed.append({"index": idx, "file_path": it.get("file_path"), "error": str(e)})
            continue

    # 保存失败列表
    if rank == 0:
        if failed:
            failed_path = os.path.join(output_dir, "failed.json")
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
            print(f"[DONE] failed={len(failed)} (saved to {failed_path})")
        else:
            print("[DONE] all success")

        # 清理 tmp_dir（若为空）
        try:
            if os.path.isdir(tmp_dir) and len(os.listdir(tmp_dir)) == 0:
                os.rmdir(tmp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
