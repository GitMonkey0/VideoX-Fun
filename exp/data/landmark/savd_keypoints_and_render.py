import os    
import json    
import time    
import platform    
import cv2    
import numpy as np  
from pathlib import Path  
from concurrent.futures import ProcessPoolExecutor, as_completed

from mediapipe.tasks.python.core import base_options as base_options_module    
from mediapipe.tasks.python.vision import hand_landmarker as hand_landmarker_module    
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions    
import mediapipe as mp    
    
# Hand landmark connections from MediaPipe    
HAND_CONNECTIONS = [    
    [0, 1], [0, 5], [9, 13], [13, 17], [5, 9], [0, 17],  # Palm    
    [1, 2], [2, 3], [3, 4],  # Thumb    
    [5, 6], [6, 7], [7, 8],  # Index finger    
    [9, 10], [10, 11], [11, 12],  # Middle finger    
    [13, 14], [14, 15], [15, 16],  # Ring finger    
    [17, 18], [18, 19], [19, 20]  # Pinky finger    
]    
    
def infer_hand_landmarks_video(    
    video_path: str,    
    model_path: str,    
    output_json_path: str = "hand_landmarks.json",    
    output_vis_video_path: str = None,  # Optional video output    
    num_hands: int = 2,    
    min_hand_detection_confidence: float = 0.5,    
    min_hand_presence_confidence: float = 0.5,    
    min_tracking_confidence: float = 0.5,    
    prefer_gpu: bool = False,    
):    
    assert os.path.exists(video_path), f"Video not found: {video_path}"    
    assert os.path.exists(model_path), f"Model not found: {model_path}"    
    
    # GPU delegate: 目前官方Python任务API一般支持 Linux / macOS(Darwin)，Windows会报NotImplementedError    
    use_gpu = prefer_gpu and platform.system() in ["Linux", "Darwin"]    
    
    base_options = base_options_module.BaseOptions(    
        model_asset_path=model_path,    
        delegate=(    
            base_options_module.BaseOptions.Delegate.GPU    
            if use_gpu    
            else base_options_module.BaseOptions.Delegate.CPU    
        ),    
    )    
    
    options = hand_landmarker_module.HandLandmarkerOptions(    
        base_options=base_options,    
        running_mode=hand_landmarker_module._RunningMode.VIDEO,    
        num_hands=num_hands,    
        min_hand_detection_confidence=min_hand_detection_confidence,    
        min_hand_presence_confidence=min_hand_presence_confidence,    
        min_tracking_confidence=min_tracking_confidence,    
    )    
    
    cap = cv2.VideoCapture(video_path)    
    if not cap.isOpened():    
        raise RuntimeError(f"Failed to open video: {video_path}")    
    
    fps = cap.get(cv2.CAP_PROP_FPS)    
    if fps <= 1e-6:    
        fps = 30.0    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    
    writer = None    
    if output_vis_video_path:    
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")    
        writer = cv2.VideoWriter(output_vis_video_path, fourcc, fps, (w, h))    
    
    # 存储结构：每帧 -> 多只手 -> 21点    
    all_frames = []    
    frame_idx = 0    
    
    with hand_landmarker_module.HandLandmarker.create_from_options(options) as landmarker:    
        while True:    
            ok, frame_bgr = cap.read()    
            if not ok:    
                break    
    
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)    
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)    
    
            # VIDEO模式必须提供单调递增timestamp_ms。这里用 frame_idx / fps 推算    
            timestamp_ms = int(frame_idx * 1000.0 / fps)    
    
            result = landmarker.detect_for_video(mp_image, timestamp_ms)    
    
            frame_record = {    
                "frame_index": frame_idx,    
                "timestamp_ms": timestamp_ms,    
                "hands": []    
            }    
    
            # 可视化：如果保存视频，创建黑背景并画连线和点    
            if writer is not None:    
                vis = np.zeros((h, w, 3), dtype=np.uint8)  # Black background    
            else:    
                vis = None    
    
            if result.hand_landmarks:    
                for hand_i, hand_lms in enumerate(result.hand_landmarks):    
                    # handedness    
                    handedness = None    
                    if result.handedness and len(result.handedness) > hand_i and len(result.handedness[hand_i]) > 0:    
                        cat = result.handedness[hand_i][0]    
                        handedness = {"category_name": cat.category_name, "score": float(cat.score)}    
    
                    # 21个归一化点 (x,y,z)    
                    lm_list = []    
                    for lm in hand_lms:    
                        lm_list.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})    
    
                    # 世界坐标(米) 21点 (x,y,z)    
                    world_list = None    
                    if result.hand_world_landmarks and len(result.hand_world_landmarks) > hand_i:    
                        world_list = []    
                        for wl in result.hand_world_landmarks[hand_i]:    
                            world_list.append({"x": float(wl.x), "y": float(wl.y), "z": float(wl.z)})    
    
                    frame_record["hands"].append({    
                        "hand_index": hand_i,    
                        "handedness": handedness,    
                        "landmarks": lm_list,    
                        "world_landmarks": world_list,    
                    })    
    
                    # Draw connections and landmarks if saving video    
                    if writer is not None:    
                        # Convert normalized coordinates to pixel coordinates    
                        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]    
                            
                        # Draw connections    
                        for connection in HAND_CONNECTIONS:    
                            start_idx, end_idx = connection    
                            if start_idx < len(points) and end_idx < len(points):    
                                cv2.line(vis, points[start_idx], points[end_idx], (0, 255, 0), 2)    
                            
                        # Draw landmark points    
                        for px, py in points:    
                            cv2.circle(vis, (px, py), 3, (0, 0, 255), -1)    
    
            all_frames.append(frame_record)    
    
            if writer is not None:    
                writer.write(vis)    
    
            frame_idx += 1    
    
    cap.release()    
    if writer is not None:    
        writer.release()    
    
    with open(output_json_path, "w", encoding="utf-8") as f:    
        json.dump(    
            {    
                "video_path": video_path,    
                "model_path": model_path,    
                "fps": fps,    
                "width": w,    
                "height": h,    
                "delegate": "GPU" if use_gpu else "CPU",    
                "frames": all_frames,    
            },    
            f,    
            ensure_ascii=False,    
            indent=2,    
        )    
    
    print(f"Done. delegate={'GPU' if use_gpu else 'CPU'} frames={frame_idx}")    
    print(f"Keypoints saved to: {output_json_path}")    
    if output_vis_video_path:    
        print(f"Visualization video saved to: {output_vis_video_path}")  
  
  
def infer_hand_landmarks_directory(  
    input_dir: str,  
    model_path: str,  
    output_json_dir: str,  
    output_vis_video_dir: str = None,  
    num_hands: int = 2,  
    min_hand_detection_confidence: float = 0.5,  
    min_hand_presence_confidence: float = 0.5,  
    min_tracking_confidence: float = 0.5,  
    prefer_gpu: bool = False,  
    video_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')  
):  
    """  
    Process all videos in a directory and save results to output directories.  
      
    Args:  
        input_dir: Directory containing input videos  
        model_path: Path to hand landmarker model  
        output_json_dir: Directory to save JSON files  
        output_vis_video_dir: Directory to save visualization videos (optional)  
        video_extensions: Supported video file extensions  
    """  
    assert os.path.exists(input_dir), f"Input directory not found: {input_dir}"  
    assert os.path.exists(model_path), f"Model not found: {model_path}"  
      
    # Create output directories if they don't exist  
    Path(output_json_dir).mkdir(parents=True, exist_ok=True)  
    if output_vis_video_dir:  
        Path(output_vis_video_dir).mkdir(parents=True, exist_ok=True)  
      
    # Find all video files in the input directory  
    video_files = []  
    for ext in video_extensions:  
        video_files.extend(Path(input_dir).glob(f'*{ext}'))  
        video_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))  
      
    if not video_files:  
        print(f"No video files found in {input_dir}")  
        return  
      
    print(f"Found {len(video_files)} video files to process")  
      
    # Process each video  
    for video_file in video_files:  
        print(f"\nProcessing: {video_file.name}")  
          
        # Generate output paths  
        base_name = video_file.stem  
        json_path = os.path.join(output_json_dir, f"{base_name}_landmarks.json")  
        vis_path = None  
        if output_vis_video_dir:  
            vis_path = os.path.join(output_vis_video_dir, f"{base_name}_skeleton.mp4")  
          
        try:  
            # Process the video  
            infer_hand_landmarks_video(  
                video_path=str(video_file),  
                model_path=model_path,  
                output_json_path=json_path,  
                output_vis_video_path=vis_path,  
                num_hands=num_hands,  
                min_hand_detection_confidence=min_hand_detection_confidence,  
                min_hand_presence_confidence=min_hand_presence_confidence,  
                min_tracking_confidence=min_tracking_confidence,  
                prefer_gpu=prefer_gpu,  
            )  
        except Exception as e:  
            print(f"Error processing {video_file.name}: {str(e)}")  
            continue  
      
    print(f"\nBatch processing completed. Processed {len(video_files)} videos.")  

def _process_one_video(args):
    (video_path, model_path, output_json_dir, output_vis_video_dir,
     num_hands, min_det, min_pres, min_track, prefer_gpu) = args

    video_path = str(video_path)
    base_name = Path(video_path).stem
    json_path = os.path.join(output_json_dir, f"{base_name}_landmarks.json")

    vis_path = None
    if output_vis_video_dir:
        vis_path = os.path.join(output_vis_video_dir, f"{base_name}_skeleton.mp4")

    infer_hand_landmarks_video(
        video_path=video_path,
        model_path=model_path,
        output_json_path=json_path,
        output_vis_video_path=vis_path,
        num_hands=num_hands,
        min_hand_detection_confidence=min_det,
        min_hand_presence_confidence=min_pres,
        min_tracking_confidence=min_track,
        prefer_gpu=prefer_gpu,
    )
    return video_path

def infer_hand_landmarks_directory_parallel(
    input_dir: str,
    model_path: str,
    output_json_dir: str,
    output_vis_video_dir: str = None,
    num_hands: int = 2,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    prefer_gpu: bool = False,
    video_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'),
    max_workers: int = None,   # 默认 os.cpu_count()
):
    assert os.path.exists(input_dir), f"Input directory not found: {input_dir}"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    Path(output_json_dir).mkdir(parents=True, exist_ok=True)
    if output_vis_video_dir:
        Path(output_vis_video_dir).mkdir(parents=True, exist_ok=True)

    # collect videos
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f'*{ext}'))
        video_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))

    video_files = sorted(set(video_files))
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    tasks = []
    for vf in video_files:
        tasks.append((
            vf, model_path, output_json_dir, output_vis_video_dir,
            num_hands,
            min_hand_detection_confidence,
            min_hand_presence_confidence,
            min_tracking_confidence,
            prefer_gpu,
        ))

    print(f"Found {len(video_files)} videos. Start parallel with max_workers={max_workers}")

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one_video, t) for t in tasks]
        for fu in as_completed(futures):
            try:
                vp = fu.result()
                ok += 1
                print(f"[OK] {Path(vp).name}")
            except Exception as e:
                fail += 1
                print(f"[FAIL] {e}")

    print(f"Done. ok={ok} fail={fail}")

if __name__ == "__main__":    
    import numpy as np  # Add numpy import    
    MODEL_PATH = "/mnt/bn/douyin-ai4se-general-wl/lht/ckpt/hand_landmarker.task"   

    # VIDEO_PATH = "/mnt/bn/douyin-ai4se-general-wl/lht/data/interhand/raw_videos/Capture0__ROM01_No_Interaction_2_Hand__cam400262.mp4"    
    # # Example 1: Process single video  
    # print("=== Single Video Processing ===")  
    # infer_hand_landmarks_video(    
    #     video_path=VIDEO_PATH,    
    #     model_path=MODEL_PATH,    
    #     output_json_path="hand_landmarks_with_video.json",    
    #     output_vis_video_path="hand_skeleton.mp4",    
    #     num_hands=2,    
    # )  
      
    # Example 2: Process directory of videos  
    print("\n=== Directory Batch Processing ===")  
    INPUT_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/workspace/VideoX-Fun/exp/evaluate/baseline/interhand_fun_control_top100"  
    OUTPUT_JSON_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/workspace/VideoX-Fun/exp/evaluate/baseline/interhand_fun_control_top100_keypoints"  
    OUTPUT_VIS_DIR = "/mnt/bn/douyin-ai4se-general-wl/lht/data/interhand/landmark_control_videos_v2"  
      
    # infer_hand_landmarks_directory(  
    #     input_dir=INPUT_DIR,  
    #     model_path=MODEL_PATH,  
    #     output_json_dir=OUTPUT_JSON_DIR,  
    #     # output_vis_video_dir=OUTPUT_VIS_DIR,  
    #     num_hands=2,  
    # )
    infer_hand_landmarks_directory_parallel(
        input_dir=INPUT_DIR,
        model_path=MODEL_PATH,
        output_json_dir=OUTPUT_JSON_DIR,
        # output_vis_video_dir=OUTPUT_VIS_DIR,
        num_hands=2,
        max_workers=8,   # 按机器 CPU/IO 调整
    )