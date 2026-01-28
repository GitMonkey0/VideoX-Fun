import os  
import glob  
import json  
import cv2  
import numpy as np  
import mediapipe as mediapipe  # Changed from 'mp' to avoid conflict  
from pathlib import Path  
from mediapipe.tasks import python  
from mediapipe.tasks.python import vision  
from concurrent.futures import ProcessPoolExecutor, as_completed  
import multiprocessing as mp  # Keep this for multiprocessing  
from typing import Optional, Dict, List, Tuple  
import time  
import logging  
  
# Configure logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
class OptimizedHandLandmarker:  
    """Optimized hand landmarker with GPU support and error handling"""  
      
    def __init__(self, model_path: str = 'hand_landmarker.task', use_gpu: bool = False):  
        self.model_path = model_path  
        self.use_gpu = use_gpu  
        self.landmarker = None  
        self._initialize_landmarker()  
      
    def _initialize_landmarker(self):  
        """Initialize landmarker with GPU fallback to CPU"""  
        try:  
            # Try GPU initialization first  
            base_options = python.BaseOptions(  
                model_asset_path=self.model_path,  
                delegate=python.BaseOptions.Delegate.GPU if self.use_gpu else None  
            )  
            options = vision.HandLandmarkerOptions(  
                base_options=base_options,  
                num_hands=2,  
                running_mode=vision.RunningMode.VIDEO  
            )  
            self.landmarker = vision.HandLandmarker.create_from_options(options)  
            logger.info("Successfully initialized GPU-accelerated hand landmarker")  
        except Exception as e:  
            logger.warning(f"GPU initialization failed: {e}. Falling back to CPU")  
            try:  
                # Fallback to CPU  
                base_options = python.BaseOptions(model_asset_path=self.model_path)  
                options = vision.HandLandmarkerOptions(  
                    base_options=base_options,  
                    num_hands=2,  
                    running_mode=vision.RunningMode.VIDEO  
                )  
                self.landmarker = vision.HandLandmarker.create_from_options(options)  
                logger.info("Successfully initialized CPU hand landmarker")  
            except Exception as cpu_e:  
                logger.error(f"CPU initialization also failed: {cpu_e}")  
                raise  
      
    def extract_hand_landmarks(self, rgb_image: np.ndarray, timestamp_ms: int) -> Dict:  
        """Extract hand landmarks with optimized processing"""  
        # Fixed: Use mediapipe.Image instead of mp.Image  
        mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_image)  
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)  
          
        output = {  
            'multi_hand_landmarks': [],  
            'multi_hand_world_landmarks': [],  
            'multi_handedness': []  
        }  
          
        if result.hand_landmarks:  
            for landmarks in result.hand_landmarks:  
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  
                output['multi_hand_landmarks'].append(landmark_array)  
          
        if result.hand_world_landmarks:  
            for world_landmarks in result.hand_world_landmarks:  
                world_landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])  
                output['multi_hand_world_landmarks'].append(world_landmark_array)  
          
        if result.handedness:  
            for handedness in result.handedness:  
                output['multi_handedness'].append({  
                    'label': handedness[0].category_name,  
                    'score': handedness[0].score  
                })  
          
        return output  
      
    def __del__(self):  
        """Cleanup resources"""  
        if self.landmarker:  
            self.landmarker.close()  
  
# Hand connections for visualization  
HAND_CONNECTIONS = [  
    [0, 1], [0, 5], [9, 13], [13, 17], [5, 9], [0, 17],  
    [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8],  
    [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16],  
    [17, 18], [18, 19], [19, 20]  
]  
  
def save_annotated_image(image: np.ndarray, landmarks: Dict, filename: str):  
    """Save annotated image with hand landmarks"""  
    annotated_image = image.copy()  
    height, width = image.shape[:2]  
      
    for hand_idx, hand_landmarks in enumerate(landmarks['multi_hand_landmarks']):  
        pixel_coords = []  
        for lm in hand_landmarks:  
            x = int(lm[0] * width)  
            y = int(lm[1] * height)  
            pixel_coords.append((x, y))  
          
        # Draw connections  
        for connection in HAND_CONNECTIONS:  
            start_idx, end_idx = connection  
            if start_idx < len(pixel_coords) and end_idx < len(pixel_coords):  
                cv2.line(annotated_image, pixel_coords[start_idx], pixel_coords[end_idx],   
                        (0, 255, 0), 2)  
          
        # Draw keypoints  
        for coord in pixel_coords:  
            cv2.circle(annotated_image, coord, 5, (0, 0, 255), -1)  
          
        # Draw hand label  
        if hand_idx < len(landmarks['multi_handedness']):  
            hand_label = landmarks['multi_handedness'][hand_idx]['label']  
            confidence = landmarks['multi_handedness'][hand_idx]['score']  
            cv2.putText(annotated_image, f"{hand_label}: {confidence:.2f}",   
                       (10, 30 + hand_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,   
                       (255, 255, 0), 2)  
      
    cv2.imwrite(filename, annotated_image)  
  
def process_single_video(args: Tuple[str, str, bool, bool]) -> Dict:  
    """Process a single video file - designed for multiprocessing"""  
    video_path, output_dir, save_annotated, use_gpu = args  
      
    try:  
        # 1. 使用 context manager 确保资源清理  
        base_options = python.BaseOptions(  
            model_asset_path='hand_landmarker.task',  
            delegate=python.BaseOptions.Delegate.GPU if use_gpu else None  
        )  
        options = vision.HandLandmarkerOptions(  
            base_options=base_options,  
            num_hands=2,  
            running_mode=vision.RunningMode.VIDEO  
        )  
          
        # 使用 with 语句！  
        with vision.HandLandmarker.create_from_options(options) as landmarker:  
            cap = None  
            try:  
                cap = cv2.VideoCapture(video_path)  
                if not cap.isOpened():  
                    raise ValueError(f"Cannot open video: {video_path}")  
                  
                # Setup output paths  
                base_name = os.path.splitext(os.path.basename(video_path))[0]  
                output_json = os.path.join(output_dir, f"{base_name}.json")  
                annotated_dir = os.path.join(output_dir, f"{base_name}_annotated_frames") if save_annotated else None  
                  
                if save_annotated and annotated_dir:  
                    Path(annotated_dir).mkdir(parents=True, exist_ok=True)  
                  
                # 2. 分批写入，不要全部存在内存中  
                with open(output_json, 'w') as f:  
                    f.write('[\n')  
                    first = True  
                    frame_index = 0  
                      
                    while True:  
                        success, frame = cap.read()  
                        if not success:  
                            break  
                          
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))  
                          
                        # 处理帧 - 提取 landmarks  
                        mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)  
                        result = landmarker.detect_for_video(mp_image, timestamp_ms)  
                          
                        # 构建输出数据  
                        frame_output = {  
                            'frame_index': frame_index,  
                            'timestamp_ms': timestamp_ms,  
                            'multi_hand_landmarks': [],  
                            'multi_hand_world_landmarks': [],  
                            'multi_handedness': []  
                        }  
                          
                        # 处理 hand landmarks  
                        if result.hand_landmarks:  
                            for landmarks in result.hand_landmarks:  
                                landmark_array = [[lm.x, lm.y, lm.z] for lm in landmarks]  
                                frame_output['multi_hand_landmarks'].append(landmark_array)  
                          
                        # 处理 world landmarks  
                        if result.hand_world_landmarks:  
                            for world_landmarks in result.hand_world_landmarks:  
                                world_landmark_array = [[lm.x, lm.y, lm.z] for lm in world_landmarks]  
                                frame_output['multi_hand_world_landmarks'].append(world_landmark_array)  
                          
                        # 处理 handedness  
                        if result.handedness:  
                            for handedness in result.handedness:  
                                frame_output['multi_handedness'].append({  
                                    'label': handedness[0].category_name,  
                                    'score': handedness[0].score  
                                })  
                          
                        # 立即序列化，不存储在列表中  
                        if not first:  
                            f.write(',\n')  
                        json.dump(frame_output, f)  
                        first = False  
                          
                        # Save annotated frame if requested  
                        if save_annotated and frame_output['multi_hand_landmarks']:  
                            annotated_path = os.path.join(annotated_dir, f"frame_{frame_index:06d}_annotated.jpg")  
                            save_annotated_image(frame, frame_output, annotated_path)  
                          
                        frame_index += 1  
                      
                    f.write('\n]')  
                  
                return {  
                    'video_path': video_path,  
                    'status': 'success',  
                    'frames_processed': frame_index,  
                    'output_file': output_json  
                }  
                  
            finally:  
                if cap:  
                    cap.release()  
                      
    except Exception as e:  
        logger.error(f"Error processing {video_path}: {e}")  
        return {  
            'video_path': video_path,  
            'status': 'error',  
            'error': str(e)  
        }
  
def batch_process_videos_optimized(  
    input_dir: str,   
    output_dir: str,   
    save_annotated: bool = False,  
    use_gpu: bool = False,  
    max_workers: Optional[int] = None,  
    video_extensions: List[str] = ['*.mp4', '*.avi', '*.mov']  
) -> Dict:  
    """  
    Optimized batch processing with GPU acceleration and concurrency  
      
    Args:  
        input_dir: Input video directory  
        output_dir: Output directory for results  
        save_annotated: Whether to save annotated frames  
        use_gpu: Whether to use GPU acceleration  
        max_workers: Number of parallel processes (default: CPU count)  
        video_extensions: Supported video extensions  
      
    Returns:  
        Dictionary with processing results and statistics  
    """  
    # Create output directory  
    Path(output_dir).mkdir(parents=True, exist_ok=True)  
      
    # Find all video files  
    video_files = []  
    for ext in video_extensions:  
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))  
        video_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))  
      
    if not video_files:  
        logger.warning(f"No video files found in {input_dir}")  
        return {'status': 'no_files', 'videos_processed': 0}  
      
    # Set default number of workers  
    if max_workers is None:  
        max_workers = min(4, len(video_files))  
      
    logger.info(f"Found {len(video_files)} video files, processing with {max_workers} workers")  
      
    # Prepare arguments for multiprocessing  
    process_args = [(video_path, output_dir, save_annotated, use_gpu) for video_path in video_files]  
      
    # Process videos in parallel  
    start_time = time.time()  
    results = []  
      
    with ProcessPoolExecutor(max_workers=max_workers) as executor:  
        future_to_video = {executor.submit(process_single_video, args): args[0] for args in process_args}  
          
        for future in as_completed(future_to_video):  
            video_path = future_to_video[future]  
            try:  
                result = future.result()  
                results.append(result)  
                if result['status'] == 'success':  
                    logger.info(f"✓ Completed: {os.path.basename(video_path)} - {result['frames_processed']} frames")  
                else:  
                    logger.error(f"✗ Failed: {os.path.basename(video_path)} - {result.get('error', 'Unknown error')}")  
            except Exception as e:  
                logger.error(f"✗ Exception processing {video_path}: {e}")  
                results.append({  
                    'video_path': video_path,  
                    'status': 'error',  
                    'error': str(e)  
                })  
      
    # Calculate statistics  
    end_time = time.time()  
    successful = [r for r in results if r['status'] == 'success']  
    failed = [r for r in results if r['status'] == 'error']  
      
    total_frames = sum(r.get('frames_processed', 0) for r in successful)  
    processing_time = end_time - start_time  
      
    stats = {  
        'status': 'completed',  
        'total_videos': len(video_files),  
        'successful_videos': len(successful),  
        'failed_videos': len(failed),  
        'total_frames_processed': total_frames,  
        'processing_time_seconds': processing_time,  
        'average_fps': total_frames / processing_time if processing_time > 0 else 0,  
        'results': results  
    }  
      
    logger.info(f"Batch processing completed in {processing_time:.2f}s")  
    logger.info(f"Success: {len(successful)}/{len(video_files)} videos")  
    logger.info(f"Total frames: {total_frames}, Average FPS: {stats['average_fps']:.2f}")  
      
    return stats  
  
if __name__ == "__main__":  
    # Example usage  
    input_directory = "train/raw_videos"  
    output_directory = "train/keypoints"  
      
    # Run optimized batch processing  
    results = batch_process_videos_optimized(  
        input_dir=input_directory,  
        output_dir=output_directory,  
        save_annotated=False,  # Set to True if you need annotated frames  
        use_gpu=False,         # Enable GPU acceleration  
        max_workers=1      # Use all available CPU cores  
    )  
      
    # Print summary  
    print("\n" + "="*50)  
    print("PROCESSING SUMMARY")  
    print("="*50)  
    print(f"Total videos: {results['total_videos']}")  
    print(f"Successful: {results['successful_videos']}")  
    print(f"Failed: {results['failed_videos']}")  
    print(f"Total frames: {results['total_frames_processed']}")  
    print(f"Processing time: {results['processing_time_seconds']:.2f}s")  
    print(f"Average FPS: {results['average_fps']:.2f}")