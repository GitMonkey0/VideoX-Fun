BASE = "/mnt/bn/douyin-ai4se-general-wl/lht/workspace/VideoX-Fun/exp/outputs/" 

VERSION="baseline" 

python exp/data/extract_landmark_keypoints.py --input_dir $BASE$VERSION/interhand_fun_control_top100 --output_dir $BASE$VERSION/interhand_fun_control_top100_keypoints
python metrics.py --pred_root $BASE$VERSION/interhand_fun_control_top100_keypoints --gt_root /mnt/bn/douyin-ai4se-general-wl/lht/data/interhand/landmark_keypoints
