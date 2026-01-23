export MODEL_NAME="/mnt/bn/douyin-ai4se-general-wl/lht/ckpt/Wan2.1-Fun-V1.1-1.3B-Control"
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/bn/douyin-ai4se-general-wl/lht/data/how2sign/train/videx_fun/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero3_save_16bit_model true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --mixed_precision="bf16" scripts/wan2.1_fun/train_control.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=5 \
  --checkpointing_steps=350 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "." \
  --enable_hl_context \
  --hl_num_classes=26 \
  --hl_num_joints=20 \
  --hl_embed_dim=256 \
  --hl_dir_dim=3 \
  --hl_dropout_prob=0.1 \
  --hl_frame_dropout_prob=0.1 \
  --hl_frame_stride=1 \
  --hl_joint_stride=1 \
  --hl_max_tokens=512 \
  --hl_file_key="hl_file_path" \
  --hl_ids_key="hl_ids" \
  --hl_dirs_key="hl_dirs" \
  --hl_latents_key="hl_latents_path" 