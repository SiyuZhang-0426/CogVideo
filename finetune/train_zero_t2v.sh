#!/bin/bash
#SBATCH -p $vp
#SBATCH --gres=gpu:4

srun -p $vp --gres=gpu:4 apptainer exec --nv ~/ubuntu.sif bash -c "
source ~/.bashrc && \
conda activate cogvideox && \
cd /mnt/petrelfs/zhangsiyu/4dgen/CogVideo/finetune && \
export TOKENIZERS_PARALLELISM=false && \
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX1.5-5B"
    --model_name "cogvideox1.5-t2v"  # ["cogvideox-t2v"]
    --model_type "t2v"
    --training_type "sft"
) && \
OUTPUT_ARGS=(
    --output_dir "/mnt/petrelfs/zhangsiyu/4dgen/CogVideo/finetune_test_zero"
    --report_to "tensorboard"
) && \
DATA_ARGS=(
    --data_root "/mnt/petrelfs/zhangsiyu/4dgen/CogVideo/Disney-VideoGeneration-Dataset"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
    --train_resolution "81x768x1360"  # (frames x height x width), frames should be 8N+1 and height, width should be multiples of 16
) && \
TRAIN_ARGS=(
    --train_epochs 10 # number of training epochs
    --seed 42 # random seed

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
) && \
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
) && \
CHECKPOINT_ARGS=(
    --checkpointing_steps 10 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
) && \
VALIDATION_ARGS=(
    --do_validation false  # ["true", "false"]
    --validation_dir "/mnt/petrelfs/zhangsiyu/4dgen/CogVideo/Disney-VideoGeneration-Dataset"
    --validation_steps 20  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --gen_fps 16
) && \
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
&& \
echo "Training completed"
"
