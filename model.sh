#!/bin/bash
#SBATCH -p $vp
#SBATCH --gres=gpu:0

srun -p $vp --gres=gpu:0 apptainer exec --nv ~/ubuntu.sif bash -c "
source ~/.bashrc && \
conda activate cogvideox && \
cd /mnt/petrelfs/zhangsiyu/4dgen/CogVideo && \
export http_proxy=http://zhangsiyu:3WqUQ2knmuHFeBXYcRKJ0rbA6EB3Jqg7Rx8bV1cjn2dtNSPZDBe9WIWbDtsn@10.1.20.50:23128/ && \
export https_proxy=http://zhangsiyu:3WqUQ2knmuHFeBXYcRKJ0rbA6EB3Jqg7Rx8bV1cjn2dtNSPZDBe9WIWbDtsn@10.1.20.50:23128/ && \
export HTTP_PROXY=http://zhangsiyu:3WqUQ2knmuHFeBXYcRKJ0rbA6EB3Jqg7Rx8bV1cjn2dtNSPZDBe9WIWbDtsn@10.1.20.50:23128/ && \
export HTTPS_PROXY=http://zhangsiyu:3WqUQ2knmuHFeBXYcRKJ0rbA6EB3Jqg7Rx8bV1cjn2dtNSPZDBe9WIWbDtsn@10.1.20.50:23128/ && \
export no_proxy=10.0.0.0/8,100.0.0.0/8,35.220.264.252/32,.pjlab.org.cn && \
python hf_download.py \
    --repo-id 'zai-org/CogVideoX1.5-5B' \
    --local-dir '/mnt/petrelfs/zhangsiyu/4dgen/CogVideo/CogVideoX1.5-5B' \
    --token '\$hf_token' \
    --include '*.safetensors' \
    --include '*.json' \
    --include '*.yaml' \
    --include '*.txt' \
    --include '*.md' \
    --include '*.log' \
"
