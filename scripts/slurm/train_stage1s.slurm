#!/bin/bash
#SBATCH --job-name=fusionwam-stage1s
#SBATCH --partition=debug        # 训练约 1.5 天：若 debug 分区有时限，换成常规分区
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

# stage-1s：视频塔 = release 权重（冻结），动作塔/proprio 从头训（scratch）。
# 假设检验：release 动作塔已把分内任务解干净，热启动使 adapter 失去被消费的
# 激励；从头训的动作塔在前缀 70% 在场的条件下是否会把它接进依赖。
# 前置：先生成 video-only 底座——
#   .venv/bin/python scripts/make_video_only_base.py

source /opt/miniforge/etc/profile.d/conda.sh
cd $HOME/FusionWAM
conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source ./.venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_zero1.sh 1 \
    task=libero_joint_2cam224_1e-4 output_dir=./runs/stage1s \
    batch_size=4 gradient_accumulation_steps=8 num_workers=2 \
    resume=checkpoints/wam_release/video_only_base.pt \
    +data.train.pretrained_norm_stats=checkpoints/wam_release/libero_uncond_2cam224_dataset_stats.json

echo "=== done ==="
