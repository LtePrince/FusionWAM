#!/bin/bash
#SBATCH --job-name=fusionwam-stage1b
#SBATCH --partition=debug        # 训练约 1.5 天：若 debug 分区有时限，换成常规分区
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

# stage-1b：把 stage-1 重建在 release 底座上（2026-09-03 判决后的方案 B）。
# - 视频塔 = LIBERO 微调过的 release 权重（冻结）——部署时联合去噪滚在数据流形上；
# - 动作塔从 release 初始化后微调；adapter 零初始化静默起步（release 档无 adapter 键，
#   FusionWAM.load_checkpoint 自动跳过）；
# - 归一化统计强制用 release 版，与热启动权重的输入约定一致（数据集会把它
#   复制进 run 目录，评测端自动解析到同一份）。
# 提交前确认 checkpoints/wam_release/ 下有 .pt（13G）和 stats json。

source /opt/miniforge/etc/profile.d/conda.sh

cd $HOME/FusionWAM

conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source ./.venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_zero1.sh 1 \
    task=libero_joint_2cam224_1e-4 output_dir=./runs/stage1b \
    batch_size=4 gradient_accumulation_steps=8 num_workers=2 \
    resume=checkpoints/wam_release/libero_uncond_2cam224.pt \
    +data.train.pretrained_norm_stats=checkpoints/wam_release/libero_uncond_2cam224_dataset_stats.json

echo "=== done ==="
