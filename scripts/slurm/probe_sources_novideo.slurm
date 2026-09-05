#!/bin/bash
#SBATCH --job-name=fusionwam-s1b-novideo
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

# 源贡献 2×2 的 no-video 侧探针（18 集，约 2 小时）：
#   [1] no-video(VLM+T5+proprio):0cm×6 + 10cm×6
#       —— 若能成任务 = 动作塔确实在读 VLM 前缀（语义消费的正面存在性证据）
#   [2] no-video + no-vlm(仅 T5+proprio,盲策略地板):0cm×6

source /opt/miniforge/etc/profile.d/conda.sh
cd $HOME/FusionWAM
conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source ./.venv/bin/activate

CKPT=runs/stage1b/checkpoints/weights/step_020000.pt
BASE="--base-ckpt checkpoints/wam_release/libero_uncond_2cam224.pt"
mkdir -p results/probe_novideo

echo "=== [1a] no-video @ 0cm ==="
for k in 0 1 2 3 4 5; do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval/displacement_eval.py \
        --ckpt $CKPT $BASE --no-video --tasks 0 --trial-list $k \
        --doses 0 --dose-index-base 0 \
        --out results/probe_novideo/d0_trial$k.json \
        || echo "trial $k CRASHED, continuing"
done

echo "=== [1b] no-video @ 10cm ==="
for k in 0 1 2 3 4 5; do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval/displacement_eval.py \
        --ckpt $CKPT $BASE --no-video --tasks 0 --trial-list $k \
        --doses 0.10 --dose-index-base 3 \
        --out results/probe_novideo/d10_trial$k.json \
        || echo "trial $k CRASHED, continuing"
done

echo "=== [2] 盲策略地板:no-video + no-vlm @ 0cm ==="
for k in 0 1 2 3 4 5; do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval/displacement_eval.py \
        --ckpt $CKPT $BASE --no-video --no-vlm --tasks 0 --trial-list $k \
        --doses 0 --dose-index-base 0 \
        --out results/probe_novideo/blind_trial$k.json \
        || echo "trial $k CRASHED, continuing"
done

echo "=== done ==="
