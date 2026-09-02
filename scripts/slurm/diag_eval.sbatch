#!/bin/bash
#SBATCH --job-name=fusionwam-diag
#SBATCH --output=slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# 如实验室要求指定分区/QoS，按你平时提交作业的写法自行加一行，例如：
# #SBATCH --partition=xxx

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/FusionWAM}"

echo "=== [1/2] rollout-gap 诊断（训练样本上三臂对照，约 10 分钟）==="
.venv/bin/python scripts/diagnose_rollout_gap.py \
    --ckpt runs/stage1/checkpoints/weights/step_020000.pt

echo "=== [2/2] 零剂量探针（6 集，约 20 分钟）==="
mkdir -p results
MUJOCO_GL=egl .venv/bin/python eval/displacement_eval.py \
    --ckpt runs/stage1/checkpoints/weights/step_020000.pt \
    --tasks 0 --trials 6 --doses 0 --out results/probe_t0b.json

echo "=== done ==="
