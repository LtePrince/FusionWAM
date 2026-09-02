#!/bin/bash
#SBATCH --job-name=fusionwam-diag
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

source /opt/miniforge/etc/profile.d/conda.sh

cd $HOME/FusionWAM

conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source ./.venv/bin/activate

echo "=== [1/2] rollout-gap 诊断（训练样本三臂对照，loss 多次抽样平均）==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/diagnose_rollout_gap.py \
    --ckpt runs/stage1/checkpoints/weights/step_020000.pt

echo "=== [2/2] 零剂量探针（6 集，每集独立进程，防原生崩溃连坐）==="
mkdir -p results
for k in 0 1 2 3 4 5; do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval/displacement_eval.py \
        --ckpt runs/stage1/checkpoints/weights/step_020000.pt \
        --tasks 0 --trial-list $k --doses 0 \
        --out results/probe_t0_trial$k.json \
        || echo "trial $k CRASHED (native abort), continuing"
done

echo "=== done ==="
