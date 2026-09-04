#!/bin/bash
#SBATCH --job-name=fusionwam-s1b-diag2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

# stage-1b 前缀毒药判别：
# [1] 三臂诊断(训练样本,无仿真):预期 C(无 VLM)≈0.01;若 B(带 VLM)≫C,
#     前缀在推理 KV-cache 路径上注入畸形信息坐实(推理侧 bug,可免重训)。
# [2] 零剂量 6 集 --no-vlm:无前缀部署,若 ≈100% 从行为层面确认同一结论。

source /opt/miniforge/etc/profile.d/conda.sh
cd $HOME/FusionWAM
conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source ./.venv/bin/activate

CKPT=runs/stage1b/checkpoints/weights/step_020000.pt

echo "=== [1/2] 三臂诊断(stage-1b 档)==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/diagnose_rollout_gap.py --ckpt $CKPT

echo "=== [2/2] 零剂量 6 集,VLM 前缀关闭(--no-vlm)==="
mkdir -p results/gate_s1b_novlm
for k in 0 1 2 3 4 5; do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval/displacement_eval.py \
        --ckpt $CKPT --no-vlm --tasks 0 --trial-list $k --doses 0 \
        --out results/gate_s1b_novlm/t0_trial$k.json \
        || echo "trial $k CRASHED, continuing"
done

echo "=== done ==="
