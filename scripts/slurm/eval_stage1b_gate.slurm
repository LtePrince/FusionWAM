#!/bin/bash
#SBATCH --job-name=fusionwam-s1b-gate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

# stage-1b 零剂量门禁：V-norm 终检 + 6 集零剂量（每集独立进程）。
# 预期：release 底座热启动后应逼近 release 自身的 ~100%。

source /opt/miniforge/etc/profile.d/conda.sh
cd $HOME/FusionWAM
conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source ./.venv/bin/activate

CKPT=runs/stage1b/checkpoints/weights/step_020000.pt

echo "=== [1/2] V-norm 终检 ==="
.venv/bin/python - <<'PY'
import glob, torch
for p in sorted(glob.glob("runs/stage1b/checkpoints/weights/step_*.pt")):
    sd = torch.load(p, map_location="cpu", weights_only=False)["vlm_adapter"]
    v = sum(float(t.float().norm()) for n, t in sd.items() if n.startswith("v_projs"))
    k = sum(float(t.float().norm()) for n, t in sd.items() if n.startswith("k_projs"))
    print(f"{p}  V-norm={v:.4f}  K-norm={k:.1f}")
PY

echo "=== [2/2] 零剂量门禁 6 集（每集独立进程）==="
mkdir -p results/gate_s1b
for k in 0 1 2 3 4 5; do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval/displacement_eval.py \
        --ckpt $CKPT --tasks 0 --trial-list $k --doses 0 --dose-index-base 0 \
        --out results/gate_s1b/t0_trial$k.json \
        || echo "trial $k CRASHED (native abort), continuing"
done

echo "=== done ==="
