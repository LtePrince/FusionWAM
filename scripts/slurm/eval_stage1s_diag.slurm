#!/bin/bash
#SBATCH --job-name=fusionwam-s1s-diag
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

# stage-1s 消费判别：V-norm + 四臂诊断(A/A2/B/C) + 零剂量门禁。
# [1] 三臂诊断(训练样本,无仿真):预期 C(无 VLM)≈0.01;若 B(带 VLM)≫C,
#     前缀在推理 KV-cache 路径上注入畸形信息坐实(推理侧 bug,可免重训)。
# [2] 零剂量 6 集 --no-vlm:无前缀部署,若 ≈100% 从行为层面确认同一结论。

source /opt/miniforge/etc/profile.d/conda.sh
cd $HOME/FusionWAM
conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source ./.venv/bin/activate

CKPT=runs/stage1s/checkpoints/weights/step_020000.pt

echo "=== [0/3] V-norm ==="
.venv/bin/python - <<'PYEOF'
import glob, torch
for p in sorted(glob.glob("runs/stage1s/checkpoints/weights/step_*.pt")):
    sd = torch.load(p, map_location="cpu", weights_only=False)["vlm_adapter"]
    v = sum(float(t.float().norm()) for n, t in sd.items() if n.startswith("v_projs"))
    print(f"{p}  V-norm={v:.4f}")
PYEOF

echo "=== [1/3] 四臂诊断(stage-1s 档)==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/diagnose_rollout_gap.py --ckpt $CKPT --base-ckpt checkpoints/wam_release/libero_uncond_2cam224.pt

echo "=== [2/3] 零剂量门禁 6 集(VLM 开启)==="
mkdir -p results/gate_s1s
for k in 0 1 2 3 4 5; do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval/displacement_eval.py \
        --ckpt $CKPT --base-ckpt checkpoints/wam_release/libero_uncond_2cam224.pt \
        --tasks 0 --trial-list $k --doses 0 \
        --out results/gate_s1s/t0_trial$k.json \
        || echo "trial $k CRASHED, continuing"
done

echo "=== done ==="
