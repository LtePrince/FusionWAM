#!/bin/bash
#SBATCH --job-name=fusionwam-s1b-full
#SBATCH --partition=debug        # 全程 8-12 小时：若 debug 分区有时限，换常规分区
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

# stage-1b 全剂量协议：5 任务 × 6 trial × 5 剂量 = 150 集，每集独立进程。
# --dose-index-base 保持 seed 族与全部历史数字配对（seed=7000+trial*10+d_i）。
# 门禁（eval_stage1b_gate.slurm）通过后再提交本作业。

source /opt/miniforge/etc/profile.d/conda.sh
cd $HOME/FusionWAM
conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source ./.venv/bin/activate

CKPT=runs/stage1b/checkpoints/weights/step_020000.pt
OUT=results/dose_s1b
mkdir -p $OUT

DOSES=(0 0.04 0.07 0.10 0.15)
for t in 0 1 2 3 4; do
  for k in 0 1 2 3 4 5; do
    for di in 0 1 2 3 4; do
      d=${DOSES[$di]}
      f=$OUT/t${t}_trial${k}_d${di}.json
      [ -s "$f" ] && { echo "skip existing $f"; continue; }   # 断点续跑
      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval/displacement_eval.py \
          --ckpt $CKPT --base-ckpt checkpoints/wam_release/libero_uncond_2cam224.pt \
          --tasks $t --trial-list $k \
          --doses $d --dose-index-base $di \
          --out $f \
          || echo "t${t} trial${k} d${di} CRASHED, continuing"
    done
  done
done

echo "=== 汇总 ==="
.venv/bin/python - <<'PY'
import collections, glob, json
rows = []
for f in glob.glob("results/dose_s1b/*.json"):
    rows += json.load(open(f))
per_dose = collections.defaultdict(list)
per_task = collections.defaultdict(list)
for r in rows:
    per_dose[r["dose"]].append(r["success"])
    per_task[(r["task"], r["dose"])].append(r["success"])
print(f"episodes: {len(rows)}/150")
for d in sorted(per_dose):
    v = per_dose[d]
    print(f"dose {int(d*100):>2d}cm: {100*sum(v)/len(v):5.1f}%  (n={len(v)})")
print("\nper-task @ each dose:")
for (t, d) in sorted(per_task):
    v = per_task[(t, d)]
    print(f"  t{t} {int(d*100):>2d}cm: {sum(v)}/{len(v)}")
PY

echo "=== done ==="
