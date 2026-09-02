#!/bin/bash
#SBATCH --job-name=fusionwam-control
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

# 对照组：已知良好的 FastWAM 发布检查点过同一条评测管线。
# 提交前先下载对照权重（约 13G，注意 df -h ~ 余量）：
#   .venv/bin/huggingface-cli download yuanty/fastwam \
#       libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json \
#       --local-dir checkpoints/wam_release

source /opt/miniforge/etc/profile.d/conda.sh

cd $HOME/FusionWAM

conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source ./.venv/bin/activate

echo "=== 对照组：release 检查点 零剂量 6 集（每集独立进程）==="
mkdir -p results
for k in 0 1 2 3 4 5; do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval/displacement_eval.py \
        --ckpt checkpoints/wam_release/libero_uncond_2cam224.pt --plain-wam \
        --tasks 0 --trial-list $k --doses 0 \
        --out results/control_t0_trial$k.json \
        || echo "trial $k CRASHED (native abort), continuing"
done

echo "=== done ==="
