#!/bin/bash
#SBATCH --job-name=fusionwam-stage1c
#SBATCH --partition=debug        # 训练约 1.5 天：若 debug 分区有时限，换常规分区
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G                # 可能与另一训练作业同节点并行,给足但不顶满
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

# stage-1c：数据压力实验——架构与 stage-1b 完全一致(release 底座热启动),
# 唯一变量 = 训练集混入枚举渲染集(data=libero_2cam_enum,含支撑外位移样本)。
# 检验 Information Collapse 假设:训练分布含支撑外样本后,VLM 前缀是否被消费
# (完后同一套 with/without-VLM 剂量消融判定)。
# 前置:gen_enum_data.slurm 已完成且软链就位。

source /opt/miniforge/etc/profile.d/conda.sh
cd $HOME/FusionWAM
conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source ./.venv/bin/activate

test -e ./data/libero_mujoco3.3.2/libero_object_enum_lerobot || { echo "enum 数据未就位"; exit 1; }

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_zero1.sh 1 \
    task=libero_joint_2cam224_1e-4 data=libero_2cam_enum \
    output_dir=./runs/stage1c \
    batch_size=4 gradient_accumulation_steps=8 num_workers=2 \
    resume=checkpoints/wam_release/libero_uncond_2cam224.pt \
    +data.train.pretrained_norm_stats=checkpoints/wam_release/libero_uncond_2cam224_dataset_stats.json

echo "=== done ==="
