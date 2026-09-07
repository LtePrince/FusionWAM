#!/bin/bash
#SBATCH --job-name=fusionwam-enumgen
#SBATCH --partition=debug        # 约 10 小时：若 debug 分区有时限，换常规分区
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=fusionwam_%j.log       # 输出日志（%j 是作业号）

# stage-1c 枚举渲染集生成：tasks 0-4 × 2cm 网格(r<=0.20,~315 点/任务),
# 脚本轨迹 + 双摄 512² 渲染 + lerobot v2.1 直写;逐集 BDDL 过滤;
# 断点续跑(meta/progress.json)。产出落 /data 大盘,结束后软链进 ./data。

source /opt/miniforge/etc/profile.d/conda.sh
cd $HOME/FusionWAM
conda activate Ziyao
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source ./.venv/bin/activate

# mp4 写入依赖(首次自愈安装)
.venv/bin/python -c "import imageio_ffmpeg, pyarrow" 2>/dev/null || .venv/bin/pip install imageio-ffmpeg pyarrow

OUT=/data/users/$USER/libero_object_enum_lerobot
mkdir -p "$(dirname "$OUT")"

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/gen_enum_dataset.py \
    --tasks 0-4 --radius 0.20 --res 512 \
    --features-from ./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot \
    --out "$OUT"

# 软链进训练目录(libero_2cam_enum.yaml 引用该路径)
ln -sfn "$OUT" ./data/libero_mujoco3.3.2/libero_object_enum_lerobot
ls -l ./data/libero_mujoco3.3.2/ | tail -3
du -sh "$OUT"

echo "=== done ==="
