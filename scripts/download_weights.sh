#!/usr/bin/env bash
# FusionWAM weight downloads — everything from public sources.
# Layout matches configs/ expectations (paths relative to repo root).
# Slow networks: export HF_ENDPOINT=https://hf-mirror.com (non-gated repos only).
set -euo pipefail
DEST="${1:-./checkpoints}"
mkdir -p "$DEST"

echo "== 1/4 PaliGemma-3B (semantic expert; LICENSE-GATED)"
echo "   Accept the license for google/paligemma-3b-pt-224 on huggingface.co,"
echo "   then 'huggingface-cli login' before running this script."
huggingface-cli download google/paligemma-3b-pt-224 \
  --local-dir "$DEST/paligemma-3b-pt-224"

echo "== 2/4 WAM release checkpoint + dataset stats (HF: yuanty/fastwam)"
huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  --local-dir "$DEST/wam_release"

echo "== 3/4 Wan2.2 base + tokenizer"
echo "   Auto-fetched into ./checkpoints/ on first model construction"
echo "   (src/fusionwam/models/loading/io.py; modelscope by default,"
echo "    or: export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface)."
echo "   To pre-fetch explicitly instead:"
echo "     huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir $DEST/Wan-AI/Wan2.2-TI2V-5B"
echo "     huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir $DEST/Wan-AI/Wan2.1-T2V-1.3B"

echo "== 4/4 ActionDiT initialization (derived from the Wan backbone)"
echo "   After the Wan base is present, run:"
echo "     python scripts/preprocess_action_dit_backbone.py --help"
echo "   and write the output to:"
echo "     $DEST/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
echo "   (path expected by configs/model/wam_joint.yaml)"
