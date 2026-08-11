#!/usr/bin/env bash
# FusionWAM weight downloads — run on the TARGET machine (nothing is vendored).
# Total ≈ 22G. Set HF_ENDPOINT=https://hf-mirror.com if huggingface is slow.
set -euo pipefail
DEST="${1:-./checkpoints}"
mkdir -p "$DEST"

echo "== 1/3 PaliGemma-3B (semantic expert, frozen in stage 1)"
huggingface-cli download google/paligemma-3b-pt-224 \
  --local-dir "$DEST/paligemma-3b-pt-224"

echo "== 2/3 FastWAM release (video 5B + action 1B, LIBERO 2cam224)"
# Same artifact the local eval uses; adjust source if you host it elsewhere.
modelscope download Wan-AI/FastWAM-libero \
  --local_dir "$DEST/fastwam_release" 2>/dev/null || cat <<'EOF'
[!] FastWAM release not on modelscope under this name — copy
    checkpoints/fastwam_release/libero_uncond_2cam224.pt from the lab
    machine (WAM/FastWAM-LoRA/checkpoints/fastwam_release/), ~12G.
EOF

echo "== 3/3 umt5-xxl text encoder (video expert conditioning, stage 1 keeps it)"
huggingface-cli download google/umt5-xxl --local-dir "$DEST/umt5-xxl" \
  --include "*.safetensors" "*.json" "spiece.model"

echo "Optional: pi0.5-finetuned PaliGemma — see scripts/convert_pi05_vlm.md"
