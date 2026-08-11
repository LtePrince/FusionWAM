#!/usr/bin/env bash
# LIBERO training data — public download (HF: yuanty/LIBERO-fastwam, ~4 tars).
# Produces the layout expected by configs/data/libero_2cam.yaml.
set -euo pipefail
DEST="${1:-./data/libero_mujoco3.3.2}"
mkdir -p "$DEST"

huggingface-cli download yuanty/LIBERO-fastwam \
  --repo-type dataset --local-dir "$DEST"

echo "== extracting"
for f in "$DEST"/*.tar.gz; do
  tar -xzf "$f" -C "$DEST" && rm -f "$f"
done
ls "$DEST"

cat <<'EOF'
== next: precompute the umt5 text-embedding cache (once per task):
  .venv/bin/python scripts/precompute_text_embeds.py task=libero_joint_2cam224_1e-4
(multi-GPU: torchrun --standalone --nproc_per_node=N scripts/precompute_text_embeds.py ...)
Expected output dir: ./data/text_embeds_cache/libero
EOF
