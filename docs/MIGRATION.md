# Migration Runbook — from checkout to first training run

Assumptions: Linux, CUDA ≥ 12.1, ≥2×A100-80G (1×80G suffices for smoke),
~150 G free disk, [uv](https://docs.astral.sh/uv/) installed, lab machine
reachable as `LAB` for artifact transfer.

All commands run from the machine's work root unless stated otherwise.

## 1. Code

FusionWAM requires FastWAM as a sibling directory (path dependency + hydra
config tree). The lab's FastWAM contains local modifications — transfer it,
do not clone upstream.

```bash
mkdir -p WAM && cd WAM
rsync -a --info=progress2 \
  --exclude '.venv' --exclude 'data' --exclude 'checkpoints' \
  --exclude 'evaluate_results' --exclude 'runs' \
  LAB:Alvin/WAM/FastWAM ./
rsync -a LAB:Alvin/WAM/FusionWAM --exclude '.venv' ./
cd FusionWAM
```

## 2. Environment

```bash
uv venv
uv sync                          # torch (CUDA), transformers, accelerate, ...
uv pip install -e ../FastWAM     # fastwam package + diffsynth deps
uv pip install -e .              # fusionwam importable without sys.path tricks
```

## 3. Weights (≈35 G)

```bash
# 3a. PaliGemma is LICENSE-GATED: accept the license for
#     google/paligemma-3b-pt-224 on huggingface.co with your account, then:
.venv/bin/huggingface-cli login
bash scripts/download_weights.sh ./checkpoints

# 3b. FastWAM artifacts (Wan2.2 base + pretrained ActionDiT), from the lab:
rsync -a --info=progress2 LAB:Alvin/WAM/FastWAM/checkpoints/ ../FastWAM/checkpoints/
# If the Wan2.2 base is absent on the lab machine, diffsynth can fetch it:
#   export DIFFSYNTH_MODEL_BASE_PATH=../FastWAM/checkpoints
#   export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
```

## 4. Training data (≈40 G)

```bash
# 4a. LIBERO lerobot quadruples + precomputed umt5 text embeddings.
#     Transfer (fast, conventions guaranteed identical to the campaign):
mkdir -p ../FastWAM/data
rsync -a --info=progress2 LAB:Alvin/WAM/FastWAM/data/ ../FastWAM/data/
#     The data config expects, relative to the FastWAM root:
#       data/<lerobot dirs listed in configs/data/libero_2cam.yaml: dataset_dirs>
#       data/text_embeds_cache/libero        (umt5 embeddings)
#     Rebuild path (only if not transferring): convert LIBERO HDF5 with the
#     FastWAM lerobot pipeline, then:
#       (cd ../FastWAM && .venv-or-this-venv python scripts/precompute_text_embeds.py ...)

# 4b. Enumeration-rendered set (support carrier, ~35 G) — can be added after
#     the first smoke training; see scripts/prepare_data.md §2.
```

## 5. Tests — run in this order, do not skip

```bash
# 5a. Structure only (no weights, seconds):
.venv/bin/python scripts/smoke_test.py --no-weights

# 5b. VLM adapter forward (loads PaliGemma, ~1 min):
.venv/bin/python scripts/smoke_test.py

# 5c. Two-step training sanity run (single GPU, batch 1, ~10 min):
#     validates hydra composition, dataset paths, the training_loss override,
#     and checkpoint writing end to end.
cd ../FastWAM   # hydra resolves data/checkpoint paths relative to FastWAM root
../FusionWAM/.venv/bin/python ../FusionWAM/train.py \
    --fastwam-root . \
    --fusion-config ../FusionWAM/configs/stage1.yaml \
    task=libero_joint_2cam224_1e-4 \
    output_dir=/tmp/fusion_smoke max_steps=2 save_every=1 batch_size=1
# PASS = 2 steps logged with finite loss + a weights checkpoint in
# /tmp/fusion_smoke. On failure, the two usual suspects are §8 of README.md.
```

## 6. Stage-1 training (the real run)

```bash
cd ../FastWAM
accelerate launch --num_processes 2 --mixed_precision bf16 \
  ../FusionWAM/train.py \
    --fastwam-root . \
    --fusion-config ../FusionWAM/configs/stage1.yaml \
    task=libero_joint_2cam224_1e-4 \
    output_dir=../FusionWAM/runs/stage1
# Notes:
# - configs/stage1.yaml already sets max_steps=20000, weights-only saves,
#   source dropout p=0.3/0.3, LayerScale init 1e-3.
# - Watch, besides loss: the VLM-branch LayerScale norm (should move off
#   1e-3 within ~2k steps). If it stays put, the semantic source is not
#   being consumed — check dropout config before burning the full budget.
```

## 7. Acceptance evaluation (sim stack, separate environment)

The dose protocol runs in the LIBERO simulator. Transfer the eval harness and
its patched sim stack; it maintains its own venv:

```bash
cd .. && rsync -a --info=progress2 \
  --exclude 'runs' --exclude 'data_enum' --exclude '.venv' \
  LAB:Alvin/WAM/FastWAM-LoRA LAB:Alvin/WAM/LIBERO ./
# Recreate FastWAM-LoRA's venv per its README (includes the LIBERO
# torch.load patch); then, with a trained checkpoint merged for inference:
cd FastWAM-LoRA
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python \
  experiments/cross_embodiment/eval_relcond.py \
    --ckpt <merged-or-weights checkpoint> \
    --tasks 0-4 --trials 6 --doses 0,0.04,0.07,0.10,0.15 \
    --out evaluate_results/fusion_stage1_dose.json
# Decision rule: FusionWAM/eval/README.md (10 cm cell: ≥60% → stage 2;
# ≤45% → fusion ineffective, diagnose before scaling).
```

## 8. Disk discipline

Weights-only checkpointing is pre-configured (`save_full_state: false`).
Before enabling full trainer-state saves anywhere, verify >30 G headroom on
the checkpoint volume: state rotation needs a 2×13 G transient window
(2026-08-09 incident, see lab experiment log).
