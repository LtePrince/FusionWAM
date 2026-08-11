# Setup Runbook — from `git clone` to first training run

FusionWAM is self-contained: the wam training stack is vendored under
`src/fusionwam/wam/` (MIT, notice preserved there), its hydra tree under
`configs/wam/`, and all weights and data download from public sources.
No other repository is required.

Assumptions: Linux, CUDA ≥ 12.1, ≥2×A100-80G (1×80G suffices for smoke tests),
~150 G free disk, [uv](https://docs.astral.sh/uv/) installed.

## 1. Clone and environment

```bash
git clone <your-github-url>/FusionWAM && cd FusionWAM
uv venv && uv sync
uv pip install -e .
```

If your cluster requires the exact upstream CUDA build:
`uv pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128`

## 2. Weights (public; ≈35 G total)

```bash
# PaliGemma is license-gated: accept the license for
# google/paligemma-3b-pt-224 on huggingface.co, then:
.venv/bin/huggingface-cli login
bash scripts/download_weights.sh ./checkpoints
```

The script fetches PaliGemma and the WAM release checkpoint; the Wan2.2
base (`Wan-AI/Wan2.2-TI2V-5B` + `Wan-AI/Wan2.1-T2V-1.3B` tokenizer)
auto-downloads into `./checkpoints/` on first model construction
(`DIFFSYNTH_DOWNLOAD_SOURCE=huggingface|modelscope` selects the source).
Then derive the ActionDiT initialization:

```bash
.venv/bin/python scripts/preprocess_action_dit_backbone.py \
  # (see --help; output path expected by configs/wam/model/wam_joint.yaml:)
  # ./checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

## 3. Training data (public; ≈40 G)

```bash
bash scripts/download_data.sh                 # HF: yuanty/LIBERO-fastwam
.venv/bin/python scripts/precompute_text_embeds.py task=libero_joint_2cam224_1e-4
```

First-ever run note (upstream convention): with no
`pretrained_norm_stats`, a `dataset_stats.json` is generated in the run
directory on the first training run; point
`configs/wam/data/libero_2cam.yaml: pretrained_norm_stats` at it for
subsequent runs — or use the released
`checkpoints/wam_release/libero_uncond_2cam224_dataset_stats.json`.

## 4. Tests — run in order, do not skip

```bash
# 4a. Structure only (seconds):
.venv/bin/python scripts/smoke_test.py --no-weights

# 4b. VLM adapter forward (loads PaliGemma, ~1 min):
.venv/bin/python scripts/smoke_test.py

# 4c. Two-step training sanity run (single GPU, ~10 min; validates hydra
#     composition, dataset paths, the training_loss override, checkpointing):
.venv/bin/python train.py \
    task=libero_joint_2cam224_1e-4 \
    output_dir=/tmp/fusion_smoke max_steps=2 save_every=1 batch_size=1
# PASS = 2 steps with finite loss + a weights checkpoint in /tmp/fusion_smoke.
```

## 5. Stage-1 training

```bash
bash scripts/train_zero1.sh 2 \
    task=libero_joint_2cam224_1e-4 output_dir=./runs/stage1
```

`configs/stage1.yaml` already sets `max_steps=20000`, weights-only saves,
source dropout p = 0.3/0.3, and LayerScale init 1e-3. Besides the loss, watch
the VLM-branch LayerScale norm: it should move off 1e-3 within ~2k steps; if
it does not, the semantic source is not being consumed — stop and check the
dropout configuration before spending the full budget.

## 6. Acceptance evaluation

The displacement dose protocol requires a LIBERO simulator environment
(separate stack: LIBERO + robosuite + MuJoCo/EGL). See `eval/README.md` for
the protocol contract, reference anchors, and the pre-registered decision
rule (10 cm cell: ≥60% → stage 2; ≤45% → diagnose fusion before scaling).

## 7. Disk discipline

Weights-only checkpointing is pre-configured (`save_full_state: false`).
Before enabling full trainer-state saves, verify >30 G headroom on the
checkpoint volume: state rotation needs a 2×13 G transient window.
