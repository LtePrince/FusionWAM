# FusionWAM

**Tri-expert fusion for wide-envelope manipulation: a semantic VLM (PaliGemma), a
video dynamics model (Wan2.2 DiT, WAM weights), and a flow-matching action
DiT that attends over both.**

This repository is **self-contained**: model code, the vendored video-action
training stack, hydra configuration, data/weight download scripts, and the
acceptance evaluation all live here. Weights and datasets are not committed;
they download from public sources via `scripts/`.

> **Status.** Structure-level tests pass locally (`scripts/smoke_test.py
> --no-weights`). A full 9B forward/backward has **not** been executed — the
> development machine (2×RTX 3090) cannot hold the stage-1 configuration.
> Run the smoke test, then a 1-batch training step, as the first action on the
> target machine (docs/MIGRATION.md §4). The vendored stack is pinned by
> inclusion, so there is no upstream-drift risk.

---

## 1. Motivation

Three findings from our 2026-08 displacement-envelope study motivate this
architecture:

1. **Envelope width tracks pretraining support, not architecture.** Under a
   seed-paired object-displacement protocol (n = 300/cell), π0.5 sustains
   73.3% success at 10 cm displacement where WAM reaches 28.7% and
   π0-base 35.3%; π0-base ≈ WAM shows the architecture family is not the
   differentiator — the diverse-pretrained VLM binding is.
2. **The video model's binding cannot be rewritten at finetune scale.** Three
   escalating interventions on WAM (stratified augmentation; augmentation
   plus a severed proprioception shortcut; doubled step budget) all plateau at
   37–47% @10 cm.
3. **The consumption mechanism is not the bottleneck.** A 4.5 M-parameter
   state-conditioned policy trained on enumerated support saturates the
   feasibility ceiling of the same protocol (93.3% @10 cm = 100% of ceiling).

Consequence: attach the *semantic binding* of a VLM and the *dynamics
representation* of a video model to one action expert, and close the residual
support gap with enumeration-density synthetic data at the joint-training
level — each element addressing the failure mode the campaign measured for it.

## 2. Architecture

```
PaliGemma VLM (3B, frozen in stage 1) ──semantic KV──┐
                                                     ├──> Action DiT (1B, flow matching)
Wan2.2 video DiT (5B, WAM weights) ──dynamics KV─┘     Q attends [VLM ⊕ video ⊕ action]
```

- **Stage 1 (this package's default).** Per-layer KV coupling (openpi's
  mechanism, generalized to heterogeneous towers): the frozen VLM runs its
  own forward over [image + instruction]; each MoT layer i attends a KV
  prefix computed from the depth-mapped Gemma layer g(i) by a trainable
  per-layer adapter (V zero-initialized: silent start). The action expert
  reads the VLM's whole depth — early layers carry displaced-object
  position dose-flat, late layers carry instruction binding (probe verdict
  2026-08-12) — so no single-layer tap has to be chosen. Video rows never
  attend the VLM prefix (the frozen video tower keeps its training
  distribution). Trainable: action expert + the KV adapter.
- **Stage 2 (large-machine option).** Open the video->VLM attention rows
  (semantics condition dynamics, replacing umt5) and optionally unfreeze
  the VLM with a web-scale vision-language co-training mix (§5.4).

### Design rules derived from campaign evidence (do not remove)

| Rule | Mechanism | Evidence |
|---|---|---|
| Source-level dropout, both KV sources (p = 0.3 each) | `p_drop_vlm`, `p_drop_video` | Available information is not consumed unless consumption is forced (Gate 1, B2, P3-B) |
| VLM frozen by default | stage-1 freeze policy | Wide-support binding collapses under narrow-distribution finetuning (P1 mechanism verdict) |
| Acceptance = dose protocol, never in-distribution success | `eval/README.md` | A 99.3% in-distribution model measured 28.7% at 10 cm |
| Weights-only checkpointing by default | `save_full_state: false` | Full trainer-state saves need a 2×13 G transient window (2026-08-09 incident) |

## 3. Repository layout

The directory structure mirrors the architecture — three experts, their
fusion, and the machinery around them:

```
src/fusionwam/
  models/                       # one file per component (openpi-style)
    fusion.py                   #   FusionWAM: tri-expert composition + tri-MoT masks
    paligemma.py                #   semantic expert (frozen VLM + trainable projection)
    video_dit.py                #   dynamics expert (Wan2.2 video DiT)
    action_dit.py               #   action expert (flow-matching DiT)
    wam.py / wam_joint.py       #   video⊕action bi-expert composition (MoT)
    wam_idm.py                  #   inverse-dynamics variant
    mot.py                      #   mixture-of-transformers joint attention
    wan22.py / vae.py / text_encoder.py / scheduler.py   # Wan2.2 plumbing
    loading/                    #   weight download / state-dict conversion
  training/
    trainer.py                  # base training loop (accelerate/deepspeed)
    fusion_trainer.py           # keeps the fusion adapter trainable under freeze mode
    runtime.py                  # model/dataset builders (hydra targets)
  data/                         # lerobot dataset stack (loaders, processors, transforms)
  shared/                       # logging, fs, samplers, video io, misc utilities
configs/
  train.yaml + data/ model/ task/   # hydra tree
  stage1.yaml                       # stage-1 fusion overrides
train.py                        # entry: compose configs/, swap model+trainer targets
scripts/                        # downloads, precompute, smoke test, multi-GPU launcher
eval/                           # dose-protocol acceptance eval + decision rule
docs/MIGRATION.md               # checkout-to-training runbook
```

## 4. Installation

FusionWAM is **self-contained**: the full training stack lives under
`src/fusionwam/`, its hydra configuration tree under `configs/`, and the
acceptance-evaluation scripts under `eval/`. All weights and datasets download
from public sources (Hugging Face / ModelScope). See `docs/MIGRATION.md` for
the complete checkout-to-training runbook:

```bash
git clone <this repo> && cd FusionWAM
uv venv && uv sync && uv pip install -e .
.venv/bin/huggingface-cli login          # PaliGemma is license-gated
bash scripts/download_weights.sh ./checkpoints
bash scripts/download_data.sh
.venv/bin/python scripts/smoke_test.py --no-weights
.venv/bin/python scripts/smoke_test.py
```

## 5. Before training

### 5.1 Gated weights
`google/paligemma-3b-pt-224` is license-gated on Hugging Face. Accept the
license with a logged-in account (`huggingface-cli login`) before running the
download script; mirrors do not serve gated repositories.

### 5.2 Data
`bash scripts/download_data.sh` fetches the public LIBERO lerobot dataset;
then run `scripts/precompute_text_embeds.py` once (docs/MIGRATION.md §3).
The enumeration-rendered support set (~35 G) can be added later per
`scripts/prepare_data.md` §2. Observe the convention checklist in that file.

### 5.3 Optional: π0.5-finetuned VLM
Run a frozen-probe experiment first (train a position probe on PaliGemma
features over displaced scenes, evaluate out-of-support). Only if base
PaliGemma features fail to carry out-of-support object position is the JAX→HF
conversion (`scripts/convert_pi05_vlm.md`, est. 1–2 days) worth doing.

### 5.4 Stage 2 prerequisites
Unfreezing the VLM requires a robot:web co-training mix (starting ratio 1:1)
and envelope monitoring during training (dose cells 4/10 cm every N steps);
loss curves alone do not detect binding collapse.

## 6. Training

```bash
# smoke (single GPU):
.venv/bin/python train.py task=libero_joint_2cam224_1e-4 \
    output_dir=/tmp/fusion_smoke max_steps=2 save_every=1 batch_size=1
# full run (multi-GPU):
bash scripts/train_zero1.sh 2 task=libero_joint_2cam224_1e-4 output_dir=./runs/stage1
```

`train.py` composes the hydra tree (`configs/`), swaps the
model target to `FusionWAM` and the trainer to `FusionTrainer`, and applies
`configs/stage1.yaml` on top. All vendored-stack overrides (task, data,
optimizer) keep their upstream meaning.

Hardware budget, stage 1: 3B frozen (bf16 ≈ 6 G) + 5B frozen (≈ 10 G) + 1B
training with AdamW (≈ 12 G) + activations. Recommended ≥ 2×A100-80G (or
4×A6000) with gradient checkpointing; batch size 1 fits a single 80 G device
for smoke purposes.

## 7. Acceptance

Every trained artifact is evaluated on the displacement dose protocol
(`eval/README.md`): tasks 0–4 × 6 trials × {0, 4, 7, 10, 15} cm with the
legacy seed family, zero-dose control, and feasibility-adjusted far cells.
Pre-registered decision rule for stage 1 at the 10 cm cell:

| Result | Reading | Action |
|---|---|---|
| ≥ 60% | VLM source genuinely consumed | proceed to stage 2 |
| 45–60% | partial consumption | attention-entropy diagnosis before scaling |
| ≤ 45% | fusion ineffective (best 6B-LoRA baseline: 46.7%) | verify source dropout is active and the adapter V-projection norms have moved off zero |

Reference anchors on the same protocol: WAM base 20% @10 cm; best
6B-LoRA variant 46.7%; π0.5 73.3%; enumerated-core upper bound (ground-truth
anchors) 93.3% = 100% of the feasibility ceiling.

## 8. Known limitations

- No end-to-end 9B execution has occurred; the two integration surfaces to
  validate first are the hydra composition in `train.py` and the
  `training_loss` override path in `models/fusion.py`.
- Attention-scale balance between the two KV sources is handled only by the
  the adapter V-projection norms: they start at exactly zero and must move
  off it within the first ~2k steps for the prefix to contribute.
- Cross-segment relative position is undefined by construction (per-source
  positional encodings); action-to-source attention is content-addressed.
  This is a deliberate choice, not an oversight.
