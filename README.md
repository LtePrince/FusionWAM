# FusionWAM

**Tri-expert fusion for wide-envelope manipulation: a semantic VLM (PaliGemma), a
video dynamics model (Wan2.2 DiT, FastWAM weights), and a flow-matching action
DiT that attends over both.**

This repository is a *migration package*: it contains all code, configuration,
and preparation scripts required to train FusionWAM on a multi-GPU machine.
Model weights and datasets are intentionally not vendored; they are fetched or
built by the scripts in `scripts/`. The package follows the lab convention:
code and configs in git, artifacts by download script.

> **Status.** Structure-level tests pass locally (`scripts/smoke_test.py
> --no-weights`). A full 9B forward/backward has **not** been executed — the
> development machine (2×RTX 3090) cannot hold the stage-1 configuration.
> Run the smoke test, then a 1-batch training step, as the first action on the
> target machine. Interfaces were written against FastWAM at commit state of
> 2026-08-11; if upstream drifts, `train.py` and `src/fusionwam/trainer.py`
> are the only two integration surfaces to re-check.

---

## 1. Motivation

Three findings from the 2026-08 envelope campaign (experiment logs:
`WAM/note/experiment_log/`, summarized in the paper draft) motivate this
architecture:

1. **Envelope width tracks pretraining support, not architecture.** Under a
   seed-paired object-displacement protocol (n = 300/cell), π0.5 sustains
   73.3% success at 10 cm displacement where FastWAM reaches 28.7% and
   π0-base 35.3%; π0-base ≈ FastWAM shows the architecture family is not the
   differentiator — the diverse-pretrained VLM binding is.
2. **The video model's binding cannot be rewritten at finetune scale.** Three
   escalating interventions on FastWAM (stratified augmentation; augmentation
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
Wan2.2 video DiT (5B, FastWAM weights) ──dynamics KV─┘     Q attends [VLM ⊕ video ⊕ action]
```

- **Stage 1 (this package's default).** Minimal-surgery fusion: the
  video↔action coupling remains exactly FastWAM's MoT joint attention; the
  VLM enters as additional projected tokens in the action expert's
  cross-attention `context`, behind a LayerScale initialized near zero.
  Both backbones are frozen; trainable parameters are the action expert and
  the VLM adapter (projection, norm, segment embedding, LayerScale).
- **Stage 2 (large-machine option).** Full tri-segment MoT
  (`build_tri_mot_attention_mask`): the video expert also reads the VLM,
  replacing umt5 conditioning; unfreezing requires a web-scale
  vision-language co-training mix (§5.4).

### Design rules derived from campaign evidence (do not remove)

| Rule | Mechanism | Evidence |
|---|---|---|
| Source-level dropout, both KV sources (p = 0.3 each) | `p_drop_vlm`, `p_drop_video` | Available information is not consumed unless consumption is forced (Gate 1, B2, P3-B) |
| VLM frozen by default | stage-1 freeze policy | Wide-support binding collapses under narrow-distribution finetuning (P1 mechanism verdict) |
| Acceptance = dose protocol, never in-distribution success | `eval/README.md` | A 99.3% in-distribution model measured 28.7% at 10 cm |
| Weights-only checkpointing by default | `save_full_state: false` | Full trainer-state saves need a 2×13 G transient window (2026-08-09 incident) |

## 3. Repository layout

```
src/fusionwam/fusion_model.py   FusionWAM model: VLM context injection, MoT dropout,
                                stage-2 tri-MoT mask builder
src/fusionwam/vlm_adapter.py    Frozen PaliGemma encoder + trainable projection
src/fusionwam/trainer.py        FusionTrainer: keeps the adapter trainable under
                                the upstream DiT-only freeze mode
train.py                        Entry point; composes FastWAM's hydra config tree
configs/stage1.yaml             Stage-1 fusion and trainer overrides
scripts/download_weights.sh     Weight fetch (PaliGemma is gated — see §5.1)
scripts/prepare_data.md         Data preparation, incl. enumerated re-rendering
scripts/convert_pi05_vlm.md     Optional π0.5-finetuned PaliGemma conversion
scripts/smoke_test.py           Structure checks + 1-batch forward/backward
eval/README.md                  Acceptance protocol and pre-registered decision rule
```

## 4. Installation on the target machine

```bash
# 1. Transfer the WAM directory (FusionWAM assumes FastWAM as a sibling):
rsync -a <lab>:Alvin/WAM/FastWAM <lab>:Alvin/WAM/FusionWAM ./WAM/
cd WAM/FusionWAM

# 2. Environment (uv-managed):
uv venv && uv sync
uv pip install -e ../FastWAM        # brings diffsynth & upstream deps

# 3. Weights (≈22 G total; PaliGemma requires license acceptance, §5.1):
bash scripts/download_weights.sh ./checkpoints

# 4. Verify before any long run:
.venv/bin/python scripts/smoke_test.py --no-weights   # structure only
.venv/bin/python scripts/smoke_test.py                # + VLM adapter forward
```

## 5. Before training

### 5.1 Gated weights
`google/paligemma-3b-pt-224` is license-gated on Hugging Face. Accept the
license with a logged-in account (`huggingface-cli login`) before running the
download script; mirrors do not serve gated repositories.

### 5.2 Data
Follow `scripts/prepare_data.md`: (1) LIBERO quadruples via the existing
lerobot pipeline — observe the convention checklist (axis-angle cover, gripper
encoding, normalization-stats provenance, agent-view flip); (2) the
enumeration-rendered set (~35 G, ~10 h on one EGL GPU) — this is the component
that carries displacement support into joint training; (3) free subtask labels
from the scripted-primitive phases if stage-2 hierarchical conditioning is
planned.

### 5.3 Optional: π0.5-finetuned VLM
Run the frozen-probe experiment first (pre-registered:
`WAM/note/experiment_log/2026-08-11_PaliGemma探针_预注册.md`). Only if base
PaliGemma features fail to carry out-of-support object position is the JAX→HF
conversion (`scripts/convert_pi05_vlm.md`, est. 1–2 days) worth doing.

### 5.4 Stage 2 prerequisites
Unfreezing the VLM requires a robot:web co-training mix (starting ratio 1:1)
and envelope monitoring during training (dose cells 4/10 cm every N steps);
loss curves alone do not detect binding collapse.

## 6. Training

```bash
.venv/bin/python train.py \
    --fastwam-root ../FastWAM \
    --fusion-config configs/stage1.yaml \
    task=<fastwam task> model=<fastwam model config> output_dir=./runs/stage1
```

`train.py` composes FastWAM's own hydra tree, swaps the model target to
`FusionWAM` and the trainer to `FusionTrainer`, and applies
`configs/stage1.yaml` on top. All upstream overrides (task, data, optimizer)
keep their meaning.

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
| ≤ 45% | fusion ineffective (best 6B-LoRA baseline: 46.7%) | verify source dropout is active and LayerScale has moved off its initialization |

Reference anchors on the same protocol: FastWAM base 20% @10 cm; best
6B-LoRA variant 46.7%; π0.5 73.3%; enumerated-core upper bound (ground-truth
anchors) 93.3% = 100% of the feasibility ceiling.

## 8. Known limitations

- No end-to-end 9B execution has occurred; the two integration surfaces to
  validate first are the hydra composition in `train.py` and the
  `training_loss` override path in `fusion_model.py`.
- Attention-scale balance between the two KV sources is handled only by the
  VLM-branch LayerScale; if the VLM slice's attention mass stays ≈ 0 after
  warm-up, raise `vlm_layerscale_init`.
- Cross-segment relative position is undefined by construction (per-source
  positional encodings); action-to-source attention is content-addressed.
  This is a deliberate choice, not an oversight.
