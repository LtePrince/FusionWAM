"""FusionWAM stage-1 training entry (adaptation of FastWAM's trainer).

Freezes: Wan2.2 video expert, PaliGemma. Trains: action expert, VLM
projection/segment/layerscale. Loss: action flow matching (+ optional video
co-generation kept OFF in stage 1 — the video expert is frozen, its
co-generation objective belongs to stage 2).

Run (big machine):
  python train.py --config configs/stage1.yaml

This file is a thin orchestrator: dataset/optimizer/checkpointing reuse
FastWAM's trainer utilities verbatim (import path below). The three fusion
hooks are marked # FUSION.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
# Expect FastWAM installed (pip install -e ../FastWAM) or on PYTHONPATH.

from fusionwam.fusion_model import FusionWAM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    model = FusionWAM.from_wan22_pretrained(**cfg["fastwam"])          # video+action
    model.attach_vlm(
        cfg["fusion"]["vlm_path"],
        p_drop_vlm=cfg["fusion"]["p_drop_vlm"],
        p_drop_video=cfg["fusion"]["p_drop_video"],
        layerscale_init=cfg["fusion"]["layerscale_init"],
    )

    # FUSION: freeze policy
    model.video_expert.requires_grad_(False)
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")

    # FUSION: the training loop below is FastWAM's, with two changes:
    #   1) context = model.fused_context(umt5_ctx, umt5_mask, frame0, prompts)
    #   2) mot_mask = model.maybe_drop_video_block(mot_mask, video_seq_len)
    # See src/fastwam/trainer.py::train_step in the FastWAM repo for the
    # surrounding code; port those two lines into your copy, or use the patch
    # in scripts/trainer_stage1.patch if the upstream file is unchanged.
    raise SystemExit(
        "Stage-1 orchestrator: wire the two FUSION hooks into FastWAM's "
        "trainer (see comments above), then delete this guard. Kept explicit "
        "so the port is a conscious step on the target machine, not a silent "
        "assumption that upstream trainer internals haven't drifted."
    )


if __name__ == "__main__":
    main()
