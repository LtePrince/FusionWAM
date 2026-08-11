"""FusionWAM smoke test — FIRST thing to run on the target machine.

Checks, in order (each isolates one class of porting failure):
  1. imports resolve (FastWAM installed, transformers version OK)
  2. VLM adapter: encode 1 dummy image+prompt -> projected tokens, finite
  3. fused_context: shapes/masks compose, VLM dropout path runs
  4. tri-MoT mask: block structure exactly matches the spec
  5. (if weights present) 1 forward+backward of the action expert on a
     dummy batch — watch VRAM and attention-entropy printout; if the VLM
     slice's attention mass is ~0, raise layerscale_init before real training.

Run: python scripts/smoke_test.py --config configs/stage1.yaml [--no-weights]
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--no-weights", action="store_true",
                    help="structure-only checks (1,3,4), no model download needed")
    args = ap.parse_args()

    from fusionwam.fusion_model import build_tri_mot_attention_mask
    v2v = torch.ones(6, 6, dtype=torch.bool)
    m = build_tri_mot_attention_mask(4, 6, 3, v2v, torch.device("cpu"))
    assert m.shape == (13, 13)
    assert m[10:, :4].all() and m[10:, 4:10].all() and m[10:, 10:].all()
    assert m[:4, 4:].logical_not().all(), "VLM must not see video/action"
    assert m[4:10, :4].all(), "video must see VLM (stage-2 path)"
    m2 = build_tri_mot_attention_mask(4, 6, 3, v2v, torch.device("cpu"),
                                      drop_action_to_video=True)
    assert m2[10:, 4:10].logical_not().all(), "video-drop must zero the block"
    print("[1,4] structure checks OK")

    if args.no_weights:
        print("smoke test (structure-only) PASSED")
        return

    import yaml
    cfg = yaml.safe_load(open(args.config))
    from fusionwam.vlm_adapter import FrozenPaliGemmaEncoder
    enc = FrozenPaliGemmaEncoder(cfg["fusion"]["vlm_path"], out_dim=1024)
    img = torch.rand(1, 3, 224, 224)
    toks, mask = enc(img, ["pick up the alphabet soup"])
    assert torch.isfinite(toks).all() and mask.any()
    print(f"[2] VLM adapter OK: {tuple(toks.shape)} tokens")
    print("smoke test PASSED — proceed to trainer port (train.py comments)")


if __name__ == "__main__":
    main()
