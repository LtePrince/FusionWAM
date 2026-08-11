"""FusionWAM smoke test — FIRST thing to run on the target machine.

Checks, in order (each isolates one class of porting failure):
  1. package imports resolve (torch and the full fusionwam tree)
  2. tri-MoT mask: block structure and source dropout exactly match the spec
  3. hydra: configs/ composes and every _target_ resolves to an importable
     object (no model construction)
  4. [weights] transformers loads PaliGemma; one dummy image+prompt through
     the VLM adapter -> finite projected tokens

A full forward/backward of the 9B stack is deliberately NOT here — that is
the 2-step training sanity run (docs/MIGRATION.md §4c), which exercises the
real dataset, trainer, and checkpoint path.

Run: python scripts/smoke_test.py [--config configs/stage1.yaml] [--no-weights]
"""

import argparse
import importlib
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def check_package_imports():
    root = REPO / "src"
    failed = []
    for py in sorted(root.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        rel = py.relative_to(root).with_suffix("")
        parts = rel.parts[:-1] + ((rel.parts[-1],) if rel.parts[-1] != "__init__" else ())
        try:
            importlib.import_module(".".join(parts))
        except Exception as e:  # noqa: BLE001
            failed.append((".".join(parts), repr(e)[:100]))
    if failed:
        for n, e in failed:
            print(" FAIL", n, e)
        raise SystemExit(f"[1] {len(failed)} modules failed to import")
    print("[1] full package import OK")


def check_masks():
    from fusionwam.models.fusion import build_tri_mot_attention_mask
    v2v = torch.ones(6, 6, dtype=torch.bool)
    m = build_tri_mot_attention_mask(4, 6, 3, v2v, torch.device("cpu"))
    assert m.shape == (13, 13)
    assert m[10:, :4].all() and m[10:, 4:10].all() and m[10:, 10:].all()
    assert m[:4, 4:].logical_not().all(), "VLM must not see video/action"
    assert m[4:10, :4].all(), "video must see VLM (stage-2 path)"
    m2 = build_tri_mot_attention_mask(4, 6, 3, v2v, torch.device("cpu"),
                                      drop_action_to_video=True)
    assert m2[10:, 4:10].logical_not().all(), "video-drop must zero the block"
    print("[2] tri-MoT mask structure OK")


def check_hydra_targets():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=libero_joint_2cam224_1e-4"])
    targets = []

    def collect(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "_target_":
                    targets.append((path, v))
                collect(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                collect(v, f"{path}[{i}]")

    collect(OmegaConf.to_container(cfg, resolve=False))
    bad = []
    for path, t in targets:
        mod, _, attr = t.rpartition(".")
        try:
            assert hasattr(importlib.import_module(mod), attr)
        except Exception as e:  # noqa: BLE001
            bad.append((path, t, repr(e)[:80]))
    if bad:
        for b in bad:
            print(" BAD", *b)
        raise SystemExit(f"[3] {len(bad)} unresolvable _target_ refs")
    print(f"[3] hydra compose OK, {len(targets)} _target_ refs resolve")


def check_vlm(config_path: str):
    import yaml
    cfg = yaml.safe_load(open(REPO / config_path))
    from fusionwam.models.paligemma import FrozenPaliGemmaEncoder
    enc = FrozenPaliGemmaEncoder(cfg["fusion"]["vlm_path"], out_dim=1024)
    toks, mask = enc(torch.rand(1, 3, 224, 224), ["pick up the alphabet soup"])
    assert torch.isfinite(toks).all() and mask.any()
    print(f"[4] VLM adapter OK: {tuple(toks.shape)} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--no-weights", action="store_true",
                    help="skip the PaliGemma load (checks 1-3 only)")
    args = ap.parse_args()

    check_package_imports()
    check_masks()
    check_hydra_targets()
    if not args.no_weights:
        check_vlm(args.config)
    print("smoke test PASSED — next: the 2-step training sanity run "
          "(docs/MIGRATION.md §4c)")


if __name__ == "__main__":
    main()
