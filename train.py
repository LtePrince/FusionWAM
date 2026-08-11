"""FusionWAM stage-1 training entry.

Composes FastWAM's own hydra config tree (so every dataset/optimizer/model
option keeps its upstream meaning), swaps the model target for FusionWAM and
the trainer for FusionTrainer, then runs the standard loop.

Usage (from FusionWAM/, FastWAM installed as sibling):
  .venv/bin/python train.py \
      --fastwam-root ../FastWAM \
      --fusion-config configs/stage1.yaml \
      task=<fastwam task name> output_dir=./runs/stage1 [more hydra overrides]
"""

import argparse
import sys
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastwam-root", default=str(REPO.parent / "FastWAM"))
    ap.add_argument("--fusion-config", default="configs/stage1.yaml")
    ap.add_argument("overrides", nargs="*", help="hydra dotlist overrides")
    args = ap.parse_args()

    fastwam_root = Path(args.fastwam_root).resolve()
    sys.path.insert(0, str(fastwam_root / "src"))

    fusion_cfg = yaml.safe_load(open(REPO / args.fusion_config))

    with initialize_config_dir(config_dir=str(fastwam_root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=list(args.overrides))

    OmegaConf.set_struct(cfg, False)
    cfg.model._target_ = "fusionwam.fusion_model.FusionWAM.from_wan22_pretrained"
    for k, v in fusion_cfg.get("fusion", {}).items():
        cfg.model[k] = v
    for k, v in fusion_cfg.get("train", {}).items():
        cfg[k] = v

    from fastwam.runtime import (  # noqa: E402  (imported after sys.path insert)
        _mixed_precision_to_model_dtype,
        _normalize_mixed_precision,
        _resolve_train_device,
        build_datasets,
    )
    from hydra.utils import instantiate

    from fusionwam.trainer import FusionTrainer  # noqa: E402

    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(mixed_precision),
        device=_resolve_train_device(),
    )
    train_ds, val_ds = build_datasets(cfg.data)
    FusionTrainer(cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds).train()


if __name__ == "__main__":
    main()
