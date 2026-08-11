"""FusionWAM stage-1 training entry (self-contained).

Composes the hydra tree (configs/), swaps the model
target for FusionWAM and the trainer for FusionTrainer, then runs the
standard loop. No external repository is required.

Usage (from the repository root):
  .venv/bin/python train.py \
      task=libero_joint_2cam224_1e-4 output_dir=./runs/stage1 [overrides...]
Multi-GPU:
  bash scripts/train_zero1.sh <nproc> task=libero_joint_2cam224_1e-4 ...
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
    ap.add_argument("--fusion-config", default="configs/stage1.yaml")
    ap.add_argument("--config-dir", default=str(REPO / "configs"),
                    help="hydra config tree")
    ap.add_argument("overrides", nargs="*", help="hydra dotlist overrides")
    args = ap.parse_args()

    fusion_cfg = yaml.safe_load(open(REPO / args.fusion_config))

    with initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=list(args.overrides))

    OmegaConf.set_struct(cfg, False)
    cfg.model._target_ = "fusionwam.models.fusion.FusionWAM.from_wan22_pretrained"
    for k, v in fusion_cfg.get("fusion", {}).items():
        cfg.model[k] = v
    for k, v in fusion_cfg.get("train", {}).items():
        cfg[k] = v

    from fusionwam.training.runtime import (  # noqa: E402
        _mixed_precision_to_model_dtype,
        _normalize_mixed_precision,
        _resolve_train_device,
        build_datasets,
    )
    from hydra.utils import instantiate

    from fusionwam.training.fusion_trainer import FusionTrainer  # noqa: E402

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
