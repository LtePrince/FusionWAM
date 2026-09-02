"""Teacher-forced vs deployment-rollout gap diagnostic.

The zero-dose gate failed 0/6 with healthy action magnitudes and a smooth
"drift away and park" rollout — the signature of conditioning the action
expert cannot read at deployment. This script separates the hypotheses on
TRAINING samples (no simulator needed):

  arm A  teacher-forced training_loss (real video, as in training)
  arm B  deployment-style infer_action rollout vs the GT action chunk
  arm C  arm B with the VLM detached (prefix contribution isolated)

Reading: low A + B~zero-baseline  -> train/deploy conditioning gap (the
frozen generic video tower's rollout distribution).  B >> C broken-vs-ok
would have implicated the VLM prefix.  High A -> training never fit.

Run on the training machine:
  .venv/bin/python scripts/diagnose_rollout_gap.py \
      --ckpt runs/stage1/checkpoints/weights/step_020000.pt
"""
import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--num-inference-steps", type=int, default=10)
    args = ap.parse_args()

    import numpy as np
    import torch
    import yaml
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[
            "task=libero_joint_2cam224_1e-4",
            "model.load_text_encoder=false",
            "model.skip_dit_load_from_pretrain=false",
        ])
    OmegaConf.set_struct(cfg, False)
    cfg.model._target_ = "fusionwam.training.runtime.create_fusion_model"
    for k, v in yaml.safe_load(open(REPO / "configs/stage1.yaml"))["fusion"].items():
        cfg.model[k] = v

    stats = pathlib.Path(args.ckpt).resolve().parents[2] / "dataset_stats.json"
    assert stats.is_file(), f"dataset_stats.json not found in the run dir: {stats}"
    cfg.data.train.pretrained_norm_stats = str(stats)

    from fusionwam.shared import misc
    misc.register_work_dir("/tmp/fusion_diag")

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(args.ckpt)
    model = model.to("cuda").eval()

    from fusionwam.training.runtime import build_datasets
    from fusionwam.training.trainer import Wan22Trainer
    train_ds, _ = build_datasets(cfg.data)
    to_batch = Wan22Trainer._to_batched_eval_sample

    def rollout_l1(dev, gt_action, with_vlm):
        video0 = dev["video"][0]  # [3,T,H,W] in (-1,1)
        kwargs = dict(
            prompt=None,
            context=dev["context"][0],
            context_mask=dev["context_mask"][0],
            input_image=video0[:, 0].unsqueeze(0),
            num_video_frames=int(video0.shape[1]),
            action_horizon=int(gt_action.shape[0]),
            proprio=dev["proprio"][:, 0] if dev.get("proprio") is not None else None,
            num_inference_steps=args.num_inference_steps,
            negative_prompt="", text_cfg_scale=1.0,
            seed=42, rand_device="cpu", tiled=False,
        )
        if with_vlm:
            kwargs["vlm_prompt"] = dev["prompt"][0]
            pred = model.infer_action(**kwargs)
        else:
            enc, ad = model.vlm_encoder, model.vlm_adapter
            model.vlm_encoder, model.vlm_adapter = None, None
            try:
                pred = model.infer_action(**kwargs)
            finally:
                model.vlm_encoder, model.vlm_adapter = enc, ad
        pa = pred["action"]
        pa = pa[0] if pa.ndim == 3 else pa
        L = min(int(pa.shape[0]), int(gt_action.shape[0]))
        return float((pa[:L].float().cpu() - gt_action[:L]).abs().mean())

    tf, rb, rc, zb = [], [], [], []
    idxs = np.linspace(0, len(train_ds) - 1, args.num_samples).astype(int)
    for i in idxs:
        sample = to_batch(train_ds[int(i)])
        dev = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in sample.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = model.training_loss(dev)
        gt = dev["action"][0].float().cpu()
        b = rollout_l1(dev, gt, with_vlm=True)
        c = rollout_l1(dev, gt, with_vlm=False)
        z = float(gt.abs().mean())
        tf.append(float(loss)); rb.append(b); rc.append(c); zb.append(z)
        comp = " ".join(f"{k}={float(v):.4f}" for k, v in loss_dict.items())
        print(f"sample {i}: A_teacher_loss={float(loss):.4f} ({comp}) "
              f"B_rollout_L1={b:.4f} C_rollout_L1_noVLM={c:.4f} zero_L1={z:.4f}",
              flush=True)

    print("\n=== summary (means) ===")
    print(f"A teacher-forced loss : {np.mean(tf):.4f}")
    print(f"B rollout L1 (VLM on) : {np.mean(rb):.4f}")
    print(f"C rollout L1 (no VLM) : {np.mean(rc):.4f}")
    print(f"  zero-action baseline: {np.mean(zb):.4f}")
    print("read: B/C << baseline = policy informative at deployment; "
          "B,C ~ baseline with low A = train/deploy conditioning gap; "
          "B >> C = the VLM prefix itself is hurting.")


if __name__ == "__main__":
    main()
