"""Strip the release checkpoint down to the video tower only.

Produces the warm-start base for stage-1s (scratch action expert): the
LIBERO-finetuned video tower is kept, while the action expert and the
proprio encoder are dropped so training re-initializes them (the ActionDiT
linear-interp construction init applies). load_checkpoint is strict=False,
so the filtered payload overlays cleanly.

  .venv/bin/python scripts/make_video_only_base.py \
      --src checkpoints/wam_release/libero_uncond_2cam224.pt \
      --out checkpoints/wam_release/video_only_base.pt
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="checkpoints/wam_release/libero_uncond_2cam224.pt")
    ap.add_argument("--out", default="checkpoints/wam_release/video_only_base.pt")
    args = ap.parse_args()

    payload = torch.load(args.src, map_location="cpu", mmap=True, weights_only=False)
    mot = payload["mot"]
    kept = {k: v for k, v in mot.items() if k.startswith("mixtures.video.")}
    dropped = [k for k in mot if not k.startswith("mixtures.video.")]
    if not kept:
        raise SystemExit("no mixtures.video.* keys found — key naming changed?")
    n_kept = sum(v.numel() for v in kept.values())
    n_drop = sum(mot[k].numel() for k in dropped)
    torch.save({"mot": kept, "step": payload.get("step"),
                "torch_dtype": payload.get("torch_dtype"),
                "video_only_base": True}, args.out)
    print(f"kept {len(kept)} video-tower tensors ({n_kept/1e9:.2f}B params); "
          f"dropped {len(dropped)} tensors ({n_drop/1e9:.2f}B params: action expert"
          f"{' + proprio_encoder' if 'proprio_encoder' in payload else ''})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
