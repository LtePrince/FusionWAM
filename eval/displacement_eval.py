"""Displacement dose-protocol acceptance eval (plain path).

Evaluates a trained checkpoint on the object-suite static-displacement
protocol: init -> settle 5 -> displace the target object by `dose` in a
seeded random direction -> settle 3 -> policy runs (BDDL success, 400-step
cap). Zero-dose control included via the dose list. Legacy seed family
(7000 + trial*10 + dose_index) keeps results seed-paired with historical
numbers.

Requires the external sim stack (LIBERO + robosuite + MuJoCo/EGL) and the
artifacts from scripts/download_weights.sh + scripts/download_data.sh
(dataset stats json, text-embed cache).

Usage (from the repository root):
  MUJOCO_GL=egl .venv/bin/python eval/displacement_eval.py \
      --ckpt <weights checkpoint> --tasks 0-4 --trials 6 \
      --doses 0,0.04,0.07,0.10,0.15 --out results/dose.json
"""

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Vendored layout: repo_root/eval/ ; wam package lives in repo_root/src.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "eval"))         # libero_utils sibling

DEVICE = "cuda"
MASK_PROPRIO = False  # set by --mask-proprio (eval-side bypass block diagnostic)


def load_model_and_processor(ckpt: str, plain_wam: bool = False, base_ckpt: str = None):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import eval_libero_single as els

    import yaml
    from omegaconf import OmegaConf

    if plain_wam:
        # Known-good control path: the released uncond checkpoint (full MoT in
        # the payload) through the original build — uncond task, factory from
        # sim_libero defaults (skip_dit_load_from_pretrain=true), release
        # normalization stats. Splits "our policy is weak" from "this
        # machine's sim harness is broken".
        stats_json = REPO_ROOT / "checkpoints/wam_release/libero_uncond_2cam224_dataset_stats.json"
        assert stats_json.is_file(), (
            f"control stats missing: {stats_json}\nDownload with:\n"
            "  huggingface-cli download yuanty/fastwam libero_uncond_2cam224.pt "
            "libero_uncond_2cam224_dataset_stats.json --local-dir checkpoints/wam_release")
        overrides = [
            "task=libero_uncond_2cam224_1e-4",
            f"ckpt={ckpt}",
            f"EVALUATION.dataset_stats_path={stats_json}",
            "gpu_id=0",
            "model.load_text_encoder=false",
        ]
        with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
            cfg = compose(config_name="sim_libero.yaml", overrides=overrides)
        model_device = els._resolve_eval_device(cfg)
        model_dtype = els._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
        model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
        payload = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
        model.mot.load_state_dict(payload["mot"], strict=False)
        if model.proprio_encoder is not None and "proprio_encoder" in payload:
            model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        del payload
        model = model.to(model_device).eval()
        from fusionwam.data.lerobot.utils.normalizer import load_dataset_stats_from_json
        stats = load_dataset_stats_from_json(str(els._resolve_dataset_stats_path(cfg)))
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(stats)
        return model, processor, cfg, model_device

    # Build the model exactly the way train.py does for stage 1: the JOINT
    # task (WAMJoint mask geometry the checkpoint was trained with), the
    # FusionWAM factory, and the fusion block (vlm_path) from stage1.yaml.
    overrides = [
        "task=libero_joint_2cam224_1e-4",
        f"ckpt={ckpt}",
        "gpu_id=0",
        # No T5 (saves ~11GB host RAM): task prompts come from the
        # precomputed embedding cache, mirroring training.
        "model.load_text_encoder=false",
        # Stage-1 checkpoints persist trainable parts only; the frozen video
        # tower must reload from the Wan2.2 base, not stay randomly initialised.
        "model.skip_dit_load_from_pretrain=false",
        # Dataset stats: resolved from the checkpoint's run directory
        # (runs/<run>/dataset_stats.json written by training) — the stats the
        # policy was actually normalised with.
    ]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="sim_libero.yaml", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.model._target_ = "fusionwam.training.runtime.create_fusion_model"
    fusion_cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "stage1.yaml"))
    for k, v in fusion_cfg.get("fusion", {}).items():
        cfg.model[k] = v
    model_device = els._resolve_eval_device(cfg)
    model_dtype = els._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    if getattr(model, "vlm_adapter", None) is None:
        raise RuntimeError("Model was built without a VLM adapter; check configs/stage1.yaml fusion.vlm_path")

    # FusionWAM.load_checkpoint: MoT (strict=False over a trainable-only
    # payload), proprio encoder, and the VLM adapter — errors if the adapter
    # is missing instead of silently evaluating a fusion-less model.
    # Trainable-only checkpoints do NOT carry the frozen towers. If training
    # warm-started them from a different base (stage-1b: resume=<release.pt>),
    # that base must be layered in FIRST or the action expert reads features
    # from the wrong (generic) tower — bug #15.
    if base_ckpt:
        print(f"[eval] layering base checkpoint first: {base_ckpt}", flush=True)
        model.load_checkpoint(base_ckpt)
    model.load_checkpoint(ckpt)
    model = model.to(model_device).eval()

    from fusionwam.data.lerobot.utils.normalizer import load_dataset_stats_from_json
    stats = load_dataset_stats_from_json(str(els._resolve_dataset_stats_path(cfg)))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    return model, processor, cfg, model_device




def load_task_context(lang, model, cache_dir):
    """Cached T5 context for the task prompt, in the training convention
    (masked positions zeroed, then an all-ones mask — mirrors the dataset and
    `encode_prompt`)."""
    import hashlib
    from fusionwam.data.lerobot.robot_video_dataset import DEFAULT_PROMPT
    prompt = DEFAULT_PROMPT.format(task=lang)
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    payload = torch.load(pathlib.Path(cache_dir) / f"{hashed}.t5_len128.wan22ti2v5b.pt",
                         map_location="cpu")
    msk0 = payload["mask"].bool()
    ctx = payload["context"].clone()
    ctx[~msk0] = 0.0
    ctx = ctx.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    msk = torch.ones((1, msk0.shape[0]), dtype=torch.bool, device=model.device)
    return ctx, msk


def target_object(bddl_path: str) -> str:
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    objs = list(robosuite_parse_problem(str(bddl_path))["obj_of_interest"])
    return next((o for o in objs if "bowl" in o), objs[0])


def run_episode(env, init_state, task_lang, task_context, dose, seed_key, model,
                processor, cfg, model_device, target,
                max_steps_override=None, save_video_path=None):
    import eval_libero_single as els
    from libero_utils import get_libero_dummy_action
    from fusionwam.data.lerobot.robot_video_dataset import DEFAULT_PROMPT
    # == training sample["prompt"]; None for plain WAM (its infer_action has
    # no vlm_prompt kwarg and no VLM to feed).
    vlm_prompt = (DEFAULT_PROMPT.format(task=task_lang)
                  if getattr(model, "vlm_encoder", None) is not None else None)

    max_steps = max_steps_override or els._get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    # Mirror eval_single_process: action_horizon defaults to num_frames-1 when
    # unset (the cfg key exists but is null), input dims come from video_size.
    ah_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(ah_cfg) if ah_cfg is not None else int(cfg.data.train.num_frames) - 1
    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(5):  # settle (benchmark inits are pre-settle)
        obs, _, _, _ = env.step(get_libero_dummy_action())
    sim = env.env.sim
    joint = f"{target}_joint0"
    if dose > 0:
        q = np.array(sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
        ang = np.random.RandomState(seed_key).uniform(0, 2 * np.pi)
        q[0] += dose * np.cos(ang)
        q[1] += dose * np.sin(ang)
        q[2] += 0.002
        sim.data.set_joint_qpos(joint, q)
        sim.data.set_joint_qvel(joint, np.zeros(6))
        sim.forward()
        for _ in range(3):
            obs, _, _, _ = env.step(get_libero_dummy_action())
    frames = []
    if save_video_path:
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
    pending, done, t = [], False, 0
    while t < max_steps:
        if not pending:
            chunk = predict_chunk(obs, task_context, model, processor, cfg,
                                  action_horizon=action_horizon, input_w=input_w,
                                  input_h=input_h, model_device=model_device,
                                  vlm_prompt=vlm_prompt)
            if save_video_path and t == 0:
                c = np.asarray(chunk)
                print(f"    [debug] chunk shape={c.shape} min={c.min():.3f} "
                      f"max={c.max():.3f} |mean|={np.abs(c).mean():.3f} "
                      f"gripper_head={np.round(c[:5, -1], 3).tolist()}", flush=True)
            pending = chunk[:replan_steps].tolist()
        obs, _, done, _ = env.step(pending.pop(0))
        t += 1
        if save_video_path:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        if done:
            break
    if save_video_path and frames:
        import imageio
        imageio.mimsave(save_video_path, frames, fps=20)
    return bool(done), t


def predict_chunk(obs, task_context, model, processor, cfg, *, action_horizon,
                  input_w, input_h, model_device, vlm_prompt=None):
    """eval_libero_single._predict_action_chunk with cached text context."""
    import inspect
    import eval_libero_single as els

    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps")
                              or cfg.get("eval_num_inference_steps", 20))
    context, context_mask = task_context
    image, proprio, _ = els._obs_to_model_input(
        obs, cfg=cfg, processor=processor, width=input_w, height=input_h,
        device=model_device, dtype=model.torch_dtype,
    )
    if MASK_PROPRIO:
        # Eval-side bypass block (diagnostic): zero the absolute eef position
        # dims so the policy MUST read the relative token. Separates "never
        # learned to use relative" from "learned it but the shortcut suppresses
        # it at eval time" (training masked these with p=0.5; eval left them on).
        proprio = proprio.clone()
        proprio[..., :3] = 0.0
    infer_kwargs = {
        "prompt": None,
        "context": context,
        "context_mask": context_mask,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (None if cfg.EVALUATION.get("sigma_shift") is None
                        else float(cfg.EVALUATION.get("sigma_shift"))),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if vlm_prompt is not None:
        infer_kwargs["vlm_prompt"] = vlm_prompt  # instruction text for the VLM prefix
    if "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = els._get_num_video_frames(cfg)
    with torch.no_grad():
        pred = model.infer_action(**infer_kwargs)
    action = els._denormalize_action(pred["action"], processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    from libero_utils import invert_gripper_action as _inv
    action = _inv(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base-ckpt", default=None,
                        help="full checkpoint the frozen towers were warm-started "
                             "from (loaded before --ckpt; required for trainable-only "
                             "checkpoints trained with resume=<base>)")
    parser.add_argument("--no-video", action="store_true",
                        help="diagnostic: block action->video attention at eval "
                             "(the video-source-dropout path; VLM prefix + T5 + "
                             "proprio only)")
    parser.add_argument("--no-vlm", action="store_true",
                        help="diagnostic: detach the VLM prefix after loading the "
                             "checkpoint (evaluates the source-dropout fallback path)")
    parser.add_argument("--plain-wam", action="store_true",
                        help="control mode: build the plain uncond WAM (release "
                             "checkpoint with full MoT payload) instead of FusionWAM")
    parser.add_argument("--mask-proprio", action="store_true",
                        help="zero absolute eef position dims at eval (diagnostic)")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", default="0-4")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--dose-index-base", type=int, default=0,
                        help="index of the first --doses entry within the canonical "
                             "dose list (seed-family pairing for isolated runs)")
    parser.add_argument("--trial-list", default=None,
                        help="comma-separated trial indices (overrides --trials); "
                             "lets a wrapper isolate one episode per process")
    parser.add_argument("--doses", default="0,0.04,0.07,0.10,0.15",
                        help="acceptance protocol incl. zero-dose control")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override suite max episode steps (timeout diagnostic)")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="override flow-matching denoise steps (latency screen)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-video", default=None,
                        help="dir for per-episode agentview mp4s (diagnostic; "
                             "also prints first-chunk action stats)")
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero_utils import get_libero_env

    global MASK_PROPRIO
    MASK_PROPRIO = args.mask_proprio
    model, processor, cfg, model_device = load_model_and_processor(
        args.ckpt, plain_wam=args.plain_wam, base_ckpt=args.base_ckpt)
    if args.no_vlm and not args.plain_wam:
        # Detach after load_checkpoint (which requires the adapter to accept the
        # payload); run_episode then auto-omits vlm_prompt and infer_action
        # passes straight through — the dropout-trained no-prefix path.
        model.vlm_encoder = None
        model.vlm_adapter = None
        print("[diag] VLM prefix detached (--no-vlm)", flush=True)
    if args.no_video and not args.plain_wam:
        model.force_drop_video = True
        print("[diag] action->video attention blocked (--no-video)", flush=True)
    cfg.EVALUATION.task_suite_name = args.suite
    if args.num_inference_steps is not None:
        cfg.EVALUATION.num_inference_steps = int(args.num_inference_steps)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    bddl_root = pathlib.Path(get_libero_path("bddl_files"))
    lo, _, hi = args.tasks.partition("-")
    task_ids = list(range(int(lo), int(hi or lo) + 1))
    doses = [float(d) for d in args.doses.split(",")]

    results = []
    for tid in task_ids:
        task = suite.get_task(tid)
        bddl = str(bddl_root / task.problem_folder / task.bddl_file)
        target = target_object(bddl)
        task_context = load_task_context(task.language, model,
                                         REPO_ROOT / "data/text_embeds_cache/libero")
        inits = suite.get_task_init_states(tid)
        trial_indices = ([int(x) for x in args.trial_list.split(",")]
                         if args.trial_list else list(range(args.trials)))
        for trial in trial_indices:
            for d_i, dose in enumerate(doses):
                    # Same seed family as the baseline curve. d_i must be the
                    # dose's index in the canonical 5-dose list; when a wrapper
                    # isolates one dose per process it passes the true index
                    # via --dose-index-base to keep seed pairing intact.
                    seed_key = 7000 + trial * 10 + (args.dose_index_base + d_i)
                    vid = None
                    if args.save_video:
                        vdir = pathlib.Path(args.save_video)
                        vdir.mkdir(parents=True, exist_ok=True)
                        vid = str(vdir / f"t{tid}_trial{trial}_dose{int(dose*100)}cm.mp4")
                    # Fresh env per episode: reusing one MuJoCo/EGL env across
                    # episodes intermittently SIGABRTs in native code at the
                    # second episode (with and without frame capture).
                    env, _ = get_libero_env(task, resolution=256, seed=0)
                    try:
                        success, steps = run_episode(
                            env, inits[trial], task.language, task_context, dose,
                            seed_key, model, processor, cfg, model_device,
                            target, max_steps_override=args.max_steps,
                            save_video_path=vid)
                    finally:
                        env.close()
                    results.append({"task": tid, "trial": trial, "dose": dose,
                                    "success": success, "steps": steps})
                    print(f"t{tid} trial{trial} dose{int(dose*100)}cm: "
                          f"{'OK' if success else 'fail'}", flush=True)
                    pathlib.Path(args.out).write_text(json.dumps(results))

    per_dose = collections.defaultdict(list)
    for r in results:
        per_dose[r["dose"]].append(r["success"])
    print("\n=== displacement eval (success rate per dose) ===")
    for dose, v in sorted(per_dose.items()):
        print(f"dose {int(dose*100):>2d}cm: {np.mean(v)*100:5.1f}%  (n={len(v)})")


if __name__ == "__main__":
    main()
