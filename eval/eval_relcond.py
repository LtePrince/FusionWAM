"""Stage B2③: displacement dose-curve eval with relative conditioning.

Evaluates a (merged) WAM checkpoint on the object-suite static-displacement
protocol (the Stage-A baseline curve: 0/4/7/10cm -> 100/88/38/26%):
init -> settle 5 -> displace target by `dose` in a seeded random direction ->
settle 3 -> policy runs. For relative-conditioned models (--relative), the
frozen Stage B1 detector predicts the anchor once per episode from the
post-displacement agentview frame, and every replan passes
relative = raw_eef_pos − anchor into `infer_action` — mirroring training
(anchor = frame-0 detector prediction, held fixed).

Usage (from WAM-LoRA root, same env vars as eval):
  ... eval_relcond.py --ckpt runs/.../merged.pt --relative --tasks 0-4 \
        --trials 10 --doses 0,0.07,0.10 --out evaluate_results/relcond.json
  ... eval_relcond.py --ckpt runs/.../merged_ctrl.pt --tasks 0-4 ...   # control
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
# Anchor-detector modes (--oracle-anchor etc.) came from the lab's Stage-B
# experiments and need detector weights that are not published; the plain
# displacement eval used for FusionWAM acceptance has no such dependency.

DEVICE = "cuda"
MASK_PROPRIO = False  # set by --mask-proprio (eval-side bypass block diagnostic)
ORACLE_ANCHOR = False  # set by --oracle-anchor (P2(a): anchor := GT object pos)


def load_model_and_processor(ckpt: str, relative: bool):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import eval_libero_single as els

    overrides = [
        "task=libero_uncond_2cam224_1e-4",
        f"ckpt={ckpt}",
        "EVALUATION.dataset_stats_path=./checkpoints/wam_release/libero_uncond_2cam224_dataset_stats.json",
        "gpu_id=0",
        # No T5 (saves ~11GB host RAM on this shared box): task prompts come
        # from the precomputed embedding cache, mirroring training.
        "model.load_text_encoder=false",
    ]
    if relative:
        overrides.append("model.relative_dim=3")
    with initialize_config_dir(config_dir=str(LORA_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="sim_libero.yaml", overrides=overrides)
    model_device = els._resolve_eval_device(cfg)
    model_dtype = els._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)

    payload = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    model.mot.load_state_dict(payload["mot"], strict=False)
    if model.proprio_encoder is not None and "proprio_encoder" in payload:
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    if relative:
        if getattr(model, "relative_encoder", None) is None:
            raise RuntimeError("--relative set but the instantiated model has no relative_encoder")
        if "relative_encoder" in payload:
            model.relative_encoder.load_state_dict(payload["relative_encoder"], strict=True)
        else:
            print("WARNING: ckpt has no relative_encoder weights; zero-init (base-equivalent).")
    del payload
    model = model.to(model_device).eval()

    from fusionwam.wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    stats = load_dataset_stats_from_json(str(els._resolve_dataset_stats_path(cfg)))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    return model, processor, cfg, model_device


def load_detector():
    raise NotImplementedError(
        "Anchor-detector modes are lab-internal (unpublished detector weights) "
        "and are not part of the vendored acceptance eval. Run the plain "
        "displacement protocol without --oracle-anchor/anchor flags."
    )


def detect_anchor_now(det, obs, lang_embed, sim):
    """One anchor prediction from the current agentview obs (dataset convention)."""
    from detect_anchor import backproject, cell_to_224, windowed_softargmax, IMG, RENDER
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    agent = obs["agentview_image"][::-1, ::-1]
    img = torch.from_numpy(np.ascontiguousarray(agent)).permute(2, 0, 1).float() / 255.0
    img = F.interpolate(img.unsqueeze(0), size=(IMG, IMG), mode="bilinear", align_corners=False)
    img = (img * 2 - 1).to(DEVICE)
    # Camera matrix in the detector's RENDER-pixel convention (intrinsics scale
    # with the assumed resolution, so this is env-render-resolution independent).
    M = torch.tensor(get_camera_transform_matrix(sim, "agentview", RENDER, RENDER),
                     dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        _, probs, z = det(img, lang_embed.unsqueeze(0).to(DEVICE))
        r, c = windowed_softargmax(probs)
        xy = backproject(M, cell_to_224(r), cell_to_224(c), z)
    return torch.cat([xy, z.unsqueeze(1)], dim=1)[0].cpu().numpy()  # [3]


def load_task_context(lang, model, cache_dir):
    """Cached T5 context for the task prompt, in the training convention
    (masked positions zeroed, then an all-ones mask — mirrors the dataset and
    `encode_prompt`)."""
    import hashlib
    from fusionwam.wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
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
                processor, cfg, model_device, det, lang_embed, target,
                max_steps_override=None):
    import eval_libero_single as els
    from libero_utils import get_libero_dummy_action

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
    gt_pos = np.array(sim.data.get_joint_qpos(joint)[:3])

    anchor, anchor_err = None, None
    if ORACLE_ANCHOR:
        # P2(a) oracle-grounding: perfect perception by construction. If the
        # dose curve does not flatten even with GT anchors, no perception
        # mount can rescue the current binding (closes the "detector's ~1cm
        # error explains B2's null" loophole).
        anchor = gt_pos.copy()
        anchor_err = 0.0
    elif det is not None:
        anchor = detect_anchor_now(det, obs, lang_embed, sim)
        anchor_err = float(np.linalg.norm(anchor - gt_pos))

    pending, done, t = [], False, 0
    while t < max_steps:
        if not pending:
            relative = None
            if anchor is not None:
                raw8 = np.asarray(els._extract_sim_state(obs)).reshape(-1)
                relative = torch.tensor(raw8[:3] - anchor, dtype=torch.float32).unsqueeze(0)
            chunk = predict_chunk(obs, task_context, model, processor, cfg,
                                  action_horizon=action_horizon, input_w=input_w,
                                  input_h=input_h, model_device=model_device,
                                  relative=relative)
            pending = chunk[:replan_steps].tolist()
        obs, _, done, _ = env.step(pending.pop(0))
        t += 1
        if done:
            break
    return bool(done), anchor_err, t


def predict_chunk(obs, task_context, model, processor, cfg, *, action_horizon,
                  input_w, input_h, model_device, relative):
    """eval_libero_single._predict_action_chunk + relative kwarg + cached context."""
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
    if relative is not None:
        infer_kwargs["relative"] = relative
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
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--mask-proprio", action="store_true",
                        help="zero absolute eef position dims at eval (bypass-block diagnostic)")
    parser.add_argument("--oracle-anchor", action="store_true",
                        help="P2(a): use GT object position as the anchor (perfect perception)")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", default="0-4")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--doses", default="0,0.07,0.10")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override suite max episode steps (timeout diagnostic)")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="override flow-matching denoise steps (latency screen)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero_utils import get_libero_env
    from detect_anchor import load_lang_embeds

    global MASK_PROPRIO, ORACLE_ANCHOR
    MASK_PROPRIO = args.mask_proprio
    ORACLE_ANCHOR = args.oracle_anchor
    model, processor, cfg, model_device = load_model_and_processor(args.ckpt, args.relative)
    cfg.EVALUATION.task_suite_name = args.suite
    if args.num_inference_steps is not None:
        cfg.EVALUATION.num_inference_steps = int(args.num_inference_steps)
    det = load_detector() if (args.relative and not args.oracle_anchor) else None

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
        lang_embed = (load_lang_embeds([task.language],
                                       cache_dir=str(LORA_ROOT / "data/text_embeds_cache/libero"))
                      [task.language] if det is not None else None)
        task_context = load_task_context(task.language, model,
                                         LORA_ROOT / "data/text_embeds_cache/libero")
        env, _ = get_libero_env(task, resolution=256, seed=0)
        try:
            inits = suite.get_task_init_states(tid)
            for trial in range(args.trials):
                for d_i, dose in enumerate(doses):
                    seed_key = 7000 + trial * 10 + d_i  # same family as the baseline curve
                    success, anchor_err, steps = run_episode(
                        env, inits[trial], task.language, task_context, dose,
                        seed_key, model, processor, cfg, model_device, det,
                        lang_embed, target, max_steps_override=args.max_steps)
                    results.append({"task": tid, "trial": trial, "dose": dose,
                                    "success": success, "anchor_err": anchor_err,
                                    "steps": steps})
                    print(f"t{tid} trial{trial} dose{int(dose*100)}cm: "
                          f"{'OK' if success else 'fail'}"
                          + (f" anchor_err={anchor_err*100:.1f}cm" if anchor_err is not None else ""),
                          flush=True)
                    pathlib.Path(args.out).write_text(json.dumps(results))
        finally:
            env.close()

    per_dose = collections.defaultdict(list)
    for r in results:
        per_dose[r["dose"]].append(r["success"])
    print("\n=== relcond eval (success rate per dose) ===")
    for dose, v in sorted(per_dose.items()):
        print(f"位移 {int(dose*100):>2d}cm: {np.mean(v)*100:5.1f}%  (n={len(v)})")
    errs = [r["anchor_err"] for r in results if r["anchor_err"] is not None]
    if errs:
        e = np.array(errs) * 100
        print(f"anchor err: mean {e.mean():.1f}cm median {np.median(e):.1f}cm")


if __name__ == "__main__":
    main()
