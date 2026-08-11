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


def load_model_and_processor(ckpt: str):
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
    pending, done, t = [], False, 0
    while t < max_steps:
        if not pending:
            chunk = predict_chunk(obs, task_context, model, processor, cfg,
                                  action_horizon=action_horizon, input_w=input_w,
                                  input_h=input_h, model_device=model_device)
            pending = chunk[:replan_steps].tolist()
        obs, _, done, _ = env.step(pending.pop(0))
        t += 1
        if done:
            break
    return bool(done), t


def predict_chunk(obs, task_context, model, processor, cfg, *, action_horizon,
                  input_w, input_h, model_device):
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
    parser.add_argument("--mask-proprio", action="store_true",
                        help="zero absolute eef position dims at eval (diagnostic)")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", default="0-4")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--doses", default="0,0.04,0.07,0.10,0.15",
                        help="acceptance protocol incl. zero-dose control")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override suite max episode steps (timeout diagnostic)")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="override flow-matching denoise steps (latency screen)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero_utils import get_libero_env

    global MASK_PROPRIO
    MASK_PROPRIO = args.mask_proprio
    model, processor, cfg, model_device = load_model_and_processor(args.ckpt)
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
        env, _ = get_libero_env(task, resolution=256, seed=0)
        try:
            inits = suite.get_task_init_states(tid)
            for trial in range(args.trials):
                for d_i, dose in enumerate(doses):
                    seed_key = 7000 + trial * 10 + d_i  # same family as the baseline curve
                    success, steps = run_episode(
                        env, inits[trial], task.language, task_context, dose,
                        seed_key, model, processor, cfg, model_device,
                        target, max_steps_override=args.max_steps)
                    results.append({"task": tid, "trial": trial, "dose": dose,
                                    "success": success, "steps": steps})
                    print(f"t{tid} trial{trial} dose{int(dose*100)}cm: "
                          f"{'OK' if success else 'fail'}", flush=True)
                    pathlib.Path(args.out).write_text(json.dumps(results))
        finally:
            env.close()

    per_dose = collections.defaultdict(list)
    for r in results:
        per_dose[r["dose"]].append(r["success"])
    print("\n=== displacement eval (success rate per dose) ===")
    for dose, v in sorted(per_dose.items()):
        print(f"dose {int(dose*100):>2d}cm: {np.mean(v)*100:5.1f}%  (n={len(v)})")


if __name__ == "__main__":
    main()
