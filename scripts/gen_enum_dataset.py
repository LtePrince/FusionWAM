"""Server-side enumeration dataset generator (stage-1c data).

Scripted pick-place trajectories over a 2cm grid of target displacements
(r <= --radius, 15cm interior when radius=0.20), rendered with BOTH cameras
and written in the lerobot-v2.1 layout of the original datasets — training
mixes it in by adding the output dir to `dataset_dirs`. Only BDDL-successful
episodes are kept. Calibration comes from scripts/enum_calibration.json
(produced by enum_calibrate.py where the original demo HDF5s live).

Conventions (each fixed a historical accident — do not "simplify"):
  - axis-angle +pi cover (dim0 positive) for eef state;
  - stored action gripper in {0,1} (1=open): env {-1(open),+1(close)} via (1-a)/2;
  - agent/wrist images flipped [::-1, ::-1] (LIBERO render convention);
  - episode meta carries `subtask_spans` (approach/grasp/transport/release) —
    the free hierarchical labels of the scripted route.

Usage (from the repo root, MUJOCO_GL=egl):
  .venv/bin/python scripts/gen_enum_dataset.py --tasks 0-4 --radius 0.20 \
      --features-from ./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot \
      --out /data/users/$USER/libero_object_enum_lerobot
"""

import argparse
import json
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "eval"))

from enum_core import (axisangle_cover, cal_from_json, displace_target,  # noqa: E402
                       grid_points, run_scripted)

FPS = 20


def capture_step(obs):
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    aa = axisangle_cover(obs["robot0_eef_quat"]).astype(np.float32)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    joints = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
    return {
        "state8": np.concatenate([eef_pos, aa, grip]),
        "ee_state": np.concatenate([eef_pos, aa]),
        "joint_state": joints,
        "gripper_state": grip,
        "agent": obs["agentview_image"][::-1, ::-1].copy(),
        "wrist": obs["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
    }


class LerobotWriter:
    def __init__(self, out: pathlib.Path, tasks: list, features: dict):
        self.out = out
        (out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in ("observation.images.image", "observation.images.wrist_image"):
            (out / f"videos/chunk-000/{cam}").mkdir(parents=True, exist_ok=True)
        (out / "meta").mkdir(exist_ok=True)
        self.tasks = tasks
        self.features = features
        with open(out / "meta/tasks.jsonl", "w") as f:
            for i, t in enumerate(tasks):
                f.write(json.dumps({"task_index": i, "task": t}) + "\n")
        self.ep_meta = []
        ep_file = out / "meta/episodes.jsonl"
        if ep_file.exists():
            self.ep_meta = [json.loads(l) for l in open(ep_file)]
        self.next_ep = len(self.ep_meta)
        self.total_frames = sum(e["length"] for e in self.ep_meta)
        self.global_index = self.total_frames

    def write_episode(self, steps: list, task_index: int, extra_meta: dict) -> int:
        ep = self.next_ep
        n = len(steps)
        df = pd.DataFrame({
            "observation.state": [s["state8"] for s in steps],
            "observation.states.ee_state": [s["ee_state"] for s in steps],
            "observation.states.joint_state": [s["joint_state"] for s in steps],
            "observation.states.gripper_state": [s["gripper_state"] for s in steps],
            "action": [s["action"] for s in steps],
            "timestamp": np.arange(n, dtype=np.float32) / FPS,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep, dtype=np.int64),
            "index": np.arange(self.global_index, self.global_index + n, dtype=np.int64),
            "task_index": np.full(n, task_index, dtype=np.int64),
        })
        df.to_parquet(self.out / f"data/chunk-000/episode_{ep:06d}.parquet")
        for cam, key in (("observation.images.image", "agent"),
                         ("observation.images.wrist_image", "wrist")):
            w = imageio.get_writer(
                self.out / f"videos/chunk-000/{cam}/episode_{ep:06d}.mp4",
                fps=FPS, codec="libx264", quality=8, macro_block_size=1)
            for s in steps:
                w.append_data(s[key])
            w.close()
        self.ep_meta.append({"episode_index": ep,
                             "tasks": [self.tasks[task_index]], "length": n,
                             **extra_meta})
        self._write_ep_stats(ep, df)
        self.next_ep += 1
        self.total_frames += n
        self.global_index += n
        self._flush_meta()
        return ep

    def _write_ep_stats(self, ep: int, df: pd.DataFrame):
        stats = {}
        for col in ("observation.state", "observation.states.ee_state",
                    "observation.states.joint_state",
                    "observation.states.gripper_state", "action"):
            arr = np.stack(df[col].values).astype(np.float64)
            stats[col] = {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
                          "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
                          "count": [len(arr)]}
        for col in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            arr = df[col].values.astype(np.float64)
            stats[col] = {"min": [float(arr.min())], "max": [float(arr.max())],
                          "mean": [float(arr.mean())], "std": [float(arr.std())],
                          "count": [len(arr)]}
        with open(self.out / "meta/episodes_stats.jsonl", "a") as f:
            f.write(json.dumps({"episode_index": ep, "stats": stats}) + "\n")

    def _flush_meta(self):
        with open(self.out / "meta/episodes.jsonl", "w") as f:
            for e in self.ep_meta:
                f.write(json.dumps(e) + "\n")
        info = {
            "codebase_version": "v2.1", "robot_type": "panda",
            "total_episodes": len(self.ep_meta), "total_frames": self.total_frames,
            "total_tasks": len(self.tasks), "total_videos": 2 * len(self.ep_meta),
            "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
            "splits": {"train": f"0:{len(self.ep_meta)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self.features,
        }
        json.dump(info, open(self.out / "meta/info.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--tasks", default="0-4")
    ap.add_argument("--radius", type=float, default=0.20,
                    help="enumeration radius (m); 0.20 keeps the 15cm dose interior")
    ap.add_argument("--limit", type=int, default=None, help="points per task cap (smoke)")
    ap.add_argument("--res", type=int, default=512,
                    help="render resolution (matches the original datasets' raw 512)")
    ap.add_argument("--calibration", default=str(REPO / "scripts/enum_calibration.json"))
    ap.add_argument("--features-from", required=True,
                    help="existing lerobot dataset dir to copy meta features from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero_utils import get_libero_env
    from displacement_eval import target_object

    cal_all = json.loads(pathlib.Path(args.calibration).read_text())["tasks"]
    features = json.load(open(pathlib.Path(args.features_from) / "meta/info.json"))["features"]

    suite = benchmark.get_benchmark_dict()[args.suite]()
    bddl_root = pathlib.Path(get_libero_path("bddl_files"))
    lo, _, hi = args.tasks.partition("-")
    task_ids = list(range(int(lo), int(hi or lo) + 1))
    langs = [suite.get_task(t).language for t in range(suite.n_tasks)]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = LerobotWriter(out, langs, features)
    progress_file = out / "meta/progress.json"
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}

    for tid in task_ids:
        if str(tid) not in cal_all:
            print(f"task {tid}: no calibration — SKIPPED (reported)", flush=True)
            continue
        cal = cal_from_json(cal_all[str(tid)])
        task = suite.get_task(tid)
        bddl = str(bddl_root / task.problem_folder / task.bddl_file)
        joint = f"{target_object(bddl)}_joint0"
        pts = grid_points(args.radius)
        if args.limit:
            rng = np.random.default_rng(7)
            pts = [pts[i] for i in rng.permutation(len(pts))[: args.limit]]
        start_i = int(progress.get(str(tid), 0))
        if start_i >= len(pts):
            print(f"task {tid}: complete ({start_i}/{len(pts)}), skip", flush=True)
            continue
        env, _ = get_libero_env(task, resolution=args.res, seed=0)
        try:
            inits = suite.get_task_init_states(tid)
            ok, per_bin = 0, {}
            for i in range(start_i, len(pts)):
                dx, dy = pts[i]
                obs, tpos = displace_target(env, joint, dx, dy,
                                            init_state=inits[i % len(inits)])
                steps = []

                def recorder(o, action_env, phase):
                    rec = capture_step(o)
                    a = action_env.copy()
                    a[6] = (1.0 - a[6]) / 2.0   # env {-1 open,+1 close} -> {0,1}, 1=open
                    rec["action"] = a
                    steps.append(rec)

                success, n, spans = run_scripted(env, obs, cal, tpos, recorder=recorder)
                r_bin = int(np.hypot(dx, dy) / 0.03)
                per_bin.setdefault(r_bin, [0, 0])[1] += 1
                if success and n >= 40:
                    writer.write_episode(steps, tid, extra_meta={
                        "subtask_spans": spans,
                        "displacement_cm": [round(dx * 100, 1), round(dy * 100, 1)],
                        "scripted": True,
                    })
                    per_bin[r_bin][0] += 1
                    ok += 1
                progress[str(tid)] = i + 1
                if (i + 1) % 10 == 0:
                    progress_file.write_text(json.dumps(progress))
                if (i + 1) % 25 == 0:
                    print(f"task {tid}: {i+1}/{len(pts)} ({ok} ok since resume)", flush=True)
            progress_file.write_text(json.dumps(progress))
            yield_str = " ".join(f"{3*b}-{3*b+3}cm:{v[0]}/{v[1]}"
                                 for b, v in sorted(per_bin.items()))
            print(f"task {tid} DONE: +{ok} episodes (per-bin {yield_str})", flush=True)
        finally:
            env.close()

    print(f"\nDONE: {len(writer.ep_meta)} episodes total in {out}")


if __name__ == "__main__":
    main()
