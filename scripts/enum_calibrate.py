"""Extract + validate per-task calibration for the enumeration generator.

Runs where the ORIGINAL LIBERO demo HDF5s live (they are only needed here —
the server-side generator reads the resulting JSON). Per task: try the first
8 demos, extract {grasp offset, grasp quat, drop waypoint, transport
altitude} by state playback, and accept the first candidate whose scripted
trajectory succeeds at BOTH validation displacements (2cm and 8cm) — the
auto-selection that fixed tasks 2-4 from 0% to 86-98% in the envelope
campaign.

Usage (machine with LIBERO + the demo HDF5s; venv must have libero/h5py):
  <venv-python> scripts/enum_calibrate.py \
      --hdf5-dir <path>/libero_original/libero_object \
      --out scripts/enum_calibration.json
"""

import argparse
import json
import pathlib
import sys

import h5py
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "eval"))

from enum_core import (SETTLE_PRE, DUMMY, cal_to_json, displace_target,  # noqa: E402
                       find_grasp_step, run_scripted)


def calibrate(env, states, actions, joint):
    """Playback demo; extract grasp offset/quat and release waypoint."""
    grasp = find_grasp_step(actions)
    release = grasp + 1
    while release < len(actions) and actions[release, 6] > 0:
        release += 1
    release = min(release, len(states) - 1)

    def probe(t):
        obs = env.set_init_state(states[t])
        return (np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy(),
                np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy(),
                np.asarray(env.env.sim.data.get_joint_qpos(joint)[:3]).copy())

    g_eef, g_quat, g_obj = probe(grasp)
    _, _, o_end = probe(len(states) - 1)
    # Transport altitude: the demo's own max eef height cleared every obstacle
    # (smoke v5: transporting at grasp+10cm slammed the can into the basket wall).
    z_max = max(probe(t)[0][2] for t in range(grasp, len(states), 5))
    # Drop waypoint: center the CARRIED OBJECT over where the demo's object
    # ended; the demo's release-time EEF pose is NOT a valid release pose
    # (demos terminate on the predicate with the gripper still closed).
    off = g_eef - g_obj
    drop = np.array([o_end[0] + off[0], o_end[1] + off[1], o_end[2] + off[2] + 0.06])
    return {"grasp_offset": off, "grasp_quat": g_quat,
            "drop_pos": drop, "transport_z": float(z_max) + 0.02}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--tasks", default="0-4")
    ap.add_argument("--hdf5-dir", required=True)
    ap.add_argument("--out", default=str(REPO / "scripts/enum_calibration.json"))
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero_utils import get_libero_env
    from displacement_eval import target_object

    suite = benchmark.get_benchmark_dict()[args.suite]()
    bddl_root = pathlib.Path(get_libero_path("bddl_files"))
    hdf5_dir = pathlib.Path(args.hdf5_dir)
    lo, _, hi = args.tasks.partition("-")
    task_ids = list(range(int(lo), int(hi or lo) + 1))

    result = {}
    for tid in task_ids:
        task = suite.get_task(tid)
        bddl = str(bddl_root / task.problem_folder / task.bddl_file)
        joint = f"{target_object(bddl)}_joint0"
        h5_path = next(hdf5_dir.glob(f"{task.language.replace(' ', '_')}*.hdf5"), None)
        if h5_path is None:
            print(f"task {tid}: no HDF5 found — SKIPPED", flush=True)
            continue
        env, _ = get_libero_env(task, resolution=256, seed=0)
        try:
            inits = suite.get_task_init_states(tid)

            def validate(cal_):
                """Two scripted episodes (2cm / 8cm) must both succeed."""
                wins = 0
                for vd in ((0.014, 0.014), (-0.056, 0.056)):
                    obs, tp = displace_target(env, joint, vd[0], vd[1],
                                              init_state=inits[0])
                    okv, _, _ = run_scripted(env, obs, cal_, tp)
                    wins += int(okv)
                return wins == 2

            cal = None
            with h5py.File(h5_path, "r") as f:
                keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
                for k in keys[:8]:
                    cand = calibrate(env, f["data"][k]["states"][:],
                                     f["data"][k]["actions"][:], joint)
                    if validate(cand):
                        cal = cand
                        print(f"task {tid}: calibrated from {k}", flush=True)
                        break
                    print(f"task {tid}: demo {k} failed validation", flush=True)
            if cal is None:
                print(f"task {tid}: NO demo passes validation — SKIPPED (reported)",
                      flush=True)
                continue
            result[str(tid)] = cal_to_json(cal)
        finally:
            env.close()

    pathlib.Path(args.out).write_text(json.dumps(
        {"suite": args.suite, "resolution_note": "validated at res 256, seed 0",
         "tasks": result}, indent=2))
    print(f"wrote {args.out} ({len(result)}/{len(task_ids)} tasks)")


if __name__ == "__main__":
    main()
