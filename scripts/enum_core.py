"""Scripted pick-place core for the enumeration dataset (shared by the local
calibration tool and the server-side generator).

Per task, a calibration (grasp offset in the object frame, grasp quaternion,
release waypoint, transport altitude) drives a scripted
reach->grasp->lift->transport->release trajectory tracked by the OSC
P-controller. Every constant below was tuned through the envelope campaign's
smoke tests; the inline comments record the failure each one fixes.
"""

import numpy as np
import robosuite.utils.transform_utils as T

SETTLE_PRE = 5
SETTLE_POST = 3
LIFT_Z = 0.002
HOVER = 0.10          # hover height above grasp z
STEP_LEN = 0.015      # ~1.5cm per waypoint step (tracker-friendly)
CLOSE_DWELL = 8
OPEN_DWELL = 45       # gripper open ~10 steps + fall + predicate settle; 12
                      # ended episodes with the object still between the fingers
GRID = 0.02           # enumeration grid spacing
CONV_TOL = 0.008
CONV_MAX = 40
DUMMY = [0.0] * 6 + [-1.0]


def axisangle_cover(quat):
    """Axis-angle in the training data's +pi cover (dim0 positive)."""
    aa = np.asarray(T.quat2axisangle(np.asarray(quat, dtype=np.float64)))
    if aa[0] < 0:
        theta = np.linalg.norm(aa)
        if theta > 1e-8:
            aa = aa * (theta - 2.0 * np.pi) / theta
    return aa


def state8_of(obs):
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        axisangle_cover(obs["robot0_eef_quat"]).astype(np.float32),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    ])


def find_grasp_step(actions, sustain=10):
    close = actions[:, 6] > 0
    for i in range(len(close) - sustain):
        if close[i:i + sustain].all():
            return i
    return len(actions) - 1


def interp(a, b, step=STEP_LEN):
    d = np.linalg.norm(b - a)
    n = max(2, int(np.ceil(d / step)))
    return [a + (b - a) * (i / n) for i in range(1, n + 1)]


def script_waypoints(start, target_pos, cal):
    """(pos, grip, converge, phase) waypoints. converge=True repeats the
    waypoint until the eef is within CONV_TOL (the P-tracker lags 1-2cm —
    closing the gripper before convergence was the 0/25 smoke failure).
    `phase` labels the free subtask annotation: approach/grasp/transport/release.
    """
    grasp = target_pos + cal["grasp_offset"]
    z_t = max(cal["transport_z"], grasp[2] + HOVER)
    hover_g = np.array([grasp[0], grasp[1], z_t])
    drop = cal["drop_pos"]
    hover_d = np.array([drop[0], drop[1], z_t])
    wp = []
    wp += [(p, -1.0, False, "approach") for p in interp(start, hover_g)]
    wp += [(hover_g, -1.0, True, "approach")]
    wp += [(p, -1.0, False, "approach") for p in interp(hover_g, grasp)]
    wp += [(grasp, -1.0, True, "approach")]        # converge BEFORE closing
    wp += [(grasp, 1.0, False, "grasp")] * CLOSE_DWELL
    wp += [(p, 1.0, False, "transport") for p in interp(grasp, hover_g)]
    wp += [(p, 1.0, False, "transport") for p in interp(hover_g, hover_d)]
    wp += [(p, 1.0, False, "transport") for p in interp(hover_d, drop)]
    wp += [(drop, 1.0, True, "transport")]
    wp += [(drop, -1.0, False, "release")] * OPEN_DWELL
    # Retreat upward with fingers open: shears off friction-wedged objects
    # (smoke v4: fingers fully open, can still stuck between them at z=0.17).
    retreat = drop + np.array([0.0, 0.0, 0.12])
    wp += [(p, -1.0, False, "release") for p in interp(drop, retreat)]
    wp += [(retreat, -1.0, False, "release")] * 20
    return wp, grasp


def run_scripted(env, obs, cal, target_pos, max_steps=400, recorder=None):
    """Track the scripted waypoints. `recorder(obs, action_env, phase)` is
    called once per executed step BEFORE env.step (obs/action pairing matches
    the demo datasets). Returns (success, n_steps, subtask_spans)."""
    controller = env.env.robots[0].controller
    output_max = np.abs(np.asarray(controller.output_max, dtype=np.float64))
    start = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
    goal_quat = cal["grasp_quat"]
    waypoints, _ = script_waypoints(start, target_pos, cal)
    n, done = 0, False
    spans, cur_phase = [], None

    def act_toward(pos_t, grip, phase):
        nonlocal obs, done, n, cur_phase
        cur_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        cur_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        rot_err = T.quat2mat(goal_quat) @ T.quat2mat(cur_quat).T
        daa = T.quat2axisangle(T.mat2quat(rot_err))
        action = np.zeros(7)
        action[:3] = np.clip((pos_t - cur_pos) / output_max[:3], -1.0, 1.0)
        action[3:6] = np.clip(daa / output_max[3:6], -1.0, 1.0)
        action[6] = grip
        if phase != cur_phase:
            spans.append([phase, n])
            cur_phase = phase
        if recorder is not None:
            recorder(obs, action.astype(np.float32), phase)
        n += 1
        obs, _, done, _ = env.step(action.tolist())
        return np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - pos_t)

    for pos_t, grip, converge, phase in waypoints:
        err = act_toward(pos_t, grip, phase)
        if converge:
            k = 0
            while err > CONV_TOL and k < CONV_MAX and not done and n < max_steps:
                err = act_toward(pos_t, grip, phase)
                k += 1
        if done or n >= max_steps:
            break
    return bool(done), n, spans


def displace_target(env, joint, dx, dy, settle_pre=SETTLE_PRE, settle_post=SETTLE_POST,
                    init_state=None):
    """Reset -> (optional init state) -> settle -> displace joint -> settle.
    Returns (obs, target_pos)."""
    env.reset()
    obs = env.set_init_state(init_state) if init_state is not None else None
    if obs is None:
        for _ in range(settle_pre):
            obs, _, _, _ = env.step(DUMMY)
    else:
        for _ in range(settle_pre):
            obs, _, _, _ = env.step(DUMMY)
    sim = env.env.sim
    q = np.array(sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
    if abs(dx) > 1e-9 or abs(dy) > 1e-9:
        q[0] += dx
        q[1] += dy
        q[2] += LIFT_Z
        sim.data.set_joint_qpos(joint, q)
        sim.data.set_joint_qvel(joint, np.zeros(6))
        sim.forward()
        for _ in range(settle_post):
            obs, _, _, _ = env.step(DUMMY)
    return obs, np.array(sim.data.get_joint_qpos(joint)[:3])


def grid_points(radius, grid=GRID):
    ax = np.arange(-radius, radius + 1e-9, grid)
    return [(x, y) for x in ax for y in ax if x * x + y * y <= radius ** 2]


def cal_to_json(cal):
    return {"grasp_offset": [float(v) for v in cal["grasp_offset"]],
            "grasp_quat": [float(v) for v in cal["grasp_quat"]],
            "drop_pos": [float(v) for v in cal["drop_pos"]],
            "transport_z": float(cal["transport_z"])}


def cal_from_json(d):
    return {"grasp_offset": np.asarray(d["grasp_offset"], dtype=np.float64),
            "grasp_quat": np.asarray(d["grasp_quat"], dtype=np.float64),
            "drop_pos": np.asarray(d["drop_pos"], dtype=np.float64),
            "transport_z": float(d["transport_z"])}
