"""The small subset of OpenPI's official LIBERO evaluator frozen by Topic 09."""
from __future__ import annotations

import math
import pathlib

import numpy as np

from .state_contract import hash_sim_state

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return q[:3] * 2.0 * math.acos(q[3]) / den


def _allow_libero_init_state_unpickling() -> None:
    """LIBERO stores its fixed initial states as pickled numpy arrays.

    `torch.load` defaults to `weights_only=True` from torch 2.6, which refuses to
    reconstruct numpy arrays and makes `get_task_init_states` raise. The stored data are
    the benchmark's own released initial states, so allowlisting the numpy reconstructors
    restores the official behaviour without changing any of it.
    """
    import torch

    try:
        import numpy.core.multiarray as _ma
    except ModuleNotFoundError:  # numpy>=2 layout
        import numpy._core.multiarray as _ma

    globs = [_ma._reconstruct, np.ndarray, np.dtype]
    globs += [getattr(np, n) for n in ("float64", "float32", "int64") if hasattr(np, n)]
    if hasattr(np, "dtypes"):
        globs += [np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType]
    torch.serialization.add_safe_globals(globs)


def get_task_suite(suite: str):
    from libero.libero import benchmark

    _allow_libero_init_state_unpickling()
    return benchmark.get_benchmark_dict()[suite]()


def make_env(task, *, resolution: int, env_seed: int):
    """Build the environment exactly as OpenPI's official LIBERO evaluator does.

    The official script also sets the global numpy seed once before evaluating, and its
    comment notes that `env.seed` affects object positions even with a fixed initial
    state. Both are reproduced here so nothing about scene construction is left to
    whatever RNG state the process happened to be in.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(int(env_seed))

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(int(env_seed))
    return env


def settle_initial_state(env, initial_state, *, env_seed: int, wait_steps: int = 10):
    """Reproduce OpenPI's reset -> set_init_state -> dummy settling sequence.

    Deliberate deviation from the official script: it seeds the env once at construction
    and then resets repeatedly, so each episode inherits a different RNG state. Topic 09
    re-seeds before every reset instead, because the entire design depends on one
    `init_idx` producing *the same physical state* in every checkpoint process and every
    repeat. The returned `sim_state_hash` is what actually proves that held.
    """
    env.seed(int(env_seed))
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    for _ in range(int(wait_steps)):
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        if done:
            raise RuntimeError("LIBERO episode terminated during settling")
    sim_state = env.sim.get_state()
    flat = sim_state.flatten() if hasattr(sim_state, "flatten") else np.asarray(sim_state).reshape(-1)
    return obs, hash_sim_state(np.asarray(flat, dtype=np.float64))


def policy_element(obs: dict, task_description: str, *, resize_size: int = 224) -> dict:
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(task_description),
    }


def parse_int_spec(spec: str) -> list[int]:
    """Parse a compact integer list such as '0-3,7,9-10'."""
    out: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = [int(x) for x in piece.split("-", 1)]
            if hi < lo:
                raise ValueError(piece)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(piece))
    if len(out) != len(set(out)):
        raise ValueError("integer spec contains duplicates")
    return out
