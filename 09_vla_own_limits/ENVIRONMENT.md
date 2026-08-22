# Environment — reproducible setup

Built and verified 2026-08-22 on `fvcrc21` (4x RTX PRO 6000 Blackwell, 97 GB each).

Two isolated venvs, matching OpenPI's own client/server split. The server needs JAX +
PyTorch + openpi; the client needs LIBERO + robosuite + MuJoCo. Their pinned dependencies
conflict (LIBERO wants `numpy<2` and an old `robosuite`), so they must not share a venv.

Nothing here lives inside the repository. Checkouts, venvs and checkpoints are external
paths so that no large artifact can be committed.

## Paths

```text
/home/xiang/projects/openpi_t09      openpi @ 15a9616 (pinned in LOCKED_CONFIG.json)
/home/xiang/projects/openpi_t09/.venv  server venv (uv sync)
/home/xiang/projects/LIBERO_t09      LIBERO checkout
/home/xiang/venvs/t09_client         client venv (python 3.11)
/home/xiang/projects/t09_ckpts       downloaded + converted checkpoints
```

## Server venv

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi /home/xiang/projects/openpi_t09
cd /home/xiang/projects/openpi_t09 && git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
uv sync
```

Verified: `torch 2.7.1+cu126`, CUDA visible, `pi05_libero` config loads with
`action_horizon=10`, `action_dim=32`.

## Client venv

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /home/xiang/projects/LIBERO_t09
uv venv --python 3.11 /home/xiang/venvs/t09_client
VIRTUAL_ENV=/home/xiang/venvs/t09_client uv pip install \
  "numpy<2" "robosuite==1.4.1" "mujoco==3.1.6" "gym==0.25.2" "setuptools<70" \
  imageio imageio-ffmpeg opencv-python pandas scikit-learn scipy pytest \
  tyro PyYaml easydict cloudpickle matplotlib torch websockets msgpack einops
VIRTUAL_ENV=/home/xiang/venvs/t09_client uv pip install --no-deps -e /home/xiang/projects/openpi_t09/packages/openpi-client
```

### Three setup details that are easy to get wrong

1. **LIBERO is not installable with `pip install .`.** Its `libero/` directory has no
   `__init__.py`, so `find_packages()` finds nothing and the install silently produces an
   importable-looking but empty package. Put the checkout on the path instead:

   ```bash
   echo /home/xiang/projects/LIBERO_t09 > /home/xiang/venvs/t09_client/lib/python3.11/site-packages/libero_repo.pth
   ```

2. **LIBERO prompts interactively on first import** and writes `~/.libero/config.yaml`.
   Under a non-interactive shell that import raises `EOFError`. Write the file up front:

   ```yaml
   benchmark_root: /home/xiang/projects/LIBERO_t09/libero/libero
   bddl_files: /home/xiang/projects/LIBERO_t09/libero/libero/./bddl_files
   init_states: /home/xiang/projects/LIBERO_t09/libero/libero/./init_files
   datasets: /home/xiang/projects/LIBERO_t09/libero/libero/../datasets
   assets: /home/xiang/projects/LIBERO_t09/libero/libero/./assets
   ```

   The `datasets` warning is expected and harmless; we never load demonstrations.

3. **`torch>=2.6` cannot read LIBERO's initial states.** They are pickled numpy arrays and
   `torch.load` now defaults to `weights_only=True`. `src/libero_common.py` allowlists the
   numpy reconstructors before building the benchmark. Without that,
   `get_task_init_states` raises `WeightsUnpickler error` — which would look like a
   missing-data problem rather than a serialization default.

## Rendering

Use EGL. OSMesa is not usable on these nodes (`libOSMesa` missing; PyOpenGL raises
`'NoneType' object has no attribute 'glGetError'`).

```bash
export MUJOCO_GL=egl
```

## Verified state identity

With `MUJOCO_GL=egl`, on `libero_10` task 0:

```text
same init_idx, two resets   -> identical settled sim_state_hash
different init_idx          -> different settled sim_state_hash
same init_idx, two resets   -> bit-identical agentview image
```

All 10 LIBERO-10 tasks expose 50 fixed initial states, so the frozen
discovery/confirmation/reserve split (`0-14` / `15-29` / `30-49`) is valid.

## Checkpoints

The released pi0.5 LIBERO checkpoints are JAX/orbax and are ~12.4 GB each:

```text
brandonyang/openpi-libero-2000
brandonyang/openpi-libero-3000
brandonyang/openpi-libero-9000
```

Feature capture uses PyTorch forward hooks, so each must be converted with OpenPI's
official `examples/convert_jax_model_to_pytorch.py`.
