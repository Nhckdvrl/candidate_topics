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

`pi05_libero` loads with `action_horizon=10`, `action_dim=32`.

### These GPUs need the cu128 torch build

`uv sync` resolves `torch==2.7.1+cu126`, whose compiled arch list stops at `sm_90`:

```text
arch list:  sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90
device cap: (12, 0)   # RTX PRO 6000 Blackwell
```

Inference then dies with `CUDA error: no kernel image is available for execution on the
device` — at the first image-normalization op, which makes it look like a data problem
rather than a build-target problem. Install the same pinned version from the cu128 index:

```bash
VIRTUAL_ENV=/home/xiang/projects/openpi_t09/.venv \
  uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

The version pin is unchanged, so openpi's `transformers_replace` patches still apply.

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

### The server venv needs openpi's patched transformers

`uv sync` alone is not enough. openpi ships modified `transformers` model files and both
conversion and inference refuse to run without them:

```bash
cd /home/xiang/projects/openpi_t09
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
```

Without this, `convert_jax_model_to_pytorch.py` fails with
`ValueError: transformers_replace is not installed correctly`. It fails loudly, which is
the good case — but it fails *after* loading a 12 GB orbax checkpoint, so it costs several
minutes per attempt.

### torch.compile is disabled on purpose

`Pi0Config.pytorch_compile_mode` defaults to `"max-autotune"`, which wraps `sample_actions`
in `torch.compile`. `src/openpi_instrumented_server.py` overrides it to `None`. The reason
is correctness of the measurement, not speed — see the note in that file and in
`VALIDATION.md`. Pass `--compile-mode max-autotune` to restore upstream behaviour, but the
feature capture is not trustworthy under it.

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

## Verified inference identity (P0, 2026-08-22)

With the stack above, `run_preflight.sh 2k 3k 9k` passes on all three checkpoints:

```text
checkpoint  state hash   same-seed action  same-seed feature  diff-seed rms  denoise steps
2k          reproducible  0.0               0.0                0.0047         10
3k          reproducible  0.0               0.0                0.0046         10
9k          reproducible  0.0               0.0                0.0207         10
```

The action and feature agreement is exactly `0.0`, not merely small. Combined with ten
captured denoising activations per inference at dim 1024, that confirms the forward hook
fires under eager mode and nothing on the path is nondeterministic.

The three checkpoints are different models at the activation level, not merely different
files. On one shared observation and noise seed:

```text
pair    action rms   layer-11 feature rms
2k-3k   0.0224       0.766
2k-9k   0.0960       3.324
3k-9k   0.0780       2.728
```

Worth carrying forward: **9k responds about four times more strongly to action noise than
2k or 3k** (diff-seed action RMS 0.0207 vs ~0.0047). The checkpoints are not equally
stochastic, so the sampling variance of `p_hat` differs by checkpoint. The crossover rule
is symmetric in A/B and the noise null is computed from the observed rollouts, so this does
not bias the G0 gate — but it is the kind of asymmetry worth knowing before reading any
result.

## Checkpoints

The released pi0.5 LIBERO checkpoints are JAX/orbax and are ~12.4 GB each:

```text
brandonyang/openpi-libero-2000
brandonyang/openpi-libero-3000
brandonyang/openpi-libero-9000
```

Feature capture uses PyTorch forward hooks, so each must be converted with OpenPI's
official `examples/convert_jax_model_to_pytorch.py`.
