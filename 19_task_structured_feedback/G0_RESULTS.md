# Topic 19 — G0 Results

## Final project status

**ARCHIVED / PRIMARY METRIC IDENTIFICATION FAILURE**

The frozen numerical gate on the observed configs lands deep in the preregistered KILL region, but that number cannot be interpreted as a clean falsification of task-structured feedback. The run exposed that the primary joint-axis projection metric does not identify task-space correction once the policy responds through different joint coordinations.

No rescue experiment, metric replacement, epsilon sweep, alternate end-effector choice, nonlinear fitting, latent probing, or G0b was run after this was recognized.

## Frozen numerical result

| Quantity | Value |
| --- | ---: |
| `R_task` | `0.9632` |
| `R_null` | `0.9671` |
| `ΔR = R_task - R_null` | **`-0.0038`** |
| config-bootstrap 95% CI | **`[-0.0275, +0.0178]`** |
| frozen GO threshold | `mean >= 0.20` and CI low `> 0` |
| frozen KILL threshold | CI high `<= 0.10` |

On the **8 observed configs**, the frozen scalar score is therefore a tight numerical null, not a gray-zone result.

## Prerequisite gates

### P0 — local policy competence

**PASS: 10/10** CloseDoor level-0 under the official evaluation path, matching the upstream report.

### P1 — paired stochastic inference

**PASS.**

- identical state + identical seed: first-action max absolute difference `0.0`;
- identical state + different seed: first-action difference about `6.0e-3`.

Thus common-random-number pairing was deterministic while the underlying generative sampler remained stochastic.

### Geometry gate

**PASS: 48/48 selected states.** No selected state was rejected by the frozen finite-geometry gate.

Typical finite perturbation geometry:

- task wrist translation: about `31 mm`;
- null wrist translation: about `0.29 mm`;
- null wrist rotation: about `0.09 deg`;
- task/null translation ratio: about `109`.

The numerical null is therefore not explained by failing to instantiate a strong task-vs-null kinematic contrast.

## The diagnostic that invalidated the primary interpretation

The original primary metric was

```text
A(d) = <a(q+d)-a(q), d> / ||d||^2
R(d) = 1 - A(d)
ΔR   = R_task - R_null
```

where `a` is Psi0's first absolute right-arm joint target.

The run showed that the target **does respond** to the perturbations, but mostly outside the injected joint-space axis:

| branch | `||Δa||` | component along `d` | component orthogonal to `d` | alignment fraction |
| --- | ---: | ---: | ---: | ---: |
| task | `0.0334 rad` | `0.0049` | `0.0328` | `15.3%` |
| null | `0.0198 rad` | `0.0069` | `0.0177` | `36.0%` |

This breaks the identification step that allowed `R(d)` to be called an "implied correction fraction".

A simple counterexample is enough. Let `J` be the end-effector Jacobian and let the policy response satisfy

```text
Δa ⟂ d
J Δa = - J d
```

Then the policy can use a *different joint coordination* to cancel the task-space error, while the frozen scalar still gives `A=0, R=1`.

Conversely, another orthogonal response can preserve or reproduce the task-space displacement while producing the same `A=0, R=1`.

Therefore the observed `ΔR≈0` establishes only:

> On the observed CloseDoor states, Psi0 did not show differential restoration **along the exact injected joint-space perturbation axes**.

It does **not** establish:

> Psi0 lacks task-structured feedback or a minimal-intervention-like task-space response.

The latter was the scientific question, so the project stops at a measurement/identification failure rather than a substantive hypothesis kill.

## Second conceptual limitation: wrist-moving is not automatically task-relevant

`δ_task` was chosen as the top singular direction of the wrist positional Jacobian. That cleanly maximizes local wrist translation for a fixed joint-space norm, but CloseDoor task relevance is defined by door/contact geometry, not generic wrist displacement.

A 31 mm wrist displacement can be large yet vary in how much it threatens the active door-closing objective depending on contact normal, hinge tangent, hand-door relative pose, and phase.

Thus the original shorthand

```text
end-effector-changing ~= task-relevant
```

was too strong. A future fresh topic would need to define correction directly in task/outcome space from the beginning rather than repair Topic 19 post hoc.

## Sample / implementation deviations

### D1 — end-effector frame representation

The authored `right_hand_palm_link` fixed joint is folded in the MuJoCo representation and is not available as a standalone body for `mj_jacBody`.

The collector therefore evaluates the authored palm point with `mj_jac` at `right_wrist_yaw_link` plus the fixed offset `[0.0415, -0.003, 0]`, with a runtime `verify_ee_frame` check.

This is an implementation correction to represent the intended physical point, not a scientific contrast change. Without the check, the code would silently evaluate a point displaced by about `4.15 cm` from the intended palm.

### D2 — only 8/10 configs contributed to G0a

The final G0a sample contains:

- **16 successful rollouts**;
- **8 distinct level-0 configs**;
- **48 selected states**;
- four frozen common-random-number seeds per state.

Configs `1` and `7` failed both collector attempts, while the other eight configs succeeded in both repeats. This conflicts with P0's 10/10 success under the official CLI and indicates a remaining collector-vs-official-eval difference.

The missing configs are therefore **systematically missing**, not random independent losses. The protocol amendment had frozen 10 config-level means as the intended primary bootstrap units. The reported 8-config CI is preserved as the result actually obtained, but the preregistered 10-config analysis was not fully executed.

No repeated reruns were used to force configs `1` and `7` into the sample.

### D3 — environment-only compatibility changes

Recorded pre-run compatibility changes:

1. public submodule transport rewritten from `git@github.com:` to HTTPS while preserving repository identities and audited revisions;
2. Psi0 used the same PyTorch version with the CUDA 12.8 wheel instead of the lockfile's CUDA 12.6 wheel so the Blackwell `sm_120` GPU had a usable kernel path; `flash-attn 2.7.4.post1` was smoke-tested on `sm_120`.

These do not change the model checkpoint, scientific intervention, pair seeds, action semantics, or frozen metric.

## Frozen-discipline decisions

After the result, the following were **not** done:

- no epsilon search;
- no joint subset or end-effector search;
- no frame/time-point selection by response;
- no alternate projection metric chosen after seeing the orthogonal response;
- no task-space/Jacobian-output replacement metric;
- no nonlinear response model;
- no hidden-state probe;
- no G0b RTC-history confirmation.

Those would constitute post-hoc rescue of Topic 19.

## Final answer to the six preregistered questions

1. **Did local Psi0 competence pass?** Yes, 10/10 under the official P0 path.
2. **Did deterministic paired inference pass?** Yes, same-seed diff `0.0`; different seed changed the sampled action.
3. **How many units passed geometry?** 48/48 selected states across 16 successful rollouts and 8 configs.
4. **What are the primary numbers?** `R_task=0.9632`, `R_null=0.9671`, `ΔR=-0.0038`, 8-config bootstrap 95% CI `[-0.0275,+0.0178]`.
5. **Frozen scalar verdict?** Numerically inside the frozen KILL region on the observed configs.
6. **Project-level verdict?** **ARCHIVED / PRIMARY METRIC IDENTIFICATION FAILURE**, not a clean falsification of task-structured feedback, because the policy response was largely orthogonal to the injected joint-space direction and the metric cannot identify task-space correction under that response geometry.
