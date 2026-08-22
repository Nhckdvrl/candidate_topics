# Data contract

The first validation intentionally keeps the data model small.

## G0 behavioral panel

CSV columns:

```text
task,seed,checkpoint,success
```

Requirements:

- every `(task, seed)` state is evaluated by every checkpoint included in a comparison;
- `checkpoint` is a stable label such as `2k`, `3k`, `9k`;
- `success` is binary `0/1`, using the official LIBERO success definition;
- discovery and confirmation seeds are disjoint;
- do not drop hard states or failed rollouts after seeing outcomes.

Example:

```csv
task,seed,checkpoint,success
libero_10_0,1000,2k,0
libero_10_0,1000,3k,1
libero_10_0,1000,9k,1
libero_10_0,1001,2k,1
libero_10_0,1001,3k,0
libero_10_0,1001,9k,1
```

## G1 representation panel

Each physical state has two rows, one per frozen checkpoint in the selected pair:

```text
state_id,task,seed,checkpoint,success,feature
```

`state_id` must uniquely identify the shared physical initial state. `feature` is the frozen predeclared hidden representation for that checkpoint/state.

For practical storage, features may be stored separately in an `.npz` file with arrays:

```text
state_id      [N]
checkpoint    [N]
success       [N]
feature       [N, D]
```

The two checkpoint rows belonging to the same `state_id` must always stay on the same side of any train/test split.

## What must not enter the primary analysis

Do not condition state inclusion on:

- probe score;
- hidden-state geometry;
- failure subtype;
- camera or visual perturbation response;
- hand-selected interesting trajectories.

The primary same-state panel is defined before representation analysis.
