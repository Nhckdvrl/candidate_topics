# Topic 12 archive summary

## Final status

**ARCHIVED — valid frozen G-0 landed in `INCONCLUSIVE_DO_NOT_TUNE`; no G-1 and no model/metric rescue.**

Topic 12 asked whether the transformer layers most causally necessary for existing mathematical competence are also the layers where isolated RL updates have the greatest causal adaptation leverage.

The experiment was technically valid and the functional-necessity profile itself was highly reproducible across MATH500 and GSM8K. However, the relation to the published single-layer-RL leverage curve was only a moderate raw depth correlation and disappeared after the preregistered broad-depth control. The specific layer-by-layer peaks and troughs did not align.

The correct conclusion is therefore not that the two quantities are unrelated in every sense, but that **functional necessity did not predict fine-grained causal RL adaptation leverage strongly enough to support the proposed project-level claim.**

No G-1 is justified under the frozen protocol.

## Scientific question

> Are the transformer layers that are required for mathematical competence also the layers in which isolated RL updates most efficiently improve that competence?

The project compared two different causal quantities on the same `Qwen3-1.7B-Base` model:

- `I_l`: functional necessity, measured as `P(ablated wrong | baseline correct)` after bypassing layer `l`;
- `C_l`: causal RL adaptation leverage, taken from the published complete single-layer GRPO sweep and recomputed on the same MATH500+GSM8K task support.

The important distinction was between:

```text
broad depth organization
and
fine-grained layer-level correspondence
```

A result where both curves are generally larger in similar depth regions was not enough. The project required the same individual layers to line up beyond that coarse depth trend.

## Frozen G-0 protocol

Model and evaluation contract:

```text
model               Qwen/Qwen3-1.7B-Base
revision            912d2727784ca0a6f718845aa14d4d9e5f48fe26
layers              28
tasks               MATH500 + GSM8K
examples/task       256
seed                 20260822
decoding             greedy
max input            2048
max new tokens       1536
intervention         residual_scale = 0.0
GPUs                 4 independent workers
engineering batch    128
```

Integrity passed:

- `28/28` layers present;
- no input truncation;
- frozen run contract matched across conditions;
- Math-Verify fallback after the grader repair was `0%`;
- baseline scores were compatible with the published Table-13 base model.

Observed baseline:

```text
GSM8K   78.1%
MATH500 58.6%
```

Published Table-13 base scores were `74.4%` and `57.4%`, respectively, and both differences were inside the predeclared compatibility bounds.

## Grader bug and why the repair was legitimate

During the run, a real evaluator bug was found for MATH500 answers stored as bare expressions such as `(-1,6)`.

`Math-Verify` can return an empty parse when such a dataset gold string is passed directly, even though the expression is valid. The repair changed **only gold serialization**: if the raw gold failed to parse, it was normalized as a boxed expression before parsing. Model responses were left untouched.

Frozen generated responses were then regraded; no inference was rerun and no scientific parameter changed. A regression test was added.

The repair reduced the apparent MATH500 fallback rate from about `8.98%` to `0%` and restored the intended grader semantics. This is an engineering/measurement correction, not outcome-dependent metric shopping.

## Decisive result

Primary locked statistics:

```text
Spearman rho(I, C)                         0.355
paired bootstrap 90% CI                  [0.300, 0.402]
Kendall tau                                0.225
rho after removing quadratic depth trend -0.238
partial-rank depth diagnostic             -0.093
circular-shift p                           0.071
top-5 overlap                              1
random top-5 overlap expectation           0.89
MATH500 vs GSM8K necessity-profile rho     0.878
```

Task-specific relations were also only modest:

```text
GSM8K   rho(I_task, C_task) = 0.293
MATH500 rho(I_task, C_task) = 0.279
```

Locked robustness checks agreed with the same interpretation:

```text
legacy net-accuracy-drop I: rho = 0.358
published four-task C_math: rho = 0.315
```

## Why this is not an underpowered null

The strongest evidence that the measurement worked is the cross-task necessity agreement:

```text
rho(I_MATH500, I_GSM8K) = 0.878
```

So the functional-necessity profile was not merely noise from 256 examples per task. The model displayed a stable layer-importance structure across two math benchmarks.

What failed was the **mapping from that stable structure to RL adaptation leverage**.

The raw `rho=0.355` is real enough to suggest some shared coarse depth organization, but the preregistered fine-grained test changed sign after removing the broad quadratic depth trend:

```text
raw rho            +0.355
depth-residual rho -0.238
```

The top-5 overlap was also only `1`, essentially the random expectation. This is exactly the pattern the depth-control gate was designed to distinguish from a genuine layer-level law.

## What the full curve says

Several local mismatches make the conclusion concrete.

Layers around `9–12` are among the strongest published RL-leverage layers, but their functional necessity is only moderate rather than uniquely high. Conversely, layers around `20–21` show high functional necessity while published RL leverage is already low.

Therefore the result is not well summarized as:

> important layers learn more.

A more accurate description is:

> existing mathematical competence and isolated RL plasticity share some broad depth structure, but the layers most necessary for using the competence are not the same individual layers that are most effective adaptation sites.

That observation is interesting as a negative, but in this one-model G-0 it is not clean or surprising enough to carry a full project.

## Intervention pathology and why G-1 was not run

Hard layer bypass caused substantial generation pathology for some individual layers. For example, several layers showed high output truncation, and necessity correlated with parser/truncation damage:

```text
rho(necessity, parser fallback) = 0.586
rho(necessity, truncation)      = 0.515
max parser fallback             = 25.6%
max truncation                  = 98.4%
```

These outcomes were deliberately retained rather than filtered because intervention-induced generation failure is causal damage.

However, the predeclared destructive-intervention criteria were **not** crossed globally; the run was classified `INFORMATIVE`, not `INCONCLUSIVE_INTERVENTION`.

Therefore the frozen protocol did not authorize an `alpha=0.5` sweep merely because the substantive result was unattractive. Running G-1 now would turn the predeclared mild intervention into a post-hoc search for a more favorable measurement.

## Why 4B was not used as a rescue

A larger Qwen model could technically support the same analysis if a complete published single-layer-RL curve is available. But after a valid frozen 1.7B discovery landed in the no-tune region, switching to another model to search for a positive correspondence would violate the candidate-selection logic.

Independent-model replication is appropriate after a compelling discovery. It is not an appropriate way to turn a gray-zone discovery into a project.

## Why the project stops here

This topic did several things correctly:

- the scientific question was natural;
- the compared quantities were causal and distinct;
- the complete layer profiles avoided cherry-picked layer subsets;
- task support was matched;
- measurement integrity was checked before interpretation;
- the necessity profile showed strong cross-task reproducibility;
- the broad-depth confound was explicitly separated from fine-grained correspondence.

The negative decision therefore carries information.

What did not materialize was the strong result that would make the project worth pursuing: a clear layer-level law relating computation necessity to adaptation leverage, or a clean enough opposite/dissociation pattern to support a new architectural principle.

The frozen verdict is:

```text
INCONCLUSIVE_DO_NOT_TUNE
```

For candidate selection, that is a stop condition.

## Transferable lessons

1. **Two stable structures do not imply a meaningful mapping between them.** Functional necessity was highly reproducible across tasks, yet it did not align at fine layer resolution with RL leverage.
2. **Control the obvious global geometry before interpreting correspondence.** If two layer-wise quantities both vary strongly with depth, raw correlation can overstate a specific mechanistic relationship. Compare local deviations after removing the predeclared broad depth trend.
3. **Complete-profile comparisons are efficient G-0s.** When a theory predicts that two layer-wise properties correspond, compare the entire profiles before investing in selected layers, mechanisms, or retraining.
4. **A valid moderate correlation can still be too weak for a project.** Statistical evidence that `rho > 0` is not the same as evidence for the stronger scientific law the paper would need.
5. **A reproducible target measurement makes a negative more informative.** The `0.878` cross-task necessity correlation rules out the easiest claim that the mismatch arose because `I_l` itself was too noisy.
6. **Do not use a second model as a lottery ticket.** Cross-model replication should validate a strong discovered effect, not search for a model where an inconclusive relation becomes publishable.
7. **Measurement repairs are allowed when the bug is objective and outcome-independent.** Fix the evaluator, regrade frozen outputs, add a regression test, and keep the scientific contract unchanged.
8. **Generation fragility and task-specific computation are not identical.** Hard ablations can reveal genuine causal dependence while still mixing reasoning damage with generic formatting/termination damage; do not overclaim a clean mechanistic dissociation from such a measurement.

## Reopen condition

Do not reopen Topic 12 by trying:

- another layer subset;
- another weighting of MATH500/GSM8K;
- another correlation statistic;
- `alpha=0.5` solely because `alpha=0` was inconclusive;
- Qwen3-4B or another model solely to search for a positive result;
- another definition of layer importance chosen after seeing the current curves.

Reopen only if an independent external result creates a genuinely new scientific premise — for example, a strong theoretical or empirical reason to predict a specific necessity/plasticity relationship different from the one tested here. That should be registered as a new question rather than treated as a rescue of Topic 12.

## Preserved evidence

The full frozen G-0 outputs are kept under:

`results/g0_qwen3_1p7b_bypass_gold_normalized_bs128/`

Key files:

- `REPORT.md`
- `integrity_report.json`
- `relation_metrics.json`
- `layer_relation.csv`
- `relation.png`

The validation code and G-1 launcher are preserved for reproducibility, but the archive decision explicitly closes further Topic-12 experimentation under the existing hypothesis.