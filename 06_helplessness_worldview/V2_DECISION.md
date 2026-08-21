# Topic 06 v2 decision

Date: 2026-08-21

The one permitted v2 acquisition gate was run with local cached
`Qwen/Qwen3-32B`, non-thinking inference, TP=4 across GPUs 0–3, intervention
cost `2`, 24 pairs per diversity, 128 client concurrency, temperature `0.7`,
and seed `20260821`. The four-cell design, yoking, families, rewards, test,
and endpoint were unchanged.

Technical gates passed:

- 4,416 rows / 96 sessions;
- invalid action rate: `0%` overall and test;
- yoke mismatch count: `0`;
- master success exposure gap: `0.3125pp`.

The cost change removed the v1 step-1 ceiling, but acquisition was not clearly
separated:

```text
late active:           C1 60.94% vs U1 58.85%  (+2.08pp)
                        C10 60.42% vs U10 55.73% (+4.69pp)

late effective action:  C1 33.85% vs U1 31.77%  (+2.08pp)
                        C10 35.42% vs U10 30.73% (+4.69pp)
```

The primary novel-test quantities were `H1=4.17pp`, `H10=0pp`, and
`D=-4.17pp`, with analyzer bootstrap interval `[-12.5pp, 0pp]`. The analyzer
labels this `BOUNDARY_RESULT`, but the preregistered v2 acquisition gate is
not met strongly enough to justify a larger pilot. Therefore do not run S2,
250-pair confirmation, model sweeps, probes, memory additions, or further
reward/environment redesign. Archive Topic 06 as an acquisition failure;
the natural question is unresolved rather than supported as a worldview.
