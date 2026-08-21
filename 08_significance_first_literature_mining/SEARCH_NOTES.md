# Search Notes

## Pivot: from anomaly explanation to phenomenon discovery

The primary mining target is **not** merely an unexplained anomaly in a recent paper.

We prefer a stronger pattern:

```text
established law / robust phenomenon in one literature
+ a new AI experimental axis that cleanly isolates a previously inaccessible variable
-> a concrete new phenomenon prediction
```

The goal is to discover and establish new empirical regularities ourselves, not only explain other papers' odd results.

### Preferred phenomenon-generation templates

1. **Axis transplant**
   - A phenomenon is established over human developmental time / repeated practice / memory delay / resource limits.
   - Modern AI provides a cleaner axis such as dense training checkpoints, denoising time, exact memory capacity, or controlled post-training.
   - Ask whether the same qualitative law holds, reverses, or breaks under the new axis.

2. **Factor completion**
   - Literature has a robust effect of X and a theoretically coupled variable Y.
   - Existing work varies X but leaves Y fixed.
   - Modern models let us independently manipulate X and Y.
   - Predict a simple interaction, reversal, threshold, or invariance.

3. **Structural law**
   - Start from a natural computational structure: dependency depth, compositional depth, interference load, task diversity, controllability, memory capacity, etc.
   - Derive a qualitative prediction such as monotonicity, ordering, scaling, or phase transition.
   - Test that prediction directly; no hidden-state construct is needed for the primary phenomenon.

4. **Cross-system dissociation**
   - Two learning systems solve the same task but have qualitatively different computational constraints.
   - Predict a behavioral law that should differ because of that constraint.
   - The contribution is the new dissociation itself, not an after-the-fact mechanism story.

### Anti-patterns

Reject if the proposal is primarily:

- `paper A found weird thing -> we explain it with mechanism B`;
- `model may contain hidden X -> probe X`;
- `A and B may stabilize at different times` without independent reason to expect a meaningful dissociation;
- a classic psychology effect simply rerun on an LLM;
- a phenomenon whose existence depends on searching many layers / thresholds / prompts;
- a result that would be unsurprising even if perfectly confirmed.

### Required pre-experiment questions

Before registering a candidate:

1. What concrete phenomenon are we predicting?
2. Why should it exist, based on prior literature rather than intuition alone?
3. Why is it not already obvious from the premises?
4. If confirmed, what broader understanding changes?
5. Can the primary phenomenon be observed with one clean contrast or one simple curve?
6. Can a strong null result cleanly kill the proposed phenomenon without requiring mechanism rescue?

