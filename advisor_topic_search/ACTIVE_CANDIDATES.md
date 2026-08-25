# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> 2026-08-25 起，所有候选必须先通过 README 新增的 **External-Construct / Model-Invariance Gate**。旧排名不受保护。

Last updated: 2026-08-25

---

# 0. Current interpretation

这次校正后，**暂时不把任何“某模型 / 某 benchmark 的怪行为”直接称为 TOP science**。

最重要的新规则：

> 研究问题必须先有模型外部定义；换模型后问题仍然成立。模型差异应成为 boundary condition / explanatory factor，而不是让题目消失。

因此 Round 08 的前三全部重新审级。Topic 28 随后的冻结全量 G0 已把
progressive truthful-clue reversal 从廉价 screen 提升为有大样本支持的真实实验对象；
它仍需 controlled order 和跨任务边界验证，不能提前写成一般论文结论。

---

# A. ACTIVE PRIORITY

## A1. Information Non-Monotonicity under Progressive Truthful Clues

**External construct:** accumulating truthful evidence 下的 information
monotonicity / belief updating stability。

**Observed object:** 冻结全量 G0 在 128 个 released response configs 上得到
`8,102/120,353` 个相邻 `correct -> wrong` event（`6.7319%`），覆盖 760 个问题、
93 个 AI configs；3,871 个 event 通过 strict-alias sanity。top config 只占
3.84%，50/50 official added-clue audit 全通过。

**Structure result:** corpus-IDF specificity quartile 的 reversal rate 从
4.35% 单调升至 8.35%；7.73% event 的 wrong prediction 被新增 clue 直接、首次
exact 引入；56.91% 后续恢复，43.09% 在 released trajectory 中未恢复。proper
noun / number 的泛化 anchor-override 解释不受支持。

**Current gate:** preregister clue multiset、gold、prompt、model 全固定的 order
intervention。只进入 behavioral analysis，不进入 activation patching。之后仍需
跨 task family / matched local-model reproduction 才能把 QuizBowl object 提升为一般
evidence-integration claim。

**Status:** `PRIORITY_1 / G0_OBJECT_ESTABLISHED / STRUCTURE_ANALYZED / CONTROLLED_ORDER_NEXT`.

---

# B. CONDITIONAL SURVIVOR

## B1. What Learning Signal Governs In-Context Association? Cue Competition / Kamin Blocking as a Diagnostic

**External construct:** associative learning 中的 cue competition；经典问题是学习是否只由共现驱动，还是依赖 prediction error / informativeness。

**为什么仍然自然：** 这个问题不依赖任何具体 LLM 或 benchmark；blocking / unblocking / overshadowing 是诊断 learning rule 的实验 manipulation，而不是研究对象本身。

**Reframed mother question:**

> Is in-context associative learning governed by mere co-occurrence, or does it exhibit prediction-error-sensitive cue competition?

**Model-generality requirement:** 不能只在一个模型上找 blocking。第一阶段可单模型 sanity，但正式晋级至少覆盖 2–3 个不同 family；模型间是否出现 blocking 本身应被解释为 learning-rule boundary condition。

**Frozen minimal design:**

```text
BLOCKING
Phase 1: A -> X
Phase 2: A+B -> X
Test:    B -> ?

MATCHED CONTROL
B-X exposure count matched
but no established competing predictor A -> X
```

**Status:** `CONDITIONAL_KEEP / EXTERNAL_CONSTRUCT_PASS / CROSS_MODEL_G0_REQUIRED`.

---

# C. DEMOTED FOR REFRAMING

## C1. Positional Imprinting of Parametric Knowledge

旧问题：第一次学习一个事实时所处位置，是否在后续 exposure 完全 equalize 后仍留下 imprint？

**External construct candidate:** learning-history / hysteresis versus final exposure statistics。

**New audit:** 比 Quiz Bowl 更有一般性，但仍可能依赖 autoregressive architecture、training recipe 或特定 synthetic setup。若只在一个 model family 出现，就不足以支撑一般 learning-history claim。

保留官方 artifact 与 frozen receipt，但暂不继续大实验。下一次晋级前必须明确：

- 为什么这是一般 learning-history question，而非 transformer positional quirk；
- 至少两个 architecture / model family 的确认设计；
- null 是否能区分 historical-path vs final-statistics explanation。

**Status:** `HOLD / EXTERNAL_CONSTRUCT_PLAUSIBLE / GENERALITY_GATE_PENDING`.

---

# D. SCIENTIFIC HOLD

## D1. Does Knowledge Arbitration Have a Training History?

**External construct:** evidence history / reliability history 对 source arbitration 的长期影响。

科学上仍可能自然，但 accessible official reproduction artifact 未验证，且训练-history manipulation 成本高。

**Status:** `SCIENCE_HOLD / ARTIFACT+PREREQUISITE_GATE`.

## D2. Parametric Encoding Specificity Across Input Structures

**External construct:** knowledge representation 的 format invariance / access specificity。

目前 seed 较弱（Findings）且 artifact / construct 尚未充分验证；容易滑向 prompt/format-specific behavior。

**Status:** `WATCH / NOT_ACTIVE`.

---

# E. SEARCH PRIORITY AFTER THE 2026-08-25 CORRECTION

下一轮不再以 `surprising LLM failure` 为主要检索入口。优先从以下外部对象找问题：

```text
semantic / lexical resource gaps
language structure / tokenization / morphology
representation compression / retrieval / storage trade-offs
language change / social-language phenomena
ontology / category / source-monitoring / classic cognition
old NLP tasks whose measurement is outdated in the LLM era
multilingual / low-resource semantic transfer
real system objectives with stable failure definitions
```

只有外部 construct 先站住，才进入模型实验、representation 或 mechanism。
