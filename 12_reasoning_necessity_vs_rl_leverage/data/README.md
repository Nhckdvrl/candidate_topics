# Locked external RL data

`qwen3_1p7b_table13_math.csv` is a transcription of **Table 13** from:

> Zijian Zhang et al., *Is One Layer Enough? Training A Single Transformer
> Layer Can Match Full-Parameter RL Training*, arXiv:2607.01232v2, 2026.

The table reports Qwen3-1.7B-Base scores after full GRPO and after independently
training each decoder layer on NuminaMath-CoT. The `c_math` column is the
paper's layer contribution on the unweighted average of MATH500, GSM8K,
OlympiadBench, and AMC.

Topic 12 treats this CSV as **immutable external evidence**. Do not edit the
numbers to fit our necessity sweep. If the paper publishes a corrected version,
add a new versioned CSV rather than replacing this file silently.

For Topic 12's **primary** statistic, the analyzer does not mix task supports:
it applies the paper's same `C(k)` formula to the MATH500+GSM8K columns only,
because those are exactly the tasks used for the necessity sweep. The published
four-task `c_math` column remains a locked robustness target.
