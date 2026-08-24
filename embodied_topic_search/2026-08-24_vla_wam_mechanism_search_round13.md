# Round 13（2026-08-24）— E 通过 P0 / P0b，注册为 Topic 24

这一轮没有找新题。这一轮做的是：**把 Round 12 提出的 provisional E 的 instrument 打穿，然后才注册。**

结论：

```text
P0   replay fidelity     PASS
P0b  WBC seam liveness   PASS
  -> 注册为 Topic 24（24_hierarchical_feedback_attribution）
```

## 1. 这一轮想验证的不是 hypothesis，是 instrument

Round 12 里 E 的注册条件写得很明确：**不施加任何 push，先证明两个 replay 能复现原系统。**

跑的是：

```text
task     simple/G1WholebodyCloseDoorTeleop-v0
policy   Psi0 ckpt_40000
configs  dr-level-0 episodes 0-9（10 个）
cells    fresh / vla_replay / actuator_replay
force    0 N（全程没有任何扰动）
```

结果：

```text
fresh            10/10
vla_replay       10/10
actuator_replay  10/10
两个 gap         0.00
门角轨迹偏离      0.000 rad
终态 base 偏离     0.000 m
```

结构性检查（这些是 contract，不是可调阈值）全部成立：

- 每条 replay 行 `server_queries == 0` —— VLA 确实离开了闭环，这是记录下来的证据，不是声明；
- 每条 replay 行 `steps == tape_len`，没有 tape 提前耗尽；
- 每一行 `force_n == 0`。

两个 intervention 都是**上游数据路径本身**，不是重写：`vla_replay` 把录好的 tape 预填进
`SonicDecoupledWbcAgent._action_queue`，而 released agent 只在队列空时才去查 policy
server，所以 VLA 离开闭环而 `_build_wbc_observation` 每 tick 仍然读实时 proprioception。

## 2. P0 通过之后，发现 P0 证明的比它看起来少

这是这一轮真正的收获，而且它是 Topic 23 教训的同一个形状，只是这次对着我们自己：

> `max door deviation = 0.0` 这个“完美”同时意味着：**世界没被扰动时，两个 replay 都没有真正调用到任何 feedback。**

推论很不舒服：如果 WBC 的输出其实是 `vla_cmd` 的纯前馈函数，`vla_replay` 也会给出一模一样的
`0.0`，P0 也会以同样的 10/10 通过。而在那个世界里：

```text
S_vla_replay - S_actuator_replay
```

结构上恒为 0，不是一个 informative 的量。

所以 P0 证明的是 **管路无损**，不是 **两个 seam 承载的 feedback 不同**。后者需要自己的
gate。这就是 P0b。

## 3. P0b：纯 command-level 的 paired seam test

关键设计是**不推、不演化**，因为要测的是一个函数有没有对 observation 的依赖，不是在测 recovery。
一旦让 simulator 演化，就会混进 dynamics、contact、controller history、trajectory phase。

```text
同 config、同 tick、同一条固定 vla_cmd、
同 WBC internal state、同 clock
        |
canonical proprio  -> WBC -> target_q^0
perturbed proprio  -> WBC -> target_q^1
        |
D = |target_q^1 - target_q^0|
```

canonical 那一边**重新调用一次 live WBC**，而不是拿 perturbed 输出去减 tape 里录的
`target_q`。后者会混进 interpolation internal state、clock positioning、前一 tick history。
只有“同 state + 同 command，只换 observation”才是干净的。

冻结项（跑之前写死）：tick `= round(0.4 * len(tape))`，只由未扰动 canonical rollout 推出；
扰动是**单一** `+0.05 rad` body-frame roll offset，加在 floating-base 姿态和 torso IMU
四元数上；一个 magnitude，不扫。

gate 不是科学阈值，是结构性的：

```text
repeatability   同 observation + 同 command + 还原后的 state -> 差异在数值零
liveness        perturbed observation + 同 command -> 差异明显超过这个底
```

另外加了一个 restore probe：在 perturbed 那次**之后**再跑一遍 canonical。如果 state 还原不完整，
它就不会返回原向量，于是报 instrument failure，而不是报出一个 liveness 结论。

结果：

| config | tick | repeat floor | restore probe | D | joints changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| dr-level-0:0 | 110/276 | 0.0 | 0.0 | 4.56e-02 rad | 15 |
| dr-level-0:1 | 123/307 | 0.0 | 0.0 | 2.43e-02 rad | 15 |
| dr-level-0:2 | 112/281 | 0.0 | 0.0 | 3.93e-02 rad | 15 |

repeatability floor 是**精确的 0.0**，restore probe 也是精确 0.0。所以这个分离不是“相对噪声很大”，
是**根本没有噪声底**。

一处偏离照实报告：cfg2 在 tick 112 时门已漂 `7.2e-04 rad`，我实现的检查是精确相等所以判 false。
那是被动 settling，比任务需要的 ~0.95 rad 行程低三个量级。cfg0/cfg1 是精确 0.0，结论不靠 cfg2。

## 4. P0b 顺带钉死了 Topic 24 能声称什么

每个 config 变的都是**同样的 15 个 joint**：12 腿 + 3 腰。两条胳膊、两只手 —— 精确 0.0。

这不是巧合，源码里写着。`G1DecoupledWholeBodyPolicy.set_observation`：

> Upper body policy is open loop (just interpolation), so we don't need to set the observation

只有 `lower_body_policy` 收 observation。所以：

```text
vla_replay - actuator_replay
    只可能承载 locomotion / balance 层的 state feedback。
    结构上不可能包含 arm-level corrective reaching。
```

一个只能靠“重新伸手去够”吸收的扰动，在 WBC 层根本没有路径，那种 recovery 必然全部落进
`fresh - vla_replay`。这条已经写进 Topic 24 的 README，是在冻 G0 **之前**写的，不是出了数字之后
才“发现”的一个方便读法。

命名也在看到任何数字之前固定了：

```text
fresh - vla_replay             VLA-level online feedback contribution
vla_replay - actuator_replay   WBC / reference-generation feedback contribution
actuator_replay residual       servo + actuator dynamics + mechanics + task tolerance
```

不用 “low-level controller contribution”，因为 `actuator_replay` 下面仍然是闭环。

## 5. 一个 engineering confound，故意不并进题目

搭 instrument 时撞到：`Psi0DecoupledWbcAgent.get_action` 打的是
`target_time = time.monotonic() + 1/control_freq`，而 `InterpolationPolicy` 按**真实调用时刻**
采样样条。也就是说 model inference latency 本身就是 controller 的输入。当前 stack 严格来说不是
`a_t = f(o_t)`，而更像 `a_t = f(o_t, Δt_compute)`。

这个延迟在 `fresh` 里有、在两个 replay 里没有，放着不管会被误读成 replay fidelity 失败。所以
runner 用冻结的 nominal virtual control clock（每次 WBC 调用推进正好一个控制周期，三个 condition
统一）。

测出来的量级比预期狠：同一个 config `dr-level-0:0`，同 seed、同 checkpoint，在有并发负载的
real-clock run 里 fail，在负载较轻的 run 里 success，三个 condition 一致；virtual clock 下跨 run
逐 bit 复现。而且不是 graceful degradation —— 失败的 rollout 全部跑满 450 步，门一动没动。

处理方式：写进 P0_RESULTS 作为 engineering / system observation，**不作为 hypothesis evidence**。
并进去的话，一个干净的 attribution question 会立刻变成
“VLA latency + WBC clock + system scheduling + recovery attribution”，正好长回这题当初是为了避开的
那条控制链。

它未来可能自己长成一题（*are robot-policy benchmarks accidentally benchmarking compute latency?*），
但现在压住，不碰。

## 6. 这一轮沉淀的通用规则

写进了 [`../FAILURES_AND_LESSONS.md`](../FAILURES_AND_LESSONS.md) 第 15–18 条：

1. **从已验证的 seam 出发排序搜索，而不是从概念出发。** 09 / 19 / 23 都是先提二阶概念、后找承载物。
2. **先证明 instrument，而且要证明对的那件事。** undisturbed 条件下完美复现 ≠ 切开了任何 causal
   东西；seam liveness 是独立的 gate。
3. **按它实际切开的东西命名差值。** 并且检查被切那层能影响哪些输出通道。
4. **wall-clock coupling 是测量 confound，不是细节。**

## 7. 下一步

Topic 24 的 physical-disturbance G0 已冻结（30 configs、`{50,100,150} N × {left,right} × 0.2 s`
全网格上报、push timing 只由未扰动 canonical rollout 的 first object contact 提前 1 秒推出、三个
condition 同 450 步预算、force=0 控制列在 G0 代码路径下重新验证 replay fidelity）。

成本照实估：630 rollouts × ~220 s ≈ 38 GPU-hours。不是便宜实验，没有当便宜实验说。
