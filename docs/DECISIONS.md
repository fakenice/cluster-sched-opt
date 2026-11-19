# Decision log

本仓库关键口径与协议决定的记录。每条注明**仓内依据**（文件/代码位置）与可核对的日期
（= git 提交日期，`git log` 可逐条对上）；只收当时已留下书面或代码痕迹的决定。

## 2026-10-13（`25157e7` 建仓）

1. **全部结论只从 trace 驱动的离线仿真得出，且以可行性 oracle 为前提**：
   MILP vs FIFO/EASY backfill 对比全部在 capacity-feasible 的 replay 实例上做
   （`README.md` §设计；`docs/trace_provenance.md`）。
2. **trace 引用义务双写**：本仓代码/结果可单独引用，但复用 Alibaba PAI GPU trace
   必须同时引 Weng et al., NSDI '22（`README.md` §8 How to cite；2026-11-18 因新增
   §7 External baselines 而顺延编号）。
3. **idle 指标必须带分母**："固定窗口 GPU idle rate" 对任何策略都恰等于 `1 − ρ`，
   不能用于排序策略——据此改为三档并报
   （`gpu_idle_rate_fixed_window` / `gpu_idle_rate_busy_period` /
   `idle_with_backlog_card_hours`），头图与 headline delta 只用
   **idle-with-backlog**；这个决定有事故依据：第一版仓库画了五根一模一样的柱子，
   `scripts/06_backlog_idle.py` 因它而存在（`README.md` §口径条）。
4. **解失败不许变成排程**：`scipy.optimize.milp` 无 incumbent 时返回 None +
   `solver_feasible` 标志；离线档回退 capacity-aware first fit 且**显式标注**；
   `results/metrics.csv` 每行携带 `capacity_violation_slots` /
   `gpu_overcommit_slots` / `max_gpu_overcommit_cards`（`README.md` §口径条；
   `scripts/03_optimize.py` 逐策略打印可行性并拒绝隐藏违规）。

## 2026-11-03/04（`3a2862f`、`fe7dea6`、`bebe7a7`）

5. **排行榜改在 72 个窗口 × seeds 上评，而不是挑出来的 6 个**（`fe7dea6`）：
   选择效应本身成为被消除的对象。
6. **request-only 优先级基线（size/aging，无 solver）与优化器同场重测**（`3a2862f`；
   `scripts/07_priority_baselines.py`）：headline"一行在线规则捕获离线最优的
   77–114 %"因此有在线可部署性支撑。
7. **aging 阈值（1/2/4 h）按观察选定、不做留出调参**，并把这个上限写死在措辞里：
   它们证明"均值等待收益不是 solver 特有"，**不证明最优阈值**（`README.md` §结论条末）。
8. **可部署配方是 "size priority + aging + max-wait cap"，不是"跑 ILP"**（`README.md` 结论 4 区）。

## 2026-11-04（`bebe7a7` 主机级模型）

9. **碎片用主机级模型计价，不当假设抹掉**：一机一列、作业机器集合终身固定
   （`sum_m z[j,m,t] = h_j·x[j,t]`）；容器不迁移 ⇒ **时间一致的 83.0–95.7 % 与逐槽
   重定的 `packing_feasibility_rate`（14.5–93.5 %）是两个不同问题的答案，
   禁止互比**（`scripts/09_host_level.py`；`README.md` 结论 12）。
10. **host LP bound 与 aggregate LP bound 在六个实例上相等**（第 6 个差 1e-9 s，
    记为浮点噪声）⇒ 碎片的账是纯 integrality 效应，并以区间发表；
    aggregate 最优用 `gap=0` 重解，证不出的报 dual bound（`README.md` 结论 12；§口径条）。
11. **ILP gap 模糊只记录、不平均、不声称一致**：任何依赖 ILP 数字的主张自带这句
    fuzz 声明；重跑 ILP-flow 六行 5 复现、P100 ρ=0.85 落在 2.02 % 更好的 incumbent
    （两跑都撞墙钟），留痕 `results/host_level_ilp_check.csv`。

## 2026-11-07（`376fcea`）

12. **在源头修调度器：不伪造容量、拒绝隐藏**（提交信息即决定）。

## 2026-11-18（`80b7931`）

13. **书目改为逐条可验证**（`bibliography.bib` 全部经核，未核条目不进引用）；
    trace 出处独立成文档 `docs/trace_provenance.md`。

## 2026-11-19（留出实验，撤回 07 决定 7 的措辞上限）

14. **aging 阈值改为留出实测，不再只写"按观察选定"**（`scripts/10_aging_threshold.py`，
    `results/aging_holdout.csv/.json`）：0.5–6 h + 无 aging 的密网格在 72 个实例上重放
    576 次，6 个调参格与 66 个留出格分开计。回放有锚——FIFO 与 priority-gpu 两列与
    `results/sweep_metrics.csv` 共 144 格逐格相符（最大相对偏差 0.0）。决定 7 的"不证明最优
    阈值"由猜测变成测量结果，且是负向的：留出集上均值等待偏好**完全不做 aging**（1 h 的
    中位后悔 +9.9 %，66 格里 44 格输给无 aging），尾部最大等待才偏好 ~1 h（66 格里 42 格
    优于无 aging，符号检验 p < 0.001，中位尾部后悔 0.0 % 对 14.3 %），并且越激进的 aging
    错过截止率越高（0.230 → 0.329 → 0.439）。调参格本会选出的 3 h 只在 66 个留出格里 5 次
    命中逐格最优，所以"阈值随窗口而变"是结论的一部分，而不是待修的缺陷。
    决定 7 记录的历史选择保持不变，本条只替换它的措辞上限。
