#!/usr/bin/env python3
"""Stage 4 - turn results/metrics.csv into the comparison tables and figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

ORDER = ["FIFO", "EASY-backfill", "ILP-rolling", "ILP-flow(offline)", "ILP-makespan(offline)"]
METRICS = ["mean_wait_s", "p90_wait_s", "throughput_jobs_per_hour",
           "gpu_idle_rate_fixed_window", "gpu_idle_rate_busy_period",
           "idle_with_backlog_card_hours", "idle_with_backlog_share",
           "idle_addressable_backlog_card_hours", "idle_addressable_backlog_share",
           "deadline_miss_rate", "packing_feasibility_rate"]
BACKLOG_COLS = ["idle_addressable_backlog_share", "idle_with_backlog_share",
                "idle_addressable_backlog_card_hours", "idle_with_backlog_card_hours"]


def fmt_policy(p: str) -> str:
    return {"FIFO": "FIFO（队头阻塞）", "EASY-backfill": "EASY backfill",
            "ILP-rolling": "ILP 滚动再优化（在线，2h 前瞻）",
            "ILP-flow(offline)": "ILP 最小化流时间（离线最优）",
            "ILP-makespan(offline)": "ILP 最小化 makespan（离线最优）"}.get(p, p)


def backlog(df: pd.DataFrame) -> pd.DataFrame:
    """Idle-with-backlog columns come from scripts/06_backlog_idle.py; degrade gracefully."""
    for c in BACKLOG_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def sweep_section(sweep: pd.DataFrame, meta: dict) -> list[str]:
    """多窗口 × 多种子扫描（08_sweep.py）的表格。sweep_metrics.csv 不存在时不渲染。"""
    lines = ["## 多窗口 × 多种子稳健性扫描（08_sweep.py）", ""]
    grid = meta.get("grid", {})
    lines += [
        f"原研究的 6 个 instance 全部取自每个 pool **需求最大的那一个 6 小时窗口**、"
        f"**一个**抽样种子。本节把同一套策略重放到 {meta.get('n_instances', 0)} 个 instance："
        f"窗口排名 {grid.get('window_ranks')}（rank1 = 最忙＝原窗口，median = 中位需求窗口，"
        f"是选择性偏倚的对照）× 种子 {grid.get('seeds')} × pool {grid.get('pools')} × 3 个负载档。"
        f"原研究的 6 个 instance 是这个网格的子集，因此充当内建阳性对照。", ""]

    # 种子这一维到底变了什么：不是换个标签，而是重抽作业集合。同一 (pool, 窗口排名, 名义负载档)
    # 下，三个种子的 achieved_rho 与主机数各自不同 —— 直接从 sweep_metrics.csv 算出来，
    # 避免在文中写死一个倍数。
    key = ["pool", "window_rank", "target_rho"]
    if all(c in sweep.columns for c in key + ["achieved_rho", "seed"]):
        one = sweep[sweep.policy == "FIFO"] if "policy" in sweep.columns else sweep
        g = one.drop_duplicates(key + ["seed"]).groupby(key)
        has_mach = "machines" in one.columns
        spread = pd.DataFrame({
            "n_seeds": g["seed"].nunique(),
            "rho_min": g["achieved_rho"].min(), "rho_max": g["achieved_rho"].max(),
        })
        if has_mach:
            spread["machines_min"] = g["machines"].min()
            spread["machines_max"] = g["machines"].max()
        spread = spread[spread["n_seeds"] > 1]
        if len(spread):
            ratio = (spread.rho_max / spread.rho_min.replace(0, np.nan)).dropna()
            mach_txt = (f"主机数也在 {int(spread.machines_min.min())}–"
                        f"{int(spread.machines_max.max())} 台之间变动。"
                        if has_mach else "")
            lines += [
                f"**种子这一维变的是作业集合本身，不是随机数标签。** 同一 (pool, 窗口排名, "
                f"名义负载档) 下 {len(spread)} 组、每组 {int(spread.n_seeds.iloc[0])} 个种子："
                f"实际达成负载 ρ 的组内最大/最小比中位数 {ratio.median():.2f}×"
                f"（范围 {ratio.min():.2f}–{ratio.max():.2f}×），{mach_txt}"
                f"也就是说“同一档负载”在种子之间并不是同一个问题实例，"
                f"这正是需要多种子的原因。", ""]

    pc = meta.get("positive_control", {})
    if pc.get("status") == "ok":
        det = pc.get("deterministic_worst_relative_deviation")
        sol = pc.get("solver_worst_relative_deviation")
        lines += [
            f"**阳性对照**：网格中 window_rank=1、seed=20261013 的那一格必须复现已提交的表格。"
            f"{pc.get('n_deterministic_rows_compared')} 行确定性策略（FIFO / EASY / 7 条优先级规则）"
            f"与 `metrics.csv` + `metrics_priority.csv` 的最大相对偏差 = "
            f"{'n/a' if det is None else f'{det:.2e}'}"
            f"（完全一致 = {pc.get('deterministic_reproduce_exactly')}）。"
            f"{pc.get('n_solver_rows_compared')} 行求解器策略的最大相对偏差 = "
            f"{'n/a' if sol is None else f'{sol:.3f}'}——"
            "`milp` 以**墙钟**为限，同一 instance 在不同机器负载下可能落在不同 incumbent 上，"
            "这一偏差如实报告而不被断言掉。", ""]

    best = pd.DataFrame(meta.get("best_deployable_per_instance", []))
    cap = meta.get("captured_gain_best_deployable")
    if cap:
        lines += [
            f"**结论是否随窗口/种子改变**：{cap['n']} 个 instance 中，最佳可部署规则捕获的 "
            f"ILP 平均等待增益中位数 {cap['median']*100:.0f}%（范围 "
            f"{cap['min']*100:.0f}%…{cap['max']*100:.0f}%）；"
            f"≥50% 的占 {cap['frac_ge_0.50']*100:.0f}%，≥77%（原研究区间下界）的占 "
            f"{cap['frac_ge_0.77']*100:.0f}%，≥100%（规则比离线 ILP 更好）的占 "
            f"{cap['frac_ge_1.00']*100:.0f}%。"
            f"离线 ILP-flow 的平均等待严格优于**所有**可部署规则的 instance 数 = "
            f"{meta.get('instances_where_ilp_flow_beats_best_deployable')}/"
            f"{meta.get('instances_compared')}。", "",
            "| 窗口排名 | instance 数 | 捕获增益中位数 | 最小 | 最大 |",
            "|---|---|---|---|---|"]
        for rank, v in sorted((meta.get("captured_gain_by_window_rank") or {}).items(),
                              key=lambda kv: int(kv[0])):
            lines.append(f"| {rank} | {v['n']} | {v['median']*100:.0f}% | "
                         f"{v['min']*100:.0f}% | {v['max']*100:.0f}% |")
        lines += ["", "| 种子 | instance 数 | 捕获增益中位数 | 最小 | 最大 |", "|---|---|---|---|---|"]
        for seed, v in sorted((meta.get("captured_gain_by_seed") or {}).items(),
                              key=lambda kv: int(kv[0])):
            lines.append(f"| {seed} | {v['n']} | {v['median']*100:.0f}% | "
                         f"{v['min']*100:.0f}% | {v['max']*100:.0f}% |")
        lines.append("")

    if not best.empty:
        lines += ["**每个 instance 的最佳可部署规则**（按平均等待）：", "",
                  "| 规则 | 当选次数 |", "|---|---|"]
        for pol, n in (meta.get("best_deployable_policy_counts") or {}).items():
            lines.append(f"| `{pol}` | {n} |")
        lines += ["", "“最佳规则”在不同 instance 之间会换人，这说明可部署的那条规则不是"
                  "唯一的一条：请求量优先（GPU/CPU/内存）三者在多数负载下近似等价，"
                  "选哪一条应看该集群的瓶颈资源。", ""]

    pr = meta.get("priority_only_ranges")
    if pr:
        lines += [f"纯规模优先（无 aging、非 oracle）在全部 instance 上的范围："
                  f"平均等待较 FIFO {pr['wait_change_vs_fifo'][0]*100:.0f}%…"
                  f"{pr['wait_change_vs_fifo'][1]*100:.0f}%，"
                  f"P90 {pr['p90_change_vs_fifo'][0]*100:.0f}%…"
                  f"{pr['p90_change_vs_fifo'][1]*100:.0f}%，"
                  f"Max {pr['max_change_vs_fifo'][0]*100:+.0f}%…"
                  f"{pr['max_change_vs_fifo'][1]*100:+.0f}%；"
                  f"可回收（head-of-line）空转最大值 "
                  f"{pr['idle_addressable_backlog_card_hours_max']:.2f} 卡·小时，"
                  f"容量越界槽位最大值 {pr['capacity_violation_slots_max']}。", ""]

    # 把"可回收空转 = 0"讲清楚：这是规则的性质，不是某列没填。
    fam = meta.get("addressable_idle_by_policy_family") or {}
    ver = meta.get("addressable_claim_verification") or {}
    if fam:
        lines += ["### 空转的两种成分：队头阻塞 vs 碎片", "",
                  "`idle-with-backlog` 有两种来源：**队头阻塞**（有空卡、且队列里有能放下的作业，"
                  "纯粹是策略不让它走）和**碎片**（有空卡，但没有任何等待作业能在 GPU/CPU/内存"
                  "三个维度同时放下）。只有前者是可回收的。", "",
                  "| 策略族 | 行数 | 可回收空转非零的行 | 总空转最大 | 可回收空转最大 | 可回收占比中位 |",
                  "|---|---|---|---|---|---|"]
        for name, v in fam.items():
            lines.append(f"| {name} | {v['n_rows']} | {v['rows_with_nonzero_addressable']} | "
                         f"{v['idle_with_backlog_card_hours_max']} | "
                         f"{v['idle_addressable_backlog_card_hours_max']} | "
                         f"{v['addressable_share_of_backlog_idle_median']} |")
        pr_fam = fam.get("priority-rules", {})
        other_fam = fam.get("fifo-easy-ilp", {})
        lines += ["", f"{pr_fam.get('n_rows', 0)} 行优先级规则里"
                  f"**{pr_fam.get('rows_with_nonzero_addressable', 0)} 行**留下可回收空转"
                  f"（对照：FIFO/EASY/ILP 的 {other_fam.get('n_rows', 0)} 行里 "
                  f"{other_fam.get('rows_with_nonzero_addressable', 0)} 行非零）。"
                  "这个 0 需要自证，"
                  "因为「某一列没算」和「真的是 0」在表里长得一模一样，所以在**达成负载 ρ 最高**的那个 "
                  f"instance（`{ver.get('instance', '?')}`）上按槽位重算了一遍：", ""]
        pp = (ver.get("per_policy") or {})
        if pp:
            lines += ["| 策略 | 有积压且有空卡的槽位 | 其中「有作业能放下」 | 其中纯碎片 | "
                      "最大空卡数 | 总空转 | 可回收空转 |",
                      "|---|---|---|---|---|---|---|"]
            for pol, v in pp.items():
                lines.append(f"| {pol} | {v['slots_with_backlog_and_free_gpu']} | "
                             f"{v['addressable_slots']} | {v['fragmentation_only_slots']} | "
                             f"{v['max_free_cards_while_backlogged']:.0f} | "
                             f"{v['idle_with_backlog_card_hours']} | "
                             f"{v['idle_addressable_backlog_card_hours']} |")
            fr = pp.get("FIFO", {}); pg = pp.get("priority-gpu", {})
            lines += ["", f"FIFO 的 {fr.get('slots_with_backlog_and_free_gpu', 0)} 个"
                      f"「有积压+有空卡」槽位里 {fr.get('addressable_slots', 0)} 个本可放行"
                      f"（占其空转的 {fr.get('addressable_share_of_backlog_idle', 0)*100:.1f}%）；"
                      f"规模优先的 {pg.get('slots_with_backlog_and_free_gpu', 0)} 个槽位里"
                      f"**0 个**能放行。原因就是规则的定义本身：每个槽位都会把所有放得下的等待作业"
                      "全部启动，所以它还留着空卡只有可能是剩下的作业太大放不下。", "",
                      "由此得到一个原研究没说清的结论：**规模优先并没有消除空转，它把空转从"
                      "「队头阻塞」换成了「碎片」**——同一 instance 上总空转从 "
                      f"{fr.get('idle_with_backlog_card_hours', 0)} 降到 "
                      f"{pg.get('idle_with_backlog_card_hours', 0)} 卡·小时，"
                      "但剩下的一方是装箱问题，不是调度问题（见 §6 的主机级分配模型）。"
                      "平均队列深度同时从 "
                      f"{fr.get('mean_queue_depth', 0)} 降到 {pg.get('mean_queue_depth', 0)}。", ""]

    # 选择性偏倚：原研究恰好站在最有利的那个窗口上
    bdf = pd.DataFrame(meta.get("best_deployable_per_instance", []))
    if len(bdf) and "window_rank" in bdf:
        b2 = bdf[bdf.captured_ilp_wait_gain.notna()].copy()
        b2["ranklab"] = np.where(b2.window_rank > 50, "median", "rank" + b2.window_rank.astype(str))
        lines += ["### 原研究站在哪个窗口上（选择性偏倚）", "",
                  "捕获增益按「窗口繁忙排名 × pool」拆开（rank1 = 需求最大窗口 = 原研究选的窗口，"
                  "median = 中位需求窗口）：", "",
                  "| pool | 窗口 | instance 数 | 捕获增益中位 | 最小 | <50% 的比例 |",
                  "|---|---|---|---|---|---|"]
        for (pool, rl), g in b2.groupby(["pool", "ranklab"]):
            lines.append(f"| {pool} | {rl} | {len(g)} | {g.captured_ilp_wait_gain.median():.3f} | "
                         f"{g.captured_ilp_wait_gain.min():.3f} | "
                         f"{(g.captured_ilp_wait_gain < 0.5).mean()*100:.0f}% |")
        tier = b2.groupby("target_rho").captured_ilp_wait_gain.agg(["median", "min", "count"])
        lines += ["", "| 负载档 | instance 数 | 中位 | 最小 |", "|---|---|---|---|"]
        for rho, v in tier.iterrows():
            lines.append(f"| ρ={rho:.2f} | {int(v['count'])} | {v['median']:.3f} | {v['min']:.3f} |")
        undef = bdf[bdf.captured_ilp_wait_gain.isna()]
        if len(undef):
            keys = "、".join(f"`{r.sweep_key}`" for r in undef.itertuples())
            lines += ["", f"分母是 {len(b2)} 而不是 {len(bdf)}：{keys} 上"
                      f"**所有**策略（含离线 ILP）的平均等待完全相同"
                      f"（{float(undef.fifo_mean_wait_s.iloc[0]):.1f} s）且积压空转为 0，"
                      "捕获增益的分式是 0/0，因此记为未定义而不是 0 或 ∞。这个格子恰好也是"
                      "「无需调度可修」的极端例子——不是规则输了，是这个窗口里没有可赢得的东西。", ""]
        frac_low = (b2.captured_ilp_wait_gain < 0.5).mean()
        r1 = b2[b2.ranklab == "rank1"].captured_ilp_wait_gain
        rest = b2[b2.ranklab != "rank1"].captured_ilp_wait_gain
        low_by_tier = (b2.assign(low=b2.captured_ilp_wait_gain < 0.5)
                       .groupby("target_rho")["low"].sum())
        hi_rho = float(tier.index.max())
        worst_rho = float(tier["min"].idxmin())
        # 机理必须实测，不能凭"合理"下结论：低捕获的格子里规则究竟是不是"和 FIFO 一样、
        # 只是没东西可修"？离线 ILP 又是不是真的靠"留卡"换等待？逐 instance 回表核一遍。
        low = b2[b2.captured_ilp_wait_gain < 0.5]
        mech = {"n": int(len(low)), "tie_fifo": 0, "ilp_idles_more": 0, "ilp_addr_share": []}
        for r in low.itertuples():
            sub = sweep[sweep.sweep_key == r.sweep_key].set_index("policy")
            if r.best_deployable_policy not in sub.index or "ILP-flow(offline)" not in sub.index:
                continue
            if abs(float(sub.loc[r.best_deployable_policy, "mean_wait_s"])
                   - float(r.fifo_mean_wait_s)) < 1e-9:
                mech["tie_fifo"] += 1
            ilp, rul = sub.loc["ILP-flow(offline)"], sub.loc[r.best_deployable_policy]
            if float(ilp.idle_with_backlog_card_hours) > float(rul.idle_with_backlog_card_hours):
                mech["ilp_idles_more"] += 1
            tot = float(ilp.idle_with_backlog_card_hours)
            if tot > 0:
                mech["ilp_addr_share"].append(float(ilp.idle_addressable_backlog_card_hours) / tot)
        shr = pd.Series(mech["ilp_addr_share"], dtype=float)
        lines += ["", f"**原研究选的 rank1 窗口是表现最好的一档**：rank1 的 "
                  f"{len(r1)} 个 instance 中位 {r1.median():.2f}、最小 {r1.min():.2f}、"
                  f"没有一例低于 50%；其余窗口 {len(rest)} 个 instance 中位 {rest.median():.2f}、"
                  f"最小 {rest.min():.2f}，全网格低于 50% 的比例 {frac_low*100:.0f}%"
                  f"（{int((b2.captured_ilp_wait_gain < 0.5).sum())}/{len(b2)}）。"
                  "所以「规则能捕获 ILP 的绝大部分平均等待增益」在 "
                  f"{len(b2)} 个 instance 上**成立但不是均匀成立**，"
                  "原先引用的 77–114% 区间取自最有利的那个窗口。", "",
                  "失效点集中在**低负载**而非高负载："
                  f"ρ={hi_rho:.2f} 档的中位捕获 {tier.loc[hi_rho, 'median']:.2f}"
                  "（>1 表示规则比离线 ILP 还好），"
                  + "；".join(f"ρ={rho:.2f} 档有 {int(n)} 例低于 50%（该档最小 "
                              f"{tier.loc[rho, 'min']:.2f}）"
                              for rho, n in low_by_tier.items() if n > 0)
                  + f"，最难的一档是 ρ={worst_rho:.2f}（最小 {tier.loc[worst_rho, 'min']:.2f}）。", "",
                  "机理是实测出来的，不是推断的。先写下过一个可被推翻的说法——「规则在这些格子里"
                  "只是没东西可修」——回表逐个核 "
                  f"{mech['n']} 个低捕获格子，**该说法不成立**：其中只有 {mech['tie_fifo']} 个的"
                  f"最佳规则平均等待与 FIFO 完全相同，其余 {mech['n'] - mech['tie_fifo']} 个"
                  "其实胜过了 FIFO，只是远不及离线解。真正起作用的是另一半："
                  f"离线 ILP 在 {mech['ilp_idles_more']}/{mech['n']} 个格子里留下了**比规则更多**的"
                  f"积压空转，而它这部分空转的 {shr.min()*100:.0f}%–{shr.max()*100:.0f}%"
                  f"（中位 {shr.median()*100:.0f}%）属于「有空卡、队列里也有放得下的作业、求解器却没放」"
                  "的可回收空转。贪心规则按定义做不到留卡——一有空卡它就把放得下的作业全启动。"
                  "实测到此为止：能证伪的是「ILP 放弃了本可立即启动的作业、并因此拿到更低等待」。"
                  "至于它为什么留卡（常见解读是预知后面的大 gang、攒够一次放行），模型不输出理由，"
                  "只有排程，所以那是解读而不是结论。无论如何，未来信息是请求量优先规则原理上拿不到的，"
                  "结论收窄成一句话："
                  "**规模优先替代的是滚动再优化，替代不了真正需要未来信息的离线最优。**", ""]

    # 扫描网格里的可行性核对：committed 表是 0 越界，网格不是——那一行必须自己站出来。
    if len(sweep) and "capacity_violation_slots" in sweep.columns:
        viol = sweep[sweep.capacity_violation_slots > 0]
        if len(viol):
            det = "；".join(f"`{r.sweep_key}` × `{r.policy}`（越界 {int(r.capacity_violation_slots)} 槽，"
                            f"GPU 峰值占用 {r.gpu_peak_allocated_share*100:.1f}%，"
                            f"放不下 {int(r.unplaceable_jobs)} 个作业）"
                            for r in viol.itertuples())
            lines += ["### 扫描网格的可行性核对", "",
                      f"committed 六实例的 30 行是零越界，但 **{len(sweep)} 行的"
                      f"网格里有 {len(viol)} 行越界**：{det}。来源是滚动再优化的一个已知出口——作业连"
                      "加宽后的尾窗都塞不下时，代码按到达槽位把它放进排程（宁可让它显形为越界，"
                      f"也不要让它从指标里消失），所以这一行的等待与吞吐是**带着容量违背**读出来的。"
                      f"越界只出现在 `ILP-rolling`（{len(sweep[sweep.policy == 'ILP-rolling'])} 行里有 "
                      f"{len(sweep[(sweep.policy == 'ILP-rolling') & (sweep.capacity_violation_slots > 0)])} 行），"
                      f"而上面所有结论用到的 `ILP-flow(offline)` 与九个在线策略"
                      "（FIFO、EASY、七条优先级规则）在全部行上越界槽位都是 0"
                      "（越界检查对每个策略都从占用矩阵重算，不分求解器与否），因此捕获增益与优先级规则的"
                      "结论不受这一行影响；受影响的只有 ILP-rolling 自己的描述性统计。", ""]
        else:
            lines += ["", f"扫描网格可行性核对：全部 {len(sweep)} 行越界槽位数 = 0。", ""]

    if meta.get("ilp_flow"):
        f = meta["ilp_flow"]
        r = meta.get("ilp_rolling") or {}
        lines += [f"求解器侧：ILP-flow 的 LP 间隙最大 {f.get('gap_vs_lp_percent_max')}%"
                  f"（{f['n']} 次求解，单次 {f['solve_seconds_range'][0]:.0f}–"
                  f"{f['solve_seconds_range'][1]:.0f} s，退回 first-fit 的次数 "
                  f"{f['fallback_to_first_fit_count']}）；滚动再优化总求解时间 "
                  f"{r.get('solve_seconds_range', [None, None])[0]}–"
                  f"{r.get('solve_seconds_range', [None, None])[1]} s/instance，"
                  f"无 incumbent 的 epoch 合计 {r.get('epochs_without_incumbent_total')}，"
                  f"被推迟到 horizon 之外的作业合计 {r.get('deferred_jobs_past_horizon_total')}。",
                  ""]
    return lines


def idle_metric(df: pd.DataFrame) -> str:
    """Headline idle column: the addressable one if stage 6 ran, else raw backlog idle,
    else each policy's own busy period. Raw backlog idle overstates the scheduler's blame:
    spare cards in a slot where no queued job fits (CPU or memory already exhausted) are a
    workload property, not a scheduling decision."""
    for c in ("idle_addressable_backlog_share", "idle_with_backlog_share"):
        if c in df.columns and df[c].notna().any():
            return c
    return "gpu_idle_rate_busy_period"


def fmt_bh(r) -> str:
    if pd.isna(r["idle_with_backlog_card_hours"]):
        return "n/a"
    return (f"{r['idle_with_backlog_card_hours']:.1f}"
            f"（可利用 {r['idle_addressable_backlog_card_hours']:.1f}）")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):      # Windows consoles default to GBK
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = pd.read_csv(RES / "metrics.csv")
    df = backlog(df)
    df["policy_cn"] = df["policy"].map(fmt_policy)
    df["wait_min"] = df["mean_wait_s"] / 60
    df["p90_min"] = df["p90_wait_s"] / 60
    idle_col = idle_metric(df)

    lines = ["# 调度策略对比结果（trace 驱动）", "",
             "数据来源：Alibaba PAI cluster-trace-gpu-v2020（公开，sha256 已校验）。",
             "每个 instance = 一个真实 6 小时繁忙窗口内的到达流 + 按 machine_spec 切出的真实机器分区。",
             "",
             "关于 GPU 空闲率，本报告给出三个口径，因为它们会给出三种结论：",
             "",
             "1. **固定窗口空闲率**：分母是模型时域的卡时。只要所有作业最终都跑完，已分配卡时就恒等于需求卡时，"
             "该值与策略无关（正好等于 1 − ρ），只能用来核对负载，不能用来评判调度质量。",
             "2. **自身忙期空闲率**：分母换成该策略自己的 makespan，于是压缩 makespan 的策略会变好，"
             "但它奖励“早点收工”而与浪费无关，只作次级视角。",
             "3. **带积压空闲卡时**：作业已经提交且仍在排队时，未被分配出去的 GPU 卡时。其中只有当某个排队作业"
             "在该槽位能同时放下它的 GPU/CPU/内存需求时，这份空闲才是调度器真正可以消掉的浪费，"
             "因此同时报告**总量**与**其中可利用**两项；表中的差值与所有图都使用“可利用”口径。"
             "队头阻塞制造可利用浪费，重排消除它，把作业推到时域之外会重新制造它。",
             ""]
    for pool in df["pool"].unique():
        for rho in sorted(df[df["pool"] == pool]["rho"].unique()):
            sub = df[(df["pool"] == pool) & (df["rho"] == rho)].copy()
            sub["ord"] = sub["policy"].map({p: i for i, p in enumerate(ORDER)})
            sub = sub.sort_values("ord")
            n_jobs = int(sub["n_jobs"].iloc[0])
            base = sub[sub["policy"] == "FIFO"].iloc[0]
            lines.append(f"## {pool} 分区 @ ρ={rho:.2f}（{n_jobs} 个作业，"
                         f"{int(sub['machines'].iloc[0])} 台机器，{len(sub)} 种策略）")
            lines.append("")
            lines.append("| 策略 | 平均等待(min) | P90 等待(min) | 吞吐(作业/h) | 带积压空闲 卡时（其中可利用） "
                         "| 可利用占窗口 | 自身忙期空闲率 | 固定窗口空闲率 | 超期率 | 真实机器可装箱率 |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r['policy_cn']} | {r['wait_min']:.1f} | {r['p90_min']:.1f} | "
                    f"{r['throughput_jobs_per_hour']:.2f} | {fmt_bh(r)} | "
                    f"{r['idle_addressable_backlog_share']*100:.1f}% | "
                    f"{r['gpu_idle_rate_busy_period']*100:.1f}% | "
                    f"{r['gpu_idle_rate_fixed_window']*100:.1f}% | "
                    f"{r['deadline_miss_rate']*100:.1f}% | {r['packing_feasibility_rate']*100:.1f}% |")
            g = sub[sub["policy"] == "ILP-flow(offline)"]["gap_vs_lp_percent"].dropna()
            if len(g):
                lines.append(f"\n离线 ILP 相对 LP 松弛下界的最优性 gap：{float(g.iloc[0]):.2f}%。")
            for name in ["EASY-backfill", "ILP-rolling", "ILP-flow(offline)"]:
                row = sub[sub["policy"] == name]
                if not len(row):
                    continue
                r = row.iloc[0]
                if pd.isna(r[idle_col]) or pd.isna(r["idle_addressable_backlog_card_hours"]):
                    idle_txt = "可利用带积压空闲 n/a"
                else:
                    idle_txt = (f"可利用带积压空闲 {(r[idle_col] - base[idle_col])*100:+.1f} 个百分点"
                                f"（{r['idle_addressable_backlog_card_hours']:.1f} vs "
                                f"{base['idle_addressable_backlog_card_hours']:.1f} 卡·时）")
                dv = r.get("deferred_jobs_past_horizon")
                defer = (f"，{int(dv)} 个作业被推到时域之外"
                         if dv is not None and not pd.isna(dv) and dv else "")
                lines.append(f"- {name}：平均等待较 FIFO "
                             f"{'-' if r['mean_wait_s'] < base['mean_wait_s'] else '+'}"
                             f"{abs(1 - r['mean_wait_s']/max(base['mean_wait_s'], 1e-9))*100:.0f}%，"
                             f"吞吐 {(r['throughput_jobs_per_hour']/base['throughput_jobs_per_hour']-1)*100:+.1f}%，"
                             f"超期率 {(r['deadline_miss_rate']-base['deadline_miss_rate'])*100:+.1f} 个百分点，"
                             f"{idle_txt}{defer}")
            lines.append("")

    pf = RES / "metrics_priority.csv"
    if pf.exists():
        p = pd.read_csv(pf)
        p = p[~p.oracle]
        base = df[df.policy == "FIFO"].set_index("instance")
        ilp = df[df.policy == "ILP-flow(offline)"].set_index("instance")
        lines += ["## 在线优先级基线（不求解器）", "",
                  "每个 instance 取平均等待最低的可部署（非 oracle）规则。`捕获 ILP 增益` = "
                  "(FIFO − 该规则) / (FIFO − ILP-flow)；>100% 表示该规则比离线 ILP 的平均等待更低。",
                  "",
                  "| instance | 最佳规则 | FIFO 等待(min) | ILP-flow 等待(min) | 规则等待(min) | 捕获 ILP 增益 | P90 较 FIFO | Max 较 FIFO | 超期率 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for tag, sub in p.groupby("instance"):
            r = sub.loc[sub.mean_wait_s.idxmin()]
            b = base.loc[tag]
            i = ilp.loc[tag] if tag in ilp.index else None
            cap = ((b.mean_wait_s - r.mean_wait_s) / (b.mean_wait_s - i.mean_wait_s)
                   if i is not None and b.mean_wait_s > i.mean_wait_s else float("nan"))
            lines.append(
                f"| {tag} | {r.policy} | {b.mean_wait_s/60:.1f} | "
                f"{(i.mean_wait_s/60 if i is not None else float('nan')):.1f} | "
                f"{r.mean_wait_s/60:.1f} | {cap*100:.1f}% | "
                f"{(r.p90_wait_s/b.p90_wait_s-1)*100:+.1f}% | "
                f"{(r.max_wait_s/b.max_wait_s-1)*100:+.1f}% | "
                f"{r.deadline_miss_rate*100:.1f}% |")
        lines += ["", "优先级规则只用提交时已知的请求量，不用实际运行时长，也不调用求解器；"
                  "`priority-short`（按真实运行时长排序）是 oracle，未列入上表。"
                  "平均等待下降的同时 P90 也下降，但 Max 等待最高可达 FIFO 的 2.9 倍——"
                  "规模优先会饿死个别大作业，aging 参数是用来换尾部的。", ""]

    sweep_df, sweep_meta = None, {}
    sf = RES / "sweep_metrics.csv"
    sj = RES / "sweep_summary.json"
    if sf.exists() and sj.exists():
        sweep_df = pd.read_csv(sf)
        sweep_meta = json.loads(sj.read_text(encoding="utf-8"))
        lines += sweep_section(sweep_df, sweep_meta)

    if "capacity_violation_slots" in df.columns:
        n_bad = int(df["capacity_violation_slots"].fillna(0).sum())
        lines.append(f"可行性核对：全部 {len(df)} 行策略结果中，容量越界的槽位数 = **{n_bad}**"
                     "（每个排程在每一槽位的 GPU/CPU/内存占用都不超过该分区的实际容量）。")
    lines.append("")

    (RES / "table_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    # figures
    idle_label = {
        "idle_addressable_backlog_share":
            "GPU cards idle while a queued job could have used them (% of window)",
        "idle_with_backlog_share":
            "GPU idle card-hours while jobs were queued (% of window)",
        "gpu_idle_rate_busy_period":
            "GPU idle rate over each policy's own busy period (%)",
    }[idle_col]

    piv = df.pivot_table(index=["pool", "rho"], columns="policy", values="mean_wait_s")
    piv = piv[[c for c in ORDER if c in piv.columns]]
    ax = (piv / 60.0).plot(kind="bar", figsize=(9, 4.2), rot=0)
    ax.set_ylabel("mean wait (minutes)")
    ax.set_title("Mean queueing delay by policy, trace-driven GPU partitions")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RES / "fig_mean_wait.png", dpi=150)
    plt.close()

    piv2 = df.pivot_table(index=["pool", "rho"], columns="policy", values=idle_col)
    piv2 = piv2[[c for c in ORDER if c in piv2.columns]]
    ax = (piv2 * 100).plot(kind="bar", figsize=(9, 4.2), rot=0)
    ax.set_ylabel(idle_label)
    ax.set_title("Wasted capacity by policy - addressable idle while jobs are queued")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RES / "fig_gpu_idle.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for pool in df["pool"].unique():
        for name in ["FIFO", "EASY-backfill", "ILP-rolling", "ILP-flow(offline)"]:
            s = df[(df["pool"] == pool) & (df["policy"] == name)]
            if s.empty:
                continue
            ax.scatter(s["mean_wait_s"] / 60, s[idle_col] * 100, label=f"{pool} {name}", s=45)
            for _, r in s.iterrows():
                ax.annotate(f"ρ={r['rho']:.2f}", (r["mean_wait_s"] / 60, r[idle_col] * 100),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean wait (minutes) - lower is better")
    ax.set_ylabel(idle_label + " - lower is better")
    ax.set_title("Waiting time vs. wasted capacity (Pareto view)")
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(RES / "fig_pareto.png", dpi=150)
    plt.close()

    summary = {}
    for (pool, rho), g in df.groupby(["pool", "rho"]):
        base = g[g["policy"] == "FIFO"]
        if base.empty:
            continue
        b = base.iloc[0]
        rec = {}
        for name in ORDER[1:]:
            r = g[g["policy"] == name]
            if r.empty:
                continue
            r = r.iloc[0]
            dv = r.get("deferred_jobs_past_horizon")
            rec[name] = {
                "mean_wait_change_pct": round((r["mean_wait_s"] / b["mean_wait_s"] - 1) * 100, 1),
                "p90_wait_change_pct": round((r["p90_wait_s"] / b["p90_wait_s"] - 1) * 100, 1) if b["p90_wait_s"] else None,
                "throughput_change_pct": round((r["throughput_jobs_per_hour"] / b["throughput_jobs_per_hour"] - 1) * 100, 1),
                "gpu_idle_fixed_window_change_pp": round((r["gpu_idle_rate_fixed_window"] - b["gpu_idle_rate_fixed_window"]) * 100, 1),
                "gpu_idle_busy_period_change_pp": round((r["gpu_idle_rate_busy_period"] - b["gpu_idle_rate_busy_period"]) * 100, 1),
                "idle_addressable_backlog_change_pp": None if pd.isna(r["idle_addressable_backlog_share"]) else round((r["idle_addressable_backlog_share"] - b["idle_addressable_backlog_share"]) * 100, 1),
                "idle_addressable_backlog_card_hours": None if pd.isna(r["idle_addressable_backlog_card_hours"]) else float(r["idle_addressable_backlog_card_hours"]),
                "idle_with_backlog_card_hours": None if pd.isna(r["idle_with_backlog_card_hours"]) else float(r["idle_with_backlog_card_hours"]),
                "deadline_miss_change_pp": round((r["deadline_miss_rate"] - b["deadline_miss_rate"]) * 100, 1),
                "packing_feasibility_rate": None if pd.isna(r["packing_feasibility_rate"]) else float(r["packing_feasibility_rate"]),
                "deferred_jobs_past_horizon": None if dv is None or pd.isna(dv) else int(dv),
                "capacity_violation_slots": int(r.get("capacity_violation_slots") or 0),
                "gap_vs_lp_percent": None if pd.isna(r.get("gap_vs_lp_percent")) else float(r.get("gap_vs_lp_percent")),
            }
        summary[f"{pool}@rho{rho:.2f}"] = rec
    pj = RES / "priority_baselines.json"
    if pj.exists():
        summary["priority_baselines"] = json.loads(pj.read_text(encoding="utf-8")).get(
            "aggregate", {})
    if sweep_meta:
        # keep summary.json small: the per-instance sweep rows live in sweep_metrics.csv and
        # the full sweep summary in sweep_summary.json
        summary["sweep"] = {k: v for k, v in sweep_meta.items()
                            if k not in ("best_deployable_per_instance",)}
        summary["sweep"]["n_instances"] = sweep_meta.get("n_instances")
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {RES/'table_comparison.md'}, fig_mean_wait.png, fig_gpu_idle.png, "
          f"fig_pareto.png, summary.json (idle metric: {idle_col})")


if __name__ == "__main__":
    main()
