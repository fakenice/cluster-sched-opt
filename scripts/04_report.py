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
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {RES/'table_comparison.md'}, fig_mean_wait.png, fig_gpu_idle.png, "
          f"fig_pareto.png, summary.json (idle metric: {idle_col})")


if __name__ == "__main__":
    main()
