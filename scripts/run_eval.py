# -*- coding: utf-8 -*-
"""
支撑/阻力位识别 —— 走查回测评测入口

用法:
    python scripts/run_eval.py --quick          # 20 只股票冒烟测试
    python scripts/run_eval.py                  # 200 只股票完整评测
    python scripts/run_eval.py --n 400 --out out/eval400

输出:
    <out>/levels.parquet     逐关键位明细（每行 = 一个关键位在一个决策点的前瞻结果）
    <out>/geo.parquet        逐决策点几何精度
    <out>/report.txt         汇总报告（人可读）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.srlab.data import build_universe
from src.srlab.detectors import (
    FixedPlacebo, ShiftPlacebo, V1Density, V2Fusion, V2Params,
    V3Fusion, V3Params, ablate,
)
from src.srlab.metrics import (
    IsotonicCalibrator, backtest_touch_strategy, brier_score,
    calibration_table, compare, compare_distance_neutral, ece,
    summarize_by_kind, wilson,
)
from src.srlab.walkforward import run_walkforward, summarize_geo

ASHARE_DIR = r"D:\A股_K线数据\parquet\stocks"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def build_detectors(with_ablation: bool = False):
    v2 = V2Fusion()
    v3 = V3Fusion()
    dets = [
        V1Density(window=40, n_zones=6),                           # 现网默认
        V1Density(window=120, n_zones=6, name="V1_density_w120"),   # 放大窗口的 V1
        v2,
        v3,
        # 每个候选算法都配自己的平移对照，避免拿 A 的对照去比 B
        ShiftPlacebo(base=v2, name="placebo_shift_v2"),
        ShiftPlacebo(base=v3, name="placebo_shift_v3"),
        ShiftPlacebo(base=V1Density(window=40, n_zones=6), name="placebo_shift_v1"),
        FixedPlacebo(),
    ]
    if with_ablation:
        dets += [ablate(v2, k) for k in ("pivot", "hold", "vp", "close", "round", "recency")]
    return dets, v3


def section(f, title):
    line = "=" * 92
    for s in (line, f"  {title}", line):
        print(s)
        f.write(s + "\n")


def emit(f, obj):
    s = obj if isinstance(obj, str) else obj.to_string()
    print(s)
    f.write(s + "\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--rebalance", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--hold-bars", type=int, default=10)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--out", default="out/eval")
    args = ap.parse_args()

    if args.quick:
        args.n = 20
        args.rebalance = 40

    os.makedirs(args.out, exist_ok=True)
    rpt = open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8")
    t0 = time.time()

    section(rpt, "0. 数据与配置")
    uni = build_universe(ASHARE_DIR, "daily", n_symbols=args.n,
                         min_bars=1200, start=args.start, end=args.end)
    cfg = (f"标的池: train={len(uni['train'])} test={len(uni['test'])} (按代码 md5 分桶，互斥)\n"
           f"区间: {args.start} ~ {args.end or '最新'}\n"
           f"再平衡: 每 {args.rebalance} 根 | 前瞻: {args.horizon} 根 | 结果窗口: {args.hold_bars} 根\n"
           f"守住判据: 先到 上沿+1.0ATR 记 hold；先到 收盘<下沿-0.5ATR 记 break")
    emit(rpt, cfg)

    dets, v2 = build_detectors(args.ablation)
    emit(rpt, "检测器: " + ", ".join(d.name for d in dets))

    # ---------- 走查 ----------
    section(rpt, "1. 走查回测")
    res = {}
    for split in ("train", "test"):
        print(f"\n--- {split} ---")
        res[split] = run_walkforward(
            uni[split], dets,
            warmup=300, rebalance_every=args.rebalance,
            horizon=args.horizon, hold_bars=args.hold_bars)
        res[split]["levels"].to_parquet(
            os.path.join(args.out, f"levels_{split}.parquet"), index=False)
        res[split]["geo"].to_parquet(
            os.path.join(args.out, f"geo_{split}.parquet"), index=False)

    lv_tr, lv_te = res["train"]["levels"], res["test"]["levels"]
    emit(rpt, f"train 关键位记录 {len(lv_tr):,} 条 | test {len(lv_te):,} 条")

    # ---------- 主表：各检测器总体表现（test 集） ----------
    for split, lv in (("train", lv_tr), ("test", lv_te)):
        section(rpt, f"2.{'12'[split=='test']} 各检测器表现 — {split} 集（全部关键位）")
        rows = []
        for d in dets:
            s = summarize_by_kind(lv[lv["detector"] == d.name]).loc["all"].to_dict()
            s["detector"] = d.name
            rows.append(s)
        t = pd.DataFrame(rows).set_index("detector")[
            ["n_levels", "med_width_atr", "avg_dist_atr", "touch_rate",
             "n_decided", "hold_rate", "hold_lo95", "hold_hi95",
             "undecided_rate", "edge_atr", "fwd_ret_pct"]]
        emit(rpt, t)

        section(rpt, f"2.{'34'[split=='test']} 同口径对比 — {split} 集（区宽 ≤ 1.5 ATR）")
        rows = []
        for d in dets:
            s = summarize_by_kind(lv[lv["detector"] == d.name], width_cap=1.5).loc["all"].to_dict()
            s["detector"] = d.name
            rows.append(s)
        t = pd.DataFrame(rows).set_index("detector")[
            ["n_levels", "med_width_atr", "touch_rate", "n_decided",
             "hold_rate", "hold_lo95", "edge_atr", "fwd_ret_pct"]]
        emit(rpt, t)

    # ---------- 支撑 / 阻力 分开 ----------
    section(rpt, "3. 支撑 vs 阻力（test 集，区宽 ≤ 1.5 ATR）")
    rows = []
    for d in dets:
        sub = summarize_by_kind(lv_te[lv_te["detector"] == d.name], width_cap=1.5)
        for kind in ("support", "resistance"):
            r = sub.loc[kind].to_dict()
            r.update({"detector": d.name, "kind": kind})
            rows.append(r)
    t = pd.DataFrame(rows).set_index(["detector", "kind"])[
        ["n_levels", "touch_rate", "n_decided", "hold_rate", "hold_lo95",
         "edge_atr", "fwd_ret_pct"]]
    emit(rpt, t)

    # ---------- 对照组提升度（未控制距离） ----------
    section(rpt, "4. 提升度与显著性（test 集，区宽 ≤ 1.5 ATR，未控距离）")
    pairs = [("placebo_shift_v3", "V3_fusion"), ("placebo_shift_v2", "V2_fusion"),
             ("placebo_shift_v1", "V1_density"), ("placebo_fixed", "V3_fusion"),
             ("V2_fusion", "V3_fusion"), ("V1_density", "V3_fusion")]
    rows = [compare(lv_te, b, c, width_cap=1.5) for b, c in pairs]
    t = pd.DataFrame(rows)[["base", "chal", "base_hold", "chal_hold", "lift_pp",
                            "base_n", "chal_n", "z", "p_value",
                            "base_edge_atr", "chal_edge_atr", "edge_lift"]]
    emit(rpt, t)

    # ---------- 距离分层对比（核心结论） ----------
    section(rpt, "4b. 控制距离后的对比 — CMH 分层检验（test 集）")
    emit(rpt, "距离是最强主效应（阻力守住率随距离 +16.5pp），不控制距离的比较没有意义。\n"
              "下表在同一距离档内比较，hold 用 CMH 合并检验，fwd_ret 用各档等权平均。")
    rows = []
    for kind in ("support", "resistance"):
        for b, c in pairs:
            r = compare_distance_neutral(lv_te, b, c, kind=kind, width_cap=1.5)
            r.pop("_strata", None)
            if "note" not in r:
                rows.append(r)
    if rows:
        t = pd.DataFrame(rows)[["kind", "base", "chal", "n_strata",
                                "base_hold_eq", "chal_hold_eq", "hold_lift_pp",
                                "cmh_p", "or_mh",
                                "base_fwd_eq", "chal_fwd_eq", "fwd_lift_pp",
                                "edge_lift"]]
        emit(rpt, t)

    # 明细分层表：V3 vs 自己的平移对照
    for kind in ("support", "resistance"):
        r = compare_distance_neutral(lv_te, "placebo_shift_v3", "V3_fusion",
                                     kind=kind, width_cap=1.5)
        if "_strata" in r:
            section(rpt, f"4c. V3 vs 平移对照 逐距离档明细 — {kind}")
            emit(rpt, r["_strata"])

    # ---------- 几何精度 ----------
    section(rpt, "5. 几何精度：预测位 vs 未来真实摆动极值（test 集）")
    emit(rpt, summarize_geo(res["test"]["geo"]))
    emit(rpt, "recall_hit_05 = 未来真实枢轴中，有多少比例落在某个预测位 0.5 ATR 内\n"
              "precision_hit_05 = 预测位中，有多少比例命中了未来真实枢轴\n"
              "median_err_atr = 真实枢轴到最近预测位的中位距离（ATR 倍数，越小越好）")

    # ---------- 概率校准（train 上 fit，test 上验证） ----------
    section(rpt, "6. 概率校准（train 拟合 → test 验证）")
    MAIN = "V3_fusion"
    tr = lv_tr[(lv_tr["detector"] == MAIN) & lv_tr["touched"] &
               lv_tr["result"].isin(["hold", "break"]) & (lv_tr["width_atr"] <= 1.5)]
    te = lv_te[(lv_te["detector"] == MAIN) & lv_te["touched"] &
               lv_te["result"].isin(["hold", "break"]) & (lv_te["width_atr"] <= 1.5)]
    if len(tr) > 200 and len(te) > 200:
        cal = IsotonicCalibrator().fit(tr["score"].to_numpy(),
                                       (tr["result"] == "hold").to_numpy(float))
        p_te = cal.predict(te["score"].to_numpy())
        y_te = (te["result"] == "hold").to_numpy(float)
        emit(rpt, calibration_table(p_te, y_te, n_bins=8))
        emit(rpt, f"test Brier = {brier_score(p_te, y_te):.4f} | "
                  f"ECE = {ece(p_te, y_te, 8):.4f} | "
                  f"基准(常数预测) Brier = {brier_score(np.full_like(y_te, y_te.mean()), y_te):.4f}")
    else:
        emit(rpt, f"样本不足，跳过校准 (train={len(tr)}, test={len(te)})")

    # ---------- 按得分分层：单调性 ----------
    section(rpt, f"7. 得分单调性（test 集，{MAIN}，区宽 ≤ 1.5 ATR）")
    if len(te) > 100:
        q = pd.qcut(te["score"], 5, labels=[f"Q{i+1}" for i in range(5)], duplicates="drop")
        t = te.assign(bucket=q).groupby("bucket", observed=True).apply(
            lambda g: pd.Series({
                "n": len(g),
                "score_mean": g["score"].mean(),
                "hold_rate": (g["result"] == "hold").mean(),
                "edge_atr": g["edge_atr"].mean(),
                "fwd_ret_pct": g["fwd_ret_pct"].mean(),
            }), include_groups=False)
        emit(rpt, t)
        emit(rpt, "得分高的关键位守住率应显著高于得分低的 —— 否则 score 没有区分度。")

    # ---------- 区宽分层：证明指标不可被"画宽区间"白嫖 ----------
    section(rpt, f"8. 区宽分层（test 集，{MAIN}）")
    d2 = lv_te[(lv_te["detector"] == MAIN) & lv_te["touched"] &
               lv_te["result"].isin(["hold", "break"])]
    if len(d2) > 100:
        bins = [0, 0.5, 0.8, 1.1, 1.5, 99]
        t = d2.assign(wb=pd.cut(d2["width_atr"], bins)).groupby("wb", observed=True).apply(
            lambda g: pd.Series({
                "n": len(g), "hold_rate": (g["result"] == "hold").mean(),
                "edge_atr": g["edge_atr"].mean()}), include_groups=False)
        emit(rpt, t)
        emit(rpt, "区间越宽，hold_rate 天然越高 —— 因此跨算法比较必须固定区宽口径。")

    # ---------- 机械交易回测 ----------
    section(rpt, "9. 机械交易回测（test 集，做多支撑，成本 0.15%/笔）")
    rows = []
    for d in dets:
        for kind in ("support", "resistance"):
            r = backtest_touch_strategy(lv_te, d.name, kind=kind,
                                        width_cap=1.5, max_dist_atr=4.0)
            r.update({"detector": d.name, "kind": kind})
            rows.append(r)
    t = pd.DataFrame(rows).set_index(["detector", "kind"])
    cols = [c for c in ["n_trades", "win_rate", "avg_ret_pct", "med_ret_pct",
                        "profit_factor", "t_stat"] if c in t.columns]
    emit(rpt, t[cols])

    # ---------- 消融 ----------
    if args.ablation:
        section(rpt, "10. 消融实验（test 集，区宽 ≤ 1.5 ATR）")
        rows = []
        for d in dets:
            if not d.name.startswith(("V2_",)):
                continue
            s = summarize_by_kind(lv_te[lv_te["detector"] == d.name], width_cap=1.5).loc["all"].to_dict()
            s["detector"] = d.name
            rows.append(s)
        t = pd.DataFrame(rows).set_index("detector")[
            ["n_levels", "touch_rate", "n_decided", "hold_rate", "hold_lo95", "edge_atr"]]
        emit(rpt, t.sort_values("hold_rate", ascending=False))

    # ---------- 导出评分标定表（供线上界面使用） ----------
    section(rpt, "11. 导出评分标定表 -> data/score_calibration.json")
    calib = export_calibration(lv_tr, lv_te, MAIN, args)
    emit(rpt, "线上不再展示任何单个关键位的『守住率』，而是展示它落在历史哪个分位档，"
              "以及该档在样本外股票上的实测表现。这样界面上每个数字都能追溯到本回测。")
    if calib:
        emit(rpt, pd.DataFrame(calib["buckets"]))

    emit(rpt, f"\n总耗时 {time.time()-t0:.0f}s | 明细已保存到 {args.out}/")
    rpt.close()


def export_calibration(lv_tr: pd.DataFrame, lv_te: pd.DataFrame,
                       detector: str, args) -> dict:
    """
    把排序分的分位边界 + 每档在 test 集的实测表现导出成 JSON。

    分位边界在 **train** 上确定（避免用 test 定边界再报告 test 表现），
    每档的实测统计在 **test** 上计算（样本外股票）。
    """
    out = {
        "meta": {
            "detector": detector,
            "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "eval_start": args.start,
            "rebalance": args.rebalance,
            "horizon": args.horizon,
            "hold_bars": args.hold_bars,
            "note": "edge_edges 在 train 股票池上确定；buckets 统计来自互斥的 test 股票池",
        },
        "edge_edges": {},
        "stall_edges": {},
        "buckets": [],
    }
    n_q = 5
    for kind in ("support", "resistance"):
        tr = lv_tr[(lv_tr["detector"] == detector) & (lv_tr["kind"] == kind) &
                   (lv_tr["width_atr"] <= 1.5)]
        te = lv_te[(lv_te["detector"] == detector) & (lv_te["kind"] == kind) &
                   (lv_te["width_atr"] <= 1.5) & lv_te["touched"]]
        if len(tr) < 500 or len(te) < 500:
            continue
        edges = np.quantile(tr["score"].to_numpy(np.float64),
                            np.linspace(0, 1, n_q + 1)[1:-1]).tolist()
        out["edge_edges"][kind] = edges
        if "m_p_stall" in tr.columns:
            out["stall_edges"][kind] = np.quantile(
                tr["m_p_stall"].dropna().to_numpy(np.float64),
                np.linspace(0, 1, n_q + 1)[1:-1]).tolist()

        b = np.digitize(te["score"].to_numpy(np.float64), edges)
        for q in range(n_q):
            g = te[b == q]
            dec = g[g["result"].isin(["hold", "break"])]
            if len(dec) == 0:
                continue
            k = int((dec["result"] == "hold").sum())
            p, lo, hi = wilson(k, len(dec))
            out["buckets"].append({
                "kind": kind, "bucket": f"Q{q+1}",
                "n": int(len(g)), "n_decided": int(len(dec)),
                "hold_rate": round(p, 4), "hold_lo95": round(lo, 4),
                "fwd_ret_pct": round(float(g["fwd_ret_pct"].mean()), 4),
                "edge_atr": round(float(g["edge_atr"].mean()), 4),
                "touch_rate": None,
            })

    if not out["buckets"]:
        return {}
    os.makedirs("data", exist_ok=True)
    with open("data/score_calibration.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("已写出 data/score_calibration.json")
    return out


if __name__ == "__main__":
    main()
