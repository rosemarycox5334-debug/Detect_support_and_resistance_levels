# -*- coding: utf-8 -*-
"""
实测：样本量够不够？要不要按时间周期分模型？

用法:
    python scripts/prob_scale_period_exp.py              # 全流程（含走查，约数分钟）
    python scripts/prob_scale_period_exp.py --reuse      # 复用 out/prob_exp 已有 parquet

输出:
    out/prob_exp/report.txt
    out/prob_exp/levels_{train,test}.parquet   (仅 V3_fusion)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.srlab.data import build_universe
from src.srlab.detectors import V3Fusion
from src.srlab.metrics import calibration_table
from src.srlab.probability import (
    FEATURES, KINDS, TARGETS, auc_score, brier_score,
    features_from_frame, train_model,
)
from src.srlab.walkforward import run_walkforward

ASHARE_DIR = r"D:\A股_K线数据\parquet\stocks"
OUT = "out/prob_exp"
DET = "V3_fusion"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def hr(f, t):
    line = "=" * 100
    for s in (line, "  " + t, line):
        print(s)
        f.write(s + "\n")


def emit(f, obj):
    s = obj if isinstance(obj, str) else (
        obj.to_string() if hasattr(obj, "to_string") else str(obj)
    )
    print(s)
    f.write(s + "\n\n")


def prep(d: pd.DataFrame, target: str):
    if target == "touch":
        sub = d
        y = d["touched"].astype(float).to_numpy()
    else:
        sub = d[d["touched"] & d["result"].isin(["hold", "break"])]
        y = (sub["result"] == "hold").astype(float).to_numpy()
    return features_from_frame(sub), y, sub


def eval_one(tr, te, target):
    Ftr, ytr, _ = prep(tr, target)
    Fte, yte, _ = prep(te, target)
    if len(Ftr) < 200 or len(Fte) < 200 or len(np.unique(ytr)) < 2:
        return None
    m = train_model(Ftr, ytr)
    p = m.predict(Fte.to_numpy())
    b = brier_score(yte, p)
    b0 = brier_score(yte, np.full(len(yte), ytr.mean()))
    a = auc_score(yte, p)
    # 校准：按预测分位 4 桶，看 |pred-actual| 均值
    order = np.argsort(p)
    bins = np.array_split(order, 4)
    cal_errs = []
    for idx in bins:
        if len(idx) < 30:
            continue
        cal_errs.append(abs(float(p[idx].mean()) - float(yte[idx].mean())))
    return {
        "auc": a,
        "brier": b,
        "brier0": b0,
        "skill": (1 - b / b0) if b0 > 0 else float("nan"),
        "n_tr": len(Ftr),
        "n_te": len(Fte),
        "base_te": float(np.mean(yte)),
        "p_lo": float(np.percentile(p, 10)),
        "p_hi": float(np.percentile(p, 90)),
        "cal_mae": float(np.mean(cal_errs)) if cal_errs else float("nan"),
        "model": m,
        "p": p,
        "yte": yte,
    }


def run_walk(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    uni = build_universe(
        ASHARE_DIR, "daily", n_symbols=10000, min_bars=1200, start="2012-01-01"
    )
    dets = [V3Fusion()]
    res = {}
    for split in ("train", "test"):
        print(f"\n--- walkforward {split} n={len(uni[split])} ---")
        t0 = time.time()
        r = run_walkforward(
            uni[split], dets,
            warmup=300, rebalance_every=20, horizon=40, hold_bars=10,
        )
        levels = r["levels"]
        levels.to_parquet(os.path.join(out_dir, f"levels_{split}.parquet"))
        res[split] = levels
        print(f"  rows={len(levels)} stocks={levels['code'].nunique()} "
              f"elapsed={time.time()-t0:.0f}s")
    meta = {
        "n_train": len(uni["train"]),
        "n_test": len(uni["test"]),
        "train_codes": sorted({s.code for s in uni["train"]}),
        "test_codes": sorted({s.code for s in uni["test"]}),
    }
    pd.Series(meta).to_json(os.path.join(out_dir, "meta.json"))
    return res["train"], res["test"], meta


def load_levels(out_dir: str):
    tr = pd.read_parquet(os.path.join(out_dir, f"levels_train.parquet"))
    te = pd.read_parquet(os.path.join(out_dir, f"levels_test.parquet"))
    tr = tr[tr["detector"] == DET].copy()
    te = te[te["detector"] == DET].copy()
    tr["date"] = pd.to_datetime(tr["date"])
    te["date"] = pd.to_datetime(te["date"])
    return tr, te


def exp_sample_size(f, tr, te):
    """固定 test 全量，train 股票数从少到多，看 AUC / 校准是否还在涨。"""
    hr(f, "A. 样本量缩放（固定互斥 test 股票池，扩大 train 股票数）")
    rows = []
    train_codes = sorted(tr["code"].unique())
    # 稳定顺序：按 code 排序后取前缀，保证可复现
    sizes = [20, 40, 80, 120, len(train_codes)]
    sizes = sorted(set(s for s in sizes if s <= len(train_codes)))
    for kind in KINDS:
        tr_k = tr[tr["kind"] == kind]
        te_k = te[te["kind"] == kind]
        for n in sizes:
            codes = set(train_codes[:n])
            sub_tr = tr_k[tr_k["code"].isin(codes)]
            for target in TARGETS:
                r = eval_one(sub_tr, te_k, target)
                if r is None:
                    continue
                rows.append({
                    "target": target, "kind": kind, "n_stocks": n,
                    "n_tr": r["n_tr"], "n_te": r["n_te"],
                    "auc": r["auc"], "skill": r["skill"],
                    "cal_mae": r["cal_mae"],
                    "span": r["p_hi"] - r["p_lo"],
                    "base_te": r["base_te"],
                })
    df = pd.DataFrame(rows)
    emit(f, df.to_string(index=False))
    # 边际收益：从 80→full / 120→full
    emit(f, "解读：若 n_stocks 从 80/120 → 全量时 AUC/cal_mae 几乎不动，"
          "说明 130 只已够用；若还在明显改善，应尽量用满可用池。")
    return df


def exp_time_period(f, tr, te):
    """
    时间切分实验（股票仍互斥：train 股票训，test 股票验）

    1) 全时段模型 vs 早期模型 vs 晚期模型，在晚期 test 上比
    2) 交叉：早期训→晚期验 / 晚期训→早期验（看制度漂移）
    3) 分时段模型 vs 统一模型：在各时段 test 上的校准误差
    """
    hr(f, "B. 时间周期：统一模型 vs 分时段模型")

    eras = [
        ("early", "2013-01-01", "2018-12-31"),
        ("late",  "2019-01-01", "2026-12-31"),
        ("bull15", "2014-07-01", "2015-12-31"),
        ("bear18", "2018-01-01", "2018-12-31"),
        ("covid", "2020-01-01", "2020-12-31"),
        ("recent", "2023-01-01", "2026-12-31"),
    ]

    # --- B1 交叉时段泛化 ---
    hr(f, "B1. 交叉时段泛化（train股票×时段 → test股票×时段）")
    rows = []
    for kind in KINDS:
        for target in TARGETS:
            for tr_era, tr_a, tr_b in [("early", "2013-01-01", "2018-12-31"),
                                      ("late", "2019-01-01", "2026-12-31"),
                                      ("all", "2013-01-01", "2026-12-31")]:
                sub_tr = tr[(tr["kind"] == kind) &
                            (tr["date"] >= tr_a) & (tr["date"] <= tr_b)]
                for te_era, te_a, te_b in [("early", "2013-01-01", "2018-12-31"),
                                          ("late", "2019-01-01", "2026-12-31"),
                                          ("recent", "2023-01-01", "2026-12-31")]:
                    sub_te = te[(te["kind"] == kind) &
                                (te["date"] >= te_a) & (te["date"] <= te_b)]
                    r = eval_one(sub_tr, sub_te, target)
                    if r is None:
                        continue
                    rows.append({
                        "target": target, "kind": kind,
                        "train_era": tr_era, "test_era": te_era,
                        "n_tr": r["n_tr"], "n_te": r["n_te"],
                        "auc": r["auc"], "skill": r["skill"],
                        "cal_mae": r["cal_mae"],
                        "base_te": r["base_te"],
                        "pred_mean": float(r["p"].mean()),
                    })
    df1 = pd.DataFrame(rows)
    emit(f, df1.to_string(index=False))

    # --- B2 分时段模型 vs 统一模型：在 late/recent test 上比 ---
    hr(f, "B2. 分时段模型 vs 统一模型（验收=test股票 × late/recent）")
    rows2 = []
    for kind in KINDS:
        for target in TARGETS:
            for te_era, te_a, te_b in [("late", "2019-01-01", "2026-12-31"),
                                      ("recent", "2023-01-01", "2026-12-31")]:
                sub_te = te[(te["kind"] == kind) &
                            (te["date"] >= te_a) & (te["date"] <= te_b)]
                # 统一：全时段 train
                r_all = eval_one(tr[tr["kind"] == kind], sub_te, target)
                # 匹配时段：只用同期 train
                r_match = eval_one(
                    tr[(tr["kind"] == kind) &
                       (tr["date"] >= te_a) & (tr["date"] <= te_b)],
                    sub_te, target,
                )
                # 错配：用另一半时段
                if te_era == "late":
                    other = tr[(tr["kind"] == kind) &
                               (tr["date"] >= "2013-01-01") &
                               (tr["date"] <= "2018-12-31")]
                else:
                    other = tr[(tr["kind"] == kind) &
                               (tr["date"] >= "2019-01-01") &
                               (tr["date"] <= "2022-12-31")]
                r_mis = eval_one(other, sub_te, target)
                for name, r in [("pooled_all", r_all),
                                ("matched_era", r_match),
                                ("mismatched", r_mis)]:
                    if r is None:
                        continue
                    rows2.append({
                        "target": target, "kind": kind, "test_era": te_era,
                        "model": name,
                        "n_tr": r["n_tr"], "n_te": r["n_te"],
                        "auc": r["auc"], "skill": r["skill"],
                        "cal_mae": r["cal_mae"],
                        "bias": float(r["p"].mean() - r["yte"].mean()),
                    })
    df2 = pd.DataFrame(rows2)
    emit(f, df2.to_string(index=False))

    # --- B3 各年 base rate 漂移（说明为什么校准可能漂）---
    hr(f, "B3. 各年真实触及率/守住率（test 股票池，看制度漂移幅度）")
    te2 = te.copy()
    te2["year"] = te2["date"].dt.year
    rows3 = []
    for kind in KINDS:
        for y, g in te2[te2["kind"] == kind].groupby("year"):
            touch = float(g["touched"].mean())
            h = g[g["touched"] & g["result"].isin(["hold", "break"])]
            hold = float((h["result"] == "hold").mean()) if len(h) else float("nan")
            rows3.append({
                "kind": kind, "year": int(y), "n": len(g),
                "touch_rate": touch, "n_hold": len(h), "hold_rate": hold,
            })
    df3 = pd.DataFrame(rows3)
    emit(f, df3.to_string(index=False))

    # 逐年：用「上一年之前」滚动训练 vs 全量，看 recent 年校准
    hr(f, "B4. 滚动训练 vs 全量（每年用此前全部 train 数据训，验当年 test）")
    rows4 = []
    years = sorted(te2["year"].unique())
    for kind in KINDS:
        for target in TARGETS:
            for y in years:
                if y < 2016:
                    continue
                sub_te = te[(te["kind"] == kind) & (te["date"].dt.year == y)]
                # expanding: train date < y
                r_exp = eval_one(
                    tr[(tr["kind"] == kind) & (tr["date"].dt.year < y)],
                    sub_te, target,
                )
                r_all = eval_one(tr[tr["kind"] == kind], sub_te, target)
                # recent-3y window
                r_win = eval_one(
                    tr[(tr["kind"] == kind) &
                       (tr["date"].dt.year >= y - 3) &
                       (tr["date"].dt.year < y)],
                    sub_te, target,
                )
                for name, r in [("expanding", r_exp), ("pooled_all", r_all),
                                ("last3y", r_win)]:
                    if r is None:
                        continue
                    rows4.append({
                        "target": target, "kind": kind, "year": int(y),
                        "model": name,
                        "n_tr": r["n_tr"], "n_te": r["n_te"],
                        "auc": r["auc"], "skill": r["skill"],
                        "cal_mae": r["cal_mae"],
                        "bias": float(r["p"].mean() - r["yte"].mean()),
                    })
    df4 = pd.DataFrame(rows4)
    # 汇总：各 model 在所有 year 上的均值
    if len(df4):
        summ = (df4.groupby(["target", "kind", "model"])
                  [["auc", "skill", "cal_mae", "bias"]]
                  .mean().reset_index())
        emit(f, "B4 汇总（跨年均值）:")
        emit(f, summ.to_string(index=False))
        emit(f, "B4 明细:")
        emit(f, df4.to_string(index=False))
    return df1, df2, df3, df4


def exp_full_vs_130(f, tr, te):
    """全可用池 vs 原 130 只（取 train 前 130）在同一 test 上比。"""
    hr(f, "C. 全可用 train 池 vs 仅 130 只（同一 test）")
    codes = sorted(tr["code"].unique())
    rows = []
    for kind in KINDS:
        te_k = te[te["kind"] == kind]
        for label, n in [("n130", min(130, len(codes))),
                         ("full", len(codes))]:
            sub = tr[(tr["kind"] == kind) & (tr["code"].isin(codes[:n]))]
            for target in TARGETS:
                r = eval_one(sub, te_k, target)
                if r is None:
                    continue
                rows.append({
                    "target": target, "kind": kind, "train_set": label,
                    "n_stocks": n, "n_tr": r["n_tr"], "n_te": r["n_te"],
                    "auc": r["auc"], "skill": r["skill"],
                    "cal_mae": r["cal_mae"],
                    "span": r["p_hi"] - r["p_lo"],
                })
    df = pd.DataFrame(rows)
    emit(f, df.to_string(index=False))
    return df


def conclude(f, df_scale, df_b2, df_b4_sum):
    hr(f, "D. 结论（基于本机实测，不是拍脑袋）")
    lines = []

    # 样本量：看 full vs 80/120 的 AUC 差
    if df_scale is not None and len(df_scale):
        for target in TARGETS:
            for kind in KINDS:
                sub = df_scale[(df_scale.target == target) & (df_scale.kind == kind)]
                if sub.empty:
                    continue
                full = sub[sub.n_stocks == sub.n_stocks.max()].iloc[0]
                ref_n = 80 if (sub.n_stocks == 80).any() else sub.n_stocks.min()
                ref = sub[sub.n_stocks == ref_n].iloc[0]
                d_auc = full["auc"] - ref["auc"]
                d_cal = ref["cal_mae"] - full["cal_mae"]  # 正=变好
                lines.append(
                    f"样本量 {target}/{kind}: stocks {int(ref_n)}→{int(full.n_stocks)} "
                    f"ΔAUC={d_auc:+.4f}  Δcal_mae(↓好)={-d_cal:+.4f} "
                    f"(full AUC={full.auc:.4f}, cal={full.cal_mae:.4f})"
                )

    # 分时段 vs 统一
    if df_b2 is not None and len(df_b2):
        for target in TARGETS:
            for kind in KINDS:
                sub = df_b2[(df_b2.target == target) & (df_b2.kind == kind) &
                            (df_b2.test_era == "recent")]
                if sub.empty:
                    continue
                pivot = sub.set_index("model")
                if "pooled_all" in pivot.index and "matched_era" in pivot.index:
                    a0, a1 = pivot.loc["pooled_all", "auc"], pivot.loc["matched_era", "auc"]
                    c0, c1 = pivot.loc["pooled_all", "cal_mae"], pivot.loc["matched_era", "cal_mae"]
                    b0, b1 = pivot.loc["pooled_all", "bias"], pivot.loc["matched_era", "bias"]
                    lines.append(
                        f"周期 recent {target}/{kind}: pooled AUC={a0:.4f} cal={c0:.4f} bias={b0:+.4f} | "
                        f"matched AUC={a1:.4f} cal={c1:.4f} bias={b1:+.4f} | "
                        f"ΔAUC(match-pool)={a1-a0:+.4f}"
                    )

    emit(f, "\n".join(lines) if lines else "(无)")
    emit(f, """
决策规则（自动应用）:
  - 若全量 vs 130 的 AUC 提升 < 0.005 且校准差不多 → 130 已够，扩池收益有限
  - 若提升明显 → 用满可用因子覆盖池重训线上模型
  - 若 matched_era 比 pooled 的 AUC/校准明显更好（AUC+>0.01 或 |bias| 小很多）
    → 建议按时段分模型或至少做滚动/近期加权
  - 若 pooled 不差甚至更好 → 保持单一全时段模型（更稳、实现简单）
""".strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="复用 out/prob_exp 已有 levels parquet")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rpt_path = os.path.join(args.out, "report.txt")
    f = open(rpt_path, "w", encoding="utf-8")
    t0 = time.time()

    hr(f, "0. 背景")
    emit(f, f"特征: {list(FEATURES)}\n检测器: {DET}\n输出: {args.out}")

    if args.reuse and os.path.exists(os.path.join(args.out, "levels_train.parquet")):
        emit(f, "复用已有走查结果")
        tr, te = load_levels(args.out)
        meta = {"n_train": tr["code"].nunique(), "n_test": te["code"].nunique()}
    else:
        emit(f, "开始全可用池走查（仅 V3_fusion）…")
        tr_raw, te_raw, meta = run_walk(args.out)
        tr, te = load_levels(args.out)

    emit(f, f"走查结果: train stocks={tr['code'].nunique()} rows={len(tr)} | "
          f"test stocks={te['code'].nunique()} rows={len(te)}\n"
          f"日期: {tr['date'].min().date()} ~ {te['date'].max().date()}")

    df_scale = exp_sample_size(f, tr, te)
    df_b1, df_b2, df_b3, df_b4 = exp_time_period(f, tr, te)
    df_c = exp_full_vs_130(f, tr, te)

    df_b4_sum = None
    if df_b4 is not None and len(df_b4):
        df_b4_sum = (df_b4.groupby(["target", "kind", "model"])
                     [["auc", "skill", "cal_mae", "bias"]].mean().reset_index())
    conclude(f, df_scale, df_b2, df_b4_sum)

    emit(f, f"总耗时 {time.time()-t0:.0f}s")
    f.close()
    print(f"\n报告已写: {rpt_path}")


if __name__ == "__main__":
    main()
