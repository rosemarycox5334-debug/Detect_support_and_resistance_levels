# -*- coding: utf-8 -*-
"""
最简可行评分搜索

目的：V3 用了 5 个证据。如果其中 1~2 个就能拿到全部排序能力，
就应该砍掉其余的 —— 更少的参数 = 更难过拟合 = 用户更容易理解和验证。

方法：直接在已落盘的走查结果上重算候选评分（证据分量都已存为 m_* 列），
在 train 股票池上挑权重，在 test 股票池上汇报。不需要重跑回测。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.srlab.metrics import wilson

pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/eval200_v3"
DET = "V3_fusion"


def load(split):
    d = pd.read_parquet(os.path.join(OUT, f"levels_{split}.parquet"))
    d = d[(d["detector"] == DET) & d["touched"] & (d["width_atr"] <= 1.5)]
    return d.copy()


# 候选评分：全部只用因果证据（m_* 都是 t 时刻算出来的）
CANDIDATES = {
    "s0_events":        lambda d: np.log1p(d["m_n_events"]),
    "s1_events_stale":  lambda d: np.log1p(d["m_n_events"]) + 0.8 * d["m_stale"],
    "s2_ev_stale_novp": lambda d: np.log1p(d["m_n_events"]) + 0.8 * d["m_stale"] - 0.5 * d["m_vp"],
    "s3_vp_only":       lambda d: d["m_vp"],
    "s4_pivot_only":    lambda d: d["m_pivot"],
    "s5_wilson_only":   lambda d: d["m_hold_lb"],
    "s6_full_v3":       lambda d: d["score"],
    "s7_ev_wilson":     lambda d: np.log1p(d["m_n_events"]) + 0.6 * d["m_hold_lb"],
}


def quintile_table(d: pd.DataFrame, col: str, n_q: int = 5) -> pd.DataFrame:
    dd = d[np.isfinite(d[col])]
    if len(dd) < 500 or dd[col].nunique() < n_q:
        return pd.DataFrame()
    try:
        q = pd.qcut(dd[col], n_q, labels=[f"Q{i+1}" for i in range(n_q)],
                    duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows = []
    for b, g in dd.assign(_b=q).groupby("_b", observed=True):
        dec = g[g["result"].isin(["hold", "break"])]
        k = int((dec["result"] == "hold").sum())
        p, lo, hi = wilson(k, len(dec))
        rows.append({"bucket": str(b), "n": len(g), "hold": p, "lo95": lo,
                     "fwd_ret": g["fwd_ret_pct"].mean(),
                     "edge_atr": g["edge_atr"].mean()})
    return pd.DataFrame(rows)


def mono(v: np.ndarray) -> float:
    """单调性：Spearman 相关（分位序号 vs 指标）"""
    if len(v) < 3 or not np.all(np.isfinite(v)):
        return np.nan
    r = np.arange(len(v))
    vr = pd.Series(v).rank().to_numpy()
    return float(np.corrcoef(r, vr)[0, 1])


def evaluate(split: str, d: pd.DataFrame, kind: str) -> pd.DataFrame:
    g = d[d["kind"] == kind]
    rows = []
    for name, fn in CANDIDATES.items():
        col = f"_{name}"
        try:
            g = g.assign(**{col: fn(g)})
        except Exception:
            continue
        t = quintile_table(g, col)
        if t.empty:
            continue
        rows.append({
            "score": name, "split": split, "kind": kind, "n": int(t["n"].sum()),
            "hold_Q1": t["hold"].iloc[0], "hold_Q5": t["hold"].iloc[-1],
            "hold_spread_pp": (t["hold"].iloc[-1] - t["hold"].iloc[0]) * 100,
            "hold_mono": mono(t["hold"].to_numpy()),
            "fwd_Q1": t["fwd_ret"].iloc[0], "fwd_Q5": t["fwd_ret"].iloc[-1],
            "fwd_spread_pp": t["fwd_ret"].iloc[-1] - t["fwd_ret"].iloc[0],
            "fwd_mono": mono(t["fwd_ret"].to_numpy()),
        })
    return pd.DataFrame(rows)


def main():
    tr, te = load("train"), load("test")
    print(f"train {len(tr):,} | test {len(te):,}  ({DET}, 已触及, 区宽<=1.5ATR)")
    print("可用证据:", [c for c in te.columns if c.startswith("m_")])

    for kind in ("support", "resistance"):
        print("\n" + "=" * 110)
        print(f"  候选评分对比 — {kind}")
        print("=" * 110)
        a = evaluate("train", tr, kind)
        b = evaluate("test", te, kind)
        both = pd.concat([a, b]).sort_values(["score", "split"])
        print(both[["score", "split", "n", "hold_Q1", "hold_Q5", "hold_spread_pp",
                    "hold_mono", "fwd_Q1", "fwd_Q5", "fwd_spread_pp",
                    "fwd_mono"]].to_string(index=False))

    # 最优候选的 test 集完整分位表
    print("\n" + "=" * 110)
    print("  推荐评分在 test 集的完整分位表")
    print("=" * 110)
    for kind in ("support", "resistance"):
        g = te[te["kind"] == kind].copy()
        for name in ("s1_events_stale", "s2_ev_stale_novp", "s6_full_v3"):
            g["_s"] = CANDIDATES[name](g)
            t = quintile_table(g, "_s")
            if t.empty:
                continue
            print(f"\n--- {kind} / {name} ---")
            print(t.to_string(index=False))


if __name__ == "__main__":
    main()
