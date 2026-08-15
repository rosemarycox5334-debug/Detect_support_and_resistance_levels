# -*- coding: utf-8 -*-
"""
单因子分析：V2 的每个"证据"到底能不能预测结果？

走查回测已经把每个关键位的证据分量（m_vp / m_pivot / m_hold / m_n_events /
m_round / m_recency）连同前瞻结果一起落盘，因此不需要重跑，直接做分层统计。

这是整个优化方案的决策依据：哪个证据有信息，就加权；没有信息的，删掉。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.srlab.metrics import two_prop_ztest, wilson

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/eval200"


def load(split):
    return pd.read_parquet(os.path.join(OUT, f"levels_{split}.parquet"))


def bucket_stats(d: pd.DataFrame, col: str, n_q: int = 5,
                 label: str = None) -> pd.DataFrame:
    """按 col 分位分层，统计 hold_rate / edge_atr / fwd_ret"""
    d = d[np.isfinite(d[col])]
    if len(d) < 200 or d[col].nunique() < 3:
        return pd.DataFrame()
    try:
        q = pd.qcut(d[col], n_q, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = d.assign(_b=q).groupby("_b", observed=True)
    rows = []
    for b, gg in g:
        dec = gg[gg["result"].isin(["hold", "break"])]
        k = int((dec["result"] == "hold").sum())
        n = len(dec)
        p, lo, hi = wilson(k, n)
        rows.append({
            "bucket": str(b), "n": len(gg), "n_dec": n,
            f"{col}_mean": gg[col].mean(),
            "hold_rate": p, "lo95": lo, "hi95": hi,
            "edge_atr": gg["edge_atr"].mean(),
            "fwd_ret_pct": gg["fwd_ret_pct"].mean(),
            "touch_rate": gg["touched"].mean(),
        })
    t = pd.DataFrame(rows)
    t.attrs["factor"] = label or col
    return t


def spread(t: pd.DataFrame, metric: str) -> float:
    """最高分位 - 最低分位"""
    if t.empty or metric not in t:
        return np.nan
    return float(t[metric].iloc[-1] - t[metric].iloc[0])


def hr(title):
    print("\n" + "=" * 100)
    print("  " + title)
    print("=" * 100)


def main():
    te = load("test")
    tr = load("train")
    print(f"test {len(te):,} 行, train {len(tr):,} 行")
    print("可用证据列:", [c for c in te.columns if c.startswith("m_")])

    v2 = te[(te["detector"] == "V2_fusion")].copy()
    v2t = v2[v2["touched"]].copy()
    print(f"\nV2 test: {len(v2):,} 个关键位, 其中触及 {len(v2t):,}")

    # ---------- 0. 全局基准 ----------
    hr("0. 全局基准（V2, test, 已触及）")
    for kind in ("support", "resistance"):
        g = v2t[v2t["kind"] == kind]
        dec = g[g["result"].isin(["hold", "break"])]
        p, lo, hi = wilson(int((dec["result"] == "hold").sum()), len(dec))
        print(f"  {kind:11s} n={len(g):6,d} hold={p:.4f} [{lo:.4f},{hi:.4f}] "
              f"edge_atr={g['edge_atr'].mean():+.4f} fwd_ret={g['fwd_ret_pct'].mean():+.4f}%")

    # ---------- 1. 逐证据分层 ----------
    factors = [
        ("m_n_events", "历史触及事件数（经典说法：测试次数越多越强）"),
        ("m_hold", "历史守住率 Wilson 下界"),
        ("m_vp", "成交量剖面强度"),
        ("m_close", "收盘价堆积强度"),
        ("m_pivot", "枢轴聚集强度"),
        ("m_round", "整数关口贴近度"),
        ("m_recency", "近期被触及的新鲜度"),
        ("width_atr", "区间宽度 / ATR"),
        ("dist_atr", "带符号距离 / ATR"),
        ("score", "融合总分"),
    ]
    summary = []
    for kind in ("support", "resistance"):
        hr(f"1.{'12'[kind=='resistance']} 单因子分层 — {kind}（V2, test, 已触及）")
        base = v2t[v2t["kind"] == kind]
        for col, desc in factors:
            if col not in base.columns:
                continue
            t = bucket_stats(base, col)
            if t.empty:
                continue
            print(f"\n--- {col}: {desc} ---")
            print(t[["bucket", "n", "n_dec", f"{col}_mean", "hold_rate",
                     "lo95", "hi95", "edge_atr", "fwd_ret_pct"]].to_string(index=False))
            # 首尾分位显著性
            first, last = t.iloc[0], t.iloc[-1]
            k1 = int(round(first["hold_rate"] * first["n_dec"]))
            k2 = int(round(last["hold_rate"] * last["n_dec"]))
            z, pv = two_prop_ztest(k2, int(last["n_dec"]), k1, int(first["n_dec"]))
            summary.append({
                "kind": kind, "factor": col, "desc": desc,
                "hold_spread_pp": spread(t, "hold_rate") * 100,
                "edge_spread": spread(t, "edge_atr"),
                "fwdret_spread_pp": spread(t, "fwd_ret_pct"),
                "z": z, "p_value": pv,
            })

    hr("2. 单因子有效性汇总（Q5 - Q1）")
    s = pd.DataFrame(summary)
    s = s.sort_values(["kind", "p_value"])
    print(s[["kind", "factor", "hold_spread_pp", "edge_spread",
             "fwdret_spread_pp", "z", "p_value", "desc"]].to_string(index=False))
    print("\n  p_value < 0.05 且 hold_spread 与 fwdret_spread 同号 => 该证据真的有信息")

    # ---------- 3. 距离是不是唯一起作用的东西 ----------
    hr("3. 控制距离后，证据还有没有增量信息？（support, test）")
    sup = v2t[v2t["kind"] == "support"].copy()
    sup["dist_bin"] = pd.qcut(sup["dist_atr"].abs(), 4, duplicates="drop")
    for col in ("m_vp", "m_pivot", "m_close", "m_n_events", "score"):
        if col not in sup.columns:
            continue
        rows = []
        for db, g in sup.groupby("dist_bin", observed=True):
            if len(g) < 400 or g[col].nunique() < 3:
                continue
            try:
                q = pd.qcut(g[col], 3, labels=["低", "中", "高"], duplicates="drop")
            except ValueError:
                continue
            gg = g.assign(_q=q).groupby("_q", observed=True)
            r = {"dist_bin": str(db)}
            for lab, h in gg:
                dec = h[h["result"].isin(["hold", "break"])]
                r[f"hold_{lab}"] = (dec["result"] == "hold").mean() if len(dec) else np.nan
                r[f"fwd_{lab}"] = h["fwd_ret_pct"].mean()
            rows.append(r)
        if rows:
            print(f"\n--- {col} 在各距离档内的分层 ---")
            print(pd.DataFrame(rows).to_string(index=False))

    # ---------- 4. 趋势/波动状态条件 ----------
    hr("4. 距离本身的效应曲线（所有检测器合并，support, test）")
    allsup = te[(te["kind"] == "support") & te["touched"]].copy()
    allsup["d"] = allsup["dist_atr"].abs()
    bins = [0, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    t = allsup.assign(db=pd.cut(allsup["d"], bins)).groupby("db", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "hold_rate": (g[g["result"].isin(["hold", "break"])]["result"] == "hold").mean(),
            "edge_atr": g["edge_atr"].mean(),
            "fwd_ret_pct": g["fwd_ret_pct"].mean(),
            "touch_rate": np.nan,
        }), include_groups=False)
    print(t.to_string())
    print("\n  如果 fwd_ret 随距离单调，说明'距离'是主效应，S/R 的具体位置是次要的。")

    # ---------- 5. 各检测器在同一距离档内对比（消除距离混淆） ----------
    hr("5. 同距离档内的检测器对比（support, test, 区宽<=1.5ATR）")
    d = te[(te["kind"] == "support") & te["touched"] & (te["width_atr"] <= 1.5)].copy()
    d["db"] = pd.cut(d["dist_atr"].abs(), bins)
    rows = []
    for (det, db), g in d.groupby(["detector", "db"], observed=True):
        if len(g) < 100:
            continue
        dec = g[g["result"].isin(["hold", "break"])]
        rows.append({"detector": det, "dist_bin": str(db), "n": len(g),
                     "hold_rate": (dec["result"] == "hold").mean() if len(dec) else np.nan,
                     "fwd_ret_pct": g["fwd_ret_pct"].mean(),
                     "edge_atr": g["edge_atr"].mean()})
    p = pd.DataFrame(rows).pivot(index="dist_bin", columns="detector",
                                 values="fwd_ret_pct")
    print("fwd_ret_pct（%）:")
    print(p.to_string())
    p2 = pd.DataFrame(rows).pivot(index="dist_bin", columns="detector",
                                  values="hold_rate")
    print("\nhold_rate:")
    print(p2.to_string())

    # ---------- 6. 触及事件数分布（诊断 m_hold 为什么是死权重） ----------
    hr("6. 诊断：历史触及事件数的分布")
    if "m_n_events" in v2.columns:
        ne = v2["m_n_events"]
        print(ne.describe().to_string())
        print("\n事件数取值分布:")
        print(ne.value_counts().sort_index().head(15).to_string())
        print(f"\n事件数 < 2 的占比: {(ne < 2).mean():.1%}  "
              f"=> 这部分 m_hold 全部落在常数兜底值 0.25，对排序毫无贡献")


if __name__ == "__main__":
    main()
