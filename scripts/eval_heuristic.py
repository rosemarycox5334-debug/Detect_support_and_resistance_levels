# -*- coding: utf-8 -*-
"""
评估"开盘缺口启发式"复权的精度 —— 以已缓存的权威因子为真值

真值 = data/adj_factors.parquet（baostock 后复权/不复权 相除，已去量化噪声）
被测 = detect_ex_rights + apply_adjustment 的纯离线启发式

对比两档阈值：
    旧: 固定 0.22（高于所有涨跌幅上限，绝不误判但会漏掉 10送2 之类）
    新: 按板块/日期取当日涨跌幅上限 + 0.5%（主板 10.5%，创业板20%时段 20.5%）

指标（都以"复权后序列 vs 真值复权序列"的比值衡量，比值恒定才算对）：
    r_cv      比值变异系数，0 = 完全一致
    max_jump  比值的最大单日阶跃 = 残余价格断层，最关键
    n_missed  漏掉的除权事件数（真值阶跃数 - 识别数）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.srlab.adjust import (
    FACTOR_CACHE_DEFAULT, FactorStore, adjustment_factors, apply_factor,
    detect_ex_rights,
)
from src.srlab.data import ashare_load

ASHARE_DIR = r"D:\A股_K线数据\parquet\stocks"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.5f}")


def cmp_series(a: np.ndarray, b: np.ndarray):
    """a, b 已对齐的两条价格序列 -> (cv, max_jump)"""
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    r = a[ok] / b[ok]
    if len(r) < 50:
        return np.nan, np.nan
    cv = float(np.std(r) / np.mean(r))
    jump = float(np.max(np.abs(np.diff(r) / r[:-1]))) if len(r) > 1 else np.nan
    return cv, jump


def main():
    store = FactorStore(FACTOR_CACHE_DEFAULT)
    print(f"权威因子缓存: {len(store)} 只  (源: baostock)")
    if len(store) == 0:
        print("缓存为空，先跑 scripts/build_factors.py")
        return

    rows = []
    for code in sorted(store._by_code.keys()):
        try:
            raw = ashare_load(ASHARE_DIR, code, "daily", adjust=False)
        except Exception:
            continue
        fac = store.get(code)
        if fac is None or len(fac) < 100:
            continue
        # 真值：权威因子后复权
        truth = apply_factor(raw, fac, mode="hfq")
        # 只在因子覆盖区间内比较
        lo = pd.Timestamp(fac["date"].min())
        mask = raw["date"] >= lo
        if mask.sum() < 300:
            continue
        n_truth_steps = int(truth.attrs.get("n_ex_rights", 0))

        rec = {"code": code, "bars": int(mask.sum()), "真值除权数": n_truth_steps}
        for label, thr in (("旧_0.22", 0.22), ("新_按板块", None)):
            ev = detect_ex_rights(raw, threshold=thr, code=code)
            f = adjustment_factors(len(raw), ev, "hfq")
            test_close = raw["close"].to_numpy(np.float64) * f
            cv, jump = cmp_series(test_close[mask.to_numpy()],
                                 truth["close"].to_numpy(np.float64)[mask.to_numpy()])
            rec[f"{label}_识别数"] = len(ev)
            rec[f"{label}_cv"] = cv
            rec[f"{label}_maxjump"] = jump
        # 完全不复权的基线
        cv0, j0 = cmp_series(raw["close"].to_numpy(np.float64)[mask.to_numpy()],
                             truth["close"].to_numpy(np.float64)[mask.to_numpy()])
        rec["不复权_cv"] = cv0
        rec["不复权_maxjump"] = j0
        rows.append(rec)

    t = pd.DataFrame(rows)
    print(f"\n有效对比样本: {len(t)} 只\n")
    cols = ["code", "bars", "真值除权数", "旧_0.22_识别数", "新_按板块_识别数",
            "不复权_maxjump", "旧_0.22_maxjump", "新_按板块_maxjump",
            "不复权_cv", "旧_0.22_cv", "新_按板块_cv"]
    print(t[cols].head(25).to_string(index=False))

    print("\n" + "=" * 100)
    print("  汇总（中位数）")
    print("=" * 100)
    summ = pd.DataFrame({
        "口径": ["不复权", "启发式 阈值0.22", "启发式 按板块限幅"],
        "残余最大断层_中位": [t["不复权_maxjump"].median(),
                              t["旧_0.22_maxjump"].median(),
                              t["新_按板块_maxjump"].median()],
        "残余最大断层_P90": [t["不复权_maxjump"].quantile(.9),
                             t["旧_0.22_maxjump"].quantile(.9),
                             t["新_按板块_maxjump"].quantile(.9)],
        "比值cv_中位": [t["不复权_cv"].median(),
                        t["旧_0.22_cv"].median(),
                        t["新_按板块_cv"].median()],
        "识别事件数_中位": [0, t["旧_0.22_识别数"].median(), t["新_按板块_识别数"].median()],
    })
    print(summ.to_string(index=False))
    print(f"\n  真值除权数 中位 = {t['真值除权数'].median():.0f}")
    print(f"  漏检率: 旧 {1 - t['旧_0.22_识别数'].sum()/max(t['真值除权数'].sum(),1):.1%}"
          f"  新 {1 - t['新_按板块_识别数'].sum()/max(t['真值除权数'].sum(),1):.1%}")

    print("\n  新口径仍有 >3% 残余断层的股票（这些必须用权威因子缓存）:")
    bad = t[t["新_按板块_maxjump"] > 0.03].sort_values("新_按板块_maxjump", ascending=False)
    print(bad[["code", "真值除权数", "新_按板块_识别数", "新_按板块_maxjump"]].head(15).to_string(index=False)
          if len(bad) else "    （无）")
    print(f"\n  >>> 占比 {len(bad)}/{len(t)} = {len(bad)/max(len(t),1):.1%}")


if __name__ == "__main__":
    main()
