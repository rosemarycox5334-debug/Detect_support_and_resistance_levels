# -*- coding: utf-8 -*-
"""
复权处理的效果验证

回答三个问题：
  1. 不复权的除权跳空，到底污染了多少比例的决策点？（决定要不要复权）
  2. 自建复权序列与 baostock 官方前复权差多少？（复权做得对不对）
  3. 后复权(因果) 与 前复权(非因果) 是否只差一个全局常数？（回测能不能用）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.srlab.adjust import (
    adjustment_factors, apply_adjustment, contamination_report,
    detect_ex_rights, validate_against_baostock,
)
from src.srlab.data import ashare_list_codes, ashare_load, _split_bucket

ASHARE_DIR = r"D:\A股_K线数据\parquet\stocks"
EVAL_START = "2012-01-01"

pd.set_option("display.width", 210)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def hr(t):
    print("\n" + "=" * 100)
    print("  " + t)
    print("=" * 100)


def main():
    codes_all = ashare_list_codes(ASHARE_DIR, "daily")
    # 取与回测同一批标的（前 300 只里符合条件的），保证结论可迁移
    sample = codes_all[:300]

    # ==========================================================
    hr("1. 污染度：有多少决策点的回看窗口跨过除权日")
    print(f"  样本 {len(sample)} 只；回测配置 warmup=300, 回看=750 根, 区间 >= {EVAL_START}\n")
    rows = []
    for code in sample:
        try:
            raw = ashare_load(ASHARE_DIR, code, "daily", adjust=False)
        except Exception:
            continue
        if len(raw) < 1500:
            continue
        ev = detect_ex_rights(raw)
        rep = contamination_report(raw, ev, warmup=300, lookback=750, start=EVAL_START)
        if rep.get("n_points", 0) == 0:
            continue
        rows.append({"code": code, "bars": len(raw), "n_ex_rights": len(ev), **rep})
    t = pd.DataFrame(rows)
    print(f"  有效样本 {len(t)} 只")
    print(f"  识别到的除权次数: 中位 {t['n_ex_rights'].median():.0f}, "
          f"均值 {t['n_ex_rights'].mean():.2f}, 最大 {t['n_ex_rights'].max()}")
    print(f"  无除权事件的股票: {(t['n_ex_rights'] == 0).sum()} / {len(t)}")
    print()
    print("  受污染决策点占比的分布:")
    q = t["pct"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    print(q.to_string())
    print(f"\n  >>> 全样本加权: {t['n_contaminated'].sum():,} / {t['n_points'].sum():,} "
          f"= {t['n_contaminated'].sum() / t['n_points'].sum() * 100:.1f}% 的决策点回看窗口内含除权跳空")
    print(f"  >>> 窗口内平均除权事件数: {t['avg_events_in_window'].mean():.3f}")
    print("\n  污染最重的 10 只:")
    print(t.nlargest(10, "pct")[["code", "bars", "n_ex_rights", "n_points",
                                 "n_contaminated", "pct"]].to_string(index=False))

    # ==========================================================
    hr("2. 尺度不变性：后复权(因果) vs 前复权(非因果) 是否只差全局常数")
    print("  若成立，回测用后复权、界面用前复权，两者检测结果的相对几何一致。\n")
    rows = []
    for code in sample[:40]:
        try:
            raw = ashare_load(ASHARE_DIR, code, "daily", adjust=False)
        except Exception:
            continue
        ev = detect_ex_rights(raw)
        if not ev:
            continue
        f_h = adjustment_factors(len(raw), ev, "hfq")
        f_q = adjustment_factors(len(raw), ev, "qfq")
        r = f_q / f_h
        rows.append({"code": code, "n_ex": len(ev),
                     "ratio_mean": float(r.mean()),
                     "ratio_cv": float(r.std() / r.mean()),
                     "max_dev": float(np.max(np.abs(r / r[0] - 1.0)))})
    t2 = pd.DataFrame(rows)
    print(t2.head(15).to_string(index=False))
    print(f"\n  >>> ratio_cv 最大值 = {t2['ratio_cv'].max():.3e}  "
          f"max_dev 最大值 = {t2['max_dev'].max():.3e}")
    print("  [OK] 两者比值恒定 => 只差全局常数，尺度不变算法输出一致"
          if t2["max_dev"].max() < 1e-9 else "  [FAIL] 不是常数，需排查")

    # ==========================================================
    hr("3. 自建复权 vs baostock 官方前复权")
    import baostock as bs
    lg = bs.login()
    print(f"  baostock 登录: {lg.error_code} {lg.error_msg}\n")
    if lg.error_code != "0":
        print("  跳过")
        return

    # 挑除权次数多的标的，差异最容易暴露
    cand = t.nlargest(12, "n_ex_rights")["code"].tolist()
    rows = []
    for code in cand:
        try:
            adj = ashare_load(ASHARE_DIR, code, "daily", adjust=True, adjust_mode="qfq")
            raw = ashare_load(ASHARE_DIR, code, "daily", adjust=False)
        except Exception as e:
            print(f"  {code}: {e}")
            continue
        r_adj = validate_against_baostock(code, adj)
        r_raw = validate_against_baostock(code, raw)
        rows.append({
            "code": code,
            "n_ex_rights": adj.attrs.get("n_ex_rights", 0),
            "raw_cv": r_raw.get("r_cv"), "raw_maxjump": r_raw.get("max_jump"),
            "adj_cv": r_adj.get("r_cv"), "adj_maxjump": r_adj.get("max_jump"),
        })
    t3 = pd.DataFrame(rows)
    t3["cv_改善倍数"] = t3["raw_cv"] / t3["adj_cv"]
    print(t3.to_string(index=False))
    print("\n  raw_cv  = 不复权本地数据 vs baostock前复权 的比值变异系数（应该很大）")
    print("  adj_cv  = 自建复权后 vs baostock前复权 的比值变异系数（应该接近 0）")
    print(f"\n  >>> 中位 cv: {t3['raw_cv'].median():.4f} -> {t3['adj_cv'].median():.4f}"
          f"  (改善 {t3['raw_cv'].median() / max(t3['adj_cv'].median(), 1e-12):.0f}x)")
    print(f"  >>> 复权后残余最大阶跃中位数: {t3['adj_maxjump'].median():.4f}"
          f"  (复权前 {t3['raw_maxjump'].median():.4f})")

    bs.logout()


if __name__ == "__main__":
    main()
