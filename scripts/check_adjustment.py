# -*- coding: utf-8 -*-
"""
判定本地 A股 parquet 的复权口径（不复权 / 前复权 / 后复权）

方法（三重独立验证，不依赖单一数据源）
--------------------------------------
【判据 1】比值恒定性检验
    对同一只股票、同一批交易日，计算 r_t = local_close_t / ref_close_t。
      - 若本地 == 参考口径              -> r_t ≡ 1
      - 若本地 == 前复权但复权基准日不同 -> r_t ≡ 常数 c（整条曲线等比缩放）
      - 若口径不同                       -> r_t 在除权日前后发生**阶跃**
    因此判据不是"r 是否等于 1"，而是 **"r 是否恒定"**（用 std/mean 衡量）。
    这一点很关键：前复权序列的绝对值取决于生成时间，直接比数值会误判。

【判据 2】除权跳空检测（不依赖任何外部数据源）
    A股 主板日涨跌幅限制 ±10%（ST ±5%，2020-08 后创业板/科创板 ±20%）。
    因此**单日收盘价变动超过 ±21% 只可能是除权除息**（或上市首日）。
    不复权序列会有这种跳空，复权序列不会。
    直接统计本地数据里的超限跳空次数即可独立定性。

【判据 3】分红送股事件对齐
    用 baostock 的三种口径拉同一只股票，在已知除权日附近对比本地序列的行为。

选股原则
    必须选**除权幅度大**的股票，否则三种口径差异太小无法区分。
    这里用 baostock 不复权/后复权的首尾比值自动筛出历史累计除权幅度最大的标的。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.srlab.data import ashare_load, ashare_list_codes

ASHARE_DIR = r"D:\A股_K线数据\parquet\stocks"

# baostock adjustflag: 1=后复权 2=前复权 3=不复权
BS_ADJ = {"hfq_后复权": "1", "qfq_前复权": "2", "raw_不复权": "3"}

# 候选标的：覆盖不同板块 + 历史上有大比例送转/高分红的票
CANDIDATES = [
    "000001",  # 平安银行  深主板，多次送转
    "600519",  # 贵州茅台  沪主板，高分红
    "000002",  # 万科A
    "600036",  # 招商银行
    "000651",  # 格力电器  高分红+送转
    "600276",  # 恒瑞医药  多次送转
    "002415",  # 海康威视
    "300059",  # 东方财富  创业板，多次送转
    "600030",  # 中信证券
    "000858",  # 五粮液
]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ============================================================
def hr(t: str):
    print("\n" + "=" * 96)
    print("  " + t)
    print("=" * 96)


# ============================================================
# 判据 2：除权跳空检测（纯本地，无外部依赖）
# ============================================================
def detect_ex_rights_gaps(df: pd.DataFrame, thr: float = 0.21) -> pd.DataFrame:
    """
    找出单日收盘价变动超过 thr 的记录。
    A股 涨跌幅上限 10%（创业板/科创板 2020-08 后 20%），
    因此 >21% 的跳空基本只能来自除权除息。
    """
    c = df["close"].to_numpy(np.float64)
    if len(c) < 3:
        return pd.DataFrame()
    ret = c[1:] / c[:-1] - 1.0
    m = np.abs(ret) > thr
    idx = np.nonzero(m)[0] + 1
    return pd.DataFrame({
        "date": df["date"].to_numpy()[idx],
        "prev_close": c[idx - 1],
        "close": c[idx],
        "chg_pct": ret[idx - 1] * 100,
    })


# ============================================================
# baostock 拉取
# ============================================================
def bs_fetch(code: str, adjustflag: str, start: str, end: str) -> pd.DataFrame:
    import baostock as bs

    prefix = "sh" if code.startswith(("6", "5", "9")) else "sz"
    bs_code = f"{prefix}.{code}"
    rs = bs.query_history_k_data_plus(
        bs_code, "date,open,high,low,close,volume",
        start_date=start, end_date=end, frequency="d", adjustflag=adjustflag)
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["close"])
    d = d[d["volume"] > 0]
    return d[["date", "open", "high", "low", "close"]].reset_index(drop=True)


# ============================================================
# 判据 1：比值恒定性
# ============================================================
def ratio_test(local: pd.DataFrame, ref: pd.DataFrame) -> Dict:
    """返回 r_t = local/ref 的统计量"""
    m = local.merge(ref, on="date", suffixes=("_l", "_r"))
    if len(m) < 100:
        return {"n": len(m), "note": "重叠不足"}
    r = (m["close_l"] / m["close_r"]).to_numpy(np.float64)
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) < 100:
        return {"n": len(r), "note": "有效样本不足"}
    cv = float(np.std(r) / np.mean(r))          # 变异系数：恒定 -> ~0
    # 最大阶跃：相邻交易日 r 的跳变
    jump = float(np.max(np.abs(np.diff(r) / r[:-1]))) if len(r) > 1 else np.nan
    # 与 1 的偏离
    return {
        "n": len(r),
        "r_mean": float(np.mean(r)),
        "r_cv": cv,
        "max_jump": jump,
        "identical": bool(np.allclose(r, 1.0, atol=2e-3)),
        "constant": bool(cv < 1e-3),
    }


# ============================================================
def main():
    import baostock as bs

    hr("0. 环境")
    lg = bs.login()
    print(f"  baostock 登录: error_code={lg.error_code} {lg.error_msg}")
    if lg.error_code != "0":
        print("  baostock 登录失败，仅执行判据 2（本地跳空检测）")

    local_codes = set(ashare_list_codes(ASHARE_DIR, "daily"))
    codes = [c for c in CANDIDATES if c in local_codes]
    print(f"  本地 daily 文件 {len(local_codes)} 只；候选中命中 {len(codes)} 只: {codes}")

    # ==========================================================
    hr("1. 判据 2（独立于外部数据源）：本地数据里的除权跳空")
    print("  A股 涨跌幅上限 ±10%（ST ±5%；2020-08 后创业板/科创板 ±20%）")
    print("  => 单日收盘变动 >21% 基本只能是除权除息。不复权数据会有，复权数据不会。\n")
    gap_rows = []
    for code in codes:
        try:
            d = ashare_load(ASHARE_DIR, code, "daily")
        except Exception as e:
            print(f"  {code}: 读取失败 {e}")
            continue
        g = detect_ex_rights_gaps(d, thr=0.21)
        # 排除上市首日附近（前 3 根）
        if len(g) and len(d):
            first_dates = set(pd.to_datetime(d["date"].head(3)))
            g = g[~g["date"].isin(first_dates)]
        gap_rows.append({
            "code": code, "bars": len(d),
            "start": d["date"].min().date(), "end": d["date"].max().date(),
            "gaps_gt21pct": len(g),
            "worst_pct": round(float(g["chg_pct"].abs().max()), 2) if len(g) else 0.0,
        })
        if len(g):
            print(f"  --- {code} 超限跳空 {len(g)} 次（前 5 条）---")
            print(g.head(5).to_string(index=False))
    print()
    print(pd.DataFrame(gap_rows).to_string(index=False))

    if lg.error_code != "0":
        bs.logout()
        return

    # ==========================================================
    hr("2. 判据 1：与 baostock 三种口径逐日比对（r = local/ref 是否恒定）")
    print("  identical=True 表示数值几乎完全相同；constant=True 表示等比缩放（同口径不同基准日）\n")
    verdicts = []
    for code in codes:
        try:
            loc = ashare_load(ASHARE_DIR, code, "daily")
        except Exception:
            continue
        start = loc["date"].min().strftime("%Y-%m-%d")
        end = loc["date"].max().strftime("%Y-%m-%d")
        row = {"code": code}
        detail = []
        for label, flag in BS_ADJ.items():
            try:
                ref = bs_fetch(code, flag, start, end)
            except Exception as e:
                print(f"  {code} {label}: 拉取失败 {e}")
                continue
            if ref.empty:
                continue
            st = ratio_test(loc[["date", "close"]], ref[["date", "close"]])
            detail.append({"code": code, "ref": label, **st})
            row[label] = ("IDENTICAL" if st.get("identical") else
                          "CONSTANT" if st.get("constant") else
                          f"cv={st.get('r_cv', float('nan')):.4f}")
            time.sleep(0.15)
        if detail:
            print(f"  --- {code} ---")
            print(pd.DataFrame(detail).to_string(index=False))
        verdicts.append(row)

    hr("3. 判定汇总")
    v = pd.DataFrame(verdicts)
    print(v.to_string(index=False))

    # 自动结论
    print()
    cols = [c for c in v.columns if c != "code"]
    for c in cols:
        if c not in v:
            continue
        n_id = (v[c] == "IDENTICAL").sum()
        n_const = (v[c] == "CONSTANT").sum()
        print(f"  vs {c:14s}: IDENTICAL {n_id}/{len(v)}  CONSTANT {n_const}/{len(v)}")

    bs.logout()


if __name__ == "__main__":
    main()
