# -*- coding: utf-8 -*-
"""
无泄漏走查回测运行器

结构性防泄漏
------------
    ctx.high = sym.high[:t+1]      <- numpy 视图，检测器**物理上拿不到** t 之后的数据
    评测函数则只读 sym.*[t+1 : t+horizon]

    枢轴是唯一需要小心的地方：ZigZag 天生需要未来才能确认极值。
    解决办法见 pivots.py —— 一次性算出全部枢轴并记录 confirm_idx，
    t 时刻只放行 confirm_idx <= t 的枢轴。等价性由 leakage.py 逐点验证。

统一后处理（对所有检测器一致，保证公平）
    1. 丢弃"现价已在区内"的关键位（support.high >= price 或 resistance.low <= price），
       这类位无法构成一次干净的测试；
    2. 丢弃距现价 > max_dist_atr 倍 ATR 的位（交易上无意义）；
    3. 每个检测器每个决策点最多取 max_out 个（按 score 降序）。
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import Ctx, Level
from .data import Symbol
from .labeling import evaluate_level, future_pivot_errors, outcome_row
from .pivots import Pivots, zigzag


def _postprocess(levels: List[Level], price: float, atr: float,
                 min_dist_atr: float, max_dist_atr: float,
                 max_out_per_side: int) -> List[Level]:
    """
    对所有检测器一致的后处理，这是公平比较的前提。

    距离带 [min_dist_atr, max_dist_atr]
        冒烟测试暴露的问题：placebo_fixed 把关键位机械地放在 ±1.5/2.5/4.0 ATR，
        几何中位误差 0.69 ATR，反而**优于** V2 的 1.18 ATR。原因不是它更聪明，
        而是未来 40 根内的摆动极值绝大多数就落在 1~4 ATR 这个区间，
        它均匀铺满了这个区间，而 V2 会把位放到 0.2 ATR 或 7 ATR 处。
        因此必须把所有检测器约束到同一个距离带，"命中率"才反映定位能力
        而不是距离分布的巧合。

    每侧限流 max_out_per_side
        输出越多，召回天然越高。按侧限流而不是按总数限流，
        避免某个检测器把配额全花在一侧。
    """
    atr = max(atr, 1e-12)
    buckets = {"support": [], "resistance": []}
    for z in levels:
        if z.kind == "support" and z.high >= price:
            continue
        if z.kind == "resistance" and z.low <= price:
            continue
        d = abs(z.center - price) / atr
        if d < min_dist_atr or d > max_dist_atr:
            continue
        if z.kind in buckets:
            buckets[z.kind].append(z)
    out: List[Level] = []
    for k in ("support", "resistance"):
        b = sorted(buckets[k], key=lambda x: -x.score)
        out.extend(b[:max_out_per_side])
    out.sort(key=lambda x: -x.score)
    return out


def run_walkforward(
    symbols: Sequence[Symbol],
    detectors: Sequence[object],
    warmup: int = 300,
    rebalance_every: int = 20,
    horizon: int = 40,
    hold_bars: int = 10,
    break_atr: float = 0.5,
    target_atr: float = 1.0,
    pivot_k_atr: float = 2.0,
    pivot_view_lookback: int = 600,
    min_dist_atr: float = 0.5,
    max_dist_atr: float = 5.0,
    max_out_per_side: int = 3,
    geo_top_k: int = 3,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    返回 {'levels': 逐关键位明细, 'geo': 逐决策点几何精度}

    geo_top_k: 几何精度评测时每侧只取得分最高的 k 个关键位，
               保证不同检测器在"输出数量"上同口径。
    """
    rows: List[Dict] = []
    geo_rows: List[Dict] = []
    t0 = time.time()

    for si, sym in enumerate(symbols):
        n = len(sym)
        if n < warmup + horizon + 10:
            continue
        piv = zigzag(sym.high, sym.low, sym.atr, k_atr=pivot_k_atr)
        ts = range(warmup, n - horizon - 1, rebalance_every)

        for t in ts:
            atr_t = float(sym.atr[t])
            if not np.isfinite(atr_t) or atr_t <= 0:
                continue
            price_t = float(sym.close[t])
            ctx = Ctx(
                code=sym.code, t=t,
                open_=sym.open_[:t + 1], high=sym.high[:t + 1],
                low=sym.low[:t + 1], close=sym.close[:t + 1],
                volume=sym.volume[:t + 1],
                atr=atr_t, tick=sym.tick,
                atr_arr=sym.atr[:t + 1],
                pivots=piv.view_at(t, max_lookback=pivot_view_lookback),
            )
            date_t = sym.date[t]

            for det in detectors:
                try:
                    lv = det.detect(ctx)
                except Exception as e:
                    if verbose:
                        print(f"  [warn] {det.name} {sym.code}@{t}: {e}")
                    continue
                lv = _postprocess(lv, price_t, atr_t, min_dist_atr,
                                  max_dist_atr, max_out_per_side)

                for z in lv:
                    oc = evaluate_level(
                        sym.high, sym.low, sym.close, t, z, atr_t,
                        horizon=horizon, hold_bars=hold_bars,
                        break_atr=break_atr, target_atr=target_atr)
                    rows.append(outcome_row(sym.code, t, date_t, det.name,
                                            z, oc, atr_t, price_t))

                g = future_pivot_errors(piv, t, horizon, lv, atr_t,
                                       top_k_per_side=geo_top_k)
                g.update({"code": sym.code, "t": t, "date": date_t,
                          "detector": det.name, "n_levels": len(lv)})
                geo_rows.append(g)

        if verbose and (si + 1) % 25 == 0:
            print(f"  [wf] {si+1}/{len(symbols)} 标的完成, "
                  f"{len(rows)} 条关键位记录, {time.time()-t0:.0f}s")

    if verbose:
        print(f"[wf] 完成: {len(symbols)} 标的, {len(rows)} 条关键位, "
              f"耗时 {time.time()-t0:.0f}s")
    return {
        "levels": pd.DataFrame(rows),
        "geo": pd.DataFrame(geo_rows),
    }


def summarize_geo(geo: pd.DataFrame) -> pd.DataFrame:
    """几何精度按检测器汇总"""
    if geo.empty:
        return pd.DataFrame()
    g = geo.copy()
    agg = g.groupby("detector").agg(
        n_points=("t", "size"),
        avg_n_out=("n_levels", "mean"),
        avg_n_eval=("n_pred_levels", "mean"),
        recall_hit_05=("recall_hit_05", "mean"),
        recall_hit_10=("recall_hit_10", "mean"),
        median_err_atr=("median_err_atr", "median"),
        precision_hit_05=("precision_hit_05", "mean"),
        recall_per_level=("recall_per_level", "mean"),
        n_truth=("n_truth_matched", "sum"),
    )
    return agg
