# -*- coding: utf-8 -*-
"""
前瞻标注：一个"支撑位"到底算不算准？

这是整套评测最容易做错的地方。旧实现的判据是
    「某根K线的 [low,high] 与区间重叠」= 触及
    「后 5 根收盘价没有跌破 区间下沿*0.98」= 守住
它有四个致命问题（已在 _audit.py 实测确认）：
  1. 逐根计数：价格在区内盘整 20 天 = 20 次"触及"（实测 000001 单个区
     在 8403 根历史里录得 1430 次"触及"）；
  2. 不分方向：从下方涨上来碰到"支撑"也算测试支撑；
  3. 阈值写死 2%：跨标的从 0.03 倍 ATR 到 0.88 倍 ATR 不等；
  4. 区宽无约束（实测 2.7~5.7 倍 ATR）：区间越宽越容易"守住"，
     于是"提高胜率"的最优策略是把区间画得无限宽 —— 指标可被算法白嫖。

本模块的定义
------------
【触及事件】(touch event)
    支撑：要求 t 时刻 close > zone.high（价格在区上方，approach from above），
          此后首次出现 low[j] <= zone.high 即为一次触及，j 为触及 bar。
    阻力：对称，要求 close < zone.low，首次 high[j] >= zone.low。
    一次评测窗口内只取**第一次**触及，天然不存在重复计数。

【结果判定】(first-touch, first-event-wins)
    从 j 开始向后最多 K 根，逐根按发生顺序判定，先满足者胜出：
      突破 break: close[m] < zone.low - break_atr * ATR_t        （支撑）
      守住 hold : high[m]  >= zone.high + target_atr * ATR_t     （支撑）
      都没发生 -> undecided（不计入胜率分母，单独统计占比）
    ATR_t 是**决策时刻 t 的 ATR**，不是未来的 ATR —— 阈值本身不含未来信息。

【区宽惩罚】
    评测同时输出 width_atr。汇总时对宽度做分层报告，并提供
    width_capped 口径（只统计 width_atr <= 1.5 的区），
    杜绝"靠画宽区间刷胜率"。

【几何精度】
    独立于交易结果，直接问："未来真实的摆动低点，离预测支撑位有多远？"
    误差以 ATR 为单位，给出 hit@0.5ATR / hit@1.0ATR 与中位误差。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from .base import Level
from .pivots import Pivots


@dataclass
class Outcome:
    """单个 Level 在一次前瞻窗口内的结果"""
    touched: bool
    touch_bar: int          # 相对 t 的偏移，未触及为 -1
    result: str             # 'hold' | 'break' | 'undecided' | 'no_touch'
    mfe_atr: float          # 触及后最大有利波动 / ATR_t（支撑=向上，阻力=向下）
    mae_atr: float          # 触及后最大不利波动 / ATR_t
    fwd_ret_pct: float      # 在区边沿进场、持有 K 根的收益率（含方向）
    edge_atr: float         # mfe - mae
    # 几何精度
    pivot_err_atr: float    # 最近的未来同向枢轴到 center 的距离 / ATR_t，无枢轴为 nan


NO_TOUCH = Outcome(False, -1, "no_touch", np.nan, np.nan, np.nan, np.nan, np.nan)


def evaluate_level(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    t: int,
    level: Level,
    atr_t: float,
    horizon: int = 40,
    hold_bars: int = 10,
    break_atr: float = 0.5,
    target_atr: float = 1.0,
) -> Outcome:
    """
    对单个 Level 做前瞻评测。只读 t+1 .. t+horizon 的数据。

    参数:
        horizon    : 等待触及的最长 bar 数
        hold_bars  : 触及后观察结果的 bar 数
        break_atr  : 判定突破所需的越界幅度（ATR 倍数）
        target_atr : 判定守住所需的反向行程（ATR 倍数）
    """
    n = len(close)
    atr_t = max(atr_t, 1e-12)
    is_sup = level.kind == "support"
    price_t = float(close[t])

    # approach 方向校验：支撑要求现价在区上方，否则本次不构成有效测试
    if is_sup:
        if price_t <= level.high:
            return NO_TOUCH
    else:
        if price_t >= level.low:
            return NO_TOUCH

    end = min(t + horizon, n - 1)
    if end <= t:
        return NO_TOUCH

    # 找首次触及
    seg = slice(t + 1, end + 1)
    if is_sup:
        hit = np.nonzero(low[seg] <= level.high)[0]
    else:
        hit = np.nonzero(high[seg] >= level.low)[0]
    if len(hit) == 0:
        return NO_TOUCH
    j = t + 1 + int(hit[0])

    # --- 结果窗口 ---
    # 守住/突破的判定从触及 bar j 开始（含 j：跳空直接穿透也是一次真突破）
    e2 = min(j + hold_bars, n - 1)
    hh = high[j:e2 + 1]
    ll = low[j:e2 + 1]
    cc = close[j:e2 + 1]

    # --- 进出场基准 ---
    # 必须用 close[j]，不能用区间边沿。
    # 原因：触及的定义就是 low[j] <= zone.high，所以"以 zone.high 为进场价"时，
    # MAE 天然有一个 >= (zone.high - low[j]) 的正下限，而 MFE 没有对称下限，
    # 于是 edge = MFE - MAE 被结构性地压成负数（冒烟测试中全体检测器 edge_atr
    # 均为负，就是这个偏差造成的）。
    # close[j] 是 A股 T+1 下真正可成交的价格，且对多空对称、无构造性偏差。
    entry = float(close[j])
    if entry <= 0:
        return NO_TOUCH
    # 波动幅度只看 j 之后（j 当天的波动已经发生，不属于"进场后"）
    e3 = min(j + 1 + hold_bars, n - 1)
    if e3 > j:
        fh = high[j + 1:e3 + 1]
        fl = low[j + 1:e3 + 1]
    else:
        fh = np.array([entry])
        fl = np.array([entry])

    if is_sup:
        brk_px = level.low - break_atr * atr_t
        tgt_px = level.high + target_atr * atr_t
        brk_hit = np.nonzero(cc < brk_px)[0]
        tgt_hit = np.nonzero(hh >= tgt_px)[0]
        mfe = (fh.max() - entry) / atr_t
        mae = (entry - fl.min()) / atr_t
        fwd = (close[e3] - entry) / entry * 100
    else:
        brk_px = level.high + break_atr * atr_t
        tgt_px = level.low - target_atr * atr_t
        brk_hit = np.nonzero(cc > brk_px)[0]
        tgt_hit = np.nonzero(ll <= tgt_px)[0]
        mfe = (entry - fl.min()) / atr_t
        mae = (fh.max() - entry) / atr_t
        fwd = (entry - close[e3]) / entry * 100

    b = int(brk_hit[0]) if len(brk_hit) else 10 ** 9
    g = int(tgt_hit[0]) if len(tgt_hit) else 10 ** 9
    if b == g == 10 ** 9:
        result = "undecided"
    elif g <= b:
        result = "hold"
    else:
        result = "break"

    return Outcome(
        touched=True, touch_bar=j - t, result=result,
        mfe_atr=float(mfe), mae_atr=float(mae),
        fwd_ret_pct=float(fwd), edge_atr=float(mfe - mae),
        pivot_err_atr=np.nan,
    )


def future_pivot_errors(
    pivots: Pivots,
    t: int,
    horizon: int,
    levels: List[Level],
    atr_t: float,
    top_k_per_side: int = 0,
) -> Dict[str, float]:
    """
    几何精度：未来窗口内真实发生的摆动极值，离预测关键位有多近？

    真值 = ZigZag 枢轴，且要求枢轴 bar 落在 (t, t+horizon]。
    注意这里用的是枢轴的 idx（发生时刻）而不是 confirm_idx —— 评测阶段
    允许使用未来信息，这正是"真值"的含义。

    参数:
        top_k_per_side: >0 时，每侧只取得分最高的 k 个关键位再评测。
            这是**跨检测器公平比较的必要条件** —— 输出的关键位越多，
            召回率天然越高（冒烟测试中 placebo_fixed 靠输出 6 个位
            拿到 0.39 召回，高于 V2 输出 4.6 个位的 0.26）。

    返回:
        n_pivot_low / n_pivot_high : 窗口内真值枢轴数
        recall_hit_05 / recall_hit_10 : 有多少比例的真值枢轴，能在
            0.5 / 1.0 倍 ATR 内找到一个预测关键位（召回）
        precision_hit_05 : 有多少比例的预测关键位，命中了某个真值枢轴（精确率）
        median_err_atr   : 真值枢轴到最近预测位的中位距离（ATR 倍数）
        recall_per_level : 召回 / 预测位数，衡量"单位关键位的信息效率"
    """
    atr_t = max(atr_t, 1e-12)
    m = (pivots.idx > t) & (pivots.idx <= t + horizon)
    pv_px, pv_kind = pivots.price[m], pivots.kind[m]

    def _side(kind: str) -> np.ndarray:
        zs = [z for z in levels if z.kind == kind]
        zs.sort(key=lambda z: -z.score)
        if top_k_per_side > 0:
            zs = zs[:top_k_per_side]
        return np.array([z.center for z in zs], dtype=np.float64)

    sup = _side("support")
    res = _side("resistance")

    errs: List[float] = []
    n_lo = int((pv_kind == -1).sum())
    n_hi = int((pv_kind == 1).sum())

    # 每个真值枢轴 -> 最近的同类预测位
    for px, kd in zip(pv_px, pv_kind):
        cands = sup if kd == -1 else res
        if len(cands) == 0:
            continue
        errs.append(float(np.min(np.abs(cands - px)) / atr_t))
    errs_a = np.array(errs, dtype=np.float64)

    # 每个预测位 -> 是否命中某个真值枢轴
    prec_hits = 0
    n_pred = 0
    for cands, kd in ((sup, -1), (res, 1)):
        truth = pv_px[pv_kind == kd]
        for c in cands:
            n_pred += 1
            if len(truth) and np.min(np.abs(truth - c)) / atr_t <= 0.5:
                prec_hits += 1

    rec05 = float((errs_a <= 0.5).mean()) if len(errs_a) else np.nan
    return {
        "n_pivot_low": n_lo,
        "n_pivot_high": n_hi,
        "n_truth_matched": len(errs_a),
        "recall_hit_05": rec05,
        "recall_hit_10": float((errs_a <= 1.0).mean()) if len(errs_a) else np.nan,
        "median_err_atr": float(np.median(errs_a)) if len(errs_a) else np.nan,
        "precision_hit_05": float(prec_hits / n_pred) if n_pred else np.nan,
        "n_pred_levels": n_pred,
        "recall_per_level": (rec05 / n_pred) if (n_pred and np.isfinite(rec05)) else np.nan,
    }


def outcome_row(code: str, t: int, date, detector: str, level: Level,
                oc: Outcome, atr_t: float, price_t: float) -> Dict:
    """展平成一行，供 DataFrame 聚合"""
    d = {
        "code": code, "t": t, "date": date, "detector": detector,
        "kind": level.kind, "center": level.center,
        "low": level.low, "high": level.high,
        "score": level.score, "confidence": level.confidence,
        "width_atr": level.width_atr, "dist_atr": level.dist_atr,
        "dist_pct": level.dist_pct,
        "atr": atr_t, "price": price_t,
    }
    d.update(asdict(oc))
    for k, v in level.meta.items():
        d[f"m_{k}"] = v
    return d
