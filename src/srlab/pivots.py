# -*- coding: utf-8 -*-
"""
因果 ZigZag 枢轴检测

为什么需要"确认时刻"(confirm_idx)：
    传统 ZigZag / fractal 枢轴天生带未来信息 —— 判定第 i 根是摆动高点，
    需要看到 i 之后价格回落了足够幅度。如果直接把枢轴喂给 t 时刻的检测器，
    就会用到未来信息，回测结果全部作废。

本模块的做法：
    对整条序列**一次性**跑 O(n) 的 ZigZag，但为每个枢轴记录它被确认的
    那一根 bar 的下标 confirm_idx（即价格反向走出 k*ATR 的那一刻）。
    t 时刻只允许使用 confirm_idx <= t 的枢轴 —— 这与"在 t 时刻实时计算"
    的结果完全等价，但省掉了 O(n^2) 的重复计算。

    等价性由 leakage.py 中的 test_pivot_causality 逐点验证。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import PivotView


@dataclass
class Pivots:
    idx: np.ndarray
    price: np.ndarray
    kind: np.ndarray
    confirm_idx: np.ndarray
    leg_atr: np.ndarray

    def view_at(self, t: int, max_lookback: int = 0) -> PivotView:
        """返回 t 时刻可见的枢轴（confirm_idx <= t）"""
        m = self.confirm_idx <= t
        if max_lookback > 0:
            m &= self.idx >= (t - max_lookback)
        return PivotView(idx=self.idx[m], price=self.price[m], kind=self.kind[m],
                         confirm_idx=self.confirm_idx[m], leg_atr=self.leg_atr[m])

    def __len__(self) -> int:
        return len(self.idx)


def zigzag(high: np.ndarray, low: np.ndarray, atr: np.ndarray,
           k_atr: float = 2.0) -> Pivots:
    """
    ATR 阈值 ZigZag。

    参数:
        k_atr: 反向确认阈值，单位为当时的 ATR。2.0 表示价格从极值反向
               走出 2 倍 ATR 才确认该极值为枢轴。越大 -> 枢轴越少越显著。

    返回 Pivots，其中：
        idx         枢轴 bar
        price       枢轴价（高点用 high，低点用 low）
        kind        +1 高点 / -1 低点
        confirm_idx 确认 bar（严格 > idx）
        leg_atr     从上一个枢轴到本枢轴的价格行程 / 本枢轴处 ATR
    """
    n = len(high)
    if n < 3:
        return Pivots(*(np.array([], dtype=t) for t in
                        (np.int64, np.float64, np.int64, np.int64, np.float64)))

    p_idx, p_price, p_kind, p_conf, p_leg = [], [], [], [], []

    # 用前若干根确定初始方向：先假设在找高点，同时跟踪低点，谁先被确认谁算
    ext_hi_i, ext_hi = 0, high[0]
    ext_lo_i, ext_lo = 0, low[0]
    direction = 0  # 0=未定, +1=正在找高点, -1=正在找低点
    last_pivot_price = np.nan

    for i in range(1, n):
        thr = k_atr * max(atr[i], 1e-12)

        if direction >= 0 and high[i] > ext_hi:
            ext_hi, ext_hi_i = high[i], i
        if direction <= 0 and low[i] < ext_lo:
            ext_lo, ext_lo_i = low[i], i

        if direction == 0:
            # 未定向：先出现哪个方向的 thr 突破就定向
            if ext_hi - low[i] >= thr:
                direction = -1
                _push(p_idx, p_price, p_kind, p_conf, p_leg,
                      ext_hi_i, ext_hi, 1, i, last_pivot_price, atr)
                last_pivot_price = ext_hi
                ext_lo, ext_lo_i = low[i], i
            elif high[i] - ext_lo >= thr:
                direction = 1
                _push(p_idx, p_price, p_kind, p_conf, p_leg,
                      ext_lo_i, ext_lo, -1, i, last_pivot_price, atr)
                last_pivot_price = ext_lo
                ext_hi, ext_hi_i = high[i], i
            continue

        if direction == 1:
            # 正在找高点：价格从 ext_hi 回落 thr -> 确认高点
            if ext_hi - low[i] >= thr:
                _push(p_idx, p_price, p_kind, p_conf, p_leg,
                      ext_hi_i, ext_hi, 1, i, last_pivot_price, atr)
                last_pivot_price = ext_hi
                direction = -1
                ext_lo, ext_lo_i = low[i], i
        else:
            # 正在找低点：价格从 ext_lo 反弹 thr -> 确认低点
            if high[i] - ext_lo >= thr:
                _push(p_idx, p_price, p_kind, p_conf, p_leg,
                      ext_lo_i, ext_lo, -1, i, last_pivot_price, atr)
                last_pivot_price = ext_lo
                direction = 1
                ext_hi, ext_hi_i = high[i], i

    return Pivots(
        idx=np.asarray(p_idx, dtype=np.int64),
        price=np.asarray(p_price, dtype=np.float64),
        kind=np.asarray(p_kind, dtype=np.int64),
        confirm_idx=np.asarray(p_conf, dtype=np.int64),
        leg_atr=np.asarray(p_leg, dtype=np.float64),
    )


def _push(p_idx, p_price, p_kind, p_conf, p_leg,
          idx, price, kind, conf, last_price, atr):
    leg = 0.0
    if not np.isnan(last_price):
        leg = abs(price - last_price) / max(atr[idx], 1e-12)
    p_idx.append(idx)
    p_price.append(float(price))
    p_kind.append(int(kind))
    p_conf.append(int(conf))
    p_leg.append(float(leg))


def fractal_pivots(high: np.ndarray, low: np.ndarray, k: int = 3) -> Pivots:
    """
    k 阶分形枢轴（对照用）：high[i] 是 [i-k, i+k] 的最大值。
    确认时刻 = i + k（必须等右侧 k 根走完才能判定），因此同样是因果的。
    """
    n = len(high)
    idx, price, kind, conf = [], [], [], []
    for i in range(k, n - k):
        w_h = high[i - k:i + k + 1]
        w_l = low[i - k:i + k + 1]
        if high[i] == w_h.max():
            idx.append(i); price.append(float(high[i])); kind.append(1); conf.append(i + k)
        if low[i] == w_l.min():
            idx.append(i); price.append(float(low[i])); kind.append(-1); conf.append(i + k)
    order = np.argsort(conf)
    return Pivots(
        idx=np.asarray(idx, dtype=np.int64)[order],
        price=np.asarray(price, dtype=np.float64)[order],
        kind=np.asarray(kind, dtype=np.int64)[order],
        confirm_idx=np.asarray(conf, dtype=np.int64)[order],
        leg_atr=np.zeros(len(idx), dtype=np.float64)[order],
    )
