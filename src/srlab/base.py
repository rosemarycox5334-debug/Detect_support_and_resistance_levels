# -*- coding: utf-8 -*-
"""
核心数据结构与检测器接口

Ctx 是"因果快照"：只包含截止 t（含 t）的数据。
检测器**只能**读 Ctx，评测器**只能**读 t 之后的数据。
这个隔离是 leakage.py 自动化测试的前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np


@dataclass
class Level:
    """一个支撑/阻力区"""
    center: float
    low: float
    high: float
    score: float                  # 融合得分（越大越强），非概率
    kind: str                     # 'support' | 'resistance'
    confidence: float = 0.0       # 校准后的"下次触及守住"概率，未校准时为 0
    width_atr: float = 0.0        # 区宽 / ATR
    dist_atr: float = 0.0         # (center - price) / ATR，带符号
    dist_pct: float = 0.0
    meta: Dict[str, float] = field(default_factory=dict)

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high


@dataclass
class Ctx:
    """
    因果上下文：截止 t 的数据切片。

    约定：所有数组长度 = t+1，最后一个元素就是 t 时刻。
    """
    code: str
    t: int                        # 在原始序列中的绝对下标
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    atr: float                    # t 时刻的 ATR
    tick: float
    # 因果 ATR 序列，长度 = t+1。atr_arr[i] 只用到 0..i 的数据
    atr_arr: Optional[np.ndarray] = None
    # 已确认枢轴（confirm_idx <= t），字段见 pivots.Pivots
    pivots: Optional["PivotView"] = None

    @property
    def price(self) -> float:
        return float(self.close[-1])

    @property
    def n(self) -> int:
        return len(self.close)


@dataclass
class PivotView:
    """t 时刻可见的已确认枢轴（全部为因果信息）"""
    idx: np.ndarray          # 枢轴所在 bar 下标
    price: np.ndarray        # 枢轴价格
    kind: np.ndarray         # +1 = 高点(潜在阻力), -1 = 低点(潜在支撑)
    confirm_idx: np.ndarray  # 该枢轴被确认的 bar 下标（<= t）
    leg_atr: np.ndarray      # 形成该枢轴的前一段行情幅度 / 当时 ATR（显著性）

    def __len__(self) -> int:
        return len(self.idx)


class Detector(Protocol):
    """检测器接口：给定因果上下文，输出打分后的支撑/阻力区列表"""

    name: str

    def detect(self, ctx: Ctx) -> List[Level]:
        ...


def annotate(levels: List[Level], price: float, atr: float) -> List[Level]:
    """统一补齐 kind / 距离 / 宽度字段"""
    atr = max(atr, 1e-12)
    for z in levels:
        z.dist_atr = (z.center - price) / atr
        z.dist_pct = (z.center - price) / price * 100 if price else 0.0
        z.width_atr = (z.high - z.low) / atr
        # kind 由区间与现价的关系决定：整个区在下方 -> 支撑
        if z.high < price:
            z.kind = "support"
        elif z.low > price:
            z.kind = "resistance"
        else:
            # 现价落在区内：按中枢方向归类，同时标记
            z.kind = "support" if z.center <= price else "resistance"
            z.meta["price_inside"] = 1.0
    return levels


def nms(levels: List[Level], min_gap_atr: float, atr: float,
        max_out: int) -> List[Level]:
    """
    非极大值抑制：按得分降序保留，剔除与已保留区中心距离 < min_gap_atr*ATR 的区。
    避免输出一堆彼此重叠、实质相同的"多个"关键位。
    """
    gap = max(min_gap_atr * atr, 1e-12)
    out: List[Level] = []
    for z in sorted(levels, key=lambda x: -x.score):
        if len(out) >= max_out:
            break
        if all(abs(z.center - k.center) >= gap for k in out):
            out.append(z)
    return out
