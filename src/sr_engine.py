# -*- coding: utf-8 -*-
"""
线上支撑/阻力位引擎（生产适配层）

职责：把已在 srlab 里验证过的 V3Fusion 包装成 Flask 层可直接用的结构，
不再复制算法逻辑 —— 避免"回测用一套、线上用另一套"这种最常见的失效模式。

与旧 density_analyzer 路径的差异（全部有实测依据，见 OPTIMIZATION_PLAN.md）：

  1. 成交量分布体积守恒（旧版把整根 K 线成交量计入每个价格箱，放大 15.8 倍）
  2. 分箱宽度 = max(tick, 0.25×ATR)，跨标的可比（旧版箱宽随窗口价幅漂移）
  3. 区宽夹逼到 [0.3, 1.2]×ATR（旧版实测 2.2~5.7 ATR，是"高胜率"的真正来源）
  4. 触及事件去重 + 接近方向校验 + 阈值 ATR 化
     （旧版单个区在 8403 根历史里录得 1430 次"触及"，且不分接近方向）
  5. 排序分只用两个实测有效的因子：log1p(触及次数) + 0.8×陈旧度
  6. 成交量剖面强度改为独立的 p_stall（停滞倾向），**不参与排序**
     —— 实测它守住率差 +8.98pp 但收益率差 -0.88pp，是"价格会停"不是"进场能赚"
  7. **不再输出任何单个关键位的守住率概率**
     —— 实测 Brier 0.1836 vs 常数预测 0.1842，零区分度
     取而代之：输出该位所属的历史分位档 + 该档在样本外股票上的实测表现

数据口径：界面用前复权（qfq，最新价 = 真实市价），回测用后复权（hfq，严格因果）。
两者只差一个全局常数，V3 是尺度不变的，由 leakage.test_adjust_scale_invariance 验证。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.srlab.base import Ctx
from src.srlab.data import atr_series, tick_size_guess
from src.srlab.detectors import V3Fusion, V3Params
from src.srlab.pivots import zigzag
from src.srlab.probability import (
    MODEL_PATH_DEFAULT, ProbabilityModels, features_from_level,
)

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(_BASE, "data", "score_calibration.json")
PROB_PATH = os.path.join(_BASE, MODEL_PATH_DEFAULT)

# 只输出交易上有意义的距离带（与回测口径一致）
MIN_DIST_ATR = 0.5
MAX_DIST_ATR = 5.0

DIRECTION_LABELS = {"long": "只做多", "short": "只做空", "both": "多空都做"}


# ============================================================
# 评分标定表
# ============================================================
class Calibration:
    """
    回测导出的分位边界 + 每档样本外实测表现。

    界面上显示的"Q4 档 · 该档历史守住率 77.8%"里的 77.8% 就来自这里，
    是 100 只**样本外**股票上真实统计出来的，不是模型的预测概率。
    """

    def __init__(self, path: str = CALIB_PATH):
        self.ok = False
        self.edge_edges: Dict[str, List[float]] = {}
        self.stall_edges: Dict[str, List[float]] = {}
        self.buckets: Dict[Tuple[str, str], dict] = {}
        self.meta: dict = {}
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                js = json.load(f)
            self.edge_edges = js.get("edge_edges", {})
            self.stall_edges = js.get("stall_edges", {})
            self.meta = js.get("meta", {})
            for b in js.get("buckets", []):
                self.buckets[(b["kind"], b["bucket"])] = b
            self.ok = bool(self.edge_edges)
        except Exception:
            self.ok = False

    def bucket_of(self, kind: str, score: float) -> Optional[str]:
        edges = self.edge_edges.get(kind)
        if not edges or not np.isfinite(score):
            return None
        return f"Q{int(np.digitize([score], edges)[0]) + 1}"

    def stall_bucket_of(self, kind: str, stall: float) -> Optional[str]:
        edges = self.stall_edges.get(kind)
        if not edges or not np.isfinite(stall):
            return None
        return f"Q{int(np.digitize([stall], edges)[0]) + 1}"

    def stats(self, kind: str, bucket: str) -> Optional[dict]:
        return self.buckets.get((kind, bucket))


_CALIB: Optional[Calibration] = None
_PROBS: Optional[ProbabilityModels] = None


def get_calibration() -> Calibration:
    global _CALIB
    if _CALIB is None:
        _CALIB = Calibration()
    return _CALIB


def get_prob_models() -> ProbabilityModels:
    """
    双概率模型（触及 + 守住），进程内只加载一次。

    实测表现（out/final，130 只训练 / 130 只互斥测试股票）：
        触及概率  support AUC 0.770 / resistance 0.732，Brier skill +21.3% / +15.1%
        守住概率  support AUC 0.587 / resistance 0.612，Brier skill +1.8% / +2.7%
    守住概率的排序能力弱，但校准误差 < 2pp（模型说 75.1% 实际 75.4%），
    所以绝对值可信，不适合用来比较两个相邻的位。
    """
    global _PROBS
    if _PROBS is None:
        _PROBS = ProbabilityModels(PROB_PATH)
    return _PROBS


# ============================================================
# 引擎
# ============================================================
@dataclass
class SREngine:
    params: V3Params = None
    n_zones: int = 6

    def __post_init__(self):
        self.det = V3Fusion(p=self.params or V3Params())

    # ---------- 构造因果上下文 ----------
    @staticmethod
    def build_ctx(df: pd.DataFrame, code: str, t: Optional[int] = None) -> Ctx:
        h = df["high"].to_numpy(np.float64)
        l = df["low"].to_numpy(np.float64)
        c = df["close"].to_numpy(np.float64)
        o = df["open"].to_numpy(np.float64)
        v = df["volume"].to_numpy(np.float64)
        atr = atr_series(h, l, c, 14)
        if t is None:
            t = len(c) - 1
        piv = zigzag(h, l, atr, k_atr=2.0)
        tick = df.attrs.get("tick_base") or tick_size_guess(c)
        return Ctx(code=code, t=t,
                   open_=o[:t + 1], high=h[:t + 1], low=l[:t + 1],
                   close=c[:t + 1], volume=v[:t + 1],
                   atr=float(atr[t]), tick=float(tick),
                   atr_arr=atr[:t + 1],
                   pivots=piv.view_at(t, max_lookback=600))

    # ---------- 主入口 ----------
    def detect(self, df: pd.DataFrame, code: str) -> Tuple[List[dict], dict]:
        """返回 (zones_json, context_info)"""
        if len(df) < 60:
            return [], {"error": f"数据不足：{len(df)} 条"}
        ctx = self.build_ctx(df, code)
        atr = max(ctx.atr, 1e-12)
        price = ctx.price

        levels = self.det.detect(ctx)
        # 与回测完全一致的后处理：距离带 + 每侧限流
        kept = {"support": [], "resistance": []}
        for z in levels:
            if z.kind == "support" and z.high >= price:
                continue
            if z.kind == "resistance" and z.low <= price:
                continue
            d = abs(z.center - price) / atr
            if d < MIN_DIST_ATR or d > MAX_DIST_ATR:
                continue
            kept[z.kind].append(z)
        out_levels = []
        per_side = max(1, self.n_zones // 2)
        for k in ("support", "resistance"):
            out_levels += sorted(kept[k], key=lambda x: -x.score)[:per_side]

        calib = get_calibration()
        probs = get_prob_models()
        zones = [self._zone_json(z, atr, price, calib, probs) for z in out_levels]
        zones.sort(key=lambda z: -z["edge_score"])

        info = {
            "atr": round(atr, 6),
            "atr_pct": round(atr / price * 100, 3) if price else None,
            "calibrated": calib.ok,
            "calib_meta": calib.meta,
            "prob_ready": probs.ok,
            "prob_meta": probs.meta,
            "adjust_mode": df.attrs.get("adjust_mode", "unknown"),
            "adjust_source": df.attrs.get("adjust_source", "unknown"),
            "n_ex_rights": int(df.attrs.get("n_ex_rights", 0)),
            # 缺复权因子时的显式告警，界面必须展示，不能静默
            "adjust_warning": df.attrs.get("adjust_warning"),
        }
        return zones, info

    # ---------- 单个关键位 -> JSON ----------
    def _zone_json(self, z, atr: float, price: float, calib: Calibration,
                   probs: ProbabilityModels) -> dict:
        m = z.meta
        n_ev = int(m.get("n_events", 0))
        bucket = calib.bucket_of(z.kind, z.score)
        stats = calib.stats(z.kind, bucket) if bucket else None
        stall_b = calib.stall_bucket_of(z.kind, m.get("p_stall", float("nan")))

        # --- 双概率 ---
        # 特征里用到了距离和带宽。这与排序分刻意排除它们并不矛盾：
        # 排序分要回答"算法有没有本事定位"（距离是对照组也有的信息，不能算本事），
        # 这里要回答"这个位会不会有效"（距离是合法且最强的预测因子）。
        p_touch = p_hold = p_eff = None
        if probs.ok:
            fv = features_from_level(
                dist_atr=z.dist_atr, width_atr=z.width_atr, n_events=n_ev,
                stale=float(m.get("stale", 0.0)), vp=float(m.get("vp", 0.5)),
                close_p=float(m.get("close", 0.5)), atr=atr, price=price)
            p_touch = probs.predict("touch", z.kind, fv)
            p_hold = probs.predict("hold", z.kind, fv)
            if p_touch is not None and p_hold is not None:
                p_eff = p_touch * p_hold

        return {
            # --- 双概率（界面主展示）---
            "p_touch": round(p_touch, 4) if p_touch is not None else None,
            "p_hold": round(p_hold, 4) if p_hold is not None else None,
            "p_effective": round(p_eff, 4) if p_eff is not None else None,
            # --- 价格与几何 ---
            "center": round(z.center, 4),
            "low": round(z.low, 4),
            "high": round(z.high, 4),
            "zone_type": z.kind,
            "zone_label": "支撑带" if z.kind == "support" else "压力带",
            "distance_pct": round(z.dist_pct, 2),
            "distance_atr": round(z.dist_atr, 2),
            "width_atr": round(z.width_atr, 2),
            "width_pct": round((z.high - z.low) / price * 100, 2) if price else None,
            # --- 排序证据（唯一参与排序的两项）---
            "edge_score": round(float(z.score), 4),
            "n_events": n_ev,
            "stale": round(float(m.get("stale", 0.0)), 3),
            # --- 停滞倾向（不参与排序，用于放止损/目标）---
            "p_stall": round(float(m.get("p_stall", 0.0)), 3),
            "stall_bucket": stall_b,
            "vp_strength": round(float(m.get("vp", 0.0)), 3),
            # --- 历史分位档 + 该档样本外实测（替代旧版的"守住率"）---
            "bucket": bucket,
            "bucket_hold_rate": stats["hold_rate"] if stats else None,
            "bucket_fwd_ret": stats["fwd_ret_pct"] if stats else None,
            "bucket_n": stats["n_decided"] if stats else None,
            # --- 多周期共振（由 attach_mtf 填充）---
            "tf_count": 1,
            "tfs": "",
        }


# ============================================================
# 多周期共振
# ============================================================
def attach_mtf(zones: List[dict], mtf: List[dict], atr: float,
               merge_atr: float = 0.8) -> List[dict]:
    """
    把其他周期的关键位按 ATR 距离并入主周期关键位，标注共振周期数。

    旧实现用 (low, high) 浮点元组做字典键合并，实测合并次数恒为 0；
    这里按中心价距离 <= merge_atr*ATR 判定同一个位。
    """
    tol = max(merge_atr * atr, 1e-12)
    for z in zones:
        tfs = set()
        for item in mtf or []:
            for oz in item.get("zones", []):
                if abs(oz["center"] - z["center"]) <= tol:
                    tfs.add(item["tf"])
                    break
        if tfs:
            z["tfs"] = "+".join(sorted(tfs))
            z["tf_count"] = 1 + len(tfs)
    return zones


# ============================================================
# 方向化摘要（替代旧 risk_score）
# ============================================================
def build_summary(zones: List[dict], df: pd.DataFrame, direction: str,
                  info: dict) -> dict:
    """
    生成面向用户的摘要。

    与旧 compute_risk_score 的本质区别：**不再给出 0-100 的"综合评分"**。
    旧评分把 40% 权重压在一个统计口径错误的"守住率"上，而实测该概率
    零区分度（Brier 0.1836 vs 常数 0.1842）。
    这里只报告可追溯的事实：最近关键位、它的历史分位档、该档样本外实测表现、
    以及趋势状态。
    """
    focus = ("support" if direction == "long"
             else "resistance" if direction == "short" else None)
    cand = [z for z in zones if focus is None or z["zone_type"] == focus]
    trend_label, trend_detail = _trend(df)

    if not cand:
        return {
            "direction": direction,
            "direction_label": DIRECTION_LABELS.get(direction, direction),
            "nearest": None,
            "trend_label": trend_label,
            "trend_detail": trend_detail,
            "headline": f"距离带内无{'支撑' if direction=='long' else '压力' if direction=='short' else '关键'}位",
            "caveat": _CAVEAT,
        }

    nearest = min(cand, key=lambda z: abs(z["distance_atr"]))
    best = max(cand, key=lambda z: z["edge_score"])
    parts = [
        f"最近{nearest['zone_label']} {nearest['center']}"
        f"（距现价 {nearest['distance_pct']:+.2f}% / {nearest['distance_atr']:+.2f} ATR，"
        f"带宽 {nearest['width_atr']:.2f} ATR）",
    ]
    if nearest.get("p_touch") is not None:
        parts.append(f"触及概率 {nearest['p_touch']*100:.0f}%")
    if nearest.get("p_hold") is not None:
        parts.append(f"守住概率 {nearest['p_hold']*100:.0f}%")
    parts.append(f"历史被测试 {nearest['n_events']} 次")
    parts.append(f"趋势{trend_label}")

    return {
        "direction": direction,
        "direction_label": DIRECTION_LABELS.get(direction, direction),
        "nearest": nearest,
        "best": best,
        "trend_label": trend_label,
        "trend_detail": trend_detail,
        "headline": " · ".join(parts),
        "caveat": _CAVEAT,
    }


_CAVEAT = (
    "概率来自 130 只训练股票拟合、130 只互斥测试股票验收的走查回测。"
    "触及概率区分度强（AUC 0.73~0.77）；守住概率区分度弱（AUC 0.59~0.61）但校准良好"
    "（模型说 75% 的那批实际 75.4% 守住），因此它的绝对值可信，"
    "但不宜用来比较两个相邻的位。"
    "另外，控制距离与带宽后本算法与「把同一个位随机平移 1~2 ATR」的对照组"
    "在守住率上无显著差异，所以关键位应当作「约 1~2 ATR 宽的区域」使用，"
    "不要当成精确到分的价格。"
)


def _trend(df: pd.DataFrame) -> Tuple[str, str]:
    c = df["close"]
    if len(c) < 26:
        return "数据不足", ""
    ema = c.ewm(span=20, adjust=False).mean()
    price = float(c.iloc[-1])
    e_now = float(ema.iloc[-1])
    e_prev = float(ema.iloc[-6])
    dev = (price - e_now) / e_now * 100 if e_now else 0.0
    slope = (e_now - e_prev) / e_prev * 100 if e_prev else 0.0
    s = dev * 0.6 + slope * 0.4
    label = "上涨" if s > 1.5 else "下跌" if s < -1.5 else "震荡"
    return label, f"现价vsEMA20 {dev:+.2f}% · EMA20斜率 {slope:+.2f}%"
