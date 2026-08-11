"""
区域历史胜率 & 触及预警模块

功能：
1. 历史区域验证：回溯每个密集区历史上的"守住"和"突破"次数
2. 区域置信度：基于历史表现给出胜率评分
3. 触及预警：计算当前价到最近支撑/压力区的距离，分级预警
"""

import sys
import os
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_WINDOW
from src.density_analyzer import find_dense_zones, classify_zones, ZoneInfo


@dataclass
class ZoneHistory:
    """单个密集区的历史表现"""
    center: float
    zone_type: str        # 'support' / 'resistance'
    held_count: int = 0   # 守住次数（触及后反弹）
    broke_count: int = 0  # 突破次数（穿透后未收回）
    touch_count: int = 0  # 总触及次数
    success_rate: float = 0.0  # 守住率


def analyze_zone_history(df: pd.DataFrame, window: int = DEFAULT_WINDOW, n_zones: int = 3) -> List[ZoneHistory]:
    """
    回溯历史：对最近一次检测到的密集区，统计全历史中该区域的表现

    算法：
    1. 用最近 window 天数据检测当前密集区
    2. 对每个密集区，在全历史中查找价格触及该区域的日期
    3. 判断触及后5天内的价格走势：收盘价收回区域 → 守住；继续穿透 → 突破
    """
    if len(df) < window + 10:
        return []

    # 当前密集区
    current_zones = find_dense_zones(df, window=window, n_zones=n_zones)
    current_price = float(df["close"].iloc[-1])
    classify_zones(current_zones, current_price)

    results = []
    for z in current_zones:
        zone_lo = z.low
        zone_hi = z.high
        held = 0
        broke = 0

        # 在全历史中（排除最近 window 天）查找触及事件
        search_df = df.iloc[:-window] if len(df) > window else df

        for i in range(len(search_df) - 5):
            row = search_df.iloc[i]
            high_i = float(row["high"])
            low_i = float(row["low"])

            # 判断是否触及
            touched = (low_i <= zone_hi and high_i >= zone_lo)

            if touched:
                # 看后续5天
                future = search_df.iloc[i+1:i+6]
                if len(future) < 3:
                    continue

                # 简单判断：触及后5天收盘价均值是否在区域内
                max_close = float(future["close"].max())
                min_close = float(future["close"].min())

                if z.zone_type == "support":
                    # 支撑区：价格应在其上方（不低于下边界过多）
                    if min_close >= zone_lo * 0.98:
                        held += 1
                    else:
                        broke += 1
                else:  # resistance
                    # 压力区：价格应在其下方（不高于上边界过多）
                    if max_close <= zone_hi * 1.02:
                        held += 1
                    else:
                        broke += 1

        total = held + broke
        rate = (held / total * 100) if total > 0 else 0

        results.append(ZoneHistory(
            center=z.center,
            zone_type=z.zone_type,
            held_count=held,
            broke_count=broke,
            touch_count=total,
            success_rate=round(rate, 1),
        ))

    return results


def generate_touch_alerts(
    current_price: float,
    zones: List[ZoneInfo],
    histories: List[ZoneHistory],
) -> List[Dict]:
    """
    生成触及预警

    返回每个区域的预警信息：
    - distance_pct: 当前距离 (%)
    - distance_price: 绝对距离
    - alert_level: 'danger'(即将触及) / 'warning'(接近) / 'normal'(远离)
    - progress: 0-100 进度条百分比
    - advice: 简短操作建议
    """
    alerts = []
    for z in zones:
        dist_pct = abs(z.distance_pct)
        dist_price = abs(current_price - z.center)

        # 分级
        if dist_pct <= 1.0:
            level = "danger"
            level_text = "⚡ 即将触及"
        elif dist_pct <= 3.0:
            level = "warning"
            level_text = "🔶 正在接近"
        else:
            level = "normal"
            level_text = "🔹 距离较远"

        # 进度条: 0% = 刚好触及, 100% = 在阈值最远处
        max_threshold = 5.0  # 超过5%认为很远
        progress = max(0, min(100, int((1 - dist_pct / max_threshold) * 100)))

        # 操作建议
        if z.zone_type == "support":
            if dist_pct <= 1.0:
                advice = "价格接近支撑区，关注反弹信号，可考虑买入"
            elif dist_pct <= 3.0:
                advice = "价格正在向支撑区靠近，准备观察入场时机"
            else:
                advice = "距离支撑区较远，等待回调"
        else:
            if dist_pct <= 1.0:
                advice = "价格接近压力区，关注突破或回落，可考虑止盈"
            elif dist_pct <= 3.0:
                advice = "价格正在向压力区靠近，关注减仓时机"
            else:
                advice = "距离压力区较远，持仓观察"

        # 附带历史胜率
        hist = next((h for h in histories if abs(h.center - z.center) < 0.01), None)
        hist_rate = hist.success_rate if hist else 0

        alerts.append({
            "zone_center": z.center,
            "zone_type": z.zone_type,
            "zone_label": "支撑区" if z.zone_type == "support" else "压力区",
            "distance_pct": round(dist_pct, 2),
            "distance_price": round(dist_price, 5),
            "alert_level": level,
            "alert_text": level_text,
            "progress": progress,
            "advice": advice,
            "history_rate": hist_rate,
            "history_label": f"历史守住率 {hist_rate}%" if hist_rate > 0 else "暂无历史数据",
        })

    return alerts


DIRECTION_LABELS = {
    "long": "只做多",
    "short": "只做空",
    "both": "多空都做",
}


def _distance_score(dist_pct: float) -> float:
    """距离得分：距离关键位越近得分越高 (0-100)"""
    d = abs(dist_pct)
    if d <= 0.5:
        return 100
    if d <= 1.0:
        return 90
    if d <= 2.0:
        return 75
    if d <= 3.0:
        return 60
    if d <= 5.0:
        return 40
    if d <= 8.0:
        return 20
    return 5


def _success_rate_score(rate: float) -> float:
    """守住率得分 (0-100)"""
    if rate <= 0:
        return 30  # 暂无历史数据
    if rate >= 80:
        return 100
    if rate >= 60:
        return 80
    if rate >= 40:
        return 60
    if rate >= 20:
        return 40
    return 15


def _strength_score(zone: ZoneInfo) -> float:
    """强度得分：基于密集区密度强度 (0-100)"""
    s = zone.strength  # 0-1 归一化密度
    return max(10, min(100, round(s * 100)))


def _compute_trend(df: pd.DataFrame, direction: str):
    """
    计算趋势得分与描述（基于 EMA20）
    返回 (score 0-100, label, detail)
    """
    closes = df["close"].values
    n = len(closes)
    if n < 20:
        return 50.0, "数据不足", ""

    s = pd.Series(closes)
    ema20 = s.ewm(span=20, adjust=False).mean().iloc[-1]
    price = float(closes[-1])

    if np.isnan(ema20):
        return 50.0, "数据不足", ""

    deviation = (price - ema20) / ema20 * 100
    slope = 0.0
    if n >= 26:
        ema20_prev = s.ewm(span=20, adjust=False).mean().iloc[-6]
        if not np.isnan(ema20_prev) and ema20_prev > 0:
            slope = (ema20 - ema20_prev) / ema20_prev * 100

    trend_strength = deviation * 0.6 + slope * 0.4

    if direction == "long":
        score = 50 + trend_strength * 8
    elif direction == "short":
        score = 50 - trend_strength * 8
    else:  # both
        score = 50 + abs(trend_strength) * 6

    score = max(0, min(100, score))

    if trend_strength > 1.5:
        label = "上涨"
    elif trend_strength < -1.5:
        label = "下跌"
    else:
        label = "震荡"

    detail = f"现价vsEMA20 {deviation:+.2f}% · EMA20斜率 {slope:+.2f}%"
    return round(score, 1), label, detail


def compute_risk_score(
    alerts: List[Dict],
    df: pd.DataFrame = None,
    zones: List[ZoneInfo] = None,
    histories=None,
    direction: str = "long",
) -> Dict:
    """
    综合风险评估：距离(20%) + 守住率(40%) + 强度(20%) + 趋势(20%)

    direction:
        'long'  - 只做多：关注支撑区（买入机会）
        'short' - 只做空：关注压力区（卖出机会）
        'both'  - 多空都做：关注最近的关键位
    """
    zones = zones or []
    histories = histories or []

    _empty = {
        "overall_score": 0,
        "level": "hold",
        "level_label": "🟡 观望",
        "summary": "暂无关键位数据",
        "direction": direction,
        "direction_label": DIRECTION_LABELS.get(direction, direction),
        "scores": {"distance": 0, "success_rate": 0, "strength": 0, "trend": 0},
        "nearest_zone": None,
        "nearest_zone_type": "",
        "nearest_distance_pct": None,
        "success_rate": 0,
        "trend_label": "—",
        "trend_detail": "",
    }

    if not zones:
        return _empty

    # 趋势得分（始终计算）
    if df is not None:
        trend_score, trend_label, trend_detail = _compute_trend(df, direction)
    else:
        trend_score, trend_label, trend_detail = 50.0, "—", ""

    # 根据方向选择关注的关键位
    if direction == "long":
        focus_zones = [z for z in zones if z.zone_type == "support"]
        focus_name = "支撑"
    elif direction == "short":
        focus_zones = [z for z in zones if z.zone_type == "resistance"]
        focus_name = "压力"
    else:
        focus_zones = list(zones)
        focus_name = "关键"

    if not focus_zones:
        overall = round(trend_score * 0.2)
        level = "buy" if overall >= 70 else "hold" if overall >= 40 else "sell"
        return {
            **_empty,
            "overall_score": overall,
            "level": level,
            "level_label": {"buy": "🟢 可关注", "hold": "🟡 观望", "sell": "🔴 回避"}[level],
            "summary": f"无{focus_name}位，趋势{trend_label}",
            "scores": {"distance": 0, "success_rate": 0, "strength": 0, "trend": round(trend_score)},
            "trend_label": trend_label,
            "trend_detail": trend_detail,
        }

    # 最近的关键位
    nearest = min(focus_zones, key=lambda z: abs(z.distance_pct))

    # 距离得分
    dist_score = _distance_score(nearest.distance_pct)

    # 守住率得分：匹配历史记录
    tol = max(0.01, abs(nearest.center) * 0.005)
    hist = next((h for h in histories if abs(h.center - nearest.center) < tol), None)
    rate = hist.success_rate if hist else 0
    rate_score = _success_rate_score(rate)

    # 强度得分
    str_score = _strength_score(nearest)

    overall = round(dist_score * 0.2 + rate_score * 0.4 + str_score * 0.2 + trend_score * 0.2)
    level = "buy" if overall >= 70 else "hold" if overall >= 40 else "sell"

    zone_label = "支撑区" if nearest.zone_type == "support" else "压力区"
    dir_action = "关注做多" if direction == "long" else "关注做空" if direction == "short" else "关注机会"
    summary = (
        f"最近{zone_label}距现价 {abs(nearest.distance_pct):.2f}%，"
        f"守住率 {rate}%，趋势{trend_label}，{dir_action}"
    )

    return {
        "overall_score": overall,
        "level": level,
        "level_label": {"buy": "🟢 可关注", "hold": "🟡 观望", "sell": "🔴 回避"}[level],
        "summary": summary,
        "direction": direction,
        "direction_label": DIRECTION_LABELS.get(direction, direction),
        "scores": {
            "distance": round(dist_score),
            "success_rate": round(rate_score),
            "strength": round(str_score),
            "trend": round(trend_score),
        },
        "nearest_zone": nearest.center,
        "nearest_zone_type": nearest.zone_type,
        "nearest_distance_pct": nearest.distance_pct,
        "success_rate": rate,
        "trend_label": trend_label,
        "trend_detail": trend_detail,
    }
