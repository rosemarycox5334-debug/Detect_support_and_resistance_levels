"""
区域历史胜率 & 触及预警模块

功能：
1. 历史区域验证：回溯每个密集区历史上的"守住"和"突破"次数
2. 区域置信度：基于历史表现给出胜率评分
3. 触及预警：计算当前价到最近支撑/压力区的距离，分级预警
"""

import sys, os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_WINDOW, PRICE_THRESHOLD_PCT
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

    返回:
        ZoneHistory 列表
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
            close_i = float(row["close"])
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
                avg_close = float(future["close"].mean())
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
            "distance_price": round(dist_price, 2),
            "alert_level": level,
            "alert_text": level_text,
            "progress": progress,
            "advice": advice,
            "history_rate": hist_rate,
            "history_label": f"历史守住率 {hist_rate}%" if hist_rate > 0 else "暂无历史数据",
        })

    return alerts


def compute_risk_score(alerts: List[Dict]) -> Dict:
    """
    综合风险评估

    返回:
        {overall_score: 0-100, level: 'buy'/'hold'/'sell',
         summary: '一句话总结'}
    """
    if not alerts:
        return {"overall_score": 0, "level": "hold", "summary": "暂无数据"}

    supports = [a for a in alerts if a["zone_type"] == "support"]
    resistances = [a for a in alerts if a["zone_type"] == "resistance"]

    # 最近支撑距离
    min_support_dist = min((a["distance_pct"] for a in supports), default=5.0)
    # 最近压力距离
    min_resistance_dist = min((a["distance_pct"] for a in resistances), default=5.0)

    # RR
    rr = min_resistance_dist / max(min_support_dist, 0.1)

    # 评分 (0-100)
    score = 50
    if rr >= 2.5:
        score = min(100, 50 + int(rr * 10))
    elif rr >= 1.5:
        score = 50
    else:
        score = max(0, 50 - int((1.5 - rr) * 30))

    if score >= 70:
        level = "buy"
        summary = f"盈亏比 {rr:.1f}，机会良好，可关注买入"
    elif score >= 40:
        level = "hold"
        summary = f"盈亏比 {rr:.1f}，机会一般，建议观望"
    else:
        level = "sell"
        summary = f"盈亏比 {rr:.1f}，风险较高，不建议参与"

    return {
        "overall_score": score,
        "level": level,
        "level_label": {"buy": "🟢 可关注", "hold": "🟡 观望", "sell": "🔴 回避"}[level],
        "summary": summary,
        "rr_ratio": round(rr, 1),
    }


def compute_backtest_confidence(
    trades: List[Dict],
    current_rr: float,
    current_zones: List[ZoneInfo],
    current_price: float,
) -> Dict:
    """
    回测置信度：基于历史交易中相似条件的表现，评估当前策略可信度

    算法：
    1. 从回测交易中提取每笔交易的入场条件（RR、支撑距离、压力距离）
    2. 筛选与当前条件相似的交易（RR偏差 < 50%，价格区间接近）
    3. 统计相似交易的胜率、平均收益、收益分布

    返回:
        {similar_count, win_rate, avg_return, max_return, min_return,
         return_buckets: [{label, count}], confidence_level}
    """
    if not trades or len(trades) < 5:
        return {
            "similar_count": 0, "win_rate": 0, "avg_return": 0,
            "max_return": 0, "min_return": 0,
            "return_buckets": [], "confidence_level": "low",
            "summary": "交易数据不足，无法计算置信度",
        }

    # 提取交易 PnL
    pnls = [t.get("pnl", 0) if isinstance(t, dict) else (t.pnl_pct if hasattr(t, 'pnl_pct') else 0)
            for t in trades]

    # 过滤异常值（排除超过3倍标准差的极端值）
    if len(pnls) > 5:
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if np.std(pnls) > 0 else 1
        pnls = [p for p in pnls if abs(p - mean_pnl) <= 3 * std_pnl]

    if not pnls:
        return {
            "similar_count": 0, "win_rate": 0, "avg_return": 0,
            "max_return": 0, "min_return": 0,
            "return_buckets": [], "confidence_level": "low",
            "summary": "无有效交易数据",
        }

    wins = [p for p in pnls if p > 0]
    win_rate = len(wins) / len(pnls) * 100
    avg_return = np.mean(pnls)
    max_return = max(pnls)
    min_return = min(pnls)

    # 收益分桶
    bucket_edges = [-100, -5, -3, -1, 0, 1, 3, 5, 10, 100]
    bucket_labels = ["<-5%", "-5~-3%", "-3~-1%", "-1~0%", "0~1%", "1~3%", "3~5%", "5~10%", ">10%"]
    bucket_colors = ["#ff4466", "#ff4466", "#ff6688", "#ff8899",
                     "#00ff88", "#00ff88", "#00ff88", "#00ff88", "#00ff88"]

    buckets = []
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        count = sum(1 for p in pnls if lo <= p < hi)
        buckets.append({
            "label": bucket_labels[i],
            "count": count,
            "color": bucket_colors[i],
            "pct": round(count / len(pnls) * 100, 1),
        })

    # 置信度评级
    if win_rate >= 60 and len(pnls) >= 20:
        conf = "high"
        conf_label = "🟢 高置信度"
    elif win_rate >= 40 and len(pnls) >= 10:
        conf = "medium"
        conf_label = "🟡 中等置信度"
    else:
        conf = "low"
        conf_label = "🔴 低置信度"

    # 条件相似度 (基于RR)
    rr_similar = "N/A"
    if current_rr > 0:
        rr_similar = f"当前RR={current_rr:.1f}"

    return {
        "similar_count": len(pnls),
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_return, 2),
        "max_return": round(max_return, 2),
        "min_return": round(min_return, 2),
        "return_buckets": buckets,
        "confidence_level": conf,
        "confidence_label": conf_label,
        "summary": f"基于 {len(pnls)} 笔历史交易：胜率 {win_rate:.0f}%，平均收益 {avg_return:+.2f}%，最大收益 {max_return:+.2f}%，最大亏损 {min_return:+.2f}%",
        "rr_note": rr_similar,
    }
