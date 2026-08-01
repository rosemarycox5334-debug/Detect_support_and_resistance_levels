"""
成交密集区核心分析模块

增强版算法（相比原始版本）：
1. 量价加权密度：按成交量加权而非简单K线重叠计数
2. 多区域检测：使用峰值检测算法识别多个密集区
3. 多时间框架：日/周/月线独立分析，大周期权重更高
4. 区域分类：自动区分支撑区（低于现价）与压力区（高于现价）
5. 盈亏比计算：基于当前价格与密集区位置关系评估交易性价比
6. 滚动窗口分析：支持回测的逐日滚动计算

核心逻辑：
每根K线的[最低价, 最高价]区间代表当日成交价格范围。在一个时间窗口内，
统计每个价格水平被多少根K线"覆盖"——重叠次数越多，该价位成交越密集，
由此形成支撑/压力区。成交量加权后更能反映真实筹码分布。
"""

import sys
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_WINDOW, DEFAULT_BINS, DEFAULT_N_ZONES,
    MIN_ZONE_PROMINENCE, VOLUME_WEIGHTED, PRICE_THRESHOLD_PCT,
    TIMEFRAMES, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
)


@dataclass
class ZoneInfo:
    """单个密集成交区信息"""
    center: float
    low: float
    high: float
    strength: float
    touch_count: int = 0
    volume_pct: float = 0.0
    zone_type: str = ""
    distance_pct: float = 0.0


def compute_density_profile(
    df: pd.DataFrame,
    n_bins: int = DEFAULT_BINS,
    volume_weighted: bool = VOLUME_WEIGHTED,
    smooth_sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    计算价格密度剖面

    将窗口内价格范围等分为 n_bins 个区间，统计每个区间
    被多少根K线的[最低价,最高价]覆盖。成交量加权时用当日
    成交量作为覆盖权重。

    参数:
        df:              K线DataFrame (需含 high, low, volume)
        n_bins:          价格分箱数
        volume_weighted: 是否按成交量加权
        smooth_sigma:    高斯平滑sigma

    返回:
        price_levels: 各分箱中心价格 (n_bins,)
        density:      各分箱归一化密度值 (n_bins,)
        price_min:    窗口最低价
        price_max:    窗口最高价
    """
    if len(df) == 0:
        return np.array([]), np.array([]), 0, 0

    price_min = float(df["low"].min())
    price_max = float(df["high"].max())
    if price_max <= price_min:
        price_max = price_min + 0.01

    price_levels = np.linspace(price_min, price_max, n_bins)
    bin_width = (price_max - price_min) / n_bins
    density = np.zeros(n_bins, dtype=np.float64)

    highs = df["high"].values
    lows = df["low"].values
    if volume_weighted and "volume" in df.columns:
        weights = df["volume"].values.astype(np.float64)
    else:
        weights = np.ones(len(df), dtype=np.float64)

    for i in range(len(df)):
        h, l, w = float(highs[i]), float(lows[i]), float(weights[i])
        if np.isnan(h) or np.isnan(l):
            continue
        lo_idx = max(0, int((l - price_min) / bin_width))
        hi_idx = min(n_bins - 1, int((h - price_min) / bin_width))
        if hi_idx >= lo_idx:
            density[lo_idx:hi_idx + 1] += w

    max_val = density.max()
    if max_val > 0:
        density = density / max_val

    if smooth_sigma > 0 and n_bins > 3:
        density = gaussian_filter1d(density, sigma=smooth_sigma)
        mx = density.max()
        if mx > 0:
            density = density / mx

    return price_levels, density, price_min, price_max


def find_dense_zones(
    df: pd.DataFrame,
    window: Optional[int] = None,
    n_zones: int = DEFAULT_N_ZONES,
    volume_weighted: bool = VOLUME_WEIGHTED,
    n_bins: int = DEFAULT_BINS,
    smooth_sigma: float = 2.0,
    min_prominence: float = MIN_ZONE_PROMINENCE,
) -> List[ZoneInfo]:
    """
    检测成交密集区

    使用 scipy.signal.find_peaks 在密度曲线上找峰值，
    每个峰值对应一个密集成交区。若检测不到峰值，
    则回退到最大密度处。

    参数:
        df:              K线数据
        window:          时间窗口，None则用全部数据
        n_zones:         最多返回几个密集区
        volume_weighted: 是否成交量加权
        n_bins:          分箱数
        smooth_sigma:    平滑参数
        min_prominence:  最小峰值显著性

    返回:
        ZoneInfo 列表，按强度降序排列
    """
    if len(df) == 0:
        return []

    data = df.tail(window) if window else df
    if len(data) < 5:
        return []

    price_levels, density, pmin, pmax = compute_density_profile(
        data, n_bins=n_bins, volume_weighted=volume_weighted,
        smooth_sigma=smooth_sigma,
    )
    if len(density) == 0:
        return []

    bin_width = (pmax - pmin) / n_bins

    # 峰值检测
    peaks, props = signal.find_peaks(
        density,
        prominence=min_prominence,
        width=2,
        distance=max(3, n_bins // 20),
    )

    if len(peaks) == 0:
        max_idx = int(np.argmax(density))
        peaks = np.array([max_idx])
        # 手动估计半峰宽
        half = density[max_idx] / 2.0
        left = max_idx
        while left > 0 and density[left] > half:
            left -= 1
        right = max_idx
        while right < n_bins - 1 and density[right] > half:
            right += 1
        props = {
            "prominences": np.array([density[max_idx]]),
            "widths": np.array([max(right - left, 1)]),
        }

    order = np.argsort(props["prominences"])[::-1][:n_zones]
    peaks = peaks[order]

    current_price = float(data["close"].iloc[-1])
    zones = []

    for i, pk in enumerate(peaks):
        center = float(price_levels[pk])
        half_max = density[pk] / 2.0

        left_idx = int(pk)
        while left_idx > 0 and density[left_idx] > half_max:
            left_idx -= 1
        right_idx = int(pk)
        while right_idx < n_bins - 1 and density[right_idx] > half_max:
            right_idx += 1

        zone_low = float(price_levels[left_idx])
        zone_high = float(price_levels[right_idx])
        if zone_high - zone_low < bin_width * 2:
            zone_low = center - bin_width
            zone_high = center + bin_width

        # 统计此区域内重叠K线数
        touch_count = 0
        vol_sum = 0.0
        total_vol = float(data["volume"].sum()) if "volume" in data.columns else 1.0
        for _, row in data.iterrows():
            if row["low"] <= zone_high and row["high"] >= zone_low:
                touch_count += 1
                vol_sum += float(row.get("volume", 0))

        volume_pct = (vol_sum / total_vol * 100) if total_vol > 0 else 0.0
        dist_pct = (center - current_price) / current_price * 100
        zone_type = "support" if dist_pct < 0 else "resistance"

        zones.append(ZoneInfo(
            center=round(center, 2),
            low=round(zone_low, 2),
            high=round(zone_high, 2),
            strength=round(float(density[pk]), 4),
            touch_count=touch_count,
            volume_pct=round(volume_pct, 1),
            zone_type=zone_type,
            distance_pct=round(dist_pct, 2),
        ))

    return zones


def classify_zones(zones: List[ZoneInfo], current_price: float) -> List[ZoneInfo]:
    """根据当前价重新分类支撑/压力"""
    for z in zones:
        dist_pct = (z.center - current_price) / current_price * 100
        z.distance_pct = round(dist_pct, 2)
        z.zone_type = "support" if dist_pct < 0 else "resistance"
    return zones


def multi_timeframe_zones(
    df_day: pd.DataFrame,
    df_week: Optional[pd.DataFrame] = None,
    df_month: Optional[pd.DataFrame] = None,
    n_zones: int = DEFAULT_N_ZONES,
) -> Dict[str, List[ZoneInfo]]:
    """
    多时间框架密集区分析

    分别对日/周/月线计算密集区，汇总结果。

    返回:
        {'day': [...], 'week': [...], 'month': [...], 'combined': [...]}
    """
    result: Dict[str, List[ZoneInfo]] = {}
    current_price = float(df_day["close"].iloc[-1])

    # 日线
    dz = find_dense_zones(df_day, window=TIMEFRAMES["day"]["window"], n_zones=n_zones)
    classify_zones(dz, current_price)
    result["day"] = dz

    # 周线
    if df_week is not None and len(df_week) >= 3:
        wz = find_dense_zones(df_week, window=TIMEFRAMES["week"]["window"], n_zones=n_zones)
        classify_zones(wz, current_price)
        result["week"] = wz
    else:
        result["week"] = []

    # 月线
    if df_month is not None and len(df_month) >= 3:
        mz = find_dense_zones(df_month, window=TIMEFRAMES["month"]["window"], n_zones=n_zones)
        classify_zones(mz, current_price)
        result["month"] = mz
    else:
        result["month"] = []

    # 汇总：合并所有时间框架，大周期权重加倍
    weighted = {}
    for z in dz:
        weighted[(z.low, z.high)] = [z.center, z.strength * 1.0]
    for z in (result.get("week") or []):
        key = (z.low, z.high)
        weighted[key] = [z.center, weighted.get(key, [z.center, 0])[1] + z.strength * 2.0]
    for z in (result.get("month") or []):
        key = (z.low, z.high)
        weighted[key] = [z.center, weighted.get(key, [z.center, 0])[1] + z.strength * 3.0]

    combined = []
    for (lo, hi), (ctr, w) in sorted(weighted.items(), key=lambda x: -x[1][1]):
        if len(combined) >= n_zones:
            break
        dist_pct = (ctr - current_price) / current_price * 100
        combined.append(ZoneInfo(
            center=round(ctr, 2), low=round(lo, 2), high=round(hi, 2),
            strength=round(w / 6.0, 4),
            zone_type="support" if dist_pct < 0 else "resistance",
            distance_pct=round(dist_pct, 2),
        ))
    result["combined"] = combined

    return result


def calc_risk_reward(
    current_price: float,
    zones: List[ZoneInfo],
    volatility_pct: float = 2.0,
) -> Dict:
    """
    计算盈亏比

    基于当前价格与最近支撑/压力区的关系：
    - 潜在盈利 = 最近压力区距离
    - 潜在亏损 = 最近支撑区距离
    - 盈亏比 = 盈利 / 亏损

    返回:
        {nearest_support, nearest_resistance, potential_profit_pct,
         potential_loss_pct, risk_reward_ratio, quality}
    """
    supports = [z for z in zones if z.zone_type == "support"]
    resistances = [z for z in zones if z.zone_type == "resistance"]

    # 最近支撑：价格下方最近的
    nearest_support = min(supports, key=lambda z: abs(z.distance_pct)) if supports else None
    nearest_resistance = min(resistances, key=lambda z: abs(z.distance_pct)) if resistances else None

    profit_pct = abs(nearest_resistance.distance_pct) if nearest_resistance else volatility_pct * 2
    loss_pct = abs(nearest_support.distance_pct) if nearest_support else volatility_pct
    rr = profit_pct / loss_pct if loss_pct > 0 else 0

    if rr >= 3:
        quality = "优秀"
    elif rr >= 2:
        quality = "良好"
    elif rr >= 1.5:
        quality = "一般"
    else:
        quality = "较差"

    return {
        "nearest_support": nearest_support.center if nearest_support else None,
        "nearest_resistance": nearest_resistance.center if nearest_resistance else None,
        "potential_profit_pct": round(profit_pct, 2),
        "potential_loss_pct": round(loss_pct, 2),
        "risk_reward_ratio": round(rr, 2),
        "quality": quality,
    }


def rolling_zone_analysis(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    step: int = 5,
    n_zones: int = DEFAULT_N_ZONES,
) -> pd.DataFrame:
    """
    滚动窗口密集区分析（用于回测）

    从第 window 天开始，每隔 step 天计算一次密集区。
    返回汇总 DataFrame。

    参数:
        df:      完整K线数据
        window:  滚动窗口大小
        step:    计算步长
        n_zones: 每次检测的密集区数量

    返回:
        DataFrame，每行一个分析时点
    """
    rows = []
    total = len(df)
    for i in range(window, total, step):
        window_df = df.iloc[i - window:i]
        zones = find_dense_zones(window_df, n_zones=n_zones)
        current_price = float(df["close"].iloc[i - 1])
        classify_zones(zones, current_price)
        rr = calc_risk_reward(current_price, zones)

        row = {
            "date": df["date"].iloc[i - 1],
            "close": current_price,
            "n_zones": len(zones),
            "rr_ratio": rr["risk_reward_ratio"],
            "rr_quality": rr["quality"],
        }
        for j, z in enumerate(zones[:3]):
            row[f"zone{j+1}_center"] = z.center
            row[f"zone{j+1}_type"] = z.zone_type
            row[f"zone{j+1}_strength"] = z.strength
            row[f"zone{j+1}_dist_pct"] = z.distance_pct
        rows.append(row)

    return pd.DataFrame(rows)
