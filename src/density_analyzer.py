"""
成交密集区核心分析模块（支撑/阻力位检测）

增强版算法：
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
    TIMEFRAMES, TF_ORDER,
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
    # 多周期共振信息（由 multi_timeframe_zones 填充）
    tf_count: int = 1          # 有多少个周期在此位置上给出了关键位
    tfs: str = ""              # 参与共振的周期代码，如 "D1+W1"


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
            center=round(center, 5),
            low=round(zone_low, 5),
            high=round(zone_high, 5),
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
    frames: List[Dict],
    n_zones: int = DEFAULT_N_ZONES,
    merge_atr: float = 0.8,
) -> Dict[str, List[ZoneInfo]]:
    """
    多时间框架密集区分析（自适应版）

    frames: [{"tf": "H1", "df": df, "weight": 1.0, "window": 60}, ...]
            第一个 frame 为主周期，用于确定 current_price 与合并尺度。
            weight: 大周期权重更高（2/3），小周期较低（0.5）。
            window: 该周期的分析窗口（K线数）。
    merge_atr: 合并容差，单位为主周期 ATR。中心价相距 <= merge_atr*ATR
               的关键位视为同一个位（多周期共振）。

    返回: {tf_code: [zones], ..., "combined": [zones]}
          combined 为跨周期按 ATR 距离聚类合并后的关键位，
          每个元素带 tf_count / tfs 表示共振情况。
    """
    if not frames:
        return {"combined": []}

    result: Dict[str, List[ZoneInfo]] = {}
    main_df = frames[0]["df"]
    current_price = float(main_df["close"].iloc[-1])

    # 合并容差用主周期 ATR 度量，跨标的、跨价位可比
    atr = _atr_last(main_df)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(current_price * 0.01, 1e-9)
    tol = merge_atr * atr

    total_weight = 0.0
    items: List[Tuple[float, float, float, float, str]] = []  # (center, low, high, w_strength, tf)

    for fr in frames:
        tf_code = fr["tf"]
        df = fr["df"]
        w = fr.get("weight", 1.0)
        win = fr.get("window", DEFAULT_WINDOW)

        if df is None or len(df) < 3:
            result[tf_code] = []
            continue

        zs = find_dense_zones(df, window=min(win, len(df)), n_zones=n_zones)
        classify_zones(zs, current_price)
        result[tf_code] = zs
        total_weight += w
        for z in zs:
            items.append((z.center, z.low, z.high, z.strength * w, tf_code))

    result["combined"] = _cluster_zones(
        items, current_price, tol,
        divisor=total_weight if total_weight > 0 else 1.0,
        n_zones=n_zones)
    return result


def _atr_last(df: pd.DataFrame, period: int = 14) -> float:
    """最后一根的 ATR（简单滚动均值），用于给多周期合并提供尺度"""
    if len(df) < 2:
        return float("nan")
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    n = min(period, len(tr))
    return float(np.mean(tr[-n:]))


def _cluster_zones(
    items: List[Tuple[float, float, float, float, str]],
    current_price: float,
    tol: float,
    divisor: float,
    n_zones: int,
) -> List[ZoneInfo]:
    """
    按中心价距离做单链凝聚聚类，容差 tol（= merge_atr * ATR）。

    旧实现用 (z.low, z.high) 浮点元组做字典键来合并，两个不同周期算出的
    区间边界几乎不可能逐位相同，实测 D1+H1+M15 共 9 个区间去重后仍是 9 个，
    **合并次数恒为 0** —— "大周期权重 2.0/3.0" 的加权逻辑从未生效。

    这里改为：按中心价排序，相邻中心距 <= tol 的归为一簇；
    簇内以强度为权重求加权中心，边界取并集，强度累加，
    并记录参与共振的周期数（tf_count）——多周期共振本身就是有用的信号。
    """
    if not items:
        return []
    items = sorted(items, key=lambda x: x[0])

    clusters: List[List[Tuple[float, float, float, float, str]]] = [[items[0]]]
    for it in items[1:]:
        if it[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(it)
        else:
            clusters.append([it])

    out: List[ZoneInfo] = []
    for cl in clusters:
        wsum = sum(x[3] for x in cl)
        if wsum <= 0:
            ctr = float(np.mean([x[0] for x in cl]))
        else:
            ctr = float(sum(x[0] * x[3] for x in cl) / wsum)
        lo = float(min(x[1] for x in cl))
        hi = float(max(x[2] for x in cl))
        tfs = sorted({x[4] for x in cl})
        dist_pct = (ctr - current_price) / current_price * 100 if current_price else 0.0
        out.append(ZoneInfo(
            center=round(ctr, 5), low=round(lo, 5), high=round(hi, 5),
            strength=round(wsum / divisor, 4),
            zone_type="support" if dist_pct < 0 else "resistance",
            distance_pct=round(dist_pct, 2),
            tf_count=len(tfs), tfs="+".join(tfs),
        ))
    # 共振周期多的优先，其次按强度
    out.sort(key=lambda z: (-z.tf_count, -z.strength))
    return out[:n_zones]


def calc_risk_reward(
    current_price: float,
    zones: List[ZoneInfo],
    volatility_pct: float = 2.0,
) -> Dict:
    """
    计算盈亏比

    - 潜在盈利 = 最近压力区距离
    - 潜在亏损 = 最近支撑区距离
    - 盈亏比 = 盈利 / 亏损
    """
    supports = [z for z in zones if z.zone_type == "support"]
    resistances = [z for z in zones if z.zone_type == "resistance"]

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
