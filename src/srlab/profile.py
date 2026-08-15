# -*- coding: utf-8 -*-
"""
价格分箱与体积守恒的成交量剖面

与旧实现的关键差异
------------------
旧: density[lo:hi+1] += volume_i
    -> 一根 K 线的成交量被完整计入它覆盖的每一个价格箱。
       实测（000001 日线 40 日窗口）密度剖面隐含总量 = 真实总量 x 15.8，
       且宽幅 K 线权重被放大 1.3~1.7 倍。振幅越大的 K 线（往往是消息面
       驱动的一次性跳空/放量长影线）反而主导了"筹码密集区"，与"筹码在
       某价位堆积"的经济含义相反。

新: density[lo:hi+1] += volume_i / n_bins_covered_i
    -> 成交量在覆盖区间内均匀摊薄，sum(density) == sum(volume)，体积守恒。
       宽幅 K 线不再获得额外权重。

分箱宽度
--------
旧: (窗口最高 - 窗口最低) / 100，箱宽随窗口波动率漂移，跨标的不可比。
新: bin_w = max(tick, alpha * ATR)，箱宽有稳定经济含义（默认 0.25 ATR）。

时间衰减
--------
旧: 无。三个月前的成交与昨天的成交等权。
新: w_i *= exp(-ln2 * age_i / half_life)，默认半衰期 60 根。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def make_bins(price_lo: float, price_hi: float, bin_w: float,
              max_bins: int = 2000) -> Tuple[float, float, int]:
    """
    返回 (p0, bin_w, n_bins)。p0 是第 0 箱的下边界。
    箱数超过 max_bins 时自动放宽箱宽，避免极端行情下内存/耗时爆炸。
    """
    span = max(price_hi - price_lo, bin_w)
    n = int(np.ceil(span / bin_w)) + 1
    if n > max_bins:
        bin_w = span / (max_bins - 1)
        n = max_bins
    return price_lo, bin_w, n


def bin_centers(p0: float, bin_w: float, n_bins: int) -> np.ndarray:
    return p0 + (np.arange(n_bins) + 0.5) * bin_w


def decay_weights(n: int, half_life: float) -> np.ndarray:
    """age=0 为最新一根，权重 1.0；age=half_life 时权重 0.5"""
    if half_life <= 0:
        return np.ones(n, dtype=np.float64)
    age = np.arange(n - 1, -1, -1, dtype=np.float64)
    return np.exp(-np.log(2.0) * age / half_life)


def volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    weight: np.ndarray,
    p0: float,
    bin_w: float,
    n_bins: int,
) -> np.ndarray:
    """
    体积守恒的成交量-价格剖面（全向量化，O(n + n_bins)）。

    每根 K 线的 weight 均摊到它覆盖的箱上，用差分数组 + cumsum 实现，
    避免 Python 逐根循环。

    保证: sum(返回值) ≈ sum(weight)（浮点误差内）
    """
    if len(high) == 0 or n_bins <= 0:
        return np.zeros(max(n_bins, 0), dtype=np.float64)

    lo_idx = np.clip(((low - p0) / bin_w).astype(np.int64), 0, n_bins - 1)
    hi_idx = np.clip(((high - p0) / bin_w).astype(np.int64), 0, n_bins - 1)
    hi_idx = np.maximum(hi_idx, lo_idx)

    cnt = (hi_idx - lo_idx + 1).astype(np.float64)
    w = weight / cnt

    diff = np.zeros(n_bins + 1, dtype=np.float64)
    np.add.at(diff, lo_idx, w)
    np.add.at(diff, hi_idx + 1, -w)
    return np.cumsum(diff)[:n_bins]


def point_profile(
    price: np.ndarray,
    weight: np.ndarray,
    p0: float,
    bin_w: float,
    n_bins: int,
    kernel_bins: float = 1.0,
) -> np.ndarray:
    """
    点价格剖面（用于收盘价堆积、枢轴价堆积）。

    每个价格点按高斯核散布到邻近箱，kernel_bins 为核宽（单位：箱）。
    收盘价比 [low,high] 全区间更能代表"市场认可的价格"，
    枢轴价则直接代表"发生过反转的价格"。
    """
    out = np.zeros(n_bins, dtype=np.float64)
    if len(price) == 0 or n_bins <= 0:
        return out
    idx = np.clip(((price - p0) / bin_w).astype(np.int64), 0, n_bins - 1)
    np.add.at(out, idx, weight)
    if kernel_bins > 0:
        out = gaussian_blur(out, kernel_bins)
    return out


def gaussian_blur(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    """一维高斯平滑（自实现，避免 scipy 边界模式差异；边界用截断归一化）"""
    if sigma_bins <= 0 or len(x) < 3:
        return x
    r = int(max(1, round(3 * sigma_bins)))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma_bins) ** 2)
    k /= k.sum()
    pad = np.pad(x, r, mode="constant")
    sm = np.convolve(pad, k, mode="same")[r:r + len(x)]
    # 边界归一化：补偿被截断的核质量
    norm = np.convolve(np.pad(np.ones_like(x), r, mode="constant"), k, mode="same")[r:r + len(x)]
    return np.where(norm > 1e-12, sm / norm, sm)


def robust_z(x: np.ndarray) -> np.ndarray:
    """
    稳健标准化到可比尺度：(x - median) / (1.4826 * MAD)，再压到 [0, 1]。
    用于把成交量剖面、枢轴剖面等不同量纲的证据放到同一尺度上融合。
    """
    if len(x) == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-12:
        mx = x.max()
        return x / mx if mx > 1e-12 else np.zeros_like(x)
    z = (x - med) / scale
    # 平滑压缩到 0-1（logistic），避免个别极端值吞掉全部权重
    return 1.0 / (1.0 + np.exp(-z / 2.0))


def find_peaks_simple(y: np.ndarray, min_distance: int = 1,
                      min_height: float = 0.0) -> np.ndarray:
    """
    极值点检测（自实现，避免 scipy.find_peaks 的 prominence 语义歧义）。

    规则：y[i] >= 两侧邻居，且 y[i] >= min_height；
          随后按高度降序做间隔 min_distance 的抑制。
    平台（连续相等）取平台中点。
    """
    n = len(y)
    if n < 3:
        return np.array([], dtype=np.int64)
    cand = []
    i = 1
    while i < n - 1:
        if y[i] >= y[i - 1] and y[i] >= min_height:
            j = i
            while j < n - 1 and y[j + 1] == y[i]:
                j += 1
            if y[i] >= y[min(j + 1, n - 1)]:
                cand.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    if not cand:
        return np.array([], dtype=np.int64)
    cand = np.array(cand, dtype=np.int64)
    order = cand[np.argsort(y[cand])[::-1]]
    kept = []
    for p in order:
        if all(abs(p - q) >= min_distance for q in kept):
            kept.append(int(p))
    return np.sort(np.array(kept, dtype=np.int64))
