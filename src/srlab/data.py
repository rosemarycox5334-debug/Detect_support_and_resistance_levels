# -*- coding: utf-8 -*-
"""
数据适配层

支持两种本地 parquet 数据集：

A) A股数据集  D:\\A股_K线数据\\parquet\\stocks
   文件名: {6位代码}_{daily|60min|15min|5min}.parquet
   列:     time(int, 单位 = unix秒 / 1000), open, high, low, close, tick_volume(int)
   注意:   time 的量化步长是 1000 秒(约16.7分钟)，因此日线需 normalize 到自然日；
           分钟线时间戳有 ±1000s 误差，只保证顺序正确，不保证精确对齐。

B) 期货/MT5 数据集
   文件名: {品种}主连_{D1|W1|H1|...}.parquet 或 {SYMBOL}_{TF}.parquet
   列:     time(unix秒), open, high, low, close, tick_volume

统一输出 DataFrame 列：
   date, open, high, low, close, volume, amount, pct_chg
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

# A股 parquet 的 time 单位换算系数：time * 1000 = unix 秒
ASHARE_TIME_SCALE = 1000
# A股 成交量单位为"手"，1手 = 100股
SHARES_PER_LOT = 100

ASHARE_TF_ALIAS = {
    "D1": "daily", "DAILY": "daily", "1D": "daily",
    "H1": "60min", "60MIN": "60min",
    "M15": "15min", "15MIN": "15min",
    "M5": "5min", "5MIN": "5min",
}

_ASHARE_FN = re.compile(r"^(?P<code>\d{6})_(?P<tf>daily|60min|15min|5min)\.parquet$")

_OHLCV = ["open", "high", "low", "close", "volume"]
_OUT_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]


# ============================================================
# A股数据集
# ============================================================
def ashare_list_codes(data_dir: str, tf: str = "daily") -> List[str]:
    """列出 A股目录下拥有指定周期文件的全部股票代码（已排序）"""
    tf = ASHARE_TF_ALIAS.get(tf.upper(), tf)
    codes = []
    if not os.path.isdir(data_dir):
        return codes
    for fn in os.listdir(data_dir):
        m = _ASHARE_FN.match(fn)
        if m and m.group("tf") == tf:
            codes.append(m.group("code"))
    return sorted(codes)


def ashare_load(data_dir: str, code: str, tf: str = "daily",
                adjust: str = "factor", adjust_mode: str = "hfq",
                factor_store=None) -> pd.DataFrame:
    """
    读取单只 A股 的 K 线，返回统一格式。

    做的清洗：
      - time -> date（日线 normalize 到自然日）
      - tick_volume -> volume，float32 -> float64（避免密度累加精度损失）
      - 丢弃 OHLC 缺失行；丢弃 volume<=0 的停牌行
      - 丢弃 high<low 或 close 越界的坏行
      - 按日期去重（保留最后一条）与排序

    adjust: 复权策略
        'factor'    （默认）用 data/adj_factors.parquet 的权威因子。
                    **因子缺失时抛 MissingFactorError** —— 见下方说明。
        'heuristic' 开盘缺口启发式（纯离线，但精度不足，仅供对比研究）
        'none'      不复权（原始数据）

    为什么 'factor' 缺失时要报错而不是回退启发式
        实测（50 只有权威因子的标的，scripts/eval_heuristic.py）：
            口径                  残余最大断层 中位 / P90    漏检率
            不复权                 0.341 / 0.509             —
            启发式 阈值0.22         0.093 / 0.282            74.9%
            启发式 按板块限幅        0.047 / 0.212            50.3%
        即便用了按板块的涨跌幅上限做阈值，仍有 **64% (32/50) 的股票残余
        >3% 的价格断层**，最差 75%。更糟的是会**误判**：000004 真实除权 0 次，
        启发式识别出 10 次，反而制造了 12% 的人为断层。
        静默地用错误的复权数据回测，比不复权更危险 —— 所以宁可报错。

    adjust_mode: 'hfq' 后复权（严格因果，回测用，默认）
                 'qfq' 前复权（最新价 = 真实市价，界面展示用）
        复权后 df.attrs 记录 n_ex_rights / adjust_mode / adjust_source。
    """
    tf_file = ASHARE_TF_ALIAS.get(tf.upper(), tf)
    path = os.path.join(data_dir, f"{code}_{tf_file}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    df["date"] = pd.to_datetime(df["time"] * ASHARE_TIME_SCALE, unit="s")
    if tf_file == "daily":
        df["date"] = df["date"].dt.normalize()
    out = _finalize(df, dedup_on_date=(tf_file == "daily"))

    if adjust is True:          # 兼容旧调用
        adjust = "factor"
    elif adjust is False:
        adjust = "none"

    if adjust == "factor":
        from .adjust import apply_factor
        store = factor_store if factor_store is not None else _default_factor_store()
        fac = store.get(code) if store is not None else None
        if fac is None or len(fac) <= 10:
            raise MissingFactorError(
                f"{code} 缺少权威复权因子。请先运行 "
                f"`python scripts/build_factors.py` 构建 data/adj_factors.parquet；"
                f"或显式指定 adjust='heuristic' / 'none'（精度代价见 ashare_load 文档）")
        out = apply_factor(out, fac, mode=adjust_mode)
    elif adjust == "heuristic":
        from .adjust import apply_adjustment
        out, _events = apply_adjustment(out, mode=adjust_mode, code=code)
        out.attrs["adjust_source"] = "gap_heuristic"
    elif adjust == "none":
        out.attrs["n_ex_rights"] = 0
        out.attrs["adjust_mode"] = "none"
        out.attrs["adjust_source"] = "none"
    else:
        raise ValueError(f"adjust 必须是 'factor'/'heuristic'/'none'，收到 {adjust!r}")
    return out


class MissingFactorError(RuntimeError):
    """缺少权威复权因子。故意报错而非静默回退，理由见 ashare_load 文档。"""


_FACTOR_STORE = None
_FACTOR_STORE_TRIED = False


def _default_factor_store():
    """惰性加载默认因子缓存（进程内只读一次）"""
    global _FACTOR_STORE, _FACTOR_STORE_TRIED
    if _FACTOR_STORE_TRIED:
        return _FACTOR_STORE
    _FACTOR_STORE_TRIED = True
    try:
        from .adjust import FACTOR_CACHE_DEFAULT, FactorStore
        base = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        path = os.path.join(base, FACTOR_CACHE_DEFAULT)
        st = FactorStore(path)
        _FACTOR_STORE = st if st.loaded else None
    except Exception:
        _FACTOR_STORE = None
    return _FACTOR_STORE


def _finalize(df: pd.DataFrame, dedup_on_date: bool = True) -> pd.DataFrame:
    for c in _OHLCV:
        if c not in df.columns:
            raise ValueError(f"缺少列 {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df.dropna(subset=_OHLCV)
    df = df[df["volume"] > 0]
    # 剔除结构性坏行
    ok = (df["high"] >= df["low"]) & \
         (df["high"] >= df["open"]) & (df["high"] >= df["close"]) & \
         (df["low"] <= df["open"]) & (df["low"] <= df["close"]) & \
         (df["low"] > 0)
    df = df[ok].copy()
    df = df.sort_values("date")
    if dedup_on_date:
        df = df.drop_duplicates(subset="date", keep="last")
    df = df.reset_index(drop=True)
    # A股 tick_volume 的单位是"手"(100股)，实测 000001 最新一根
    # 805221手 x 100股 x 10.47元 ≈ 8.4亿元，与平安银行实际日成交额量级一致。
    # amount 统一折算成"元"，使流动性阈值有明确的经济含义。
    df["amount"] = df["close"] * df["volume"] * SHARES_PER_LOT
    df["pct_chg"] = df["close"].pct_change().fillna(0.0) * 100
    return df[_OUT_COLS]


# ============================================================
# 指标：真实波幅
# ============================================================
def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_c = np.concatenate([[close[0]], close[:-1]])
    return np.maximum(high - low,
                      np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))


def atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """
    因果 ATR（Wilder 平滑的简化版：滚动均值）。
    第 i 个元素仅使用 0..i 的数据，可安全用于 t 时刻决策。
    前 period 根用累计均值填充，不产生 NaN。
    """
    tr = true_range(high, low, close)
    n = len(tr)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    csum = np.cumsum(tr)
    for i in range(min(period, n)):
        out[i] = csum[i] / (i + 1)
    if n > period:
        # 滚动均值
        roll = (csum[period:] - np.concatenate([[0.0], csum[:n - period - 1]]))
        out[period:] = roll / period
    return out


def tick_size_guess(close: np.ndarray) -> float:
    """
    估计最小价格变动单位。A股为 0.01 元；对高价品种放宽。
    仅用于给分箱宽度设下限，避免箱宽小于可成交精度。
    """
    px = float(np.nanmedian(close))
    if px <= 0:
        return 0.01
    if px < 1000:
        return 0.01
    return 10 ** (np.floor(np.log10(px)) - 4)


# ============================================================
# 标的池：确定性抽样 + 训练/测试切分
# ============================================================
@dataclass
class Symbol:
    code: str
    df: pd.DataFrame
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    open_: np.ndarray
    volume: np.ndarray
    date: np.ndarray
    atr: np.ndarray
    tick: float

    def __len__(self) -> int:
        return len(self.close)


def to_symbol(code: str, df: pd.DataFrame, atr_period: int = 14) -> Symbol:
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    d = df["date"].to_numpy()
    # 复权后 tick 需随价格尺度缩放，见 adjust.apply_factor 的说明
    tick = df.attrs.get("tick_base") or tick_size_guess(c)
    return Symbol(code=code, df=df, high=h, low=l, close=c, open_=o,
                  volume=v, date=d, atr=atr_series(h, l, c, atr_period),
                  tick=float(tick))


def _split_bucket(code: str, n_buckets: int = 10) -> int:
    """按代码哈希稳定分桶（同一代码永远落在同一桶，跨运行可复现）"""
    h = hashlib.md5(code.encode()).hexdigest()
    return int(h[:8], 16) % n_buckets


def build_universe(
    data_dir: str,
    tf: str = "daily",
    n_symbols: int = 200,
    min_bars: int = 800,
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_median_amount: float = 5e7,   # 日均成交额中位数 >= 5000万元
    train_buckets: Tuple[int, ...] = (0, 1, 2, 3, 4),
    adjust: str = "factor",
    adjust_mode: str = "hfq",
    verbose: bool = True,
) -> Dict[str, List[Symbol]]:
    """
    构建标的池并切分 train / test。

    切分方式：按股票代码 md5 分 10 桶，桶 0-4 为 train，5-9 为 test。
    → 训练集与测试集是**不同的股票**，参数在 train 上选，在 test 上报告，
      从结构上避免"同一批数据既调参又汇报"的过拟合。

    过滤条件：
      - 在 [start, end] 区间内至少 min_bars 根 K 线
      - 区间内日均成交额中位数 >= min_median_amount（剔除极端流动性差的票）
    """
    codes = ashare_list_codes(data_dir, tf)
    store = _default_factor_store() if adjust == "factor" else None
    if verbose:
        print(f"[universe] 目录内 {tf} 文件共 {len(codes)} 只  "
              f"复权={adjust}/{adjust_mode}"
              + (f"  权威因子覆盖 {len(store)} 只" if store is not None else ""))
    if adjust == "factor":
        if store is None or len(store) == 0:
            raise MissingFactorError(
                "权威复权因子缓存为空。请先运行 `python scripts/build_factors.py`。")
        # 只保留有因子的标的 —— 这是评测标的池的显式定义，不做静默降级
        codes = [c for c in codes if c in store]
        if verbose:
            print(f"[universe] 按因子覆盖过滤后可用 {len(codes)} 只")

    train: List[Symbol] = []
    test: List[Symbol] = []
    want_train = n_symbols // 2
    want_test = n_symbols - want_train
    scanned = 0
    rejected = {"bars": 0, "amount": 0, "error": 0}

    for code in codes:
        if len(train) >= want_train and len(test) >= want_test:
            break
        bucket = _split_bucket(code)
        is_train = bucket in train_buckets
        if is_train and len(train) >= want_train:
            continue
        if (not is_train) and len(test) >= want_test:
            continue
        scanned += 1
        try:
            df = ashare_load(data_dir, code, tf, adjust=adjust,
                             adjust_mode=adjust_mode, factor_store=store)
        except Exception:
            rejected["error"] += 1
            continue
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        if len(df) < min_bars:
            rejected["bars"] += 1
            continue
        if float(df["amount"].median()) < min_median_amount:
            rejected["amount"] += 1
            continue
        sym = to_symbol(code, df.reset_index(drop=True))
        (train if is_train else test).append(sym)

    if verbose:
        print(f"[universe] 扫描 {scanned} 只 -> train={len(train)} test={len(test)}  "
              f"剔除: 数据不足{rejected['bars']} 流动性{rejected['amount']} 读取失败{rejected['error']}")
    return {"train": train, "test": test}
