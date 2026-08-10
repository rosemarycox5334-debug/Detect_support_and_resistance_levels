"""
本地期货 K 线数据加载器

数据约定：
- 文件命名：{品种名}主连_{周期}.parquet
- 周期代码：D1=日线, W1=周线, M1/M3/M5/M15/M30=分钟, H1/H2/H4=小时
- parquet 列：time(unix秒), open, high, low, close, tick_volume

本加载器将原始数据统一转换为内部 DataFrame：
    date(datetime), open, high, low, close, volume, amount, turn, pct_chg
其中 amount/turn/pct_chg 可缺失（用于兼容 density_analyzer 的可选字段）。
"""

import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd


# 周期代码 → 显示名
TIMEFRAME_LABELS = {
    "D1": "日线",
    "W1": "周线",
    "M1": "1分钟",
    "M3": "3分钟",
    "M5": "5分钟",
    "M15": "15分钟",
    "M30": "30分钟",
    "H1": "1小时",
    "H2": "2小时",
    "H4": "4小时",
}

# 多时间框架分析默认使用的周期组（与原 baostock 版 day/week/month 对应）
# 期货本地数据通常没有月线，用 W1 替代月线
MTF_TIMEFRAMES = ["D1", "W1"]

# 周期层级（从小到大），与 config.TF_ORDER 保持一致
_TF_ORDER = ["M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1", "MN", "MN1"]

# 每个周期默认分析窗口（K线数）
_TF_WINDOW = {
    "M1": 120, "M3": 120, "M5": 100, "M15": 80, "M30": 60,
    "H1": 60, "H2": 50, "H4": 40, "D1": 40, "W1": 30, "MN": 12, "MN1": 12,
}


def pick_mtf_timeframes(main_tf: str, available_tfs: List[str],
                        up: int = 2, down: int = 2) -> List[Tuple[str, float, int]]:
    """
    自适应选择多时间框架对比的邻居周期。

    根据主周期在周期层级中的位置，向上（更大周期）和向下（更小周期）各取最多
    up / down 个可用邻居。权重规则：
      - 主周期：1.0
      - 向上第 1 个：2.0，第 2 个：3.0（大周期结构更重要，权重递增）
      - 向下：0.5（小周期用于精确入场，权重较低）

    返回: [(tf_code, weight, window), ...]  第一个元素始终是主周期
    """
    available = set(available_tfs)
    if main_tf not in _TF_ORDER:
        return [(main_tf, 1.0, _TF_WINDOW.get(main_tf, 40))]
    idx = _TF_ORDER.index(main_tf)

    result = [(main_tf, 1.0, _TF_WINDOW.get(main_tf, 40))]

    # 向上（更大周期），权重 2、3 递增
    weight = 2.0
    pos = idx + 1
    count = 0
    while pos < len(_TF_ORDER) and count < up:
        t = _TF_ORDER[pos]
        if t in available:
            result.append((t, weight, _TF_WINDOW.get(t, 40)))
            weight += 1.0
            count += 1
        pos += 1

    # 向下（更小周期），权重 0.5
    pos = idx - 1
    count = 0
    while pos >= 0 and count < down:
        t = _TF_ORDER[pos]
        if t in available:
            result.append((t, 0.5, _TF_WINDOW.get(t, 40)))
            count += 1
        pos -= 1

    return result

# 文件名解析（同时兼容两种命名）：
#   1. 期货本地格式：{品种}主连_{周期}.parquet   例如 螺纹钢主连_D1.parquet
#   2. MT5 导出格式：{品种}_{周期}.parquet       例如 EURUSD_H1.parquet、US100.cash_H1.parquet
_FILENAME_PATTERNS = [
    re.compile(r"^(?P<symbol>.+?)主连_(?P<tf>[A-Z0-9]+)\.parquet$"),
    re.compile(r"^(?P<symbol>.+)_(?P<tf>[A-Z0-9]+)\.parquet$"),
]


def _parse_filename(filename: str) -> Optional[Tuple[str, str]]:
    """从文件名解析 (品种, 周期)；不匹配返回 None"""
    for pat in _FILENAME_PATTERNS:
        m = pat.match(filename)
        if m:
            return m.group("symbol"), m.group("tf")
    return None


def list_instruments(data_dir: str) -> Dict[str, List[str]]:
    """
    扫描数据文件夹，返回 {品种: [可用周期列表]}

    例：{'螺纹钢': ['D1','W1','H1','M5',...], '沪金':[...]}
    """
    result: Dict[str, List[str]] = {}
    if not os.path.isdir(data_dir):
        return result
    for fn in os.listdir(data_dir):
        if not fn.endswith(".parquet"):
            continue
        parsed = _parse_filename(fn)
        if not parsed:
            continue
        symbol, tf = parsed
        result.setdefault(symbol, []).append(tf)
    # 每个品种的周期按固定顺序排序
    order = {tf: i for i, tf in enumerate(
        ["D1", "W1", "H1", "H2", "H4", "M30", "M15", "M5", "M3", "M1"])}
    for symbol in result:
        result[symbol].sort(key=lambda t: order.get(t, 99))
    # 品种按名称排序
    return {s: result[s] for s in sorted(result.keys())}


def _filepath(data_dir: str, symbol: str, tf: str) -> str:
    """返回实际存在的 parquet 文件路径（兼容两种命名）"""
    # 优先期货格式 {symbol}主连_{tf}.parquet，回退 MT5 格式 {symbol}_{tf}.parquet
    fut_path = os.path.join(data_dir, f"{symbol}主连_{tf}.parquet")
    if os.path.exists(fut_path):
        return fut_path
    mt5_path = os.path.join(data_dir, f"{symbol}_{tf}.parquet")
    if os.path.exists(mt5_path):
        return mt5_path
    # 都不存在：返回期货格式路径用于报错信息（保持原有报错文案）
    return fut_path


def load_kline(data_dir: str, symbol: str, tf: str) -> pd.DataFrame:
    """
    读取单个品种指定周期的 K 线，返回统一格式 DataFrame

    返回列：date(datetime64[ns]), open, high, low, close, volume
    额外列（可选）：amount=0, turn=0, pct_chg(自动计算)

    参数:
        data_dir: 数据文件夹
        symbol:   品种名（不含"主连"后缀）
        tf:       周期代码，如 'D1'/'W1'/'M5'
    """
    path = _filepath(data_dir, symbol, tf)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件: {path}")

    df = pd.read_parquet(path)

    # 兼容列名：tick_volume → volume
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})

    # time (unix秒) → datetime
    if "time" in df.columns:
        # 数值型 unix 时间戳（秒）
        if pd.api.types.is_numeric_dtype(df["time"]):
            df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(
                "Asia/Shanghai").dt.tz_localize(None)
        else:
            df["date"] = pd.to_datetime(df["time"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError(f"文件 {path} 缺少 time/date 列")

    # 确保所需列存在
    required = ["open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"文件 {path} 缺少列: {c}")

    # 转数值
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 过滤无效行
    df = df.dropna(subset=required)
    df = df[df["volume"] > 0].copy()

    # 可选字段填充（density_analyzer 不强制要求）
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    if "turn" not in df.columns:
        df["turn"] = 0.0
    # 涨跌幅 %
    if "pct_chg" not in df.columns:
        df["pct_chg"] = df["close"].pct_change() * 100
    df["pct_chg"] = df["pct_chg"].fillna(0)

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 统一最终列顺序
    return df[["date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_chg"]]


def iter_all_instruments(
    data_dir: str,
    tf: str = "D1",
    symbols: Optional[List[str]] = None,
):
    """
    迭代生成 (symbol, DataFrame)，用于批量分析

    参数:
        data_dir: 数据文件夹
        tf:       使用的周期
        symbols:  限定品种列表；None 表示全部品种
    """
    instruments = list_instruments(data_dir)
    target_symbols = symbols if symbols is not None else list(instruments.keys())
    for symbol in target_symbols:
        available_tfs = instruments.get(symbol, [])
        if tf not in available_tfs:
            continue
        try:
            df = load_kline(data_dir, symbol, tf)
            yield symbol, df
        except Exception as e:
            # 跳过错误文件，继续下一个
            print(f"[跳过] {symbol} ({tf}): {e}")
            continue
