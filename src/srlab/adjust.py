# -*- coding: utf-8 -*-
"""
除权复权处理

背景（已由 scripts/check_adjustment.py 实测确认）
-----------------------------------------------
本地 D:\\A股_K线数据\\parquet\\stocks 是 **不复权** 数据：
  - 与 baostock adjustflag=3(不复权) 逐日比对：10/10 只股票吻合
    （4 只数值完全相同，6 只变异系数 3e-5 ~ 9e-4，来自 parquet 的 float32 存储精度）
  - 与前复权/后复权比对：变异系数 0.61 ~ 2.02，且在除权日出现 ~50% 的阶跃
  - 独立验证（不依赖外部源）：每只股票都有 1~18 次单日 >±21% 的跳空，
    最大 -56.7%（600519 于 2006-05-25，茅台 10转10 除权日）。
    A股 涨跌幅上限 ±10%（2020-08 后创业板/科创板 ±20%），
    因此这种跳空只可能是除权除息。

为什么必须处理
--------------
本算法的每一个环节都会被除权跳空污染：
  - 成交量剖面：除权前后的价格不在同一尺度，"筹码密集区"跨除权日即失去意义
  - ZigZag 枢轴：一次 -50% 的除权跳空会被识别成一段巨大的下跌腿
  - ATR：除权日 TR 暴增，把之后 14 根的所有 ATR 阈值全部放大
  - 历史触及统计：回看 750 根很容易跨过除权日

复权方法
--------
用**开盘价缺口**识别除权日：
    ratio = open[d] / close[d-1]
除权日的除权参考价就是按这个比例计算的，缺口发生在开盘。
而涨跌停最多 ±10%（或 ±20%），因此 ratio 偏离 1 超过阈值只能是除权。

识别后做**前复权**（把历史价格乘以累积因子，使最新价保持原值）：
    factor[t] = prod(ratio[d] for d in 除权日 if d > t)
    adj_price[t] = price[t] * factor[t]
前复权而非后复权，是因为界面要显示与当前真实价格一致的关键位价格。

成交量的处理
------------
送股/拆股会使股数按 1/ratio 变化，现金分红不改变股数。
仅凭价格缺口无法区分两者，因此**不调整成交量**，理由：
  1. 成交量在本算法里只作为 250 根窗口内的**相对权重**，一次性的水平位移影响
     远小于 50% 的价格断层；
  2. 60 根半衰期的时间衰减本身就压制了跨除权日的旧数据权重；
  3. 强行按价格比例调整成交量，在纯现金分红的情形下反而引入错误。
这个取舍在 validate_against_baostock() 的残差里可以量化检验。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 识别阈值：必须大于最宽的涨跌幅限制（创业板/科创板 2020-08 后 ±20%）
DEFAULT_THRESHOLD = 0.22

_PRICE_COLS = ("open", "high", "low", "close")

# 创业板 20% 涨跌幅生效日（2020-08-24 创业板注册制改革）
_GEM_20PCT_DATE = pd.Timestamp("2020-08-24")
# 科创板设立即为 20%
_STAR_START = pd.Timestamp("2019-07-22")


def daily_limit_series(code: str, dates: pd.Series) -> np.ndarray:
    """
    返回每个交易日适用的涨跌幅上限（小数）。

    A股 涨跌幅制度：
        主板 (600/601/603/605/000/001/002/003)  ±10%
        创业板 (300/301)  2020-08-24 起 ±20%，之前 ±10%
        科创板 (688/689)  ±20%（2019-07-22 设立起）
        北交所 (8xx/4xx)  ±30%
    未考虑 ST（±5%）—— ST 限制更严，用 10% 作上限只会让识别更保守，不会误判。
    """
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    n = len(d)
    if code.startswith(("688", "689")):
        return np.full(n, 0.20)
    if code.startswith(("8", "4")):
        return np.full(n, 0.30)
    if code.startswith(("300", "301")):
        return np.where(d >= _GEM_20PCT_DATE.to_datetime64(), 0.20, 0.10)
    return np.full(n, 0.10)


@dataclass
class ExRightsEvent:
    idx: int          # 除权日在序列中的下标
    date: object
    ratio: float      # open[d] / close[d-1]，< 1 表示除权下跳
    prev_close: float
    open_: float


def detect_ex_rights(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    min_ratio: float = 0.02,
    code: Optional[str] = None,
    margin: float = 0.005,
    max_halt_days: int = 10,
) -> List[ExRightsEvent]:
    """
    用开盘缺口识别除权日。

    识别逻辑
    --------
    除权日的除权参考价按 open/prev_close 的比例计算，缺口发生在**开盘**。
    而涨跌停最多 ±limit，因此开盘跌幅超过 limit 的只能是除权。

    threshold 的取法（这是精度的关键）
        固定 0.22 太宽：它高于所有涨跌幅上限所以绝不误判，但会漏掉
        10送2（ratio=10/12=0.833，缺口 -16.7%）这类真实送股 ——
        实测残余最大阶跃中位 7.9%（最大 20%）就是这么来的。
        改为**按板块和日期取当日适用的涨跌幅上限 + margin**：
        主板 10.5%、创业板 2020-08-24 后 20.5%、科创板 20.5%、北交所 30.5%。
        阈值从 22% 降到 10.5%，10送2/10送3 这类事件不再漏掉。
        传入 threshold 可覆盖（回归对比用）。

    误判防护
        停牌复牌可能合法跳空超过涨跌幅上限，因此**跳过距上一根 K 线
        超过 max_halt_days 个自然日的 bar**（长期停牌复牌，不是除权）。
        只识别向下跳空：A股 缩股极罕见，向上大缺口更可能是复牌或脏数据，
        误当除权会把历史价格整体放大，危害更大。
    """
    if len(df) < 3:
        return []
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    dates = pd.to_datetime(df["date"])
    dates_np = dates.to_numpy()

    prev_c = c[:-1]
    cur_o = o[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prev_c > 0, cur_o / prev_c, 1.0)

    if threshold is not None:
        thr = np.full(len(ratio), float(threshold))
    else:
        lim = daily_limit_series(code or "", dates)
        thr = lim[1:] + margin

    # 停牌复牌过滤：与上一根的自然日间隔
    gap_days = (dates_np[1:] - dates_np[:-1]).astype("timedelta64[D]").astype(int)

    m = ((ratio < 1.0 - thr) & (ratio > min_ratio) & np.isfinite(ratio)
         & (gap_days <= max_halt_days))
    idxs = np.nonzero(m)[0] + 1
    return [
        ExRightsEvent(idx=int(i), date=dates_np[i], ratio=float(ratio[i - 1]),
                      prev_close=float(c[i - 1]), open_=float(o[i]))
        for i in idxs
    ]


def adjustment_factors(n: int, events: List[ExRightsEvent],
                       mode: str = "hfq") -> np.ndarray:
    """
    构造累积复权因子。

    mode='qfq' 前复权（锚定最新一根）
        factor[t] = prod(ratio[d] for d in 除权日 if d > t)
        => 最新一根 factor == 1.0，最新价保持真实市价，适合界面展示。
        **但它不是因果的**：t 时刻的因子取决于 t 之后发生的除权。

    mode='hfq' 后复权（锚定第一根）
        factor[t] = prod(1/ratio[d] for d in 除权日 if d <= t)
        => 第一根 factor == 1.0，t 时刻的因子只依赖 <= t 的事件，**严格因果**。
        回测必须用这个。

    两者只差一个**全局常数**（factor_qfq = factor_hfq * factor_qfq[0]），
    而本算法的分箱、阈值、区宽全部以 ATR 为单位，是尺度不变的，
    因此两种口径下检测结果的相对几何完全一致 ——
    由 leakage.test_adjust_scale_invariance 逐点验证。
    """
    f = np.ones(n, dtype=np.float64)
    if not events:
        return f
    if mode == "qfq":
        for ev in events:
            # 除权日之前（不含除权日）的所有 bar 乘以 ratio
            f[:ev.idx] *= ev.ratio
    elif mode == "hfq":
        for ev in events:
            # 除权日及其之后的所有 bar 除以 ratio
            f[ev.idx:] /= ev.ratio
    else:
        raise ValueError(f"mode 必须是 'qfq' 或 'hfq'，收到 {mode!r}")
    return f


def apply_adjustment(df: pd.DataFrame, events: Optional[List[ExRightsEvent]] = None,
                     threshold: Optional[float] = None,
                     mode: str = "hfq",
                     code: Optional[str] = None) -> Tuple[pd.DataFrame, List[ExRightsEvent]]:
    """
    对 OHLC 做复权，返回 (新 DataFrame, 事件列表)。

    mode 默认 'hfq'（后复权，严格因果），回测用；界面展示可用 'qfq'。

    volume 不变（见模块 docstring 的取舍说明）；
    amount 按调整后价格重算，保持 amount == close * volume * 100 的一致性。
    """
    if events is None:
        events = detect_ex_rights(df, threshold=threshold, code=code)
    out = df.copy()
    if not events:
        out.attrs["n_ex_rights"] = 0
        out.attrs["adjust_mode"] = mode
        return out, events

    f = adjustment_factors(len(df), events, mode=mode)
    for c in _PRICE_COLS:
        if c in out.columns:
            out[c] = out[c].to_numpy(np.float64) * f
    if "amount" in out.columns and "volume" in out.columns:
        out["amount"] = out["close"] * out["volume"] * 100.0
    # pct_chg 必须重算：除权日的原始涨跌幅是错的
    out["pct_chg"] = out["close"].pct_change().fillna(0.0) * 100
    out.attrs["n_ex_rights"] = len(events)
    out.attrs["adjust_mode"] = mode
    return out, events


# ============================================================
# 污染度量化
# ============================================================
def contamination_report(
    df: pd.DataFrame,
    events: List[ExRightsEvent],
    warmup: int = 300,
    lookback: int = 750,
    start: Optional[str] = None,
) -> Dict:
    """
    统计"有多少决策点的回看窗口跨过了除权日"。

    这是决定"要不要复权"的关键数字：如果只有 1% 的决策点受影响，
    直接剔除即可；如果是 50%，就必须复权。
    """
    n = len(df)
    if n <= warmup:
        return {"n_points": 0}
    ev_idx = np.array([e.idx for e in events], dtype=np.int64)
    dates = pd.to_datetime(df["date"])
    ts = np.arange(warmup, n)
    if start:
        ts = ts[dates.to_numpy()[ts] >= np.datetime64(pd.Timestamp(start))]
    if len(ts) == 0:
        return {"n_points": 0}
    if len(ev_idx) == 0:
        return {"n_points": len(ts), "n_contaminated": 0, "pct": 0.0}
    # 决策点 t 的回看窗口是 [t-lookback, t]
    lo = np.maximum(ts - lookback, 0)
    cnt = np.array([int(((ev_idx > l) & (ev_idx <= t)).sum())
                    for l, t in zip(lo, ts)])
    return {
        "n_points": len(ts),
        "n_contaminated": int((cnt > 0).sum()),
        "pct": float((cnt > 0).mean() * 100),
        "avg_events_in_window": float(cnt.mean()),
    }


# ============================================================
# 与 baostock 对照验证
# ============================================================
def validate_against_baostock(code: str, local_adj: pd.DataFrame,
                              sleep: float = 0.15) -> Dict:
    """
    把自建前复权序列与 baostock 前复权(adjustflag=2)比对。

    判据同 check_adjustment.py：比值的**变异系数**（前复权基准日不同会导致
    整体等比缩放，所以不能直接比数值）。
    """
    import time

    import baostock as bs

    prefix = "sh" if code.startswith(("6", "5", "9")) else "sz"
    start = pd.Timestamp(local_adj["date"].min()).strftime("%Y-%m-%d")
    end = pd.Timestamp(local_adj["date"].max()).strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(
        f"{prefix}.{code}", "date,close,volume",
        start_date=start, end_date=end, frequency="d", adjustflag="2")
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    time.sleep(sleep)
    if not rows:
        return {"code": code, "note": "baostock 无数据"}
    ref = pd.DataFrame(rows, columns=["date", "close", "volume"])
    ref["date"] = pd.to_datetime(ref["date"])
    for c in ("close", "volume"):
        ref[c] = pd.to_numeric(ref[c], errors="coerce")
    ref = ref[ref["volume"] > 0].dropna(subset=["close"])

    m = local_adj[["date", "close"]].merge(ref[["date", "close"]], on="date",
                                          suffixes=("_l", "_r"))
    if len(m) < 100:
        return {"code": code, "n": len(m), "note": "重叠不足"}
    r = (m["close_l"] / m["close_r"]).to_numpy(np.float64)
    r = r[np.isfinite(r) & (r > 0)]
    return {
        "code": code, "n": len(r),
        "r_cv": float(np.std(r) / np.mean(r)),
        "max_jump": float(np.max(np.abs(np.diff(r) / r[:-1]))) if len(r) > 1 else np.nan,
    }


# ============================================================
# 精确复权因子缓存（权威来源）
# ============================================================
"""
为什么需要因子缓存
------------------
开盘缺口启发式（detect_ex_rights）只能识别 >22% 的跳空，因为 A股 涨跌幅上限
±10%（创业板/科创板 ±20%）决定了阈值不能再低。但**小额现金分红**产生的除权
缺口只有 0.5% ~ 5%，全部会被漏掉，而它们会累积。

实测（10 只 2010 年后除权 ≥3 次的标的，与 baostock 官方前复权比对）：
    不复权          比值变异系数 中位 0.557，最大单日阶跃 中位 0.342
    启发式复权后    比值变异系数 中位 0.068，最大单日阶跃 中位 0.079（最大 0.20）
改善 8 倍，但残余仍有 ~8% 的价格断层 —— 对一只典型个股相当于 3~4 个 ATR，
足以毁掉一个支撑位的判定。因此必须用权威来源的精确因子。

因子怎么来
----------
    factor_hfq(t) = 后复权收盘(t) / 不复权收盘(t)
两个序列都来自同一个数据源（akshare→东方财富），两次请求约 1.2 秒。
这个因子严格因果：t 时刻的值只由 <= t 的分红送转决定
（后复权锚定第一根，之后的事件只影响之后的因子）。

去噪
----
两个序列都是 2 位小数，比值带 ~0.1% 的量化噪声，直接用会在每根 bar 上引入抖动。
真实因子是**分段常数**的，因此按 step_thr 检测阶跃点，段内取中位数，
还原成干净的阶梯函数。
"""

FACTOR_CACHE_DEFAULT = "data/adj_factors.parquet"


def fetch_factor_akshare(code: str, step_thr: float = 0.003) -> pd.DataFrame:
    """
    从 akshare（东方财富）取精确后复权因子。

    返回 DataFrame[date, factor]，factor 为分段常数（已去量化噪声）。
    第一根的 factor 归一到 1.0，因此 factor 严格因果、单调不减。
    """
    import akshare as ak

    kw = dict(symbol=code, period="daily", start_date="19900101", end_date="20991231")
    raw = ak.stock_zh_a_hist(adjust="", **kw)
    hfq = ak.stock_zh_a_hist(adjust="hfq", **kw)
    if raw is None or hfq is None or raw.empty or hfq.empty:
        return pd.DataFrame(columns=["date", "factor"])

    dcol = "日期" if "日期" in raw.columns else "date"
    ccol = "收盘" if "收盘" in raw.columns else "close"
    m = raw[[dcol, ccol]].merge(hfq[[dcol, ccol]], on=dcol, suffixes=("_r", "_h"))
    m = m.rename(columns={dcol: "date"})
    m["date"] = pd.to_datetime(m["date"])
    r = pd.to_numeric(m[f"{ccol}_r"], errors="coerce").to_numpy(np.float64)
    h = pd.to_numeric(m[f"{ccol}_h"], errors="coerce").to_numpy(np.float64)
    ok = np.isfinite(r) & np.isfinite(h) & (r > 0)
    m, r, h = m[ok].reset_index(drop=True), r[ok], h[ok]
    if len(r) < 10:
        return pd.DataFrame(columns=["date", "factor"])

    f = h / r
    f = _piecewise_constant(f, step_thr)
    f = f / f[0]                      # 归一：第一根 = 1.0（后复权锚定起点）
    return pd.DataFrame({"date": m["date"].to_numpy(), "factor": f})


def _piecewise_constant(f: np.ndarray, step_thr: float) -> np.ndarray:
    """把带量化噪声的因子序列还原成分段常数（段内取中位数）"""
    n = len(f)
    if n < 3:
        return f
    rel = np.abs(np.diff(f)) / np.maximum(f[:-1], 1e-12)
    breaks = np.nonzero(rel > step_thr)[0] + 1
    bounds = np.concatenate([[0], breaks, [n]])
    out = np.empty(n, dtype=np.float64)
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[a:b] = np.median(f[a:b])
    return out


def fetch_factor_baostock(code: str, start: str = "2004-01-01",
                          end: str = "2030-12-31",
                          step_thr: float = 0.003) -> pd.DataFrame:
    """
    从 baostock 取精确后复权因子（adjustflag 1=后复权 / 3=不复权 相除）。

    比 akshare 慢，但没有东方财富那种 IP 限流封禁
    （实测连续 800 次 akshare 请求后被 RemoteDisconnected 拒绝）。
    """
    import baostock as bs

    pre = "sh" if code.startswith(("6", "5", "9")) else "sz"
    bs_code = f"{pre}.{code}"

    def _pull(flag: str) -> pd.DataFrame:
        rs = bs.query_history_k_data_plus(
            bs_code, "date,close,volume", start_date=start, end_date=end,
            frequency="d", adjustflag=flag)
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame(columns=["date", "close"])
        d = pd.DataFrame(rows, columns=["date", "close", "volume"])
        d["date"] = pd.to_datetime(d["date"])
        for c in ("close", "volume"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        return d[d["volume"] > 0].dropna(subset=["close"])[["date", "close"]]

    hfq, raw = _pull("1"), _pull("3")
    if hfq.empty or raw.empty:
        return pd.DataFrame(columns=["date", "factor"])
    m = raw.merge(hfq, on="date", suffixes=("_r", "_h"))
    if len(m) < 10:
        return pd.DataFrame(columns=["date", "factor"])
    r = m["close_r"].to_numpy(np.float64)
    h = m["close_h"].to_numpy(np.float64)
    ok = np.isfinite(r) & np.isfinite(h) & (r > 0)
    m, r, h = m[ok].reset_index(drop=True), r[ok], h[ok]
    if len(r) < 10:
        return pd.DataFrame(columns=["date", "factor"])
    f = _piecewise_constant(h / r, step_thr)
    f = f / f[0]
    return pd.DataFrame({"date": m["date"].to_numpy(), "factor": f})


_FACTOR_FETCHERS = {
    "baostock": fetch_factor_baostock,
    "akshare": fetch_factor_akshare,
}


def build_factor_cache(codes: List[str], cache_path: str = FACTOR_CACHE_DEFAULT,
                       source: str = "baostock", sleep: float = 0.05,
                       max_retries: int = 2, verbose: bool = True) -> pd.DataFrame:
    """
    批量拉取并落盘因子缓存。已有的 code 会跳过（增量构建，可中断续跑）。

    缓存格式: parquet[code, date, factor]
    每 50 只增量落盘一次，避免中途失败白跑。
    """
    import time

    fetch = _FACTOR_FETCHERS[source]
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    old = pd.DataFrame(columns=["code", "date", "factor"])
    have: Dict[str, bool] = {}
    if os.path.exists(cache_path):
        old = pd.read_parquet(cache_path)
        have = {str(c): True for c in old["code"].unique()}

    todo = [c for c in codes if c not in have]
    if verbose:
        print(f"[factor] 源={source} 缓存已有 {len(have)} 只，待拉取 {len(todo)} 只")
    if not todo:
        return old

    parts = [old] if len(old) else []
    t0 = time.time()
    fail = 0

    def _flush():
        out = pd.concat(parts, ignore_index=True) if parts else old
        if len(out):
            out.to_parquet(cache_path, index=False)
        return out

    flush_every = 10

    for i, code in enumerate(todo, 1):
        got = None
        for attempt in range(max_retries + 1):
            try:
                f = fetch(code)
                if not f.empty:
                    got = f
                break
            except Exception as e:
                if attempt >= max_retries:
                    if verbose and fail < 5:
                        print(f"  [warn] {code}: {type(e).__name__}: {e}")
                else:
                    time.sleep(1.5 * (attempt + 1))
        if got is None:
            fail += 1
        else:
            got.insert(0, "code", code)
            parts.append(got)
        if sleep:
            time.sleep(sleep)
        if i % flush_every == 0:
            _flush()
            if verbose:
                print(f"  [factor] {i}/{len(todo)}  失败 {fail}  "
                      f"{time.time()-t0:.0f}s", flush=True)

    out = _flush()
    if verbose:
        print(f"[factor] 落盘 {cache_path}: {out['code'].nunique()} 只, "
              f"{len(out):,} 行, 失败 {fail}, 耗时 {time.time()-t0:.0f}s")
    return out


class FactorStore:
    """因子缓存的内存索引，供 ashare_load 高频调用"""

    def __init__(self, cache_path: str = FACTOR_CACHE_DEFAULT):
        self.path = cache_path
        self._by_code: Dict[str, pd.DataFrame] = {}
        self.loaded = False
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            df["date"] = pd.to_datetime(df["date"])
            for code, g in df.groupby("code"):
                self._by_code[str(code)] = g[["date", "factor"]].reset_index(drop=True)
            self.loaded = True

    def __contains__(self, code: str) -> bool:
        return code in self._by_code

    def get(self, code: str) -> Optional[pd.DataFrame]:
        return self._by_code.get(code)

    def __len__(self) -> int:
        return len(self._by_code)


def apply_factor(df: pd.DataFrame, factor: pd.DataFrame,
                 mode: str = "hfq") -> pd.DataFrame:
    """
    用权威因子做复权。

    mode='hfq' 直接乘以因子（锚定第一根，严格因果）
    mode='qfq' 再除以最后一根的因子（锚定最新，最新价 = 真实市价）

    按日期左连接；缺失日期用**前值填充**（因子是分段常数，前值填充是正确的）。

    早于因子序列起点的 bar 用因子的**首值回填**（bfill），不能填 1.0 ——
    因子缓存可能只覆盖 2005 年之后，若把更早的 bar 填 1.0，会在缓存起点
    人为制造一个巨大的价格断层（正是复权要消除的东西）。
    """
    out = df.copy()
    f = out[["date"]].merge(factor, on="date", how="left")["factor"]
    f = f.ffill().bfill().fillna(1.0).to_numpy(np.float64)
    if mode == "qfq":
        f = f / f[-1]
    elif mode != "hfq":
        raise ValueError(f"mode 必须是 'qfq' 或 'hfq'，收到 {mode!r}")
    for c in _PRICE_COLS:
        if c in out.columns:
            out[c] = out[c].to_numpy(np.float64) * f
    # 最小报价单位也要随复权缩放，否则分箱下限 max(tick, 0.25*ATR) 在
    # hfq / qfq 两种口径下可能一个绑定一个不绑定，破坏尺度不变性。
    # 真实 tick 是 0.01 元，复权后等效 tick = 0.01 * 该口径的末端因子。
    out.attrs["tick_base"] = 0.01 * float(f[-1])
    if "amount" in out.columns and "volume" in out.columns:
        out["amount"] = out["close"] * out["volume"] * 100.0
    out["pct_chg"] = out["close"].pct_change().fillna(0.0) * 100
    out.attrs["adjust_mode"] = mode
    out.attrs["adjust_source"] = "factor_cache"
    # 阶跃数 = 因子变化次数，等价于除权次数
    out.attrs["n_ex_rights"] = int((np.abs(np.diff(f)) / np.maximum(f[:-1], 1e-12) > 1e-6).sum())
    return out


# ============================================================
# 东方财富直连（curl_cffi 指纹伪装，绕过 akshare 被限流的问题）
# ============================================================
"""
akshare 底层用普通 requests，连续请求约 800 次后被东方财富按 TLS 指纹封禁
（实测持续返回 RemoteDisconnected，换参数无效）。
PA_Agent 的做法是 curl_cffi + 浏览器指纹伪装 + CDN 多节点轮换，这里照搬。
baostock 虽然稳定但全历史查询约 36 秒/只，400 只要 4 小时，不实用。
"""

_EM_HOSTS = (
    "push2delay.eastmoney.com",
    "push2his.eastmoney.com",
    "33.push2his.eastmoney.com",
    "63.push2his.eastmoney.com",
    "7.push2his.eastmoney.com",
)
_EM_UT = "fa5fd1943c7b386f172d6893dbfba10b"
_EM_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/",
}
_EM_IMPERSONATE = ("chrome120", "chrome116", "chrome131", "edge101", None)


def _em_secid(code: str) -> str:
    return f"1.{code}" if code.startswith(("6", "5", "9")) else f"0.{code}"


def _em_kline(code: str, fqt: int, timeout: float = 20.0) -> pd.DataFrame:
    """拉一条东财日线序列。fqt: 0=不复权 1=前复权 2=后复权"""
    try:
        from curl_cffi import requests as http
        has_imp = True
    except ImportError:
        import requests as http
        has_imp = False

    params = {
        "secid": _em_secid(code),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": "101", "fqt": str(fqt),
        "beg": "19900101", "end": "20500101",
        "ut": _EM_UT,
    }
    last: Optional[Exception] = None
    for host in _EM_HOSTS:
        for imp in (_EM_IMPERSONATE if has_imp else (None,)):
            url = f"https://{host}/api/qt/stock/kline/get"
            kw = {"params": params, "headers": _EM_HEADERS, "timeout": timeout}
            if imp and has_imp:
                kw["impersonate"] = imp
            try:
                r = http.get(url, **kw)
                if r.status_code != 200:
                    last = RuntimeError(f"HTTP {r.status_code}")
                    continue
                js = r.json()
            except Exception as e:
                last = e
                continue
            data = (js or {}).get("data") or {}
            kl = data.get("klines") or []
            if not kl:
                last = RuntimeError("empty klines")
                continue
            rows = []
            for line in kl:
                p = line.split(",")
                if len(p) < 3:
                    continue
                rows.append((p[0], float(p[2])))   # f51=date, f53=close
            if not rows:
                last = RuntimeError("parse failed")
                continue
            d = pd.DataFrame(rows, columns=["date", "close"])
            d["date"] = pd.to_datetime(d["date"])
            return d
    raise RuntimeError(f"东财拉取失败 {code}: {last}")


def fetch_factor_eastmoney(code: str, step_thr: float = 0.003) -> pd.DataFrame:
    """从东方财富取精确后复权因子（fqt=2 后复权 / fqt=0 不复权 相除）"""
    hfq = _em_kline(code, fqt=2)
    raw = _em_kline(code, fqt=0)
    if hfq.empty or raw.empty:
        return pd.DataFrame(columns=["date", "factor"])
    m = raw.merge(hfq, on="date", suffixes=("_r", "_h"))
    r = m["close_r"].to_numpy(np.float64)
    h = m["close_h"].to_numpy(np.float64)
    ok = np.isfinite(r) & np.isfinite(h) & (r > 0)
    m, r, h = m[ok].reset_index(drop=True), r[ok], h[ok]
    if len(r) < 10:
        return pd.DataFrame(columns=["date", "factor"])
    f = _piecewise_constant(h / r, step_thr)
    f = f / f[0]
    return pd.DataFrame({"date": m["date"].to_numpy(), "factor": f})


_FACTOR_FETCHERS["eastmoney"] = fetch_factor_eastmoney
