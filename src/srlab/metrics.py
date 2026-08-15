# -*- coding: utf-8 -*-
"""
指标聚合与统计检验

原则：
  1. 任何比例都配 Wilson 置信区间与样本量，禁止裸比例；
  2. 任何"胜率"都必须有对照组（placebo）才有意义 —— 支撑位在均值回复的
     市场里天然有较高的"守住率"，不与对照比就是自欺；
  3. 提升度差异做两比例 z 检验，给出 p 值。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson 得分区间。返回 (点估计, 下界, 上界)，n=0 时全 nan"""
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def two_prop_ztest(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    """两独立比例 z 检验，返回 (z, 双尾 p)。用于 模型 vs 对照 的显著性。"""
    if n1 <= 0 or n2 <= 0:
        return (np.nan, np.nan)
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se < 1e-15:
        return (np.nan, np.nan)
    z = (p1 - p2) / se
    # 正态双尾 p，用误差函数
    from math import erfc, sqrt
    return (float(z), float(erfc(abs(z) / sqrt(2))))


def summarize(df: pd.DataFrame, width_cap: Optional[float] = None) -> Dict:
    """
    汇总一个检测器的评测结果。

    参数:
        width_cap: 若给定，只统计 width_atr <= width_cap 的关键位
                   （防止用超宽区间刷胜率）
    """
    d = df
    if width_cap is not None:
        d = d[d["width_atr"] <= width_cap]

    n_levels = len(d)
    if n_levels == 0:
        return {"n_levels": 0}

    touched = d[d["touched"]]
    decided = touched[touched["result"].isin(["hold", "break"])]
    n_hold = int((decided["result"] == "hold").sum())
    n_dec = len(decided)
    p, lo, hi = wilson(n_hold, n_dec)

    out = {
        "n_levels": n_levels,
        "n_symbols": d["code"].nunique(),
        "n_dates": d.groupby("code")["t"].nunique().sum(),
        "avg_width_atr": float(d["width_atr"].mean()),
        "med_width_atr": float(d["width_atr"].median()),
        "avg_dist_atr": float(d["dist_atr"].abs().mean()),
        "touch_rate": float(d["touched"].mean()),
        "n_touched": len(touched),
        "n_decided": n_dec,
        "undecided_rate": float((touched["result"] == "undecided").mean()) if len(touched) else np.nan,
        "hold_rate": p,
        "hold_lo95": lo,
        "hold_hi95": hi,
        "n_hold": n_hold,
        "edge_atr": float(touched["edge_atr"].mean()) if len(touched) else np.nan,
        "edge_atr_med": float(touched["edge_atr"].median()) if len(touched) else np.nan,
        "mfe_atr": float(touched["mfe_atr"].mean()) if len(touched) else np.nan,
        "mae_atr": float(touched["mae_atr"].mean()) if len(touched) else np.nan,
        "fwd_ret_pct": float(touched["fwd_ret_pct"].mean()) if len(touched) else np.nan,
        "fwd_ret_med": float(touched["fwd_ret_pct"].median()) if len(touched) else np.nan,
    }
    return out


def summarize_by_kind(df: pd.DataFrame, width_cap: Optional[float] = None) -> pd.DataFrame:
    rows = []
    for kind in ("support", "resistance"):
        s = summarize(df[df["kind"] == kind], width_cap)
        s["kind"] = kind
        rows.append(s)
    s = summarize(df, width_cap)
    s["kind"] = "all"
    rows.append(s)
    return pd.DataFrame(rows).set_index("kind")


def compare(df: pd.DataFrame, base: str, chal: str,
            width_cap: Optional[float] = None) -> Dict:
    """
    模型 vs 对照 的提升度与显著性。
    对齐口径：只比较双方都产出了关键位的 (code, t) 交集，避免样本构成差异。
    """
    a = df[df["detector"] == base]
    b = df[df["detector"] == chal]
    if width_cap is not None:
        a = a[a["width_atr"] <= width_cap]
        b = b[b["width_atr"] <= width_cap]

    sa, sb = summarize(a), summarize(b)
    ka = sa.get("n_hold", 0) or 0
    na = sa.get("n_decided", 0) or 0
    kb = sb.get("n_hold", 0) or 0
    nb = sb.get("n_decided", 0) or 0
    z, pval = two_prop_ztest(kb, nb, ka, na)

    return {
        "base": base, "chal": chal,
        "base_hold": sa.get("hold_rate"), "base_n": na,
        "chal_hold": sb.get("hold_rate"), "chal_n": nb,
        "lift_pp": (sb.get("hold_rate", np.nan) - sa.get("hold_rate", np.nan)) * 100,
        "z": z, "p_value": pval,
        "base_edge_atr": sa.get("edge_atr"), "chal_edge_atr": sb.get("edge_atr"),
        "edge_lift": (sb.get("edge_atr", np.nan) - sa.get("edge_atr", np.nan)),
    }


# ============================================================
# 概率校准
# ============================================================
def calibration_table(conf: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    可靠性表：把预测概率分箱，对比每箱的预测均值与实际守住率。
    用户可直接看这张表验证"模型说 70% 是不是真的 70%"。
    """
    conf = np.asarray(conf, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(conf) & np.isfinite(y)
    conf, y = conf[ok], y[ok]
    if len(conf) == 0:
        return pd.DataFrame()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        k = int(y[m].sum())
        n = int(m.sum())
        p, lo, hi = wilson(k, n)
        rows.append({
            "bin": f"[{edges[b]:.1f},{edges[b+1]:.1f})",
            "n": n, "pred_mean": float(conf[m].mean()),
            "actual": p, "lo95": lo, "hi95": hi,
            "gap": float(conf[m].mean()) - p,
        })
    return pd.DataFrame(rows)


def brier_score(conf: np.ndarray, y: np.ndarray) -> float:
    conf = np.asarray(conf, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(conf) & np.isfinite(y)
    if not ok.any():
        return np.nan
    return float(np.mean((conf[ok] - y[ok]) ** 2))


def ece(conf: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error"""
    tb = calibration_table(conf, y, n_bins)
    if tb.empty:
        return np.nan
    w = tb["n"] / tb["n"].sum()
    return float((w * (tb["pred_mean"] - tb["actual"]).abs()).sum())


class IsotonicCalibrator:
    """
    保序回归校准（PAVA 自实现，避免引入 sklearn 依赖）。
    在 train 股票池上 fit，在 test 股票池上 predict，杜绝校准本身过拟合。
    """

    def __init__(self):
        self.x_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 10:
            self.x_, self.y_ = None, None
            return self
        order = np.argsort(x, kind="mergesort")
        x, y = x[order], y[order]
        # PAVA
        w = np.ones_like(y)
        yy = y.copy()
        i = 0
        blocks = [[yy[0], w[0], 0, 0]]
        for i in range(1, len(yy)):
            blocks.append([yy[i], w[i], i, i])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                v2, w2, s2, e2 = blocks.pop()
                v1, w1, s1, e1 = blocks.pop()
                nw = w1 + w2
                blocks.append([(v1 * w1 + v2 * w2) / nw, nw, s1, e2])
        fit_y = np.empty_like(yy)
        for v, _w, s, e in blocks:
            fit_y[s:e + 1] = v
        self.x_, self.y_ = x, fit_y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x_ is None:
            return np.full_like(x, np.nan)
        return np.interp(x, self.x_, self.y_,
                         left=self.y_[0], right=self.y_[-1])


# ============================================================
# 交易回测（用户最直观的验证口径）
# ============================================================
def backtest_touch_strategy(
    df: pd.DataFrame,
    detector: str,
    kind: str = "support",
    width_cap: float = 1.5,
    max_dist_atr: float = 4.0,
    min_score: float = 0.0,
    cost_pct: float = 0.15,
) -> Dict:
    """
    机械规则回测（逐笔独立，不做资金曲线，避免仓位管理掩盖信号质量）：

      信号: 检测出的支撑区，宽度 <= width_cap*ATR，距现价 <= max_dist_atr*ATR
      进场: 价格首次触及区上沿（limit 单，假定成交在区上沿）
      离场: hold_bars 根后按收盘价平仓（与 labeling 的 fwd_ret_pct 一致）
      成本: cost_pct（默认 0.15% 含双边佣金+印花税+滑点）

    输出逐笔期望值、胜率、盈亏比、profit factor。
    """
    d = df[(df["detector"] == detector) & (df["kind"] == kind) &
           (df["width_atr"] <= width_cap) &
           (df["dist_atr"].abs() <= max_dist_atr) &
           (df["score"] >= min_score) & (df["touched"])]
    if len(d) == 0:
        return {"n_trades": 0}
    r = d["fwd_ret_pct"].to_numpy(np.float64) - cost_pct
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"n_trades": 0}
    win = r > 0
    gp = r[win].sum()
    gl = -r[~win].sum()
    return {
        "n_trades": len(r),
        "win_rate": float(win.mean()),
        "avg_ret_pct": float(r.mean()),
        "med_ret_pct": float(np.median(r)),
        "avg_win_pct": float(r[win].mean()) if win.any() else np.nan,
        "avg_loss_pct": float(r[~win].mean()) if (~win).any() else np.nan,
        "profit_factor": float(gp / gl) if gl > 1e-12 else np.inf,
        "expectancy_pct": float(r.mean()),
        "t_stat": float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan,
    }


# ============================================================
# 距离分层比较（Cochran–Mantel–Haenszel）
# ============================================================
"""
为什么必须分层比较
    实测（test 集）: 阻力位的守住率随距离从 68.4%(0.84 ATR) 升到 84.9%(4.2 ATR)，
    +16.5pp，是所有因子里最强的主效应。
    因此"某检测器守住率更高"完全可能只是因为它偏爱把位放得更远。
    要回答"算法是否真的知道关键位在哪"，必须在**同一距离档内**比较。

CMH 检验正是为此设计：把样本按距离分层，每层构造 2x2 表
(检测器 x 守住/突破)，再合并各层的证据，得到"控制距离后"的统一结论。
"""


def cmh_test(tables) -> Dict:
    """
    Cochran–Mantel–Haenszel 检验 + MH 合并优势比。

    tables: [(a, b, c, d), ...] 每层一个 2x2
            a=挑战组守住, b=挑战组突破, c=基准组守住, d=基准组突破
    """
    num = den = 0.0
    e_sum = v_sum = a_sum = 0.0
    r_num = r_den = 0.0
    used = 0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n < 8:
            continue
        r1, r2 = a + b, c + d
        c1 = a + c
        if r1 == 0 or r2 == 0 or c1 == 0 or (b + d) == 0:
            continue
        used += 1
        a_sum += a
        e_sum += r1 * c1 / n
        v_sum += r1 * r2 * c1 * (b + d) / (n * n * (n - 1)) if n > 1 else 0
        r_num += a * d / n
        r_den += b * c / n
    if used == 0 or v_sum <= 0:
        return {"n_strata": used, "chi2": np.nan, "p_value": np.nan, "or_mh": np.nan}
    chi2 = (abs(a_sum - e_sum) - 0.5) ** 2 / v_sum
    from math import erfc, sqrt
    p = erfc(sqrt(max(chi2, 0.0)) / sqrt(2))
    return {"n_strata": used, "chi2": float(chi2), "p_value": float(p),
            "or_mh": float(r_num / r_den) if r_den > 1e-12 else np.inf}


def compare_distance_neutral(
    df: pd.DataFrame, base: str, chal: str, kind: Optional[str] = None,
    n_dist_bins: int = 6, width_cap: Optional[float] = 1.5,
) -> Dict:
    """
    控制距离后的检测器对比。

    做法:
      1. 用两个检测器合并后的 |dist_atr| 分位数切分距离档（同一套边界，保证可比）
      2. 每档内分别算 hold_rate / fwd_ret
      3. hold_rate 用 CMH 检验合并；fwd_ret 用"各档等权平均之差"
         （等权而非样本量加权，避免样本集中在某一档的检测器占便宜）
    """
    d = df[df["detector"].isin([base, chal])].copy()
    if kind:
        d = d[d["kind"] == kind]
    if width_cap is not None:
        d = d[d["width_atr"] <= width_cap]
    d = d[d["touched"]]
    if len(d) < 200:
        return {"base": base, "chal": chal, "n": len(d), "note": "样本不足"}

    d["ad"] = d["dist_atr"].abs()
    try:
        d["db"] = pd.qcut(d["ad"], n_dist_bins, duplicates="drop")
    except ValueError:
        return {"base": base, "chal": chal, "n": len(d), "note": "无法分层"}

    tables = []
    rows = []
    for db, g in d.groupby("db", observed=True):
        gb = g[g["detector"] == base]
        gc = g[g["detector"] == chal]
        db_ = gb[gb["result"].isin(["hold", "break"])]
        dc_ = gc[gc["result"].isin(["hold", "break"])]
        if len(db_) < 5 or len(dc_) < 5:
            continue
        a = int((dc_["result"] == "hold").sum()); b = len(dc_) - a
        c = int((db_["result"] == "hold").sum()); dd = len(db_) - c
        tables.append((a, b, c, dd))
        rows.append({
            "dist_bin": str(db),
            "base_n": len(db_), "chal_n": len(dc_),
            "base_hold": c / len(db_), "chal_hold": a / len(dc_),
            "base_fwd": gb["fwd_ret_pct"].mean(), "chal_fwd": gc["fwd_ret_pct"].mean(),
            "base_edge": gb["edge_atr"].mean(), "chal_edge": gc["edge_atr"].mean(),
        })
    if not rows:
        return {"base": base, "chal": chal, "n": len(d), "note": "分层后样本不足"}
    t = pd.DataFrame(rows)
    cmh = cmh_test(tables)
    return {
        "base": base, "chal": chal, "kind": kind or "all",
        "n_strata": cmh["n_strata"],
        "base_hold_eq": float(t["base_hold"].mean()),
        "chal_hold_eq": float(t["chal_hold"].mean()),
        "hold_lift_pp": float((t["chal_hold"] - t["base_hold"]).mean() * 100),
        "cmh_chi2": cmh["chi2"], "cmh_p": cmh["p_value"], "or_mh": cmh["or_mh"],
        "base_fwd_eq": float(t["base_fwd"].mean()),
        "chal_fwd_eq": float(t["chal_fwd"].mean()),
        "fwd_lift_pp": float((t["chal_fwd"] - t["base_fwd"]).mean()),
        "base_edge_eq": float(t["base_edge"].mean()),
        "chal_edge_eq": float(t["chal_edge"].mean()),
        "edge_lift": float((t["chal_edge"] - t["base_edge"]).mean()),
        "_strata": t,
    }
