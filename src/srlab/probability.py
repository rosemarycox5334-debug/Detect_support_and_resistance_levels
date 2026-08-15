# -*- coding: utf-8 -*-
"""
双概率模型：触及概率 + 守住概率

为什么是两个概率
----------------
用户问"这个关键位有效概率多少"，其实包含两个独立的问题：

    ① 触及概率  未来 horizon 根内，价格会不会走到这个位？
    ② 守住概率  走到之后，会不会反弹（而不是跌穿）？

实测（out/final，130 只训练股票 / 130 只互斥测试股票）两者可预测性差别很大：

    维度        AUC(support/resistance)  预测范围        P10~P90 跨度   校准误差
    触及概率    0.770 / 0.732            6.5% ~ 95.3%   61.1 / 52.1pp  —
    守住概率    0.587 / 0.612            53.8% ~ 85.3%  15.2 / 17.9pp  < 2pp

结论：
  - 触及概率强（AUC 0.77），主要由距离驱动，跨度大，用户能真正看出差别
  - 守住概率弱（AUC 0.59），但**校准很好**（模型说 75.1% 实际 75.4%），
    数字本身不骗人，只是不适合用来比较两个相邻的位
  - 一个"守住概率 85% 但触及概率 8%"的位实际用不上，所以两个都要显示

为什么这里可以用距离和带宽（而排序分不能）
------------------------------------------
排序分（V3Params.w_events / w_stale）故意排除距离和带宽，因为那是在检验
"算法有没有本事定位关键位" —— 距离和带宽是随机平移对照组也具备的信息，
放进去等于自己和自己比。
但"给用户预测某个位会不会有效"是**另一个任务**：这时距离和带宽是完全合法
且强力的预测因子（实测 resistance 守住率随距离 68.4%→84.9%，+16.5pp）。
两个口径不冲突，服务的问题不同。

模型
----
逻辑回归（自实现 Newton-Raphson + L2，不引入 sklearn 依赖）。
特征全部为 t 时刻可得的因果量，训练在 train 股票池、验收在互斥的 test 股票池。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MODEL_PATH_DEFAULT = "data/prob_models.json"

# 特征列表（顺序即模型系数顺序，改动必须重训）
FEATURES: Tuple[str, ...] = (
    "dist",      # |距现价| / ATR —— 触及概率的主驱动
    "dist_sq",   # 距离平方，捕捉非线性
    "width",     # 带宽 / ATR
    "logev",     # log1p(历史触及事件数)
    "stale",     # 陈旧度 = 1 - 近期新鲜度
    "vp",        # 成交量剖面强度（只进概率模型，不进排序分）
    "close_p",   # 收盘价堆积强度
    "atr_pct",   # ATR / 价格，波动率水平
)

TARGETS = ("touch", "hold")
KINDS = ("support", "resistance")


# ============================================================
# 逻辑回归
# ============================================================
def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3,
              iters: int = 80) -> np.ndarray:
    """带 L2 正则的 Newton-Raphson 逻辑回归。X 第 0 列须为全 1 截距。"""
    n, k = X.shape
    w = np.zeros(k, dtype=np.float64)
    for _ in range(iters):
        p = _sigmoid(X @ w)
        W = np.clip(p * (1 - p), 1e-6, None)
        g = X.T @ (y - p) - l2 * w
        H = (X * W[:, None]).T @ X + l2 * np.eye(k)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U 法算 AUC，正确处理并列"""
    ok = np.isfinite(p) & np.isfinite(y)
    y, p = np.asarray(y)[ok], np.asarray(p)[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    ranks = pd.Series(ranks).groupby(pd.Series(p)).transform("mean").to_numpy()
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(p) & np.isfinite(y)
    if not ok.any():
        return float("nan")
    return float(np.mean((np.asarray(p)[ok] - np.asarray(y)[ok]) ** 2))


# ============================================================
# 特征构造
# ============================================================
def features_from_frame(d: pd.DataFrame) -> pd.DataFrame:
    """从走查回测落盘的 levels_*.parquet 构造特征"""
    f = pd.DataFrame(index=d.index)
    ad = d["dist_atr"].abs()
    f["dist"] = ad
    f["dist_sq"] = ad ** 2
    f["width"] = d["width_atr"]
    f["logev"] = np.log1p(d.get("m_n_events", pd.Series(0, index=d.index)).fillna(0))
    f["stale"] = d.get("m_stale", pd.Series(0.0, index=d.index)).fillna(0.0)
    f["vp"] = d.get("m_vp", pd.Series(0.5, index=d.index)).fillna(0.5)
    f["close_p"] = d.get("m_close", pd.Series(0.5, index=d.index)).fillna(0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["atr_pct"] = (d["atr"] / d["price"]).clip(0, 0.2).fillna(0.02)
    return f[list(FEATURES)]


def features_from_level(dist_atr: float, width_atr: float, n_events: float,
                        stale: float, vp: float, close_p: float,
                        atr: float, price: float) -> np.ndarray:
    """线上单个关键位的特征向量（顺序必须与 FEATURES 一致）"""
    ad = abs(float(dist_atr))
    atr_pct = 0.02
    if price and np.isfinite(price) and price > 0:
        atr_pct = float(np.clip(atr / price, 0.0, 0.2))
    return np.array([
        ad,
        ad * ad,
        float(width_atr),
        float(np.log1p(max(n_events, 0.0))),
        float(stale),
        float(vp),
        float(close_p),
        atr_pct,
    ], dtype=np.float64)


# ============================================================
# 模型容器
# ============================================================
@dataclass
class LogitModel:
    """标准化 + 逻辑回归系数。predict 输入原始特征矩阵。"""
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray          # 含截距，长度 = len(FEATURES) + 1
    base_rate: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def predict(self, F: np.ndarray) -> np.ndarray:
        F = np.atleast_2d(np.asarray(F, dtype=np.float64))
        Z = (F - self.mean) / self.std
        X = np.column_stack([np.ones(len(Z)), Z])
        return _sigmoid(X @ self.coef)

    def predict_one(self, f: np.ndarray) -> float:
        return float(self.predict(f.reshape(1, -1))[0])

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "base_rate": self.base_rate,
            "metrics": self.metrics,
        }

    @staticmethod
    def from_dict(d: dict) -> "LogitModel":
        return LogitModel(
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            coef=np.asarray(d["coef"], dtype=np.float64),
            base_rate=float(d.get("base_rate", 0.75)),
            metrics=d.get("metrics", {}),
        )


def train_model(F: pd.DataFrame, y: np.ndarray, l2: float = 1e-3) -> LogitModel:
    Fv = F.to_numpy(dtype=np.float64)
    mean = Fv.mean(axis=0)
    std = Fv.std(axis=0)
    std[std < 1e-12] = 1.0
    Z = (Fv - mean) / std
    X = np.column_stack([np.ones(len(Z)), Z])
    coef = fit_logit(X, np.asarray(y, dtype=np.float64), l2=l2)
    return LogitModel(mean=mean, std=std, coef=coef,
                      base_rate=float(np.mean(y)))


# ============================================================
# 模型集合（touch/hold × support/resistance）
# ============================================================
class ProbabilityModels:
    """
    线上使用的模型集合。缺文件时 ok=False，调用方应优雅降级（不显示概率）。
    """

    def __init__(self, path: str = MODEL_PATH_DEFAULT):
        self.path = path
        self.ok = False
        self.meta: dict = {}
        self._m: Dict[Tuple[str, str], LogitModel] = {}
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                js = json.load(fh)
            if js.get("features") != list(FEATURES):
                # 特征定义变了，旧模型不能用（系数顺序会错位）
                return
            self.meta = js.get("meta", {})
            for tgt in TARGETS:
                for kind in KINDS:
                    d = (js.get("models", {}).get(tgt, {}) or {}).get(kind)
                    if d:
                        self._m[(tgt, kind)] = LogitModel.from_dict(d)
            self.ok = len(self._m) > 0
        except Exception:
            self.ok = False

    def get(self, target: str, kind: str) -> Optional[LogitModel]:
        return self._m.get((target, kind))

    def predict(self, target: str, kind: str, f: np.ndarray) -> Optional[float]:
        m = self.get(target, kind)
        if m is None:
            return None
        p = m.predict_one(f)
        return p if np.isfinite(p) else None

    def metrics(self, target: str, kind: str) -> dict:
        m = self.get(target, kind)
        return dict(m.metrics) if m else {}


def save_models(models: Dict[str, Dict[str, LogitModel]], meta: dict,
                path: str = MODEL_PATH_DEFAULT) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {
        "features": list(FEATURES),
        "meta": meta,
        "models": {tgt: {k: m.to_dict() for k, m in by_kind.items()}
                   for tgt, by_kind in models.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
