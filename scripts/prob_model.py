# -*- coding: utf-8 -*-
"""
有效概率模型：能做到多少区分度？

背景
----
之前的排序分 (log1p(n_events) + 0.8*stale) 故意**不含距离和带宽**，
因为要检验"算法是否知道关键位在哪"，而距离/带宽是对照组也有的信息，
把它们放进去等于自己和自己比。结果排序分的 Brier 只有 0.1828 vs 常数 0.1834。

但"给用户显示有效概率"是完全不同的任务：
用户问的是"价格跌到这里会不会守住"，而不是"算法有没有本事定位"。
这时距离、带宽都是**合法且强力**的预测因子：
    实测 resistance 守住率随距离 68.4%(0.84ATR) -> 84.9%(4.2ATR)，+16.5pp
    实测 support 守住率随带宽 65.8%(0.34ATR) -> 77.9%(1.2ATR)，+10.3pp
所以换成"预测任务"口径后，区分度应该显著高于纯排序分。

本脚本在 train 股票池上拟合若干候选模型，在互斥的 test 股票池上汇报：
    AUC        排序能力（0.5=瞎猜）
    Brier      概率精度（越低越好）
    Brier Skill  相对常数预测的提升 (1 - B/B0)
    P10/P90    预测概率的 10/90 分位 —— 直接反映"用户看到的数字有没有差别"
    校准表      预测 70% 的那批是不是真的 70%
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.srlab.metrics import calibration_table, wilson

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/final"
DET = "V3_fusion"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


# ============================================================
# 逻辑回归（自实现 IRLS，避免引入 sklearn）
# ============================================================
def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3,
              iters: int = 60) -> np.ndarray:
    """带 L2 的 Newton-Raphson 逻辑回归。X 已含截距列。"""
    n, k = X.shape
    w = np.zeros(k)
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        g = X.T @ (y - p) - l2 * w
        H = (X * W[:, None]).T @ X + l2 * np.eye(k)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def predict_logit(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U -> AUC"""
    ok = np.isfinite(p) & np.isfinite(y)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # 处理并列
    df = pd.DataFrame({"p": p, "r": ranks})
    ranks = df.groupby("p")["r"].transform("mean").to_numpy()
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p):
    ok = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[ok] - y[ok]) ** 2))


# ============================================================
# 特征构造
# ============================================================
def build_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    ad = d["dist_atr"].abs()
    f["dist"] = ad
    f["dist_sq"] = ad ** 2
    f["width"] = d["width_atr"]
    f["logev"] = np.log1p(d["m_n_events"].fillna(0))
    f["stale"] = d["m_stale"].fillna(0)
    f["vp"] = d["m_vp"].fillna(0.5)
    f["close_p"] = d["m_close"].fillna(0.5)
    f["atr_pct"] = (d["atr"] / d["price"]).clip(0, 0.2)
    return f


FEATURE_SETS = {
    "M0_常数": [],
    "M1_当前排序分(事件+陈旧)": ["logev", "stale"],
    "M2_只用距离": ["dist", "dist_sq"],
    "M3_距离+带宽": ["dist", "dist_sq", "width"],
    "M4_距离+带宽+事件+陈旧": ["dist", "dist_sq", "width", "logev", "stale"],
    "M5_全部因子": ["dist", "dist_sq", "width", "logev", "stale", "vp",
                    "close_p", "atr_pct"],
}


def load(split: str, kind: str) -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(OUT, f"levels_{split}.parquet"))
    d = d[(d["detector"] == DET) & (d["kind"] == kind) & d["touched"]
          & d["result"].isin(["hold", "break"])].copy()
    d["y"] = (d["result"] == "hold").astype(float)
    return d


def hr(t):
    print("\n" + "=" * 104)
    print("  " + t)
    print("=" * 104)


def main():
    summary_rows = []
    best = {}

    for kind in ("support", "resistance"):
        tr, te = load("train", kind), load("test", kind)
        hr(f"{kind}：train {len(tr):,} / test {len(te):,}  "
           f"(基础守住率 train {tr['y'].mean():.4f} / test {te['y'].mean():.4f})")

        Ftr, Fte = build_features(tr), build_features(te)
        ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
        b0 = brier(yte, np.full(len(yte), ytr.mean()))

        rows = []
        for name, cols in FEATURE_SETS.items():
            if not cols:
                p_te = np.full(len(yte), ytr.mean())
                w = None
            else:
                Xtr = np.column_stack([np.ones(len(Ftr))] + [Ftr[c].to_numpy() for c in cols])
                Xte = np.column_stack([np.ones(len(Fte))] + [Fte[c].to_numpy() for c in cols])
                mu = Xtr[:, 1:].mean(0)
                sd = Xtr[:, 1:].std(0)
                sd[sd < 1e-12] = 1.0
                Xtr[:, 1:] = (Xtr[:, 1:] - mu) / sd
                Xte[:, 1:] = (Xte[:, 1:] - mu) / sd
                w = fit_logit(Xtr, ytr)
                p_te = predict_logit(Xte, w)
            b = brier(yte, p_te)
            rows.append({
                "模型": name,
                "AUC": auc(yte, p_te),
                "Brier": b,
                "BrierSkill": 1 - b / b0,
                "P10": float(np.percentile(p_te, 10)),
                "P50": float(np.percentile(p_te, 50)),
                "P90": float(np.percentile(p_te, 90)),
                "跨度pp": float((np.percentile(p_te, 90) - np.percentile(p_te, 10)) * 100),
            })
            if name == "M5_全部因子":
                best[kind] = (cols, w, mu, sd, p_te, yte)
        t = pd.DataFrame(rows)
        print(t.to_string(index=False))
        for r in rows:
            r["kind"] = kind
            summary_rows.append(r)

    # ---------- 最优模型的校准表 ----------
    for kind in ("support", "resistance"):
        cols, w, mu, sd, p_te, yte = best[kind]
        hr(f"M5 全因子模型在 test 上的校准表 — {kind}")
        tb = calibration_table(p_te, yte, n_bins=10)
        if not tb.empty:
            print(tb.to_string(index=False))
        print(f"\n  预测概率范围: {p_te.min():.3f} ~ {p_te.max():.3f}")
        print(f"  AUC={auc(yte, p_te):.4f}  Brier={brier(yte, p_te):.4f}")

    hr("汇总：各方案能给用户看到多大的概率差异")
    s = pd.DataFrame(summary_rows)
    print(s[["kind", "模型", "AUC", "Brier", "BrierSkill", "P10", "P90", "跨度pp"]].to_string(index=False))
    print("\n  跨度pp = 90分位预测概率 − 10分位预测概率，直接代表'用户能看到的差别'")
    print("  AUC 0.5=瞎猜, 0.6=弱, 0.7=可用, 0.8=强")


if __name__ == "__main__":
    main()
