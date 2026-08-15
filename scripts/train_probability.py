# -*- coding: utf-8 -*-
"""
训练双概率模型（触及概率 + 守住概率）→ data/prob_models.json

用法:
    python scripts/train_probability.py                # 读 out/final
    python scripts/train_probability.py out/prob_exp
    python scripts/train_probability.py out/prob_exp --from 2019-01-01
        # 只用该日期及之后的样本拟合（线上默认；实测对近期校准更好）

纪律：
    - 系数只在 **train 股票池**上拟合
    - AUC / Brier / 校准表只在**互斥的 test 股票池**上汇报
    - 两个池按股票代码 md5 分桶，不是按时间切分，所以不存在同一只股票既训练又汇报
    - --from 只裁剪训练/验收的时间窗，不改变股票互斥

标签定义（与走查回测完全一致，见 srlab/labeling.py）：
  触及  未来 40 根内价格是否从正确方向走到该区间
  守住  触及后 10 根内，先到 上沿+1.0ATR 记 hold；先到 收盘<下沿-0.5ATR 记 break
        （两者都没发生的"未决"样本不进守住模型的训练/评估）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.srlab.metrics import calibration_table
from src.srlab.probability import (
    FEATURES, KINDS, MODEL_PATH_DEFAULT, TARGETS, auc_score, brier_score,
    features_from_frame, save_models, train_model,
)

DET = "V3_fusion"
pd.set_option("display.width", 210)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def hr(t: str):
    print("\n" + "=" * 100)
    print("  " + t)
    print("=" * 100)


def load_split(out_dir: str, split: str, kind: str,
               date_from: str | None = None) -> pd.DataFrame:
    p = os.path.join(out_dir, f"levels_{split}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} 不存在。先跑 python scripts/run_eval.py --n 260 --out {out_dir}"
            f" 或 python scripts/prob_scale_period_exp.py")
    d = pd.read_parquet(p)
    d = d[(d["detector"] == DET) & (d["kind"] == kind)].copy()
    if date_from:
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] >= date_from]
    return d


def prep(d: pd.DataFrame, target: str):
    """返回 (特征, 标签)。touch 用全部样本；hold 只用已判定的触及样本。"""
    if target == "touch":
        sub = d
        y = d["touched"].astype(float).to_numpy()
    else:
        sub = d[d["touched"] & d["result"].isin(["hold", "break"])]
        y = (sub["result"] == "hold").astype(float).to_numpy()
    return features_from_frame(sub), y, sub


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="out/final")
    ap.add_argument("--from", dest="date_from", default=None,
                    help="只使用该日期及之后的样本（如 2019-01-01）；"
                         "实测对近期绝对概率校准更好，排序能力几乎不变")
    args = ap.parse_args()
    out_dir = args.out_dir
    date_from = args.date_from

    print(f"数据来源: {out_dir}  检测器: {DET}"
          + (f"  时间窗: >={date_from}" if date_from else "  时间窗: 全部"))
    print(f"特征: {list(FEATURES)}")

    models = {t: {} for t in TARGETS}
    report_rows = []

    for target in TARGETS:
        label = "触及概率" if target == "touch" else "守住概率"
        for kind in KINDS:
            tr = load_split(out_dir, "train", kind, date_from)
            te = load_split(out_dir, "test", kind, date_from)
            Ftr, ytr, _ = prep(tr, target)
            Fte, yte, _ = prep(te, target)
            if len(Ftr) < 500 or len(Fte) < 500:
                print(f"[skip] {target}/{kind} 样本不足 "
                      f"(train={len(Ftr)}, test={len(Fte)})")
                continue

            m = train_model(Ftr, ytr)
            p_te = m.predict(Fte.to_numpy())
            b = brier_score(yte, p_te)
            b0 = brier_score(yte, np.full(len(yte), ytr.mean()))
            a = auc_score(yte, p_te)
            m.metrics = {
                "auc_test": round(float(a), 4),
                "brier_test": round(float(b), 5),
                "brier_baseline": round(float(b0), 5),
                "brier_skill": round(float(1 - b / b0), 4) if b0 > 0 else 0.0,
                "n_train": int(len(Ftr)),
                "n_test": int(len(Fte)),
                "base_rate_test": round(float(np.mean(yte)), 4),
                "p10": round(float(np.percentile(p_te, 10)), 4),
                "p50": round(float(np.percentile(p_te, 50)), 4),
                "p90": round(float(np.percentile(p_te, 90)), 4),
                "p_min": round(float(p_te.min()), 4),
                "p_max": round(float(p_te.max()), 4),
            }
            models[target][kind] = m

            report_rows.append({
                "目标": label, "方向": kind,
                "n_train": len(Ftr), "n_test": len(Fte),
                "基础率": np.mean(yte), "AUC": a,
                "Brier": b, "BrierSkill": 1 - b / b0 if b0 > 0 else 0,
                "P10": m.metrics["p10"], "P50": m.metrics["p50"],
                "P90": m.metrics["p90"],
                "跨度pp": (m.metrics["p90"] - m.metrics["p10"]) * 100,
            })

            hr(f"{label} · {kind} — test 集校准表")
            tb = calibration_table(p_te, yte, n_bins=10)
            if not tb.empty:
                print(tb.to_string(index=False))
            print(f"\n  AUC={a:.4f}  Brier={b:.5f} (基准 {b0:.5f}, "
                  f"skill {1-b/b0:+.2%})  预测范围 {p_te.min():.3f}~{p_te.max():.3f}")
            # 系数方向（标准化后，可直接比较重要性）
            coefs = pd.Series(m.coef[1:], index=list(FEATURES)).sort_values(
                key=np.abs, ascending=False)
            print("  标准化系数（按绝对值排序）:")
            print("    " + "  ".join(f"{k}={v:+.3f}" for k, v in coefs.items()))

    if not any(models[t] for t in TARGETS):
        print("\n没有训练出任何模型，退出")
        return 1

    meta = {
        "source": out_dir,
        "detector": DET,
        "date_from": date_from,
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "note": ("系数在 train 股票池拟合，AUC/Brier/校准在互斥的 test 股票池汇报；"
                 "两池按代码 md5 分桶互斥"
                 + (f"；训练/验收时间窗 >={date_from}" if date_from else "")),
    }
    save_models(models, meta, MODEL_PATH_DEFAULT)
    print(f"\n已写出 {MODEL_PATH_DEFAULT}")

    hr("汇总")
    t = pd.DataFrame(report_rows)
    print(t.to_string(index=False))
    print("\n  跨度pp = P90 - P10，代表用户实际能看到的概率差异")
    print("  AUC: 0.5=瞎猜  0.6=弱  0.7=可用  0.8=强")
    return 0


if __name__ == "__main__":
    sys.exit(main())
