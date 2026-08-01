"""
市场扫描器

扫描A股全市场，找出当前处于关键密集区附近的股票。
输出按盈亏比、区域强度等维度排序的股票列表。

扫描模式:
- near_support:  价格在支撑区附近（潜在买入机会）
- near_resistance: 价格在压力区附近（潜在卖出/观望）
- best_rr:       盈亏比最优
"""

import sys, os
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_WINDOW, DEFAULT_N_ZONES, PRICE_THRESHOLD_PCT, DATA_DIR
from src.data_fetcher import DataFetcher, bs_code, short_code
from src.density_analyzer import find_dense_zones, classify_zones, calc_risk_reward


def scan_market(
    top_n: int = 100,
    window: int = DEFAULT_WINDOW,
    n_zones: int = DEFAULT_N_ZONES,
    mode: str = "all",
    price_threshold: float = PRICE_THRESHOLD_PCT,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, List[Dict]]:
    """
    扫描全市场

    参数:
        top_n:     扫描股票数量（按代码排序取前N只，保证可复现）
        window:    分析窗口
        n_zones:   密集区数量
        mode:      'near_support' / 'near_resistance' / 'best_rr' / 'all'
        price_threshold: 接近区域的距离阈值（%）
        progress_callback: 进度回调 (current, total)

    返回:
        {
            'near_support': [{code, name, price, zones, rr, ...}, ...],
            'near_resistance': [...],
            'best_rr': [...],
        }
    """
    fetcher = DataFetcher()
    try:
        codes = fetcher.get_stock_codes(limit=top_n, exclude_st=True)

        results = {
            "near_support": [],
            "near_resistance": [],
            "best_rr": [],
            "all": [],
        }

        total = len(codes)
        for i, code in enumerate(codes):
            try:
                df = fetcher.get_kline(code, freq="d", start="2023-01-01", end="2025-12-31")
                if len(df) < window + 5:
                    continue

                # 股票名称
                short = short_code(code)
                name = short  # 先用代码，后面可以优化

                current_price = float(df["close"].iloc[-1])
                zones = find_dense_zones(df, window=window, n_zones=n_zones)
                classify_zones(zones, current_price)
                rr = calc_risk_reward(current_price, zones)

                # 找最近的支撑和压力
                supports = [z for z in zones if z.zone_type == "support"]
                resistances = [z for z in zones if z.zone_type == "resistance"]

                nearest_support = min(supports, key=lambda z: abs(z.distance_pct)) if supports else None
                nearest_resistance = min(resistances, key=lambda z: abs(z.distance_pct)) if resistances else None

                item = {
                    "code": short,
                    "code_full": code,
                    "price": round(current_price, 2),
                    "n_zones": len(zones),
                    "rr_ratio": rr["risk_reward_ratio"],
                    "rr_quality": rr["quality"],
                    "nearest_support": nearest_support.center if nearest_support else None,
                    "support_dist_pct": round(abs(nearest_support.distance_pct), 2) if nearest_support else None,
                    "support_strength": round(nearest_support.strength, 2) if nearest_support else 0,
                    "nearest_resistance": nearest_resistance.center if nearest_resistance else None,
                    "resistance_dist_pct": round(abs(nearest_resistance.distance_pct), 2) if nearest_resistance else None,
                    "resistance_strength": round(nearest_resistance.strength, 2) if nearest_resistance else 0,
                }

                results["all"].append(item)

                # 支撑附近
                if nearest_support and abs(nearest_support.distance_pct) <= price_threshold:
                    results["near_support"].append(item)

                # 压力附近
                if nearest_resistance and abs(nearest_resistance.distance_pct) <= price_threshold:
                    results["near_resistance"].append(item)

                # 盈亏比最优
                if rr["risk_reward_ratio"] >= 2.0:
                    results["best_rr"].append(item)

            except Exception:
                pass

            if progress_callback:
                progress_callback(i + 1, total)

        # 排序
        results["near_support"] = sorted(results["near_support"],
            key=lambda x: (x["support_dist_pct"] or 99))
        results["near_resistance"] = sorted(results["near_resistance"],
            key=lambda x: (x["resistance_dist_pct"] or 99))
        results["best_rr"] = sorted(results["best_rr"],
            key=lambda x: -x["rr_ratio"])

        return results

    finally:
        fetcher.logout()


def scan_summary(results: Dict) -> Dict:
    """生成扫描摘要统计"""
    return {
        "total_scanned": len(results.get("all", [])),
        "near_support_count": len(results.get("near_support", [])),
        "near_resistance_count": len(results.get("near_resistance", [])),
        "best_rr_count": len(results.get("best_rr", [])),
        "avg_rr": round(np.mean([r["rr_ratio"] for r in results.get("all", [])]), 2) if results.get("all") else 0,
        "support_ratio_pct": round(len(results.get("near_support", [])) / max(len(results.get("all", [])), 1) * 100, 1),
        "resistance_ratio_pct": round(len(results.get("near_resistance", [])) / max(len(results.get("all", [])), 1) * 100, 1),
    }
