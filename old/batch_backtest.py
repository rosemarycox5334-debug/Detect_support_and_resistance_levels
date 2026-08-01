#!/usr/bin/env python3
"""
批量回测脚本

支持三种规模：
  --scale small   20只股票（快速测试，~30秒）
  --scale medium  200只（中等验证，~5分钟）
  --scale full    全部A股（约5000只，~30分钟+）

用法:
    python batch_backtest.py --scale small
    python batch_backtest.py --scale medium --window 60 --n-zones 5
    python batch_backtest.py --scale full --start 2022-01-01

特性：
- 进度条实时显示
- 网络重试 + 指数退避
- 本地缓存加速
- 汇总HTML仪表盘
"""

import sys
import os
import argparse
import time
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, DEFAULT_WINDOW, DEFAULT_N_ZONES, BATCH_SMALL, BATCH_MEDIUM, BATCH_FULL

from src.data_fetcher import DataFetcher, bs_code, short_code
from src.backtest_engine import BacktestEngine, run_batch_backtest, aggregate_results
from src.dashboard import generate_batch_dashboard


SCALE_MAP = {
    "small":  BATCH_SMALL,
    "medium": BATCH_MEDIUM,
    "full":   BATCH_FULL,
}


def main():
    parser = argparse.ArgumentParser(description="批量回测 - 成交密集区策略")
    parser.add_argument("--scale", choices=["small", "medium", "full"],
                        default="medium", help="回测规模")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help=f"分析窗口(默认{DEFAULT_WINDOW})")
    parser.add_argument("--n-zones", type=int, default=DEFAULT_N_ZONES, help=f"密集区数量(默认{DEFAULT_N_ZONES})")
    parser.add_argument("--start", default="2022-01-01", help="数据起始日期")
    parser.add_argument("--end", default="2025-12-31", help="数据结束日期")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    args = parser.parse_args()

    limit = SCALE_MAP[args.scale]
    if args.scale == "full":
        confirm = input(f"\n⚠️  将回测全部A股(约5000只)，预计耗时30分钟以上。\n    确认继续？(yes/no): ")
        if confirm.lower() != "yes":
            print("已取消")
            return

    print(f"\n{'='*60}")
    print(f"  批量回测: {args.scale.upper()} ({limit if limit > 0 else '全部'} 只股票)")
    print(f"  窗口: {args.window}天 | 密集区: {args.n_zones}个")
    print(f"  区间: {args.start} ~ {args.end}")
    print(f"{'='*60}\n")

    if args.no_cache:
        from config import CACHE_ENABLED
        import config
        config.CACHE_ENABLED = False

    # ---- 获取股票列表 ----
    fetcher = DataFetcher()
    try:
        print("[1/4] 获取股票列表...")
        codes = fetcher.get_stock_codes(limit=limit)
        print(f"  ✓ 共 {len(codes)} 只股票\n")

        # ---- 批量获取K线 ----
        print("[2/4] 批量获取K线数据...")
        failed = 0
        data_dict = {}
        with tqdm(total=len(codes), desc="下载K线", unit="股") as pbar:
            for code in codes:
                try:
                    df = fetcher.get_kline(code, freq="d", start=args.start, end=args.end)
                    if len(df) >= args.window + 10:
                        data_dict[code] = df
                except Exception:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix({"成功": len(data_dict), "失败": failed})

        print(f"  ✓ 成功 {len(data_dict)} 只, 失败 {failed} 只\n")

        if len(data_dict) == 0:
            print("  ❌ 无有效数据，退出")
            return

        # ---- 回测 ----
        print("[3/4] 运行回测...")
        engine = BacktestEngine(window=args.window, n_zones=args.n_zones)
        results = []

        with tqdm(total=len(data_dict), desc="回测进度", unit="股") as pbar:
            for i, (code, df) in enumerate(data_dict.items()):
                try:
                    r = engine.run(df)
                    r["code"] = code
                    results.append(r)
                except Exception as e:
                    results.append({"code": code, "error": str(e), "metrics": {}, "trades": []})
                pbar.update(1)

        # ---- 汇总 ----
        print("\n[4/4] 汇总结果...")
        agg = aggregate_results(results)
        print(f"  ✓ 有效结果: {agg['valid_stocks']} / {agg['total_stocks']}")
        print(f"  ✓ 平均收益: {agg['avg_return_pct']:.2f}%")
        print(f"  ✓ 中位收益: {agg['median_return_pct']:.2f}%")
        print(f"  ✓ 盈利比: {agg['positive_ratio_pct']:.1f}%")
        print(f"  ✓ 平均夏普: {agg['avg_sharpe']:.3f}")
        print(f"  ✓ 最佳: {agg['best_stock']} ({agg['best_return_pct']:.2f}%)")
        print(f"  ✓ 最差: {agg['worst_stock']} ({agg['worst_return_pct']:.2f}%)")

        # ---- 保存结果 ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(OUTPUT_DIR, f"batch_{args.scale}_{timestamp}.json")

        # 简化结果用于JSON
        simplified = []
        for r in results:
            m = r.get("metrics", {})
            simplified.append({
                "code": r.get("code", ""),
                "total_return_pct": m.get("total_return_pct", 0),
                "sharpe_ratio": m.get("sharpe_ratio", 0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                "win_rate_pct": m.get("win_rate_pct", 0),
                "total_trades": m.get("total_trades", 0),
                "profit_factor": str(m.get("profit_factor", 0)),
            })

        import json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {
                    "scale": args.scale,
                    "window": args.window,
                    "n_zones": args.n_zones,
                    "start": args.start,
                    "end": args.end,
                },
                "aggregate": agg,
                "results": simplified,
            }, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 结果保存: {json_path}")

        # ---- 仪表盘 ----
        print(f"\n  生成汇总仪表盘...")
        batch_result = {"aggregate": agg, "results": simplified}
        dash_path = generate_batch_dashboard(
            batch_result,
            batch_label=f"{args.scale.upper()}规模回测 ({args.window}天窗口)",
        )
        print(f"  ✓ 仪表盘: {dash_path}")

        print(f"\n{'='*60}")
        print(f"  回测完成！")
        print(f"  结果: {json_path}")
        print(f"  图表: {dash_path}")
        print(f"{'='*60}\n")

    finally:
        fetcher.logout()


if __name__ == "__main__":
    main()
