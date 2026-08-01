#!/usr/bin/env python3
"""
单只股票密集区分析

用法:
    python analyze_single.py 600519                   # 贵州茅台，默认参数
    python analyze_single.py 600519 --name 茅台         # 自定义名称
    python analyze_single.py 600519 --window 60        # 自定义窗口
    python analyze_single.py 600519 --n-zones 5        # 检测5个密集区
    python analyze_single.py 600519 --backtest         # 同时运行回测
    python analyze_single.py 600519 --mtf              # 多时间框架分析
"""

import sys
import os
import argparse

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, DEFAULT_WINDOW, DEFAULT_N_ZONES

from src.data_fetcher import DataFetcher, bs_code, short_code
from src.density_analyzer import (
    find_dense_zones, classify_zones, multi_timeframe_zones, calc_risk_reward,
)
from src.dashboard import (
    generate_single_stock_dashboard, generate_mtf_dashboard,
)
from src.backtest_engine import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="单股成交密集区分析")
    parser.add_argument("code", help="股票代码，如 600519 / sh.600519")
    parser.add_argument("--name", default="", help="股票名称（可选）")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help=f"分析窗口(交易日，默认{DEFAULT_WINDOW})")
    parser.add_argument("--n-zones", type=int, default=DEFAULT_N_ZONES, help=f"检测密集区数量(默认{DEFAULT_N_ZONES})")
    parser.add_argument("--start", default="2023-01-01", help="数据起始日期")
    parser.add_argument("--end", default="2025-12-31", help="数据结束日期")
    parser.add_argument("--backtest", action="store_true", help="同时运行回测")
    parser.add_argument("--mtf", action="store_true", help="多时间框架分析")
    args = parser.parse_args()

    bs = bs_code(args.code)
    short = short_code(bs)
    print(f"\n{'='*60}")
    print(f"  成交密集区分析: {short}")
    print(f"{'='*60}\n")

    # ---- 获取数据 ----
    fetcher = DataFetcher()
    try:
        print("[1/4] 获取日线数据...")
        df_day = fetcher.get_kline(bs, freq="d", start=args.start, end=args.end)
        if len(df_day) < args.window:
            print(f"  ❌ 数据不足: 仅 {len(df_day)} 条，需要 >= {args.window}")
            return

        stock_name = args.name or short
        if not args.name:
            sl = fetcher.get_stock_list()
            match = sl[sl["code"] == bs]
            if len(match) > 0:
                stock_name = match.iloc[0]["code_name"]

        current_price = float(df_day["close"].iloc[-1])
        print(f"  ✓ {stock_name} 共 {len(df_day)} 条日线, 最新价: {current_price:.2f}")

        # ---- 密集区分析 ----
        print(f"\n[2/4] 检测成交密集区 (窗口={args.window}天)...")
        zones = find_dense_zones(df_day, window=args.window, n_zones=args.n_zones)
        classify_zones(zones, current_price)

        print(f"\n  {'─'*50}")
        print(f"  {'类型':<8} {'中心价':>10} {'区间':>18} {'强度':>6} {'距离%':>8} {'重叠数':>6}")
        print(f"  {'─'*50}")
        for z in zones:
            tag = "🔴压" if z.zone_type == "resistance" else "🟢支"
            print(f"  {tag:<8} {z.center:>10.2f} {f'{z.low:.2f}-{z.high:.2f}':>18} {z.strength:>6.3f} {z.distance_pct:>7.2f}% {z.touch_count:>5}")
        print(f"  {'─'*50}")

        # 盈亏比
        rr = calc_risk_reward(current_price, zones)
        print(f"\n  📊 盈亏比分析: {rr['risk_reward_ratio']} ({rr['quality']})")
        print(f"     潜在盈利: {rr['potential_profit_pct']}% | 潜在亏损: {rr['potential_loss_pct']}%")

        # ---- 回测（可选） ----
        backtest_result = None
        if args.backtest:
            print(f"\n[3/4] 运行回测...")
            engine = BacktestEngine(window=args.window, n_zones=args.n_zones)
            backtest_result = engine.run(df_day)
            m = backtest_result["metrics"]
            print(f"  ✓ 交易 {m.get('total_trades', 0)} 笔")
            print(f"  ✓ 总收益: {m.get('total_return_pct', 0):.2f}% | 年化: {m.get('annual_return_pct', 0):.2f}%")
            print(f"  ✓ 夏普: {m.get('sharpe_ratio', 0):.3f} | 最大回撤: {m.get('max_drawdown_pct', 0):.2f}%")
            print(f"  ✓ 胜率: {m.get('win_rate_pct', 0):.1f}% | 盈亏比: {m.get('profit_factor', 0)}")

        # ---- 生成仪表盘 ----
        print(f"\n[{'4' if args.backtest else '3'}/4] 生成交互式仪表盘...")
        dash_path = generate_single_stock_dashboard(
            short, stock_name, df_day, zones, backtest_result
        )
        print(f"  ✓ 仪表盘: {dash_path}")

        # 多时间框架
        if args.mtf:
            print(f"\n  多时间框架分析...")
            df_week = fetcher.get_kline(bs, freq="w", start=args.start, end=args.end)
            df_month = fetcher.get_kline(bs, freq="m", start=args.start, end=args.end)
            mtf = multi_timeframe_zones(df_day, df_week, df_month, n_zones=args.n_zones)
            mtf_path = generate_mtf_dashboard(short, stock_name, df_day, mtf)
            print(f"  ✓ 多时间框架仪表盘: {mtf_path}")

        print(f"\n{'='*60}")
        print(f"  分析完成！请在浏览器中打开 HTML 文件查看交互式图表")
        print(f"{'='*60}\n")

    finally:
        fetcher.logout()


if __name__ == "__main__":
    main()
