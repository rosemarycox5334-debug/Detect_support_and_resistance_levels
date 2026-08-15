# -*- coding: utf-8 -*-
"""
构建复权因子缓存

从 akshare（东方财富）拉取 不复权/后复权 收盘价，相除得到精确复权因子，
去量化噪声后落盘到 data/adj_factors.parquet。

用法:
    python scripts/build_factors.py                # 前 400 只（覆盖评测标的池）
    python scripts/build_factors.py --n 1000
    python scripts/build_factors.py --all          # 全部 4272 只（约 1.5 小时）
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.srlab.adjust import FACTOR_CACHE_DEFAULT, build_factor_cache
from src.srlab.data import ashare_list_codes

ASHARE_DIR = r"D:\A股_K线数据\parquet\stocks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=FACTOR_CACHE_DEFAULT)
    ap.add_argument("--source", default="baostock",
                    choices=["baostock", "akshare", "eastmoney"])
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--slice", default="",
                    help="并行分片，格式 i/n，如 0/6。baostock 单会话较慢，"
                         "多进程分片可线性加速；各分片写独立缓存文件，用 --merge 合并")
    ap.add_argument("--merge", action="store_true",
                    help="把 data/adj_factors_part*.parquet 合并进主缓存")
    args = ap.parse_args()

    if args.merge:
        import glob

        import pandas as pd
        parts = []
        if os.path.exists(args.out):
            parts.append(pd.read_parquet(args.out))
        for p in sorted(glob.glob("data/adj_factors_part*.parquet")):
            parts.append(pd.read_parquet(p))
            print(f"  合并 {p}")
        if not parts:
            print("无可合并文件")
            return
        m = pd.concat(parts, ignore_index=True)
        m = m.drop_duplicates(subset=["code", "date"], keep="last")
        m.to_parquet(args.out, index=False)
        print(f"合并完成 -> {args.out}: {m['code'].nunique()} 只, {len(m):,} 行")
        return

    codes = ashare_list_codes(ASHARE_DIR, "daily")
    if not args.all:
        codes = codes[:args.n]

    out_path = args.out
    if args.slice:
        i, n = (int(x) for x in args.slice.split("/"))
        codes = codes[i::n]
        out_path = f"data/adj_factors_part{i}.parquet"
        print(f"分片 {i}/{n}: {len(codes)} 只 -> {out_path}")

    args.out = out_path
    print(f"目标 {len(codes)} 只，源={args.source}，缓存文件 {args.out}")

    if args.source == "baostock":
        import baostock as bs
        lg = bs.login()
        print(f"baostock 登录: {lg.error_code} {lg.error_msg}")
        if lg.error_code != "0":
            sys.exit(1)
        try:
            build_factor_cache(codes, cache_path=args.out,
                               source=args.source, sleep=args.sleep)
        finally:
            bs.logout()
    else:
        build_factor_cache(codes, cache_path=args.out,
                           source=args.source, sleep=args.sleep)


if __name__ == "__main__":
    main()
