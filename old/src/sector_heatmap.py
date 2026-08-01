"""
板块热力图模块

按申万行业分类聚合分析全市场股票的密集区状态，
生成板块级别的支撑/压力分布热力图数据。
"""

import sys, os
from typing import Dict, List, Optional, Callable
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_WINDOW, DEFAULT_N_ZONES, PRICE_THRESHOLD_PCT
from src.data_fetcher import DataFetcher, bs_code, short_code
from src.density_analyzer import find_dense_zones, classify_zones


# 申万一级行业映射（基于常见股票代码前缀快速推断）
# 完整版需通过 baostock query_stock_industry() 获取
CODE_INDUSTRY_MAP = {
    "银行": ["sh.600000","sh.600015","sh.600016","sh.600036","sh.601009","sh.601166","sh.601169",
             "sh.601229","sh.601288","sh.601328","sh.601398","sh.601818","sh.601939","sh.601988",
             "sz.000001","sz.002142","sz.002807","sz.002839","sz.002936","sz.002948","sz.002958"],
    "食品饮料": ["sh.600519","sh.600809","sh.600887","sh.603288","sz.000568","sz.000858","sz.000860",
                "sz.002304","sz.002568","sz.000895"],
    "医药生物": ["sh.600276","sh.603259","sz.000538","sz.002001","sz.002007","sz.300015","sz.300122",
                "sz.300347","sz.300529","sz.300601","sz.300760"],
    "电子": ["sh.603501","sz.000725","sz.002049","sz.002241","sz.002371","sz.002415","sz.002475",
            "sz.300433","sz.300661","sz.300782"],
    "电力设备": ["sh.600406","sh.601012","sh.601615","sz.002129","sz.002202","sz.300274","sz.300750",
                "sz.300763"],
    "汽车": ["sh.600104","sh.601238","sh.601633","sz.000625","sz.002594","sz.300124"],
    "非银金融": ["sh.600030","sh.600837","sh.601066","sh.601211","sh.601318","sh.601336","sh.601601",
                "sh.601628","sh.601688","sz.000776","sz.002736","sz.300059"],
    "房地产": ["sh.600048","sh.600325","sh.600383","sh.600606","sh.601155","sz.000002","sz.000069",
              "sz.001979"],
    "计算机": ["sh.600570","sh.600588","sz.000066","sz.000938","sz.002230","sz.002236","sz.002405",
              "sz.002410","sz.300033","sz.300454"],
    "有色金属": ["sh.600111","sh.600362","sh.600489","sh.600547","sh.601600","sh.601899","sh.603799",
               "sz.000630","sz.000831","sz.002460","sz.300618"],
    "基础化工": ["sh.600309","sh.600346","sh.600352","sh.600426","sh.600989","sz.000408","sz.000830",
               "sz.002064","sz.002601"],
    "国防军工": ["sh.600118","sh.600150","sh.600685","sh.600760","sh.600893","sz.000547","sz.000768",
               "sz.002013","sz.002179"],
    "通信": ["sh.600050","sh.600105","sh.600487","sh.601138","sh.601728","sz.000063","sz.002281",
            "sz.002396","sz.300308","sz.300502"],
    "建筑装饰": ["sh.600585","sh.601186","sh.601390","sh.601668","sh.601669","sh.601800","sz.002051",
               "sz.002081","sz.002310"],
    "交通运输": ["sh.600004","sh.600009","sh.600029","sh.600115","sh.601006","sh.601111","sz.000089",
               "sz.002352"],
    "家用电器": ["sh.600690","sz.000333","sz.000651","sz.002032","sz.002050"],
    "煤炭": ["sh.600188","sh.600348","sh.600985","sh.601088","sh.601225","sh.601699"],
    "钢铁": ["sh.600010","sh.600019","sh.600282","sh.600808","sz.000709","sz.000825","sz.000932"],
    "石油石化": ["sh.600028","sh.600256","sh.600346","sh.600688","sh.601857"],
}


def build_industry_map(fetcher: DataFetcher = None) -> Dict[str, str]:
    """
    构建股票代码→行业映射
    优先使用预定义映射，也可从 baostock 查询
    """
    code_to_industry = {}
    for ind, codes in CODE_INDUSTRY_MAP.items():
        for c in codes:
            code_to_industry[c] = ind
    return code_to_industry


def scan_sector_heatmap(
    stock_count: int = 200,
    window: int = DEFAULT_WINDOW,
    n_zones: int = DEFAULT_N_ZONES,
    price_threshold: float = PRICE_THRESHOLD_PCT,
    progress_callback: Optional[Callable] = None,
) -> List[Dict]:
    """
    扫描板块热力图数据

    对每个行业，统计：
    - 处于支撑区附近的股票占比
    - 处于压力区附近的股票占比
    - 盈亏比中位数
    - 平均密集区强度

    返回:
        按行业聚合的数据列表，用于前端热力图渲染
    """
    fetcher = DataFetcher()
    try:
        codes = fetcher.get_stock_codes(limit=stock_count, exclude_st=True)
        industry_map = build_industry_map(fetcher)

        # 按行业分组
        industry_stocks = defaultdict(list)
        for code in codes:
            ind = industry_map.get(code, "其他")
            industry_stocks[ind].append(code)

        total = len(codes)
        processed = 0
        industry_data = defaultdict(lambda: {
            "total": 0, "near_support": 0, "near_resistance": 0,
            "rr_list": [], "avg_strength": [], "scanned": 0,
        })

        for code in codes:
            try:
                df = fetcher.get_kline(code, freq="d", start="2023-01-01", end="2025-12-31")
                if len(df) < window + 5:
                    processed += 1
                    if progress_callback: progress_callback(processed, total)
                    continue

                ind = industry_map.get(code, "其他")
                current_price = float(df["close"].iloc[-1])
                zones = find_dense_zones(df, window=window, n_zones=n_zones)
                classify_zones(zones, current_price)

                industry_data[ind]["total"] += 1
                industry_data[ind]["scanned"] += 1

                supports = [z for z in zones if z.zone_type == "support"]
                resistances = [z for z in zones if z.zone_type == "resistance"]

                # 支撑附近
                min_support_dist = min((abs(z.distance_pct) for z in supports), default=99)
                if min_support_dist <= price_threshold:
                    industry_data[ind]["near_support"] += 1

                # 压力附近
                min_resistance_dist = min((abs(z.distance_pct) for z in resistances), default=99)
                if min_resistance_dist <= price_threshold:
                    industry_data[ind]["near_resistance"] += 1

                # 盈亏比
                if supports and resistances:
                    nearest_s = min(supports, key=lambda z: abs(z.distance_pct))
                    nearest_r = min(resistances, key=lambda z: abs(z.distance_pct))
                    rr = abs(nearest_r.distance_pct) / max(abs(nearest_s.distance_pct), 0.1)
                    industry_data[ind]["rr_list"].append(rr)

                # 强度
                if zones:
                    industry_data[ind]["avg_strength"].append(
                        np.mean([z.strength for z in zones])
                    )

            except Exception:
                pass

            processed += 1
            if progress_callback:
                progress_callback(processed, total)

        # 汇总
        result = []
        for ind, d in sorted(industry_data.items()):
            scanned = d["scanned"]
            if scanned == 0:
                continue

            support_pct = round(d["near_support"] / scanned * 100, 1)
            resistance_pct = round(d["near_resistance"] / scanned * 100, 1)
            neutral_pct = round(100 - support_pct - resistance_pct, 1)
            avg_rr = round(np.mean(d["rr_list"]), 2) if d["rr_list"] else 0
            avg_strength = round(np.mean(d["avg_strength"]), 2) if d["avg_strength"] else 0

            # 状态标签
            if support_pct > resistance_pct + 10:
                status = "偏多"
            elif resistance_pct > support_pct + 10:
                status = "偏空"
            else:
                status = "均衡"

            result.append({
                "industry": ind,
                "scanned": scanned,
                "support_pct": support_pct,
                "resistance_pct": resistance_pct,
                "neutral_pct": neutral_pct,
                "avg_rr": avg_rr,
                "avg_strength": avg_strength,
                "status": status,
            })

        return result

    finally:
        fetcher.logout()
