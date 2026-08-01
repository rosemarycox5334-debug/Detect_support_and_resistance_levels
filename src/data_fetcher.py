"""
数据获取模块
基于 baostock 的 A 股数据加载器

功能：
- K线数据获取（日/周/月），自动前复权
- 股票列表获取（过滤ST、退市）
- 自动重试 + 指数退避（最多3次）
- 本地 pickle 缓存，避免重复下载
- 批量获取进度回调
"""

import os
import sys
import time
import pickle
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Callable

import pandas as pd
import baostock as bs

# 导入项目配置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, MAX_RETRIES, RETRY_BACKOFF,
    CACHE_ENABLED, CACHE_EXPIRE_DAYS,
)


def _retry_with_backoff(func, max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF):
    """带指数退避的重试包装器"""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                wait = backoff ** attempt
                print(f"  [重试] 第{attempt+1}/{max_retries}次失败，{wait:.1f}秒后重试... ({e})")
                time.sleep(wait)
    raise last_exception


def bs_code(code: str) -> str:
    """将通用代码格式转为 baostock 格式

    例: '600519' -> 'sh.600519',  '000001' -> 'sz.000001'
        'sh.600519' -> 'sh.600519' (已是bs格式则不变)
    """
    code = code.strip().lower()
    if code.startswith("sh.") or code.startswith("sz.") or code.startswith("bj."):
        return code
    # 判断交易所
    if code.startswith("6"):
        return f"sh.{code}"
    elif code.startswith(("0", "3")):
        return f"sz.{code}"
    elif code.startswith(("8", "4")):
        return f"bj.{code}"
    return f"sz.{code}"


def short_code(bs_code_str: str) -> str:
    """从 baostock 格式提取简短代码  'sh.600519' -> '600519'"""
    return bs_code_str.split(".")[-1] if "." in bs_code_str else bs_code_str


class DataFetcher:
    """A股数据获取器，封装 baostock 接口"""

    def __init__(self):
        self._logged_in = False

    def _ensure_login(self):
        """确保已登录 baostock"""
        if not self._logged_in:
            lg = bs.login()
            if lg.error_code != '0':
                raise ConnectionError(f"baostock 登录失败: {lg.error_msg}")
            self._logged_in = True

    def logout(self):
        """登出"""
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def _cache_file(self, key: str) -> str:
        """生成缓存文件路径"""
        h = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(DATA_DIR, f"cache_{h}.pkl")

    def _read_cache(self, path: str) -> Optional[pd.DataFrame]:
        """读取缓存，过期则返回 None"""
        if not CACHE_ENABLED or not os.path.exists(path):
            return None
        try:
            age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
            if age > CACHE_EXPIRE_DAYS:
                return None
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def _write_cache(self, path: str, df: pd.DataFrame):
        """写入缓存"""
        if CACHE_ENABLED:
            try:
                with open(path, 'wb') as f:
                    pickle.dump(df, f)
            except Exception:
                pass

    # ================================================================
    # K 线数据
    # ================================================================
    def get_kline(
        self,
        code: str,
        freq: str = "d",
        start: str = "2023-01-01",
        end: str = "2025-12-31",
        adjust: str = "2",
    ) -> pd.DataFrame:
        """
        获取 K 线数据（前复权）

        参数:
            code:   股票代码，如 '600519' / 'sh.600519'
            freq:   频率 'd'=日线, 'w'=周线, 'm'=月线
            start:  起始日期 'YYYY-MM-DD'
            end:    结束日期 'YYYY-MM-DD'
            adjust: 复权类型 '1'=后复权 '2'=前复权 '3'=不复权

        返回:
            DataFrame，列: date, open, high, low, close, volume, amount, turn, pct_chg
        """
        bs_code_str = bs_code(code)
        cache_key = f"kline_{bs_code_str}_{freq}_{start}_{end}_{adjust}"
        cache_path = self._cache_file(cache_key)

        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        self._ensure_login()

        def _fetch():
            # baostock 的 frequency 参数: d/w/m
            freq_map = {"d": "d", "w": "w", "m": "m"}
            rs = bs.query_history_k_data_plus(
                bs_code_str,
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start, end_date=end,
                frequency=freq_map.get(freq, "d"),
                adjustflag=adjust,
            )
            if rs.error_code != '0':
                raise RuntimeError(f"查询失败: {rs.error_msg}")

            rows = []
            while (rs.error_code == '0') and rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_chg"])
            # 类型转换
            for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pct_chg"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df["date"] = pd.to_datetime(df["date"])
            # 过滤掉全0行（停牌等）
            df = df[df["volume"] > 0].copy()
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

        try:
            df = _retry_with_backoff(_fetch)
            self._write_cache(cache_path, df)
            return df
        except Exception as e:
            raise RuntimeError(f"获取 {bs_code_str} K线失败: {e}")

    # ================================================================
    # 股票列表
    # ================================================================
    def get_stock_list(
        self,
        exclude_st: bool = True,
        only_active: bool = True,
    ) -> pd.DataFrame:
        """
        获取全A股股票列表

        返回:
            DataFrame，列: code, code_name, ipo_date, out_date, type, status
        """
        cache_key = f"stock_list_{exclude_st}_{only_active}"
        cache_path = self._cache_file(cache_key)

        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        self._ensure_login()

        def _fetch():
            rs = bs.query_stock_basic()
            if rs.error_code != '0':
                raise RuntimeError(f"查询股票列表失败: {rs.error_msg}")
            rows = []
            while (rs.error_code == '0') and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=["code", "code_name", "ipo_date", "out_date", "type", "status"])
            # 只保留 A 股
            df = df[df["type"] == "1"].copy()
            # 排除 ST
            if exclude_st:
                df = df[~df["code_name"].str.contains("ST", na=False)].copy()
            # 只保留在市的
            if only_active:
                df = df[df["status"] == "1"].copy()
            # 排除 B 股等
            df = df[df["code"].str.startswith(("sh.", "sz."))].copy()
            df["ipo_date"] = pd.to_datetime(df["ipo_date"])
            df.reset_index(drop=True, inplace=True)
            return df

        try:
            df = _retry_with_backoff(_fetch)
            self._write_cache(cache_path, df)
            return df
        except Exception as e:
            raise RuntimeError(f"获取股票列表失败: {e}")

    def get_stock_name(self, bs_code_str: str) -> str:
        """根据 baostock 代码查询股票名称，失败返回简短代码"""
        try:
            sl = self.get_stock_list()
            match = sl[sl["code"] == bs_code_str]
            if len(match) > 0:
                return str(match.iloc[0]["code_name"])
        except Exception:
            pass
        return short_code(bs_code_str)
