"""
回测引擎

基于成交密集区策略的回测框架：
1. 滚动窗口计算密集区 → 确定支撑/压力位
2. 价格触及支撑区 → 买入信号
3. 价格触及压力区 → 止盈 / 止损 → 卖出
4. T+1 约束：今日买入，次日才能卖出
5. 防未来信息泄露：用 T-1 数据计算区域，T 日交易

输出指标：总收益、年化收益、夏普比率、最大回撤、胜率、盈亏比
"""

import sys
import os
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_WINDOW, DEFAULT_N_ZONES, PRICE_THRESHOLD_PCT,
    INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE,
    POSITION_SIZE_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT, BACKTEST_START, BACKTEST_END,
)

from src.density_analyzer import find_dense_zones, classify_zones, ZoneInfo


@dataclass
class Trade:
    """单笔交易记录"""
    entry_date: str = ""
    entry_price: float = 0.0
    exit_date: str = ""
    exit_price: float = 0.0
    shares: int = 0
    pnl_pct: float = 0.0
    pnl_amount: float = 0.0
    exit_reason: str = ""
    holding_days: int = 0
    entry_support: float = 0.0
    entry_resistance: float = 0.0


def _compute_metrics(
    equity_curve: pd.Series,
    trades: List[Trade],
    benchmark_return: float = 0.0,
    risk_free_rate: float = 0.02,
) -> Dict:
    """计算回测绩效指标"""
    if len(equity_curve) < 2:
        return {}

    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    # 年化收益率
    n_years = len(returns) / 252
    annual_return = ((1 + total_return / 100) ** (1 / max(n_years, 0.25)) - 1) * 100

    # 夏普比率
    excess = returns - risk_free_rate / 252
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    # 最大回撤
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax * 100
    max_dd = float(drawdown.min())

    # 最大回撤持续期
    dd_start = None
    max_dd_days = 0
    in_dd = False
    dd_begin = 0
    for i, v in enumerate(drawdown.values):
        if v < 0 and not in_dd:
            in_dd = True
            dd_begin = i
        elif v >= 0 and in_dd:
            in_dd = False
            max_dd_days = max(max_dd_days, i - dd_begin)

    # 卡尔玛比率
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

    # 交易统计
    if trades:
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl_pct for t in losses])) if losses else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        avg_hold = np.mean([t.holding_days for t in trades])
        total_trades = len(trades)
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        avg_hold = 0
        total_trades = 0

    return {
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "max_dd_days": max_dd_days,
        "calmar_ratio": round(calmar, 3),
        "benchmark_return_pct": round(benchmark_return, 2),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
        "avg_holding_days": round(avg_hold, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
    }


class BacktestEngine:
    """密集区策略回测引擎"""

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        window: int = DEFAULT_WINDOW,
        n_zones: int = DEFAULT_N_ZONES,
        price_threshold: float = PRICE_THRESHOLD_PCT,
        stop_loss: float = STOP_LOSS_PCT,
        take_profit: float = TAKE_PROFIT_PCT,
        max_hold_days: int = 30,
        commission: float = COMMISSION_RATE,
        slippage: float = SLIPPAGE,
    ):
        self.initial_capital = initial_capital
        self.window = window
        self.n_zones = n_zones
        self.price_threshold = price_threshold / 100.0
        self.stop_loss = stop_loss / 100.0
        self.take_profit = take_profit / 100.0
        self.max_hold_days = max_hold_days
        self.commission = commission
        self.slippage = slippage

    def run(self, df: pd.DataFrame) -> Dict:
        """
        对单只股票运行回测

        参数:
            df: K线数据，需含 date, open, high, low, close, volume

        返回:
            {metrics: {...}, trades: [...], equity_curve: [...], benchmark_curve: [...]}
        """
        if len(df) < self.window + 10:
            return self._empty_result("数据不足")

        df = df.reset_index(drop=True).copy()
        n = len(df)

        capital = self.initial_capital
        position = 0          # 持仓股数
        entry_price = 0.0     # 买入价
        entry_date = ""       # 买入日期
        hold_days = 0         # 已持仓天数

        equity = [self.initial_capital]
        benchmark = [self.initial_capital]
        equity_dates = [str(df["date"].iloc[0])[:10]]

        trades: List[Trade] = []
        benchmark_shares = int(self.initial_capital / df["close"].iloc[self.window])

        for i in range(self.window, n - 1):
            date_str = str(df["date"].iloc[i])[:10]
            next_date = str(df["date"].iloc[i + 1])[:10]

            # 用截至 i（不含 i）的数据计算密集区，防止未来信息
            window_df = df.iloc[i - self.window:i]
            zones = find_dense_zones(window_df, n_zones=self.n_zones)
            current_price = float(df["close"].iloc[i])
            classify_zones(zones, current_price)

            # 当天交易价用次日开盘（模拟T+1实际成交）
            next_open = float(df["open"].iloc[i + 1])
            next_high = float(df["high"].iloc[i + 1])
            next_low = float(df["low"].iloc[i + 1])
            next_close = float(df["close"].iloc[i + 1])

            # ---- 卖出逻辑 ----
            if position > 0:
                hold_days += 1
                exit_reason = ""
                exit_price_realized = 0.0
                should_exit = False

                # 止盈：触及压力区
                resistances = [z for z in zones if z.zone_type == "resistance"]
                for z in resistances:
                    if next_high >= z.low and current_price < z.low:
                        exit_reason = "take_profit"
                        exit_price_realized = z.low * (1 - self.slippage)
                        should_exit = True
                        break

                # 止损
                if not should_exit and next_low <= entry_price * (1 - self.stop_loss):
                    exit_reason = "stop_loss"
                    exit_price_realized = entry_price * (1 - self.stop_loss) * (1 - self.slippage)
                    should_exit = True

                # 时间止损
                if not should_exit and hold_days >= self.max_hold_days:
                    exit_reason = "time_exit"
                    exit_price_realized = next_close * (1 - self.slippage)
                    should_exit = True

                # 无信号但持仓：按收盘价估值
                if not should_exit:
                    cash_val = capital
                    position_val = position * next_close
                else:
                    gross = position * exit_price_realized
                    cost = gross * self.commission
                    capital += gross - cost
                    trades.append(Trade(
                        entry_date=entry_date,
                        entry_price=entry_price,
                        exit_date=next_date,
                        exit_price=exit_price_realized,
                        shares=position,
                        pnl_pct=round((exit_price_realized / entry_price - 1) * 100, 2),
                        pnl_amount=round(gross - cost - position * entry_price, 2),
                        exit_reason=exit_reason,
                        holding_days=hold_days,
                    ))
                    position = 0
                    hold_days = 0
                    cash_val = capital
                    position_val = 0.0

            else:
                # ---- 买入逻辑 ----
                cash_val = capital
                position_val = 0.0

                # 价格接近支撑区 → 买入
                supports = [z for z in zones if z.zone_type == "support"]
                for z in supports:
                    dist = abs(current_price - z.center) / current_price
                    if dist <= self.price_threshold and current_price > z.center:
                        # 确认：当前价在支撑区上方且足够接近
                        buy_price = next_open * (1 + self.slippage)
                        max_shares = int(capital * POSITION_SIZE_PCT / buy_price)
                        if max_shares >= 100:
                            position = (max_shares // 100) * 100
                            cost = position * buy_price * (1 + self.commission)
                            if cost <= capital:
                                capital -= cost
                                entry_price = buy_price
                                entry_date = next_date
                                hold_days = 0
                                cash_val = capital
                                position_val = position * next_close
                                break

            # 记录权益
            equity.append(cash_val + position_val)
            benchmark.append(benchmark_shares * next_close)
            equity_dates.append(next_date)

        # 最后一天强制平仓
        if position > 0:
            last_close = float(df["close"].iloc[-1])
            gross = position * last_close
            cost = gross * self.commission
            capital += gross - cost
            trades.append(Trade(
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=str(df["date"].iloc[-1])[:10],
                exit_price=last_close,
                shares=position,
                pnl_pct=round((last_close / entry_price - 1) * 100, 2),
                pnl_amount=round(gross - cost - position * entry_price, 2),
                exit_reason="force_exit",
                holding_days=hold_days,
            ))
            equity[-1] = capital
            position = 0

        eq_series = pd.Series(equity, index=equity_dates)
        bm_series = pd.Series(benchmark, index=equity_dates)
        bench_return = (bm_series.iloc[-1] / bm_series.iloc[0] - 1) * 100

        metrics = _compute_metrics(eq_series, trades, bench_return)

        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": [{"date": d, "value": v} for d, v in zip(equity_dates, equity)],
            "benchmark_curve": [{"date": d, "value": v} for d, v in zip(equity_dates, benchmark)],
        }

    def _empty_result(self, reason: str = "") -> Dict:
        return {
            "metrics": {"total_return_pct": 0, "total_trades": 0, "error": reason},
            "trades": [],
            "equity_curve": [],
            "benchmark_curve": [],
        }


def run_batch_backtest(
    data_dict: Dict[str, pd.DataFrame],
    engine: Optional[BacktestEngine] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict]:
    """
    批量回测多只股票

    参数:
        data_dict:          {code: DataFrame} 字典
        engine:             回测引擎实例
        progress_callback:  进度回调 (current, total, code)

    返回:
        [{code, metrics, trades, equity_curve}, ...]
    """
    if engine is None:
        engine = BacktestEngine()

    results = []
    total = len(data_dict)
    for i, (code, df) in enumerate(data_dict.items()):
        try:
            r = engine.run(df)
            r["code"] = code
            results.append(r)
        except Exception as e:
            results.append({"code": code, "error": str(e), "metrics": {}, "trades": [], "equity_curve": []})
        if progress_callback:
            progress_callback(i + 1, total, code)
    return results


def aggregate_results(results: List[Dict]) -> Dict:
    """
    汇总批量回测结果

    返回:
        {total_stocks, avg_return, median_return, win_ratio,
         avg_sharpe, best/worst_stock, return_distribution, ...}
    """
    valid = [r for r in results if r.get("metrics") and r["metrics"].get("total_trades", 0) > 0]
    if not valid:
        return {"total_stocks": len(results), "valid_stocks": 0}

    returns = [r["metrics"]["total_return_pct"] for r in valid]
    sharpes = [r["metrics"].get("sharpe_ratio", 0) for r in valid]
    win_rates = [r["metrics"].get("win_rate_pct", 0) for r in valid]
    max_dds = [r["metrics"].get("max_drawdown_pct", 0) for r in valid]

    best_idx = int(np.argmax(returns))
    worst_idx = int(np.argmin(returns))

    return {
        "total_stocks": len(results),
        "valid_stocks": len(valid),
        "avg_return_pct": round(np.mean(returns), 2),
        "median_return_pct": round(np.median(returns), 2),
        "std_return_pct": round(np.std(returns), 2),
        "positive_ratio_pct": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
        "avg_sharpe": round(np.mean(sharpes), 3),
        "avg_win_rate_pct": round(np.mean(win_rates), 1),
        "avg_max_dd_pct": round(np.mean(max_dds), 2),
        "best_stock": valid[best_idx].get("code", ""),
        "best_return_pct": round(returns[best_idx], 2),
        "worst_stock": valid[worst_idx].get("code", ""),
        "worst_return_pct": round(returns[worst_idx], 2),
        "returns": returns,
        "sharpes": sharpes,
        "win_rates": win_rates,
        "stocks": [{"code": r["code"], "return_pct": m["total_return_pct"],
                     "sharpe": m.get("sharpe_ratio", 0),
                     "win_rate": m.get("win_rate_pct", 0),
                     "max_dd": m.get("max_drawdown_pct", 0),
                     "trades": m.get("total_trades", 0)}
                    for r, m in zip(valid, [v["metrics"] for v in valid])],
    }
