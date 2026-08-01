"""
交互式可视化仪表盘

基于 Plotly 生成自包含的交互式 HTML 仪表盘，无需服务器：
1. 单股分析：K线 + 密集区叠加 + 成交量剖面 + 密度曲线
2. 批量回测：汇总KPI卡 + 收益分布 + 散点图 + 排序表
3. 多时间框架：日/周/月线密集区对比
"""

import sys
import os
import json
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, COLORS, CHART_THEME

from src.density_analyzer import ZoneInfo


def _format_date(d) -> str:
    """统一日期格式"""
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def save_dashboard(fig: go.Figure, filename: str) -> str:
    """保存仪表盘为自包含HTML文件"""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(
        path,
        include_plotlyjs='cdn',
        full_html=True,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'scrollZoom': True,
            'locale': 'zh-CN',
        }
    )
    return path


# ================================================================
# 仪表盘1：单股综合分析
# ================================================================
def generate_single_stock_dashboard(
    stock_code: str,
    stock_name: str,
    kline_df: pd.DataFrame,
    zones: List[ZoneInfo],
    backtest_result: Optional[Dict] = None,
) -> str:
    """
    生成单只股票综合分析仪表盘

    包含：
    - 主图：K线 + 支撑/压力区彩色带状叠加 + 成交量柱
    - 右侧：密度剖面曲线 + 当前价格标记
    - 底部：回测权益曲线 + 交易标记（如有）
    """
    df = kline_df.tail(150).copy()  # 最近150根K线用于展示
    dates = df["date"].tolist()

    # ---- 4行布局 ----
    row_heights = [0.45, 0.15, 0.20, 0.20]
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(
            f"{stock_name}({stock_code}) 密集成交区分析",
            "成交量",
            "成交密度剖面",
            "回测权益曲线" if backtest_result else "",
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    # ---- 第1行：K线 + 区域叠加 ----
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="K线",
        increasing_line_color=COLORS["k_up"],
        decreasing_line_color=COLORS["k_down"],
        increasing_fillcolor=COLORS["k_up"],
        decreasing_fillcolor=COLORS["k_down"],
    ), row=1, col=1)

    # 密集区叠加（水平带状）
    for i, z in enumerate(zones[:5]):
        color = COLORS["support"] if z.zone_type == "support" else COLORS["resistance"]
        label = f"{z.zone_type}: {z.center:.2f}"
        fig.add_trace(go.Scatter(
            x=[dates[0], dates[-1]],
            y=[z.low, z.low],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[dates[0], dates[-1]],
            y=[z.high, z.high],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=color,
            name=f"{label} (强度:{z.strength:.2f})",
        ), row=1, col=1)

    # 当前价格线
    current_price = float(df["close"].iloc[-1])
    fig.add_hline(y=current_price, line_dash="dash", line_color="gray",
                  annotation_text=f"现价 {current_price:.2f}",
                  annotation_position="right", row=1, col=1)

    # ---- 第2行：成交量 ----
    colors_vol = [COLORS["volume_up"] if df["close"].iloc[i] >= df["open"].iloc[i]
                  else COLORS["volume_down"] for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=dates, y=df["volume"], name="成交量",
        marker_color=colors_vol, showlegend=False,
    ), row=2, col=1)

    # ---- 第3行：密度剖面 ----
    from src.density_analyzer import compute_density_profile
    price_levels, density, _, _ = compute_density_profile(df.tail(40))
    if len(price_levels) > 0:
        fig.add_trace(go.Scatter(
            x=price_levels, y=density,
            mode='lines', fill='tozeroy',
            fillcolor='rgba(41, 98, 255, 0.2)',
            line=dict(color='#2962ff', width=2),
            name='成交密度',
        ), row=3, col=1)
        # 标记峰值
        for z in zones[:5]:
            fig.add_vline(x=z.center, line_dash="dot", line_width=1,
                          line_color="green" if z.zone_type == "support" else "red",
                          annotation_text=f"{z.center:.1f}",
                          annotation_position="top", row=3, col=1)
        fig.add_vline(x=current_price, line_dash="dash", line_color="gray",
                      row=3, col=1)

    # ---- 第4行：回测结果 ----
    if backtest_result and backtest_result.get("equity_curve"):
        eq = backtest_result["equity_curve"]
        bm = backtest_result.get("benchmark_curve", [])
        eq_dates = [e["date"] for e in eq]
        eq_vals = [e["value"] for e in eq]
        fig.add_trace(go.Scatter(
            x=eq_dates, y=eq_vals, mode='lines',
            line=dict(color=COLORS["equity"], width=2),
            name='策略权益',
        ), row=4, col=1)
        if bm:
            bm_vals = [b["value"] for b in bm]
            fig.add_trace(go.Scatter(
                x=eq_dates, y=bm_vals, mode='lines',
                line=dict(color=COLORS["benchmark"], width=1, dash='dash'),
                name='买入持有',
            ), row=4, col=1)
        # 交易标记（兼容 dict 和 dataclass）
        def _trade_attr(t, key, default=None):
            if isinstance(t, dict):
                return t.get(key, default)
            return getattr(t, key, default)

        for t in backtest_result.get("trades", []):
            entry_d = _trade_attr(t, "entry_date")
            exit_d = _trade_attr(t, "exit_date")
            if entry_d:
                y_val = eq_vals[eq_dates.index(entry_d)] if entry_d in eq_dates else eq_vals[-1]
                fig.add_trace(go.Scatter(
                    x=[entry_d], y=[y_val],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='买入' if '买入' not in [tr.name for tr in fig.data] else None,
                    showlegend=False,
                ), row=4, col=1)
            if exit_d:
                y_val = eq_vals[eq_dates.index(exit_d)] if exit_d in eq_dates else eq_vals[-1]
                fig.add_trace(go.Scatter(
                    x=[exit_d], y=[y_val],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='卖出' if '卖出' not in [tr.name for tr in fig.data] else None,
                    showlegend=False,
                ), row=4, col=1)

    # 布局
    fig.update_layout(
        template=CHART_THEME,
        height=1000,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=80, b=40),
        title=dict(
            text=f"<b>{stock_name}({stock_code})</b> 成交密集区量化分析",
            x=0.5, xanchor="center",
            font=dict(size=20),
        ),
    )
    fig.update_xaxes(title_text="日期", row=4, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="密度", row=3, col=1)
    fig.update_yaxes(title_text="权益", row=4, col=1)

    # 免责声明
    fig.add_annotation(
        text="⚠️ 以上内容由AI基于公开数据生成，仅供参考，不构成投资建议。投资有风险，决策需谨慎。",
        xref="paper", yref="paper", x=0.5, y=-0.12, showarrow=False,
        font=dict(size=10, color="#999"),
    )

    filename = f"dashboard_{stock_code.replace('.', '_')}.html"
    return save_dashboard(fig, filename)


# ================================================================
# 仪表盘2：批量回测汇总
# ================================================================
def generate_batch_dashboard(
    batch_result: Dict,
    batch_label: str = "批量回测",
) -> str:
    """
    生成批量回测汇总仪表盘

    包含：KPI卡片、收益分布直方图、夏普vs胜率散点图、排序表格
    """
    agg = batch_result.get("aggregate", {})
    stocks = agg.get("stocks", [])

    if not stocks:
        # 空结果
        fig = go.Figure()
        fig.add_annotation(text="无有效回测结果", x=0.5, y=0.5, showarrow=False, font=dict(size=24))
        fig.update_layout(template=CHART_THEME, height=400)
        return save_dashboard(fig, "dashboard_batch_empty.html")

    returns = [s["return_pct"] for s in stocks]
    sharpes = [s["sharpe"] for s in stocks]
    win_rates = [s["win_rate"] for s in stocks]

    # 4象限布局
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "收益分布", "夏普比率 vs 胜率",
            "Top 10 收益排行", "Bottom 10 收益排行",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # (1,1) 收益分布
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=30,
        marker=dict(color='rgba(41, 98, 255, 0.7)', line=dict(color='white', width=1)),
        name='收益分布',
    ), row=1, col=1)
    fig.add_vline(x=np.mean(returns), line_dash="dash", line_color="red",
                  annotation_text=f"均值 {np.mean(returns):.1f}%", row=1, col=1)
    fig.add_vline(x=0, line_color="gray", line_width=1, row=1, col=1)

    # (1,2) 散点图
    fig.add_trace(go.Scatter(
        x=sharpes, y=win_rates, mode='markers',
        marker=dict(
            size=8, color=returns,
            colorscale='RdYlGn', colorbar=dict(title='收益%'),
            line=dict(width=0.5, color='white'),
        ),
        text=[s["code"] for s in stocks],
        hovertemplate='%{text}<br>夏普: %{x:.2f}<br>胜率: %{y:.1f}%<extra></extra>',
        name='股票',
    ), row=1, col=2)

    # (2,1) Top 10
    sorted_by_return = sorted(stocks, key=lambda x: x["return_pct"], reverse=True)[:10]
    fig.add_trace(go.Bar(
        x=[s["code"].split(".")[-1] for s in sorted_by_return],
        y=[s["return_pct"] for s in sorted_by_return],
        marker=dict(color=['green' if v > 0 else 'red' for v in [s["return_pct"] for s in sorted_by_return]]),
        text=[f"{v:.1f}%" for v in [s["return_pct"] for s in sorted_by_return]],
        textposition='outside',
        name='Top 10',
    ), row=2, col=1)

    # (2,2) Bottom 10
    sorted_by_return_asc = sorted(stocks, key=lambda x: x["return_pct"])[:10]
    fig.add_trace(go.Bar(
        x=[s["code"].split(".")[-1] for s in sorted_by_return_asc],
        y=[s["return_pct"] for s in sorted_by_return_asc],
        marker=dict(color=['green' if v > 0 else 'red' for v in [s["return_pct"] for s in sorted_by_return_asc]]),
        text=[f"{v:.1f}%" for v in [s["return_pct"] for s in sorted_by_return_asc]],
        textposition='outside',
        name='Bottom 10',
    ), row=2, col=2)

    # KPI 标注
    kpi_text = (
        f"总股票: {agg.get('valid_stocks', 0)} | "
        f"平均收益: {agg.get('avg_return_pct', 0):.2f}% | "
        f"中位数: {agg.get('median_return_pct', 0):.2f}% | "
        f"盈利比: {agg.get('positive_ratio_pct', 0):.1f}% | "
        f"平均夏普: {agg.get('avg_sharpe', 0):.3f}"
    )

    fig.update_layout(
        template=CHART_THEME,
        height=900,
        title=dict(
            text=f"<b>{batch_label}</b><br><span style='font-size:14px;color:#666'>{kpi_text}</span>",
            x=0.5, xanchor="center",
            font=dict(size=20),
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=120, b=40),
    )
    fig.update_xaxes(title_text="收益(%)", row=1, col=1)
    fig.update_xaxes(title_text="夏普比率", row=1, col=2)
    fig.update_yaxes(title_text="频次", row=1, col=1)
    fig.update_yaxes(title_text="胜率(%)", row=1, col=2)
    fig.update_yaxes(title_text="收益(%)", row=2, col=1)
    fig.update_yaxes(title_text="收益(%)", row=2, col=2)

    fig.add_annotation(
        text="⚠️ 以上内容由AI基于公开数据生成，仅供参考，不构成投资建议。投资有风险，决策需谨慎。",
        xref="paper", yref="paper", x=0.5, y=-0.08, showarrow=False,
        font=dict(size=10, color="#999"),
    )

    return save_dashboard(fig, "dashboard_batch_summary.html")


# ================================================================
# 仪表盘3：多时间框架对比
# ================================================================
def generate_mtf_dashboard(
    stock_code: str,
    stock_name: str,
    kline_df: pd.DataFrame,
    mtf_zones: Dict[str, List[ZoneInfo]],
) -> str:
    """
    多时间框架密集区对比仪表盘

    日/周/月线的密集区在同一张图上对比展示
    """
    df = kline_df.tail(200)
    dates = df["date"].tolist()
    current_price = float(df["close"].iloc[-1])

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("日线密集区", "周线密集区", "月线密集区"),
    )

    timeframes = ["day", "week", "month"]
    for row_idx, tf in enumerate(timeframes, 1):
        # K线
        fig.add_trace(go.Candlestick(
            x=dates, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name=f"{tf} K线", showlegend=(row_idx == 1),
            increasing_line_color=COLORS["k_up"], decreasing_line_color=COLORS["k_down"],
        ), row=row_idx, col=1)

        # 该时间框架的密集区
        for z in (mtf_zones.get(tf) or [])[:3]:
            color = COLORS["support"] if z.zone_type == "support" else COLORS["resistance"]
            label = f"{z.zone_type} {z.center:.2f}"
            fig.add_trace(go.Scatter(
                x=[dates[0], dates[-1]], y=[z.low, z.low],
                mode='lines', line=dict(width=0), showlegend=False,
            ), row=row_idx, col=1)
            fig.add_trace(go.Scatter(
                x=[dates[0], dates[-1]], y=[z.high, z.high],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor=color, name=label,
            ), row=row_idx, col=1)

        fig.add_hline(y=current_price, line_dash="dash", line_color="gray", row=row_idx, col=1)

    fig.update_layout(
        template=CHART_THEME,
        height=900,
        title=dict(
            text=f"<b>{stock_name}({stock_code})</b> 多时间框架密集区对比",
            x=0.5, font=dict(size=20),
        ),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=60, t=80, b=40),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)

    fig.add_annotation(
        text="⚠️ 以上内容由AI基于公开数据生成，仅供参考，不构成投资建议。投资有风险，决策需谨慎。",
        xref="paper", yref="paper", x=0.5, y=-0.08, showarrow=False,
        font=dict(size=10, color="#999"),
    )

    return save_dashboard(fig, f"dashboard_mtf_{stock_code.replace('.', '_')}.html")
