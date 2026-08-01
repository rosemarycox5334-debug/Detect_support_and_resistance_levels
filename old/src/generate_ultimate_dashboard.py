#!/usr/bin/env python3
"""
🎬 视频级成交密集区仪表盘 v4 — 决策闭环 + 市场扫描 + 板块热力

功能模块:
  决策支持: 区域历史胜率、触及预警、风险评估
  市场扫描: 全A股支撑/压力位扫描、盈亏比排序
  板块热力: 申万行业聚合分析、热力图
  原有图表: K线/权益/密度/饼图/3D/热力/雷达/月度 (全保留)
"""

import sys, os, json, argparse
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config import OUTPUT_DIR, DEFAULT_WINDOW, DEFAULT_N_ZONES
from src.data_fetcher import DataFetcher, bs_code, short_code
from src.density_analyzer import (
    find_dense_zones, classify_zones, compute_density_profile, calc_risk_reward,
)
from src.backtest_engine import BacktestEngine
from src.zone_history import analyze_zone_history, generate_touch_alerts, compute_risk_score, compute_backtest_confidence
from src.market_scanner import scan_market, scan_summary
from src.sector_heatmap import scan_sector_heatmap


# ═══════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════

def gather_data(code, window, n_zones, start, end):
    fetcher = DataFetcher()
    try:
        bs = bs_code(code); short = short_code(bs)
        sl = fetcher.get_stock_list()
        match = sl[sl["code"] == bs]
        name = match.iloc[0]["code_name"] if len(match) > 0 else short
        df = fetcher.get_kline(bs, freq="d", start=start, end=end)
        df_display = df.tail(150).copy()
        zones = find_dense_zones(df, window=window, n_zones=n_zones)
        current_price = float(df["close"].iloc[-1])
        classify_zones(zones, current_price)
        rr = calc_risk_reward(current_price, zones)
        price_levels, density, _, _ = compute_density_profile(df.tail(window))
        density_data = list(zip(
            [round(float(p), 2) for p in price_levels],
            [round(float(d), 4) for d in density],
        )) if len(density) > 0 else []
        engine = BacktestEngine(window=window, n_zones=n_zones)
        bt = engine.run(df)
        recent_trades = []
        for t in bt.get("trades", [])[-20:]:
            entry_d = t.entry_date if hasattr(t, 'entry_date') else t.get('entry_date', '')
            exit_d = t.exit_date if hasattr(t, 'exit_date') else t.get('exit_date', '')
            pnl = t.pnl_pct if hasattr(t, 'pnl_pct') else t.get('pnl_pct', 0)
            reason = t.exit_reason if hasattr(t, 'exit_reason') else t.get('exit_reason', '')
            days = t.holding_days if hasattr(t, 'holding_days') else t.get('holding_days', 0)
            recent_trades.append({"entry": entry_d, "exit": exit_d, "pnl": pnl, "reason": reason, "days": days})

        # Kline for charts
        kline_150 = [{"date": str(df_display["date"].iloc[i])[:10],
                      "open": float(df_display["open"].iloc[i]),
                      "high": float(df_display["high"].iloc[i]),
                      "low": float(df_display["low"].iloc[i]),
                      "close": float(df_display["close"].iloc[i]),
                      "volume": float(df_display["volume"].iloc[i]),
                      } for i in range(len(df_display))]

        # Monthly returns for heatmap
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month").agg(
            open=("open", "first"), close=("close", "last")
        ).reset_index()
        monthly["return"] = (monthly["close"] / monthly["open"] - 1) * 100
        monthly["year"] = monthly["month"].apply(lambda x: x.year)
        monthly["mon"] = monthly["month"].apply(lambda x: x.month)
        monthly_data = []
        for _, r in monthly.iterrows():
            monthly_data.append({"year": int(r["year"]), "month": int(r["mon"]), "return": round(float(r["return"]), 2)})

        # ★ NEW: 区域历史胜率 & 触及预警
        zone_histories = analyze_zone_history(df, window=window, n_zones=n_zones)
        touch_alerts = generate_touch_alerts(current_price, zones, zone_histories)
        risk_score = compute_risk_score(touch_alerts)
        history_data = [{"center": h.center, "type": h.zone_type,
                         "held": h.held_count, "broke": h.broke_count,
                         "total": h.touch_count, "rate": h.success_rate}
                        for h in zone_histories]

        # ★ NEW: 回测置信度
        backtest_confidence = compute_backtest_confidence(
            recent_trades, rr["risk_reward_ratio"], zones, current_price
        )

        return {
            "code": short, "name": name, "window": window, "n_zones": n_zones,
            "current_price": current_price, "data_rows": len(df),
            "zones": [{"type": z.zone_type, "center": z.center, "low": z.low, "high": z.high,
                       "strength": z.strength, "dist_pct": z.distance_pct,
                       "touch_count": z.touch_count, "vol_pct": z.volume_pct} for z in zones],
            "rr": rr,
            "metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in bt["metrics"].items()},
            "trades_count": len(bt["trades"]),
            "recent_trades": recent_trades,
            "equity_curve": bt.get("equity_curve", []),
            "benchmark_curve": bt.get("benchmark_curve", []),
            "kline": kline_150,
            "density_data": density_data,
            "monthly_data": monthly_data,
            # ★ NEW data
            "zone_history": history_data,
            "touch_alerts": touch_alerts,
            "risk_score": risk_score,
            "backtest_confidence": backtest_confidence,
        }
    finally:
        fetcher.logout()


# ═══════════════════════════════════════════
# CHART BUILDERS — all return dicts for Plotly.react()
# ═══════════════════════════════════════════

def chart_kline_zones(data):
    """K线 + 密集区叠加"""
    kl = data["kline"]; dates = [k["date"] for k in kl]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.72, 0.28])
    fig.add_trace(go.Candlestick(
        x=dates, open=[k["open"] for k in kl], high=[k["high"] for k in kl],
        low=[k["low"] for k in kl], close=[k["close"] for k in kl], name="K线",
        increasing_line_color="#00ff88", decreasing_line_color="#ff4466",
        increasing_fillcolor="rgba(0,255,136,0.12)", decreasing_fillcolor="rgba(255,68,102,0.12)",
    ), row=1, col=1)
    for z in data["zones"]:
        c = "rgba(0,255,136,0.10)" if z["type"]=="support" else "rgba(255,68,102,0.10)"
        lc = "#00ff88" if z["type"]=="support" else "#ff4466"
        fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[z["low"], z["low"]],
            mode="lines", line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[z["high"], z["high"]],
            mode="lines", line=dict(width=0, color=lc), fill="tonexty", fillcolor=c,
            name=f"{'🟢' if z['type']=='support' else '🔴'} {z['center']}"), row=1, col=1)
    fig.add_hline(y=data["current_price"], line_dash="dash", line_color="rgba(255,255,255,0.25)", row=1, col=1)
    cv = ["rgba(0,255,136,0.25)" if kl[i]["close"]>=kl[i]["open"] else "rgba(255,68,102,0.25)" for i in range(len(kl))]
    fig.add_trace(go.Bar(x=dates, y=[k["volume"] for k in kl], marker_color=cv, name="量"), row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=5,b=5), height=380, showlegend=True,
        legend=dict(orientation="h", y=1.05, font=dict(color="rgba(255,255,255,0.5)",size=8)),
        xaxis=dict(showgrid=False, rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", side="right"),
        xaxis2=dict(showgrid=False), yaxis2=dict(showgrid=False, showticklabels=False))
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_equity(data):
    """权益曲线"""
    eq = data["equity_curve"]; bm = data["benchmark_curve"]
    if not eq: return None
    ed = [e["date"] for e in eq]; ev = [e["value"] for e in eq]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ed, y=ev, mode="lines", line=dict(color="#00ff88", width=2.5), name="策略权益"))
    if bm:
        fig.add_trace(go.Scatter(x=ed, y=[b["value"] for b in bm], mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"), name="买入持有"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=5,b=5), height=320,
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(color="rgba(255,255,255,0.4)",size=9)))
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_density(data):
    """密度剖面"""
    dd = data["density_data"]
    if not dd: return None
    prices, vals = zip(*dd)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(prices), y=list(vals), mode="lines",
        fill="tozeroy", fillcolor="rgba(0,212,255,0.12)", line=dict(color="#00d4ff", width=2.5), name="密度"))
    for z in data["zones"]:
        c = "#00ff88" if z["type"]=="support" else "#ff4466"
        fig.add_vline(x=z["center"], line_dash="dot", line_width=1.5, line_color=c, opacity=0.7)
    fig.add_vline(x=data["current_price"], line_dash="dash", line_color="rgba(255,255,255,0.25)")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=5,b=5), height=300,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(showgrid=False, showticklabels=False))
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_pie_pnl(data):
    """交易盈亏分布饼图"""
    trades = data["recent_trades"]
    if not trades: return None
    win = sum(1 for t in trades if t["pnl"] > 0)
    loss = sum(1 for t in trades if t["pnl"] < 0)
    flat = len(trades) - win - loss
    fig = go.Figure()
    labels, values, colors = [], [], []
    if win > 0: labels.append("盈利"); values.append(win); colors.append("#00ff88")
    if loss > 0: labels.append("亏损"); values.append(loss); colors.append("#ff4466")
    if flat > 0: labels.append("持平"); values.append(flat); colors.append("#8b5cf6")
    fig.add_trace(go.Pie(labels=labels, values=values, marker_colors=colors,
        hole=0.5, textinfo="label+percent", textfont=dict(color="#fff", size=12),
        hoverinfo="label+value"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=5,b=5), height=300, showlegend=False)
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_heatmap_density(data):
    """成交密度热力图"""
    dd = data["density_data"]
    if not dd: return None
    prices, vals = zip(*dd)
    # reshape to 10x10 grid
    n = len(prices)
    grid_size = 10
    z_vals = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            idx = int((i * grid_size + j) / (grid_size * grid_size) * n)
            idx = min(idx, n - 1)
            row.append(vals[idx])
        z_vals.append(row)
    y_labels = [f"¥{prices[int(i*grid_size/grid_size/grid_size*n)]:.1f}" for i in range(grid_size)]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_vals, colorscale=[[0,"rgba(5,5,16,1)"],[0.5,"rgba(0,212,255,0.6)"],[1,"rgba(0,255,136,0.9)"]],
        showscale=False, hoverinfo="z"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=5,b=5), height=300,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False))
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_3d_surface(data):
    """3D 价格曲面"""
    kl = data["kline"][-60:]  # last 60 days
    if len(kl) < 10: return None
    # Create a simple 3D surface from OHLC
    n = len(kl)
    grid = 20
    step = max(1, n // grid)
    x_idx = list(range(0, n, step))[:grid]
    x_labels = [kl[i]["date"] for i in x_idx[:grid]]

    z_data = []
    for i in range(grid):
        row = []
        for j in range(grid):
            idx = min(x_idx[i] + j % 5, n - 1)
            row.append(kl[idx]["close"])
        z_data.append(row)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=z_data, colorscale=[[0,"#050510"],[0.5,"#00d4ff"],[1,"#00ff88"]],
        showscale=False, opacity=0.85,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.3, specular=0.4),
        lightposition=dict(x=100, y=200, z=50)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, showticklabels=False, title=""),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=0,r=0,t=0,b=0), height=380)
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_zone_bar(data):
    """密集区强度排行柱图"""
    zones = data["zones"]
    if not zones: return None
    names = [f"{'🟢' if z['type']=='support' else '🔴'} ¥{z['center']}" for z in zones]
    strengths = [z["strength"] for z in zones]
    colors = ["#00ff88" if z["type"]=="support" else "#ff4466" for z in zones]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=strengths, y=names, orientation="h",
        marker_color=colors, text=[f"{s:.2f}" for s in strengths],
        textposition="outside", textfont=dict(color="#fff", size=11)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=40,t=5,b=5), height=260,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", range=[0, max(strengths)*1.2] if strengths else None),
        yaxis=dict(showgrid=False, autorange="reversed"))
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_radar(data):
    """区域因子雷达图"""
    zones = data["zones"]
    if not zones: return None
    fig = go.Figure()
    categories = ["强度", "距离%", "重叠数", "量占比", "区间宽度"]
    for z in zones[:4]:
        width = abs(z["high"] - z["low"]) / z["center"] * 100 if z["center"] else 0
        vals = [
            z["strength"] * 100,
            abs(z["dist_pct"]),
            min(z["touch_count"], 40) / 40 * 100,
            min(z["vol_pct"], 100),
            min(width, 100),
        ]
        vals.append(vals[0])  # close the loop
        cats = categories + [categories[0]]
        color = "#00ff88" if z["type"]=="support" else "#ff4466"
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            fillcolor=color.replace(")", ",0.15)"),
            line=dict(color=color, width=2),
            name=f"{'🟢' if z['type']=='support' else '🔴'} ¥{z['center']}"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=5,b=5), height=300,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", range=[0,100], showticklabels=False),
            angularaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        ),
        legend=dict(orientation="h", y=1.08, font=dict(color="rgba(255,255,255,0.4)",size=9)))
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_monthly_heatmap(data):
    """月度收益热力图"""
    md = data.get("monthly_data", [])
    if not md: return None
    years = sorted(set(r["year"] for r in md))
    months = list(range(1, 13))
    z = np.zeros((len(years), 12))
    for r in md:
        yi = years.index(r["year"])
        z[yi][r["month"]-1] = r["return"]
    z = z.tolist()
    month_labels = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=month_labels, y=[str(y) for y in years],
        colorscale=[[0,"#ff4466"],[0.5,"#050510"],[1,"#00ff88"]],
        text=[[f"{v:.1f}%" if v!=0 else "" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=9),
        showscale=True, colorbar=dict(title="收益%", title_font=dict(color="#fff",size=9),
            tickfont=dict(color="#fff",size=8)),
        hoverongaps=False))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=5,b=5), height=320,
        xaxis=dict(showgrid=False, side="top"),
        yaxis=dict(showgrid=False, autorange="reversed"))
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_sector_heatmap(data):
    """板块热力图"""
    sectors = data.get("sectors", [])
    if not sectors: return None
    industries = [s["industry"] for s in sectors]
    support_vals = [s["support_pct"] for s in sectors]
    resistance_vals = [s["resistance_pct"] for s in sectors]
    neutral_vals = [s["neutral_pct"] for s in sectors]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="支撑占比", y=industries, x=support_vals,
        orientation="h", marker_color="#00ff88", text=[f"{v}%" for v in support_vals],
        textposition="outside", textfont=dict(size=10)))
    fig.add_trace(go.Bar(name="压力占比", y=industries, x=resistance_vals,
        orientation="h", marker_color="#ff4466", text=[f"{v}%" for v in resistance_vals],
        textposition="outside", textfont=dict(size=10)))

    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=5,b=5), height=max(300, len(industries)*28), barmode="group",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", title="占比(%)", range=[0, 100]),
        yaxis=dict(showgrid=False, autorange="reversed"),
        legend=dict(orientation="h", y=1.05, font=dict(color="rgba(255,255,255,0.5)",size=9)))
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


def chart_confidence(data):
    """回测置信度 - 收益分布柱图"""
    bc = data.get("backtest_confidence", {})
    buckets = bc.get("return_buckets", [])
    if not buckets: return None
    labels = [b["label"] for b in buckets]
    counts = [b["count"] for b in buckets]
    colors = [b["color"] for b in buckets]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=counts, marker_color=colors,
        text=[str(c) for c in counts], textposition="outside",
        textfont=dict(size=10)))
    avg = bc.get("avg_return", 0)
    fig.add_annotation(x=4, y=max(counts)*0.9 if counts else 1,
        text=f"均值: {avg:+.2f}% | 胜率: {bc.get('win_rate',0)}%<br>{bc.get('confidence_label','')}",
        showarrow=False, font=dict(color="#fff", size=10),
        bgcolor="rgba(12,12,35,0.8)", borderpad=8)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=20,b=5), height=280,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", showticklabels=False),
        showlegend=False)
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


# ═══════════════════════════════════════════
# HTML TEMPLATE — all charts embedded in DATA.charts
# ═══════════════════════════════════════════

HTML_V3 = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>成交密集区 · __NAME__ (__CODE__)</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
:root{{--g:#00ff88;--p:#8b5cf6;--c:#00d4ff;--r:#ff4466;--bg:#050510;--card:rgba(12,12,35,0.7);--border:rgba(255,255,255,0.06);--text:#d0d0d0;--dim:rgba(255,255,255,0.4)}}
body{{background:var(--bg);color:var(--text);font-family:'Noto Sans SC',sans-serif;overflow-x:hidden;min-height:100vh}}
canvas#stars{{position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none}}
#scanline{{position:fixed;top:0;left:0;width:100%;height:2px;background:linear-gradient(90deg,transparent,rgba(0,255,136,0.3),transparent);z-index:999;pointer-events:none;animation:scanDown 4s linear infinite;box-shadow:0 0 20px rgba(0,255,136,0.2)}}
@keyframes scanDown{{0%{{top:-2px}}100%{{top:100%}}}}
main{{position:relative;z-index:1;max-width:1500px;margin:0 auto;padding:28px 36px 60px}}

.title-block{{text-align:center;margin-bottom:8px}}
.glitch{{font-family:'Orbitron',monospace;font-size:clamp(28px,5vw,48px);font-weight:900;color:#fff;display:inline-block;text-shadow:0 0 7px var(--c),0 0 10px var(--c),0 0 21px var(--c),0 0 42px var(--p),0 0 80px var(--p);animation:gp 2.5s ease-in-out infinite}}
@keyframes gp{{0%,100%{{opacity:1}}50%{{opacity:.85}}}}
.subtitle-line{{font-size:13px;color:var(--dim);letter-spacing:6px;margin-top:4px}}

.price-stage{{text-align:center;margin:24px 0 28px}}
.price-ring{{display:inline-block;position:relative;padding:18px 36px}}
.price-ring::before,.price-ring::after{{content:'';position:absolute;border-radius:50%;border:1px solid rgba(0,255,136,0.15);animation:rp 3s ease-in-out infinite}}
.price-ring::before{{inset:0}} .price-ring::after{{inset:-12px;animation-delay:.5s;border-color:rgba(0,212,255,0.08)}}
@keyframes rp{{0%,100%{{transform:scale(1);opacity:.5}}50%{{transform:scale(1.08);opacity:1}}}}
.price-big{{font-family:'Orbitron',monospace;font-size:clamp(38px,6.5vw,68px);font-weight:900;background:linear-gradient(180deg,#fff 0%,var(--g) 50%,var(--c) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 0 30px rgba(0,255,136,0.35))}}
.price-meta{{font-size:11px;color:var(--dim);letter-spacing:5px;margin-top:4px}}

/* Tabs */
.tab-nav{{display:flex;justify-content:center;gap:4px;margin-bottom:24px;flex-wrap:wrap}}
.tab-btn{{font-family:'Orbitron',monospace;font-size:11px;letter-spacing:2px;padding:9px 22px;border:1px solid var(--border);border-radius:8px;background:transparent;color:var(--dim);cursor:pointer;transition:all .35s}}
.tab-btn:hover{{border-color:rgba(0,255,136,0.3);color:#fff}}
.tab-btn.active{{border-color:var(--g);color:var(--g);background:rgba(0,255,136,0.06);box-shadow:0 0 16px rgba(0,255,136,0.1)}}
.tab-panel{{display:none;animation:fs .45s ease}}
.tab-panel.active{{display:block}}
@keyframes fs{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}

/* KPI */
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:22px}}
.kpi-card{{background:var(--card);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);border:1px solid var(--border);border-radius:14px;padding:16px 18px;position:relative;overflow:hidden;transition:all .4s}}
.kpi-card:hover{{transform:translateY(-3px);border-color:rgba(0,255,136,0.25);box-shadow:0 8px 36px rgba(0,0,0,0.45)}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(0,255,136,0.5),transparent);animation:kl 3s ease-in-out infinite}}
@keyframes kl{{0%,100%{{opacity:.3}}50%{{opacity:1}}}}
.kpi-label{{font-size:10px;letter-spacing:2.5px;color:var(--dim);margin-bottom:5px;text-transform:uppercase}}
.kpi-value{{font-family:'Orbitron',monospace;font-size:clamp(19px,2.5vw,30px);font-weight:700;line-height:1.1}}
.kpi-sub{{font-size:10px;color:var(--dim);margin-top:4px}}
.kpi-value.up{{color:var(--g);text-shadow:0 0 12px rgba(0,255,136,0.25)}}
.kpi-value.down{{color:var(--r);text-shadow:0 0 12px rgba(255,68,102,0.25)}}
.kpi-value.neutral{{color:var(--c);text-shadow:0 0 12px rgba(0,212,255,0.25)}}

/* Chart boxes */
.chart-box{{background:var(--card);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);border:1px solid var(--border);border-radius:14px;padding:16px;margin-bottom:16px;transition:border-color .5s}}
.chart-box:hover{{border-color:rgba(0,255,136,0.2)}}
.chart-title{{font-family:'Orbitron',monospace;font-size:10px;letter-spacing:3px;color:var(--dim);margin-bottom:10px;display:flex;align-items:center;gap:8px}}
.chart-title::before{{content:'';display:inline-block;width:8px;height:8px;background:var(--g);border-radius:50%;box-shadow:0 0 8px var(--g)}}
.chart-div{{min-height:300px;width:100%}}
.chart-div-lg{{min-height:360px;width:100%}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
.grid-main{{display:grid;grid-template-columns:1fr 340px;gap:16px}}
@media(max-width:1100px){{.grid-3{{grid-template-columns:1fr 1fr}}}}
@media(max-width:800px){{.grid-2,.grid-3,.grid-main{{grid-template-columns:1fr}}}}

/* Zones */
.zone-card{{background:var(--card);border-radius:12px;padding:14px 16px;margin-bottom:10px;border-left:3px solid;transition:all .3s}}
.zone-card:hover{{transform:translateX(4px);box-shadow:0 4px 20px rgba(0,0,0,0.3)}}
.zone-card.support{{border-color:var(--g)}} .zone-card.resistance{{border-color:var(--r)}}
.zone-card .z-type{{font-size:9px;letter-spacing:2px;margin-bottom:4px}}
.zone-card.support .z-type{{color:var(--g)}} .zone-card.resistance .z-type{{color:var(--r)}}
.zone-card .z-price{{font-family:'Orbitron',monospace;font-size:20px;font-weight:700}}
.zone-card .z-range{{font-size:11px;color:var(--dim);margin-top:2px}}
.zone-card .z-meta{{display:flex;gap:14px;margin-top:8px;font-size:10px;color:var(--dim)}}

/* Trade table */
.trade-wrap{{max-height:380px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.1) transparent}}
.trade-tbl{{width:100%;border-collapse:collapse;font-size:11px}}
.trade-tbl th{{position:sticky;top:0;background:rgba(12,12,35,0.95);text-align:left;color:var(--dim);font-weight:400;padding:8px 10px;border-bottom:1px solid var(--border);font-size:9px;letter-spacing:1.5px;z-index:1}}
.trade-tbl td{{padding:7px 10px;border-bottom:1px solid rgba(255,255,255,0.02)}}
.trade-tbl tr:hover td{{background:rgba(0,255,136,0.03)}}
.pnl-up{{color:var(--g)}} .pnl-down{{color:var(--r)}}

.footer-note{{text-align:center;color:rgba(255,255,255,0.15);font-size:10px;margin-top:44px;padding:20px;border-top:1px solid rgba(255,255,255,0.04)}}
</style>
</head>
<body>
<canvas id="stars"></canvas>
<div id="scanline"></div>
<main>
  <div class="title-block">
    <h1 class="glitch">密集成交区 · 量化分析</h1>
    <div class="subtitle-line">__NAME__ · __CODE__ · __WINDOW__DAY WINDOW · __DATE__</div>
  </div>
  <div class="price-stage"><div class="price-ring"><div class="price-big" id="priceBig">¥__CURRENT_PRICE__</div></div>
    <div class="price-meta">REAL-TIME DENSE ZONE ANALYSIS</div></div>

  <div class="tab-nav">
    <button class="tab-btn active" data-tab="overview">▸ 概览</button>
    <button class="tab-btn" data-tab="decision">▸ 决策支持</button>
    <button class="tab-btn" data-tab="charts">▸ 分析图表</button>
    <button class="tab-btn" data-tab="scanner">▸ 市场扫描</button>
    <button class="tab-btn" data-tab="heatmap">▸ 板块热力</button>
    <button class="tab-btn" data-tab="trades">▸ 交易记录</button>
  </div>

  <!-- TAB: Overview -->
  <div class="tab-panel active" id="tabOverview">
    <div class="kpi-grid" id="kpiGrid"></div>
    <div class="grid-main">
      <div class="chart-box"><div class="chart-title">K线 · 密集区叠加</div><div class="chart-div-lg" id="chart_kline"></div></div>
      <div><div class="chart-box"><div class="chart-title">密集区详情</div><div id="zoneList"></div></div></div>
    </div>
  </div>

  <!-- TAB: Decision Support (NEW) -->
  <div class="tab-panel" id="tabDecision">
    <div class="grid-2">
      <div class="chart-box">
        <div class="chart-title">风险评估</div>
        <div id="riskScore"></div>
      </div>
      <div class="chart-box">
        <div class="chart-title">触及预警</div>
        <div id="touchAlerts"></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="chart-box">
        <div class="chart-title">回测置信度 · 收益分布</div>
        <div class="chart-div" id="chart_confidence"></div>
      </div>
      <div class="chart-box">
        <div class="chart-title">置信度统计</div>
        <div id="confidenceStats"></div>
      </div>
    </div>
    <div class="chart-box">
      <div class="chart-title">区域历史胜率</div>
      <div style="overflow-x:auto"><table class="trade-tbl" id="historyTable">
        <thead><tr><th>区域类型</th><th>中心价</th><th>守住次数</th><th>突破次数</th><th>总触及</th><th>守住率</th><th>评价</th></tr></thead>
        <tbody></tbody>
      </table></div>
    </div>
  </div>

  <!-- TAB: Charts -->
  <div class="tab-panel" id="tabCharts">
    <div class="grid-3">
      <div class="chart-box"><div class="chart-title">收益分布</div><div class="chart-div" id="chart_pie"></div></div>
      <div class="chart-box"><div class="chart-title">密度热力图</div><div class="chart-div" id="chart_heatmap"></div></div>
      <div class="chart-box"><div class="chart-title">区域雷达图</div><div class="chart-div" id="chart_radar"></div></div>
    </div>
    <div class="grid-2">
      <div class="chart-box"><div class="chart-title">权益曲线</div><div class="chart-div" id="chart_equity"></div></div>
      <div class="chart-box"><div class="chart-title">成交密度剖面</div><div class="chart-div" id="chart_density"></div></div>
    </div>
    <div class="grid-2">
      <div class="chart-box"><div class="chart-title">3D 价格曲面</div><div class="chart-div-lg" id="chart_3d"></div></div>
      <div class="chart-box"><div class="chart-title">密集区强度排行</div><div class="chart-div" id="chart_bar"></div></div>
    </div>
    <div class="chart-box"><div class="chart-title">月度收益热力图</div><div class="chart-div" id="chart_monthly"></div></div>
  </div>

  <!-- TAB: Market Scanner (NEW) -->
  <div class="tab-panel" id="tabScanner">
    <div class="kpi-grid" id="scanKpiGrid"></div>
    <div class="grid-2">
      <div class="chart-box">
        <div class="chart-title">支撑位附近 (潜在买入机会)</div>
        <div style="overflow-x:auto;max-height:400px;overflow-y:auto"><table class="trade-tbl" id="supportScanTable">
          <thead><tr><th>代码</th><th>现价</th><th>支撑价</th><th>距离%</th><th>压力价</th><th>盈亏比</th></tr></thead>
          <tbody></tbody>
        </table></div>
      </div>
      <div class="chart-box">
        <div class="chart-title">压力位附近 (关注突破/回落)</div>
        <div style="overflow-x:auto;max-height:400px;overflow-y:auto"><table class="trade-tbl" id="resistScanTable">
          <thead><tr><th>代码</th><th>现价</th><th>压力价</th><th>距离%</th><th>支撑价</th><th>盈亏比</th></tr></thead>
          <tbody></tbody>
        </table></div>
      </div>
    </div>
    <div class="chart-box">
      <div class="chart-title">盈亏比最优</div>
      <div style="overflow-x:auto;max-height:400px;overflow-y:auto"><table class="trade-tbl" id="bestRrTable">
        <thead><tr><th>代码</th><th>现价</th><th>支撑价</th><th>压力价</th><th>盈亏比</th><th>评级</th></tr></thead>
        <tbody></tbody>
      </table></div>
    </div>
  </div>

  <!-- TAB: Sector Heatmap (NEW) -->
  <div class="tab-panel" id="tabHeatmap">
    <div class="chart-box"><div class="chart-title">板块支撑/压力分布</div><div class="chart-div-lg" id="chart_sector"></div></div>
    <div class="chart-box">
      <div class="chart-title">板块详情</div>
      <div style="overflow-x:auto;max-height:360px;overflow-y:auto"><table class="trade-tbl" id="sectorTable">
        <thead><tr><th>行业</th><th>扫描数</th><th>支撑占比</th><th>压力占比</th><th>平均RR</th><th>状态</th></tr></thead>
        <tbody></tbody>
      </table></div>
    </div>
  </div>

  <!-- TAB: Trades -->
  <div class="tab-panel" id="tabTrades">
    <div class="chart-box"><div class="chart-title">交易记录</div>
      <div class="trade-wrap"><table class="trade-tbl" id="tradeTable">
        <thead><tr><th>入场日期</th><th>出场日期</th><th>盈亏%</th><th>原因</th><th>持仓天</th></tr></thead><tbody></tbody>
      </table></div>
      <div style="text-align:center;padding:12px;color:var(--dim);font-size:10px" id="tradeSummary"></div>
    </div>
  </div>

  <div class="footer-note">
    ⚠️ 基于公开历史数据自动生成 · 仅供学习研究参考 · 不构成投资建议 · 投资有风险入市需谨慎<br>
    POWERED BY AI · QUANT ANALYSIS SYSTEM · 2026
  </div>
</main>

<script>
const DATA = __DATA_JSON__;

// ═══════════ STARS ═══════════
(function(){{
  const c=document.getElementById('stars'),x=c.getContext('2d');
  let w,h,p=[];
  const C=['rgba(0,255,136,','rgba(139,92,246,','rgba(0,212,255,','rgba(255,68,102,','rgba(255,255,255,'];
  function rs(){{w=c.width=innerWidth;h=c.height=innerHeight}} rs();addEventListener('resize',rs);
  for(let i=0;i<240;i++)p.push({{x:Math.random()*w,y:Math.random()*h,vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,r:Math.random()*2.2+.3,color:C[Math.floor(Math.random()*C.length)],a:Math.random()*.5+.15}});
  let mx=-999,my=-999;addEventListener('mousemove',e=>{{mx=e.clientX;my=e.clientY}});
  function anim(){{
    x.clearRect(0,0,w,h);
    for(let o of p){{
      o.x+=o.vx;o.y+=o.vy;
      if(o.x<0)o.x=w;if(o.x>w)o.x=0;if(o.y<0)o.y=h;if(o.y>h)o.y=0;
      let dx=mx-o.x,dy=my-o.y,dt=Math.sqrt(dx*dx+dy*dy);
      if(dt<180){{o.x-=dx/dt*.6;o.y-=dy/dt*.6}}
      x.beginPath();x.arc(o.x,o.y,o.r,0,Math.PI*2);x.fillStyle=o.color+o.a+')';x.fill();
    }}
    for(let i=0;i<p.length;i++)for(let j=i+1;j<p.length;j++){{
      let dx=p[i].x-p[j].x,dy=p[i].y-p[j].y,dt=Math.sqrt(dx*dx+dy*dy);
      if(dt<100){{x.beginPath();x.moveTo(p[i].x,p[i].y);x.lineTo(p[j].x,p[j].y);x.strokeStyle='rgba(0,255,136,'+(.08*(1-dt/100))+')';x.lineWidth=.5;x.stroke()}}
    }}
    requestAnimationFrame(anim);
  }}anim();
}})();

// ═══════════ TABS ═══════════
(function(){{
  document.querySelectorAll('.tab-btn').forEach(b=>b.addEventListener('click',()=>{{
    document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    const tid='tab'+b.dataset.tab.charAt(0).toUpperCase()+b.dataset.tab.slice(1);
    const panel=document.getElementById(tid);
    panel.classList.add('active');
    // Trigger Plotly resize after tab becomes visible
    setTimeout(()=>window.dispatchEvent(new Event('resize')),150);
    // Render charts in this tab if not yet rendered
    if(b.dataset.tab==='charts' && !window._chartsRendered) renderAllCharts();
    if(b.dataset.tab==='heatmap') renderChart('chart_sector', DATA.charts.sector);
    if(b.dataset.tab==='decision') renderChart('chart_confidence', DATA.charts.confidence);
  }}));
  // Render overview chart immediately
  renderChart('chart_kline', DATA.charts.kline);
}})();

// ═══════════ PLOTLY RENDERER ═══════════
// Uses Plotly.react which handles updates gracefully
function renderChart(id, cfg) {{
  if (!cfg || !cfg.data) return;
  var el = document.getElementById(id);
  if (!el) return;
  // Ensure container is visible for proper sizing
  var panel = el.closest('.tab-panel');
  if (panel && !panel.classList.contains('active')) {{
    // Save for later render
    if (!el._pendingCfg) el._pendingCfg = cfg;
    return;
  }}
  try {{
    Plotly.react(el, cfg.data, cfg.layout || {{}}, {{
      responsive: true, displayModeBar: false,
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
    }});
  }} catch(e) {{
    console.error('Chart error:', id, e);
    el.innerHTML = '<div style="color:var(--dim);text-align:center;padding:40px">图表渲染中...</div>';
  }}
}}

function renderAllCharts() {{
  if (window._chartsRendered) return;
  window._chartsRendered = true;
  var charts = DATA.charts;
  var map = {{
    chart_equity: charts.equity,
    chart_density: charts.density,
    chart_pie: charts.pie,
    chart_heatmap: charts.heatmap,
    chart_3d: charts.surface3d,
    chart_bar: charts.bar,
    chart_radar: charts.radar,
    chart_monthly: charts.monthly,
    chart_sector: charts.sector,
    chart_confidence: charts.confidence,
  }};
  Object.keys(map).forEach(function(id) {{
    if (map[id]) renderChart(id, map[id]);
  }});
  setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 300);
}}

// ═══════════ KPI ═══════════
(function(){{
  var m=DATA.metrics,rr=DATA.rr;
  var kpis=[
    {{l:'总收益率',v:m.total_return_pct||0,s:'%',c:m.total_return_pct>=0?'up':'down',sub:'基准 '+(m.benchmark_return_pct||0).toFixed(1)+'%'}},
    {{l:'夏普比率',v:m.sharpe_ratio||0,s:'',c:m.sharpe_ratio>=1?'up':m.sharpe_ratio>=0?'neutral':'down',sub:'年化 '+(m.annual_return_pct||0).toFixed(1)+'%'}},
    {{l:'最大回撤',v:Math.abs(m.max_drawdown_pct||0),s:'%',c:'down',sub:'持续'+(m.max_dd_days||0)+'天'}},
    {{l:'胜率',v:m.win_rate_pct||0,s:'%',c:m.win_rate_pct>=50?'up':'down',sub:'共'+(m.total_trades||0)+'笔'}},
    {{l:'盈亏比',v:parseFloat(m.profit_factor)||0,s:'',c:parseFloat(m.profit_factor)>=2?'up':'neutral',sub:'持仓'+(m.avg_holding_days||0).toFixed(0)+'天'}},
    {{l:'RR评级',v:rr.risk_reward_ratio||0,s:'',c:rr.risk_reward_ratio>=2?'up':rr.risk_reward_ratio>=1.5?'neutral':'down',sub:rr.quality||''}},
  ];
  var grid=document.getElementById('kpiGrid');
  kpis.forEach(function(k,i){{
    var card=document.createElement('div');card.className='kpi-card';
    var disp=k.v%1===0?Math.round(k.v):k.v.toFixed(2);
    card.innerHTML='<div class="kpi-label">'+k.l+'</div><div class="kpi-value '+k.c+'" data-v="'+disp+'" data-s="'+k.s+'">0'+k.s+'</div><div class="kpi-sub">'+k.sub+'</div>';
    grid.appendChild(card);
  }});
  function cu(el,tgt,sfx,ms){{
    var s=0,st=tgt/(ms/16);
    function tk(){{s+=st;if((st>0&&s>=tgt)||(st<0&&s<=tgt))s=tgt;el.textContent=(tgt%1===0?Math.round(s):s.toFixed(2))+sfx;if(s!==tgt)requestAnimationFrame(tk)}}
    tk();
  }}
  setTimeout(function(){{document.querySelectorAll('.kpi-value[data-v]').forEach(function(el){{cu(el,parseFloat(el.dataset.v),el.dataset.s||'',1800)}})}},400);
}})();

// ═══════════ ZONE LIST ═══════════
(function(){{
  var c=document.getElementById('zoneList');
  if(DATA.zones.length===0){{c.innerHTML='<div style="color:var(--dim);text-align:center;padding:30px">暂无检测到的密集成交区</div>';return}}
  DATA.zones.forEach(function(z){{
    var d=document.createElement('div');d.className='zone-card '+z.type;
    d.innerHTML='<div class="z-type">'+(z.type==='support'?'🟢 支撑区 SUPPORT':'🔴 压力区 RESISTANCE')+'</div><div class="z-price">¥'+z.center.toFixed(2)+'</div><div class="z-range">区间 ¥'+z.low.toFixed(2)+' – ¥'+z.high.toFixed(2)+'</div><div class="z-meta"><span>强度 '+z.strength.toFixed(2)+'</span><span>距离 '+z.dist_pct.toFixed(1)+'%</span><span>重叠 '+z.touch_count+'次</span></div>';
    c.appendChild(d);
  }});
}})();

// ═══════════ TRADE TABLE ═══════════
(function(){{
  var trades=DATA.recent_trades||[];
  var tbody=document.querySelector('#tradeTable tbody');
  if(trades.length===0){{tbody.innerHTML='<tr><td colspan="5" style="text-align:center;color:var(--dim);padding:24px">暂无交易记录</td></tr>';return}}
  var rm={{take_profit:'止盈',stop_loss:'止损',time_exit:'时间退出',force_exit:'强制平仓'}};
  trades.forEach(function(t){{
    var pnl=t.pnl||0,cls=pnl>=0?'pnl-up':'pnl-down';
    tbody.innerHTML+='<tr><td>'+(t.entry||'-')+'</td><td>'+(t.exit||'-')+'</td><td class="'+cls+'">'+(pnl>=0?'+':'')+pnl.toFixed(2)+'%</td><td>'+(rm[t.reason]||t.reason||'-')+'</td><td>'+(t.days||'-')+'天</td></tr>';
  }});
  var wins=trades.filter(function(t){{return (t.pnl||0)>0}}).length;
  document.getElementById('tradeSummary').textContent='共 '+trades.length+' 笔 · 盈利 '+wins+' 笔 · 胜率 '+(trades.length>0?(wins/trades.length*100).toFixed(0):0)+'%';
}})();

// ═══════════ DECISION SUPPORT ═══════════
(function(){{
  var rs=DATA.risk_score||{{}};
  var score=rs.overall_score||50;
  var color=score>=70?'var(--g)':score>=40?'var(--c)':'var(--r)';
  document.getElementById('riskScore').innerHTML=
    '<div style="text-align:center;padding:20px">'+
    '<div style="font-family:Orbitron;font-size:48px;font-weight:900;color:'+color+'">'+(rs.level_label||'🟡 观望')+'</div>'+
    '<div style="margin-top:12px;font-size:14px;color:var(--text)">'+(rs.summary||'')+'</div>'+
    '<div style="margin-top:8px;font-size:11px;color:var(--dim)">盈亏比: '+(rs.rr_ratio||0).toFixed(1)+' | 综合评分: '+score+'/100</div>'+
    '<div style="margin-top:16px;background:rgba(255,255,255,0.04);border-radius:8px;height:8px">'+
    '<div style="background:'+color+';height:100%;border-radius:8px;width:'+score+'%;transition:width 2s ease"></div></div></div>';

  // Touch alerts
  var alerts=DATA.touch_alerts||[];
  var ac=document.getElementById('touchAlerts');
  if(alerts.length===0){{ac.innerHTML='<div style="color:var(--dim);text-align:center;padding:24px">暂无预警</div>'}}
  else{{
    var ah='';
    alerts.forEach(function(a){{
      var lc=a.alert_level==='danger'?'var(--r)':a.alert_level==='warning'?'var(--c)':'var(--dim)';
      ah+='<div style="border-left:3px solid '+lc+';padding:10px 14px;margin-bottom:10px;background:var(--card);border-radius:8px">'+
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'+
        '<span style="font-size:11px;color:'+lc+'">'+a.alert_text+'</span>'+
        '<span style="font-family:Orbitron;font-size:14px;font-weight:700">¥'+a.zone_center.toFixed(2)+'</span></div>'+
        '<div style="font-size:10px;color:var(--dim);margin-bottom:6px">'+a.zone_label+' · 距离 '+a.distance_pct+'% · ¥'+a.distance_price.toFixed(2)+'</div>'+
        '<div style="background:rgba(255,255,255,0.04);border-radius:4px;height:4px;margin-bottom:6px">'+
        '<div style="background:'+lc+';height:100%;border-radius:4px;width:'+a.progress+'%"></div></div>'+
        '<div style="font-size:10px;color:var(--dim)">'+a.advice+'</div>'+
        (a.history_rate>0?'<div style="font-size:9px;color:var(--dim);margin-top:4px">📊 '+a.history_label+'</div>':'')+
        '</div>';
    }});
    ac.innerHTML=ah;
  }}

  // Zone history table
  var hist=DATA.zone_history||[];
  var ht=document.querySelector('#historyTable tbody');
  if(hist.length===0){{ht.innerHTML='<tr><td colspan="7" style="text-align:center;color:var(--dim);padding:16px">暂无历史数据</td></tr>'}}
  else{{
    hist.forEach(function(h){{
      var ev=h.rate>=70?'🌟🌟🌟':h.rate>=50?'🌟🌟':h.rate>0?'🌟':'—';
      ht.innerHTML+='<tr><td>'+(h.type==='support'?'🟢 支撑':'🔴 压力')+'</td><td>¥'+h.center.toFixed(2)+'</td>'+
        '<td>'+h.held+'</td><td>'+h.broke+'</td><td>'+h.total+'</td>'+
        '<td style="color:'+(h.rate>=50?'var(--g)':'var(--r)')+'">'+h.rate+'%</td><td>'+ev+'</td></tr>';
    }});
  }}

  // Confidence stats
  var bc=DATA.backtest_confidence||{{}};
  var cs=document.getElementById('confidenceStats');
  var cl=bc.confidence_level||'low';
  var cc=cl==='high'?'var(--g)':cl==='medium'?'var(--c)':'var(--r)';
  cs.innerHTML=
    '<div style="text-align:center;padding:16px">'+
    '<div style="font-family:Orbitron;font-size:36px;font-weight:900;color:'+cc+'">'+(bc.win_rate||0).toFixed(0)+'%</div>'+
    '<div style="font-size:12px;color:var(--text);margin-top:4px">'+(bc.confidence_label||'—')+'</div>'+
    '<div style="margin-top:16px;font-size:11px;color:var(--dim)">'+(bc.summary||'暂无数据')+'</div>'+
    '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:14px">'+
    '<div style="background:var(--card);border-radius:8px;padding:10px"><div style="font-size:9px;color:var(--dim);margin-bottom:2px">交易数</div><div style="font-family:Orbitron;font-size:16px;color:var(--c)">'+(bc.similar_count||0)+'</div></div>'+
    '<div style="background:var(--card);border-radius:8px;padding:10px"><div style="font-size:9px;color:var(--dim);margin-bottom:2px">平均收益</div><div style="font-family:Orbitron;font-size:16px;color:'+(bc.avg_return>=0?'var(--g)':'var(--r)')+'">'+(bc.avg_return>=0?'+':'')+(bc.avg_return||0).toFixed(1)+'%</div></div>'+
    '<div style="background:var(--card);border-radius:8px;padding:10px"><div style="font-size:9px;color:var(--dim);margin-bottom:2px">最大回撤</div><div style="font-family:Orbitron;font-size:16px;color:var(--r)">'+(bc.min_return||0).toFixed(1)+'%</div></div></div></div>';
}})();

// ═══════════ SCANNER DATA ═══════════
(function(){{
  var scan=DATA.scanner||{{}};
  var sum=scan.summary||{{}};
  var sg=document.getElementById('scanKpiGrid');
  sg.innerHTML='<div class="kpi-card"><div class="kpi-label">扫描总数</div><div class="kpi-value neutral">'+(sum.total_scanned||0)+'</div></div>'+
    '<div class="kpi-card"><div class="kpi-label">支撑附近</div><div class="kpi-value up">'+(sum.near_support_count||0)+'</div><div class="kpi-sub">'+((sum.support_ratio_pct||0).toFixed(0))+'%</div></div>'+
    '<div class="kpi-card"><div class="kpi-label">压力附近</div><div class="kpi-value down">'+(sum.near_resistance_count||0)+'</div><div class="kpi-sub">'+((sum.resistance_ratio_pct||0).toFixed(0))+'%</div></div>'+
    '<div class="kpi-card"><div class="kpi-label">最优盈亏比</div><div class="kpi-value up">'+(sum.best_rr_count||0)+'</div><div class="kpi-sub">RR≥2.0</div></div>'+
    '<div class="kpi-card"><div class="kpi-label">平均RR</div><div class="kpi-value neutral">'+(sum.avg_rr||0).toFixed(1)+'</div></div>';

  function fillTable(tid,data,cols){{
    var tb=document.querySelector('#'+tid+' tbody');
    tb.innerHTML='';
    if(!data||data.length===0){{tb.innerHTML='<tr><td colspan="'+cols+'" style="text-align:center;color:var(--dim);padding:16px">扫描中或暂无数据</td></tr>';return}}
    data.slice(0,30).forEach(function(r){{
      var row='<tr>';
      if(cols>=6)row+='<td>'+r.code+'</td><td>¥'+r.price+'</td><td>¥'+(r.nearest_support||'-')+'</td><td style="color:var(--g)">'+(r.support_dist_pct||'-')+'%</td><td>¥'+(r.nearest_resistance||'-')+'</td><td style="color:var(--c)">'+(r.rr_ratio||0).toFixed(1)+'</td>';
      tb.innerHTML+=row+'</tr>';
    }});
  }}
  fillTable('supportScanTable',scan.near_support,6);
  (function(){{
    var tb=document.querySelector('#resistScanTable tbody');tb.innerHTML='';
    var data=scan.near_resistance||[];
    if(data.length===0){{tb.innerHTML='<tr><td colspan="6" style="text-align:center;color:var(--dim);padding:16px">扫描中或暂无数据</td></tr>';return}}
    data.slice(0,30).forEach(function(r){{tb.innerHTML+='<tr><td>'+r.code+'</td><td>¥'+r.price+'</td><td>¥'+(r.nearest_resistance||'-')+'</td><td style="color:var(--r)">'+(r.resistance_dist_pct||'-')+'%</td><td>¥'+(r.nearest_support||'-')+'</td><td style="color:var(--c)">'+(r.rr_ratio||0).toFixed(1)+'</td></tr>'}});
  }})();
  (function(){{
    var tb=document.querySelector('#bestRrTable tbody');tb.innerHTML='';
    var data=scan.best_rr||[];
    if(data.length===0){{tb.innerHTML='<tr><td colspan="6" style="text-align:center;color:var(--dim);padding:16px">扫描中或暂无数据</td></tr>';return}}
    data.slice(0,30).forEach(function(r){{tb.innerHTML+='<tr><td>'+r.code+'</td><td>¥'+r.price+'</td><td>¥'+(r.nearest_support||'-')+'</td><td>¥'+(r.nearest_resistance||'-')+'</td><td style="color:var(--c);font-weight:700">'+(r.rr_ratio||0).toFixed(1)+'</td><td>'+(r.rr_quality||'')+'</td></tr>'}});
  }})();
}})();

// ═══════════ SECTOR HEATMAP ═══════════
(function(){{
  var sectors=DATA.sectors||[];
  var st=document.querySelector('#sectorTable tbody');
  if(sectors.length===0){{st.innerHTML='<tr><td colspan="6" style="text-align:center;color:var(--dim);padding:16px">扫描中或暂无数据</td></tr>'}}
  else{{
    sectors.forEach(function(s){{
      var sc=s.status==='偏多'?'color:var(--g)':s.status==='偏空'?'color:var(--r)':'color:var(--c)';
      st.innerHTML+='<tr><td>'+s.industry+'</td><td>'+s.scanned+'</td>'+
        '<td style="color:var(--g)">'+s.support_pct+'%</td>'+
        '<td style="color:var(--r)">'+s.resistance_pct+'%</td>'+
        '<td>'+s.avg_rr+'</td><td style="'+sc+'">'+s.status+'</td></tr>';
    }});
  }}
}})();
</script>
</body>
</html>'''


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def generate(code, window, n_zones, start, end):
    print(f"\n{'='*56}\n  🎬 生成视频级仪表盘 v4: {code}\n{'='*56}\n")

    print("[1/5] 运行全量分析...")
    data = gather_data(code, window, n_zones, start, end)
    print(f"  ✓ {data['name']} · ¥{data['current_price']:.2f} · {len(data['zones'])}区 · {data['trades_count']}笔")

    print("[2/5] 市场扫描 (100只)...")
    try:
        scan_results = scan_market(top_n=100, window=window, n_zones=n_zones)
        data["scanner"] = {
            "near_support": scan_results["near_support"][:20],
            "near_resistance": scan_results["near_resistance"][:20],
            "best_rr": scan_results["best_rr"][:20],
            "summary": scan_summary(scan_results),
        }
        print(f"  ✓ 支撑{len(scan_results['near_support'])}只 压力{len(scan_results['near_resistance'])}只 RR优{len(scan_results['best_rr'])}只")
    except Exception as e:
        print(f"  ⚠ 扫描跳过: {e}")
        data["scanner"] = {"near_support": [], "near_resistance": [], "best_rr": [], "summary": {}}

    print("[3/5] 板块热力 (100只)...")
    try:
        sectors = scan_sector_heatmap(stock_count=100, window=window, n_zones=n_zones)
        data["sectors"] = sectors
        print(f"  ✓ {len(sectors)}个行业")
    except Exception as e:
        print(f"  ⚠ 板块跳过: {e}")
        data["sectors"] = []

    print("[4/5] 生成图表...")
    charts = {}
    charts["kline"] = chart_kline_zones(data)
    charts["equity"] = chart_equity(data)
    charts["density"] = chart_density(data)
    charts["pie"] = chart_pie_pnl(data)
    charts["heatmap"] = chart_heatmap_density(data)
    charts["surface3d"] = chart_3d_surface(data)
    charts["bar"] = chart_zone_bar(data)
    charts["radar"] = chart_radar(data)
    charts["monthly"] = chart_monthly_heatmap(data)
    # NEW: sector heatmap chart
    charts["sector"] = chart_sector_heatmap(data)
    charts["confidence"] = chart_confidence(data)
    data["charts"] = charts
    names = [k for k, v in charts.items() if v is not None]
    print(f"  ✓ {len(names)}个图表: {', '.join(names)}")

    print("[5/5] 渲染 HTML...")
    data_json = json.dumps(data, ensure_ascii=False, default=str)

    html = HTML_V3
    html = html.replace("__NAME__", data["name"])
    html = html.replace("__CODE__", data["code"])
    html = html.replace("__WINDOW__", str(data["window"]))
    html = html.replace("__CURRENT_PRICE__", f"{data['current_price']:.2f}")
    html = html.replace("__DATE__", datetime.now().strftime("%Y-%m-%d %H:%M"))
    html = html.replace("{{", "{").replace("}}", "}")
    html = html.replace("__DATA_JSON__", data_json)

    filename = f"ultimate_{code}_{window}d.html"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"[完成!]")
    print(f"\n{'='*56}\n  ✅ {path}\n  📏 {os.path.getsize(path)/1024:.0f} KB\n  🌐 浏览器打开即可\n{'='*56}\n")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="视频级仪表盘生成器 v3")
    p.add_argument("code", nargs="?", default="000001")
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--n-zones", type=int, default=5)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2025-12-31")
    a = p.parse_args()
    generate(a.code, a.window, a.n_zones, a.start, a.end)
