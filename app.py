"""
支撑/阻力位检测 Web 应用

基于 Flask 的 Web 服务：
- GET /                  首页（交互式分析界面）
- GET /api/analyze        单股支撑/阻力位检测，返回 JSON
- GET /api/stock_search  股票代码/名称搜索（自动补全）

运行：
    python app.py
然后浏览器访问 http://127.0.0.1:5000
"""

import os
import sys
import threading
from datetime import datetime, timedelta

from flask import Flask, render_template, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_fetcher import DataFetcher, bs_code, short_code
from src.density_analyzer import (
    find_dense_zones, classify_zones, multi_timeframe_zones,
    calc_risk_reward, compute_density_profile, ZoneInfo,
)
from src.zone_history import (
    analyze_zone_history, generate_touch_alerts, compute_risk_score,
)

app = Flask(__name__)

# baostock 登录为全局状态，需加锁串行化访问
_fetcher_lock = threading.Lock()
_fetcher: DataFetcher | None = None


def get_fetcher() -> DataFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher


def _date_range() -> tuple[str, str]:
    """默认数据时间范围：今天向前推 3 年"""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=config.DEFAULT_START_OFFSET_DAYS)).strftime("%Y-%m-%d")
    return start, end


def _zone_to_dict(z: ZoneInfo) -> dict:
    return {
        "center": z.center,
        "low": z.low,
        "high": z.high,
        "strength": z.strength,
        "touch_count": z.touch_count,
        "volume_pct": z.volume_pct,
        "zone_type": z.zone_type,
        "distance_pct": z.distance_pct,
    }


def _kline_to_json(df, n_bars: int = 150) -> dict:
    """把 K 线 DataFrame 转为前端可用的 JSON（取最近 n_bars 根）"""
    recent = df.tail(n_bars)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in recent["date"].tolist()],
        "open": [round(float(x), 4) for x in recent["open"].tolist()],
        "high": [round(float(x), 4) for x in recent["high"].tolist()],
        "low": [round(float(x), 4) for x in recent["low"].tolist()],
        "close": [round(float(x), 4) for x in recent["close"].tolist()],
        "volume": [int(x) for x in recent["volume"].tolist()],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze")
def api_analyze():
    code = (request.args.get("code") or "").strip()
    if not code:
        return jsonify({"error": "请输入股票代码"}), 400

    try:
        window = int(request.args.get("window", config.DEFAULT_WINDOW))
    except ValueError:
        window = config.DEFAULT_WINDOW
    try:
        n_zones = int(request.args.get("n_zones", config.DEFAULT_N_ZONES))
    except ValueError:
        n_zones = config.DEFAULT_N_ZONES
    use_mtf = request.args.get("mtf", "true").lower() in ("true", "1", "yes")

    window = max(10, min(window, 250))
    n_zones = max(1, min(n_zones, 8))

    try:
        bsc = bs_code(code)
    except Exception:
        return jsonify({"error": f"无效的股票代码: {code}"}), 400
    short = short_code(bsc)

    start, end = _date_range()

    with _fetcher_lock:
        fetcher = get_fetcher()
        try:
            # 日线
            df_day = fetcher.get_kline(bsc, freq="d", start=start, end=end)
            if len(df_day) < window:
                return jsonify({
                    "error": f"数据不足：仅 {len(df_day)} 条日线，需要 ≥ {window} 条"
                }), 400

            stock_name = fetcher.get_stock_name(bsc)
            current_price = float(df_day["close"].iloc[-1])
            latest_pct = float(df_day["pct_chg"].iloc[-1]) if "pct_chg" in df_day.columns else 0.0

            # 主分析：日线密集区
            zones = find_dense_zones(df_day, window=window, n_zones=n_zones)
            classify_zones(zones, current_price)

            # 密度剖面
            window_df = df_day.tail(window)
            price_levels, density, pmin, pmax = compute_density_profile(window_df)

            # 历史胜率 & 预警 & 风险
            histories = analyze_zone_history(df_day, window=window, n_zones=n_zones)
            alerts = generate_touch_alerts(current_price, zones, histories)
            risk_score = compute_risk_score(alerts)
            risk_reward = calc_risk_reward(current_price, zones)

            # 多时间框架（可选）
            mtf = None
            if use_mtf:
                try:
                    df_week = fetcher.get_kline(bsc, freq="w", start=start, end=end)
                    df_month = fetcher.get_kline(bsc, freq="m", start=start, end=end)
                    mtf_raw = multi_timeframe_zones(df_day, df_week, df_month, n_zones=n_zones)
                    mtf = {tf: [_zone_to_dict(z) for z in zs] for tf, zs in mtf_raw.items()}
                except Exception as e:
                    mtf = {"error": f"多时间框架分析失败: {e}"}

            result = {
                "stock_code": short,
                "stock_name": stock_name,
                "current_price": round(current_price, 4),
                "change_pct": round(latest_pct, 2),
                "window": window,
                "n_zones": n_zones,
                "data_bars": len(df_day),
                "kline": _kline_to_json(df_day, n_bars=150),
                "zones": [_zone_to_dict(z) for z in zones],
                "density": {
                    "price_levels": [round(float(x), 4) for x in price_levels.tolist()],
                    "density": [round(float(x), 6) for x in density.tolist()],
                    "price_min": round(float(pmin), 4),
                    "price_max": round(float(pmax), 4),
                },
                "history": [{
                    "center": h.center,
                    "zone_type": h.zone_type,
                    "held_count": h.held_count,
                    "broke_count": h.broke_count,
                    "touch_count": h.touch_count,
                    "success_rate": h.success_rate,
                } for h in histories],
                "alerts": alerts,
                "risk_score": risk_score,
                "risk_reward": risk_reward,
                "mtf": mtf,
            }
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": f"分析失败: {e}"}), 500


@app.route("/api/stock_search")
def api_stock_search():
    """股票代码/名称搜索（自动补全）"""
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"results": []})

    with _fetcher_lock:
        fetcher = get_fetcher()
        try:
            sl = fetcher.get_stock_list()
        except Exception as e:
            return jsonify({"error": f"获取股票列表失败: {e}"}), 500

    ql = q.lower()
    mask = sl["code"].str.lower().str.contains(ql, na=False) | \
           sl["code_name"].str.lower().str.contains(ql, na=False)
    matches = sl[mask].head(15)
    results = [{
        "code": str(r["code"]),
        "short": short_code(str(r["code"])),
        "name": str(r["code_name"]),
    } for _, r in matches.iterrows()]
    return jsonify({"results": results})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  支撑/阻力位检测 Web 应用")
    print("  访问: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
