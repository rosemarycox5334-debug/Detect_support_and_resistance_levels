"""
支撑/阻力位检测 Web 应用（本地期货数据版）

数据源：用户指定的本地文件夹，内含 {品种}主连_{周期}.parquet 文件
- GET /                          首页
- GET /api/data_dir              获取/设置数据文件夹路径
- POST /api/data_dir             设置数据文件夹路径
- GET /api/instruments           列出当前数据文件夹内全部品种及可用周期
- GET /api/analyze               单品种分析
- GET /api/analyze_all           一键分析全部品种（返回汇总）
"""

import os
import sys
import json
import threading
from typing import Optional

from flask import Flask, render_template, jsonify, request, session

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.local_data_loader import (
    list_instruments, load_kline, iter_all_instruments,
    TIMEFRAME_LABELS, pick_mtf_timeframes,
)
from src.density_analyzer import (
    find_dense_zones, classify_zones,
    ZoneInfo,
)
from src.zone_history import (
    analyze_zone_history, generate_touch_alerts, compute_risk_score,
    DIRECTION_LABELS,
)

app = Flask(__name__)
app.secret_key = "density-sr-local-futures"

# 全局共享状态：当前选定的数据文件夹
_state_lock = threading.Lock()
_state = {"data_dir": ""}


def get_data_dir() -> str:
    return _state["data_dir"]


def set_data_dir(path: str):
    _state["data_dir"] = path


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
    recent = df.tail(n_bars)
    return {
        "dates": [d.strftime("%Y-%m-%d %H:%M") if hasattr(d, "hour") else d.strftime("%Y-%m-%d")
                  for d in recent["date"].tolist()],
        "open": [round(float(x), 4) for x in recent["open"].tolist()],
        "high": [round(float(x), 4) for x in recent["high"].tolist()],
        "low": [round(float(x), 4) for x in recent["low"].tolist()],
        "close": [round(float(x), 4) for x in recent["close"].tolist()],
        "volume": [int(x) for x in recent["volume"].tolist()],
    }


def _analyze_df(df, window: int, n_zones: int, symbol: str, tf: str,
                do_mtf: bool = True, mtf_frames=None,
                direction: str = "long") -> dict:
    """对单个 DataFrame 执行完整分析，返回结果 dict（不含 K 线大对象，便于批量调用）"""
    if len(df) < window:
        raise ValueError(f"数据不足：仅 {len(df)} 条，需要 ≥ {window} 条")

    current_price = float(df["close"].iloc[-1])
    latest_pct = float(df["pct_chg"].iloc[-1]) if "pct_chg" in df.columns else 0.0

    zones = find_dense_zones(df, window=window, n_zones=n_zones)
    classify_zones(zones, current_price)

    histories = analyze_zone_history(df, window=window, n_zones=n_zones)
    alerts = generate_touch_alerts(current_price, zones, histories)
    risk_score = compute_risk_score(
        alerts, df=df, zones=zones, histories=histories, direction=direction
    )

    # 多周期关键位：不合并、不加权，仅用于主图叠加显示
    mtf_zones = []
    if do_mtf and mtf_frames:
        for fr in mtf_frames:
            code = fr["tf"]
            if code == tf:
                continue  # 主周期已在 zones 中
            ndf = fr.get("df")
            if ndf is None or len(ndf) < 5:
                continue
            try:
                nprice = float(ndf["close"].iloc[-1])
                nzs = find_dense_zones(ndf, window=min(fr.get("window", window), len(ndf)), n_zones=n_zones)
                classify_zones(nzs, nprice)
                mtf_zones.append({
                    "tf": code,
                    "tf_label": TIMEFRAME_LABELS.get(code, code),
                    "zones": [_zone_to_dict(z) for z in nzs],
                })
            except Exception:
                pass

    return {
        "stock_name": symbol,
        "stock_code": f"{symbol}主连",
        "current_price": round(current_price, 4),
        "change_pct": round(latest_pct, 2),
        "window": window,
        "n_zones": n_zones,
        "data_bars": len(df),
        "zones": [_zone_to_dict(z) for z in zones],
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
        "mtf_zones": mtf_zones,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data_dir", methods=["GET", "POST"])
def api_data_dir():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        path = (data.get("path") or "").strip()
        if not path:
            return jsonify({"error": "未提供路径"}), 400
        if not os.path.isdir(path):
            return jsonify({"error": f"文件夹不存在: {path}"}), 400
        set_data_dir(path)
        return jsonify({"ok": True, "path": path})
    # GET
    path = get_data_dir()
    return jsonify({"path": path, "set": bool(path)})


@app.route("/api/pick_dir")
def api_pick_dir():
    """弹出系统原生文件夹选择对话框，返回所选绝对路径（本地应用专用）"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="选择数据文件夹")
        root.destroy()
        return jsonify({"path": path or ""})
    except Exception as e:
        return jsonify({"error": f"无法打开文件夹选择器: {e}", "path": ""}), 500


@app.route("/api/instruments")
def api_instruments():
    data_dir = get_data_dir()
    if not data_dir:
        return jsonify({"error": "请先选择数据文件夹"}), 400
    if not os.path.isdir(data_dir):
        return jsonify({"error": f"文件夹不存在: {data_dir}"}), 400
    instruments = list_instruments(data_dir)
    return jsonify({
        "data_dir": data_dir,
        "count": len(instruments),
        "instruments": [{"symbol": s, "timeframes": tfs} for s, tfs in instruments.items()],
        "tf_labels": TIMEFRAME_LABELS,
    })


@app.route("/api/analyze")
def api_analyze():
    data_dir = get_data_dir()
    if not data_dir:
        return jsonify({"error": "请先选择数据文件夹"}), 400

    symbol = (request.args.get("symbol") or "").strip()
    if not symbol:
        return jsonify({"error": "请选择品种"}), 400

    tf = (request.args.get("tf") or "D1").strip().upper()
    try:
        window = int(request.args.get("window", config.DEFAULT_WINDOW))
    except ValueError:
        window = config.DEFAULT_WINDOW
    try:
        n_zones = int(request.args.get("n_zones", config.DEFAULT_N_ZONES))
    except ValueError:
        n_zones = config.DEFAULT_N_ZONES
    use_mtf = request.args.get("mtf", "true").lower() in ("true", "1", "yes")
    direction = (request.args.get("direction") or "long").strip().lower()
    if direction not in DIRECTION_LABELS:
        direction = "long"

    window = max(10, min(window, 500))
    n_zones = max(1, min(n_zones, 8))

    try:
        df = load_kline(data_dir, symbol, tf)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"读取数据失败: {e}"}), 500

    # 多时间框架：自适应选择邻居周期并加载
    mtf_frames = None
    if use_mtf:
        instruments = list_instruments(data_dir)
        available_tfs = instruments.get(symbol, [])
        neighbors = pick_mtf_timeframes(tf, available_tfs)
        mtf_frames = []
        for ntf, nweight, nwin in neighbors:
            if ntf == tf:
                mtf_frames.append({"tf": ntf, "df": df, "weight": nweight, "window": nwin})
            else:
                try:
                    ndf = load_kline(data_dir, symbol, ntf)
                    if len(ndf) >= 3:
                        mtf_frames.append({"tf": ntf, "df": ndf, "weight": nweight, "window": nwin})
                except Exception:
                    pass  # 该周期数据不存在或读取失败，跳过

    try:
        result = _analyze_df(df, window=window, n_zones=n_zones,
                             symbol=symbol, tf=tf,
                             do_mtf=use_mtf, mtf_frames=mtf_frames,
                             direction=direction)
        result["tf"] = tf
        result["tf_label"] = TIMEFRAME_LABELS.get(tf, tf)
        result["kline"] = _kline_to_json(df, n_bars=150)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"分析失败: {e}"}), 500


@app.route("/api/analyze_all")
def api_analyze_all():
    """一键分析全部品种，返回汇总（不含完整 K 线/density，仅关键指标）"""
    data_dir = get_data_dir()
    if not data_dir:
        return jsonify({"error": "请先选择数据文件夹"}), 400

    tf = (request.args.get("tf") or "D1").strip().upper()
    try:
        window = int(request.args.get("window", config.DEFAULT_WINDOW))
    except ValueError:
        window = config.DEFAULT_WINDOW
    try:
        n_zones = int(request.args.get("n_zones", config.DEFAULT_N_ZONES))
    except ValueError:
        n_zones = config.DEFAULT_N_ZONES
    direction = (request.args.get("direction") or "long").strip().lower()
    if direction not in DIRECTION_LABELS:
        direction = "long"
    window = max(10, min(window, 500))
    n_zones = max(1, min(n_zones, 8))

    instruments = list_instruments(data_dir)
    rows = []
    errors = []

    for symbol, available_tfs in instruments.items():
        if tf not in available_tfs:
            continue
        try:
            df = load_kline(data_dir, symbol, tf)
            if len(df) < window:
                errors.append({"symbol": symbol, "reason": f"数据不足({len(df)}条)"})
                continue

            current_price = float(df["close"].iloc[-1])
            latest_pct = float(df["pct_chg"].iloc[-1]) if "pct_chg" in df.columns else 0.0
            zones = find_dense_zones(df, window=window, n_zones=n_zones)
            classify_zones(zones, current_price)
            histories = analyze_zone_history(df, window=window, n_zones=n_zones)
            alerts = generate_touch_alerts(current_price, zones, histories)
            risk_score = compute_risk_score(
                alerts, df=df, zones=zones, histories=histories, direction=direction
            )

            supports = [z for z in zones if z.zone_type == "support"]
            resistances = [z for z in zones if z.zone_type == "resistance"]
            nearest_sup = min(supports, key=lambda z: abs(z.distance_pct)) if supports else None
            nearest_res = min(resistances, key=lambda z: abs(z.distance_pct)) if resistances else None

            rows.append({
                "symbol": symbol,
                "tf": tf,
                "current_price": round(current_price, 4),
                "change_pct": round(latest_pct, 2),
                "n_zones": len(zones),
                "nearest_support": nearest_sup.center if nearest_sup else None,
                "nearest_support_dist_pct": nearest_sup.distance_pct if nearest_sup else None,
                "nearest_resistance": nearest_res.center if nearest_res else None,
                "nearest_resistance_dist_pct": nearest_res.distance_pct if nearest_res else None,
                "overall_score": risk_score["overall_score"],
                "level": risk_score["level"],
                "trend_label": risk_score["trend_label"],
                "success_rate": risk_score["success_rate"],
                "nearest_distance_pct": risk_score["nearest_distance_pct"],
                "direction": direction,
                "data_bars": len(df),
                "zones": [_zone_to_dict(z) for z in zones],
            })
        except Exception as e:
            errors.append({"symbol": symbol, "reason": str(e)})
            continue

    # 排序：综合评分从高到低
    rows.sort(key=lambda r: -(r["overall_score"] or 0))

    summary = {
        "total": len(rows) + len(errors),
        "analyzed": len(rows),
        "failed": len(errors),
        "tf": tf,
        "tf_label": TIMEFRAME_LABELS.get(tf, tf),
        "window": window,
        "n_zones": n_zones,
        "direction": direction,
        "direction_label": DIRECTION_LABELS.get(direction, direction),
        "data_dir": data_dir,
    }

    return jsonify({"summary": summary, "rows": rows, "errors": errors})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "data_dir": get_data_dir()})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  支撑/阻力位检测 Web 应用（本地期货数据版）")
    print("  访问: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
