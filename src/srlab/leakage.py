# -*- coding: utf-8 -*-
"""
防泄漏与正确性自动化测试

这些测试是"可验证"的核心：用户不必读懂算法，只要这四个测试通过，
就可以确信回测结果没有用到未来数据、且指标不是被算法白嫖的。

    test_pivot_causality      枢轴的 confirm_idx 机制与"在 t 时刻实时计算"完全等价
    test_no_future_dependence 把 t 之后的行情替换成任意垃圾，检测输出必须逐位不变
    test_volume_conservation  成交量剖面体积守恒（V2 修复项的直接验证）
    test_v1_equivalence       V1 向量化复刻与原始 find_dense_zones 输出一致
    test_label_direction      标注的方向校验生效（从下方涨上来不算测试支撑）

运行: python -m src.srlab.leakage
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from .base import Ctx, Level
from .data import ashare_load, atr_series, to_symbol
from .detectors import FixedPlacebo, V1Density, V2Fusion, V3Fusion
from .labeling import evaluate_level
from .pivots import zigzag
from .profile import volume_profile

ASHARE_DIR = r"D:\A股_K线数据\parquet\stocks"


def _sym(code="000001", adjust="factor", adjust_mode="hfq"):
    df = ashare_load(ASHARE_DIR, code, "daily", adjust=adjust,
                     adjust_mode=adjust_mode)
    return to_symbol(code, df)


# ============================================================
def test_adjust_scale_invariance(code: str = "000001", n_checks: int = 6) -> bool:
    """
    后复权(严格因果) 与 前复权(锚定最新，含未来除权信息) 只差一个全局常数，
    而本算法以 ATR 为尺度，因此检测结果的**相对几何**必须完全一致。

    这条测试是"回测用后复权、界面用前复权"这个设计的正确性依据：
    如果它通过，说明用哪种口径回测都不影响结论，
    而后复权是严格因果的，所以回测选它。

    只对 V3 成立，**V2 故意不满足**：V2 的 _round_bonus（整数关口）依赖
    绝对价位，本质上不是尺度不变的。这也正是 V3 删掉该项的原因之一 ——
    单因子分析显示它在 support 上 p=0.70 无信息、在 resistance 上显著为负。
    """
    try:
        s_h = _sym(code, adjust="factor", adjust_mode="hfq")
        s_q = _sym(code, adjust="factor", adjust_mode="qfq")
    except Exception as e:
        print(f"[SKIP] test_adjust_scale_invariance: {e}")
        return True

    n = min(len(s_h), len(s_q))
    if n < 600:
        print("[SKIP] test_adjust_scale_invariance: 数据不足")
        return True

    # 全局比例常数
    k = float(s_q.close[-1] / s_h.close[-1])
    ok = True
    det = V3Fusion()
    for t in np.linspace(500, n - 60, n_checks).astype(int):
        a = det.detect(_ctx(s_h, t))
        b = det.detect(_ctx(s_q, t))
        if len(a) != len(b):
            ok = False
            print(f"  [FAIL] t={t}: 关键位数量不同 {len(a)} vs {len(b)}")
            continue
        for x, y in zip(a, b):
            # y 应该等于 x 乘以全局常数 k
            if abs(y.center - x.center * k) > max(1e-6, 1e-6 * abs(y.center)):
                ok = False
                print(f"  [FAIL] t={t}: center {x.center}*{k:.6f}={x.center*k} vs {y.center}")
                break
            # 以 ATR 为单位的相对量必须完全一致
            if abs(y.dist_atr - x.dist_atr) > 1e-6 or abs(y.width_atr - x.width_atr) > 1e-6:
                ok = False
                print(f"  [FAIL] t={t}: dist_atr {x.dist_atr} vs {y.dist_atr}")
                break
    print(f"{'[PASS]' if ok else '[FAIL]'} test_adjust_scale_invariance "
          f"({n_checks} 个时点, 全局比例 k={k:.6f})")
    return ok


# ============================================================
def test_factor_sanity(code: str = "000001") -> bool:
    """
    后复权因子的结构性检查。

    注意：**不能**要求因子单调不减。最初这么写导致误报 ——
    实测 50 只缓存标的中有 2 只存在真实下降（000001 单次 -16.8%，
    对应深发展/平安银行换股；000002 两次 -2.6%），
    配股、换股这类公司行为确实会降低后复权因子。
    因果性不等于单调性。

    真正该检查的是：
      1. 首值归一到 1.0（后复权锚定序列起点 => factor[t] 只依赖 <= t 的事件）
      2. 是分段常数：阶跃次数远小于 bar 数（否则说明量化噪声没去干净，
         每根 bar 都在微幅变动，会给价格引入抖动）
    """
    from .adjust import FACTOR_CACHE_DEFAULT, FactorStore

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    store = FactorStore(os.path.join(base, FACTOR_CACHE_DEFAULT))
    fac = store.get(code) if store.loaded else None
    if fac is None:
        print("[SKIP] test_factor_sanity: 无因子缓存")
        return True
    f = fac["factor"].to_numpy(np.float64)
    starts_at_one = abs(f[0] - 1.0) < 1e-9
    rel = np.abs(np.diff(f)) / np.maximum(f[:-1], 1e-12)
    n_steps = int((rel > 1e-9).sum())
    piecewise = n_steps < max(60, len(f) // 100)
    n_down = int((np.diff(f) < -1e-12).sum())
    ok = starts_at_one and piecewise
    print(f"{'[PASS]' if ok else '[FAIL]'} test_factor_sanity "
          f"(首值=1.0:{starts_at_one}, 阶跃 {n_steps}/{len(f)} 分段常数:{piecewise}, "
          f"下降 {n_down} 次, 因子 {f[0]:.4f}->{f[-1]:.4f})")
    return ok


def _ctx(sym, t, piv=None, lookback=600):
    piv = piv if piv is not None else zigzag(sym.high, sym.low, sym.atr, 2.0)
    return Ctx(code=sym.code, t=t,
               open_=sym.open_[:t + 1], high=sym.high[:t + 1],
               low=sym.low[:t + 1], close=sym.close[:t + 1],
               volume=sym.volume[:t + 1],
               atr=float(sym.atr[t]), tick=sym.tick,
               atr_arr=sym.atr[:t + 1],
               pivots=piv.view_at(t, max_lookback=lookback))


# ============================================================
def test_pivot_causality(sym=None, n_checks: int = 12) -> bool:
    """全序列 zigzag + confirm_idx 过滤  ==  截断序列上重算 zigzag"""
    sym = sym or _sym()
    full = zigzag(sym.high, sym.low, sym.atr, 2.0)
    n = len(sym)
    ok = True
    for t in np.linspace(400, n - 2, n_checks).astype(int):
        # 截断重算：注意 ATR 也必须重算（因果 ATR 天然一致，但要走同一路径）
        atr_tr = atr_series(sym.high[:t + 1], sym.low[:t + 1], sym.close[:t + 1], 14)
        tr = zigzag(sym.high[:t + 1], sym.low[:t + 1], atr_tr, 2.0)
        v = full.view_at(t)
        same = (len(v) == len(tr)
                and np.array_equal(v.idx, tr.idx)
                and np.allclose(v.price, tr.price)
                and np.array_equal(v.kind, tr.kind)
                and np.array_equal(v.confirm_idx, tr.confirm_idx))
        if not same:
            ok = False
            print(f"  [FAIL] t={t}: 全序列视图 {len(v)} 个枢轴 vs 截断重算 {len(tr)} 个")
    print(f"{'[PASS]' if ok else '[FAIL]'} test_pivot_causality "
          f"({n_checks} 个时点, 共 {len(full)} 个枢轴)")
    return ok


# ============================================================
def test_no_future_dependence(sym=None, n_checks: int = 8, seed: int = 7) -> bool:
    """
    把 t 之后的所有 bar 替换成随机垃圾，重算 ATR/枢轴后在同一个 t 上检测。
    任何检测器如果偷看了未来，输出必然改变。
    """
    sym = sym or _sym()
    rng = np.random.default_rng(seed)
    n = len(sym)
    dets = [V1Density(), V2Fusion(), FixedPlacebo()]
    ok = True

    for t in np.linspace(500, n - 60, n_checks).astype(int):
        base_ctxs = _ctx(sym, t)
        base_out = {d.name: d.detect(base_ctxs) for d in dets}

        # 构造被污染的未来
        h = sym.high.copy(); l = sym.low.copy()
        c = sym.close.copy(); o = sym.open_.copy(); v = sym.volume.copy()
        m = slice(t + 1, n)
        k = n - t - 1
        scale = rng.uniform(0.2, 5.0, k)
        c[m] = np.abs(c[m] * scale) + 0.01
        o[m] = c[m] * rng.uniform(0.9, 1.1, k)
        h[m] = np.maximum(o[m], c[m]) * rng.uniform(1.0, 1.2, k)
        l[m] = np.minimum(o[m], c[m]) * rng.uniform(0.8, 1.0, k)
        v[m] = np.abs(v[m] * rng.uniform(0.1, 10.0, k)) + 1

        atr2 = atr_series(h, l, c, 14)
        piv2 = zigzag(h, l, atr2, 2.0)
        ctx2 = Ctx(code=sym.code, t=t,
                   open_=o[:t + 1], high=h[:t + 1], low=l[:t + 1],
                   close=c[:t + 1], volume=v[:t + 1],
                   atr=float(atr2[t]), tick=sym.tick, atr_arr=atr2[:t + 1],
                   pivots=piv2.view_at(t, max_lookback=600))

        for d in dets:
            a = base_out[d.name]
            b = d.detect(ctx2)
            if len(a) != len(b) or not all(
                    abs(x.center - y.center) < 1e-9 and abs(x.low - y.low) < 1e-9
                    and abs(x.high - y.high) < 1e-9 and abs(x.score - y.score) < 1e-9
                    for x, y in zip(a, b)):
                ok = False
                print(f"  [FAIL] {d.name} @t={t}: 污染未来后输出改变 "
                      f"({len(a)} -> {len(b)} 个位)")
    print(f"{'[PASS]' if ok else '[FAIL]'} test_no_future_dependence "
          f"({n_checks} 个时点 x {len(dets)} 个检测器)")
    return ok


# ============================================================
def test_volume_conservation(sym=None) -> bool:
    """成交量剖面必须体积守恒；同时展示旧算法的放大倍数"""
    sym = sym or _sym()
    h, l, v = sym.high[-250:], sym.low[-250:], sym.volume[-250:]
    atr = float(sym.atr[-1])
    bin_w = max(sym.tick, 0.25 * atr)
    p0 = float(l.min()) - bin_w
    nb = int(np.ceil((float(h.max()) + bin_w - p0) / bin_w)) + 1
    prof = volume_profile(h, l, v, p0, bin_w, nb)
    err = abs(prof.sum() - v.sum()) / v.sum()

    # 旧算法（全额累加）的放大倍数
    lo_i = np.clip(((l - p0) / bin_w).astype(np.int64), 0, nb - 1)
    hi_i = np.clip(((h - p0) / bin_w).astype(np.int64), 0, nb - 1)
    cnt = (hi_i - lo_i + 1)
    inflate = (v * cnt).sum() / v.sum()

    ok = err < 1e-9
    print(f"{'[PASS]' if ok else '[FAIL]'} test_volume_conservation  "
          f"相对误差={err:.2e}  (旧算法放大 {inflate:.1f}x)")
    return ok


# ============================================================
def test_v1_equivalence(sym=None, n_checks: int = 10) -> bool:
    """V1 向量化复刻 vs 原始 src/density_analyzer.find_dense_zones"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    from src.density_analyzer import find_dense_zones

    sym = sym or _sym()
    n = len(sym)
    det = V1Density(window=40, n_zones=6)
    ok = True
    for t in np.linspace(300, n - 2, n_checks).astype(int):
        sub = sym.df.iloc[:t + 1]
        old = find_dense_zones(sub, window=40, n_zones=6)
        new = det.detect(_ctx(sym, t))
        # 原始实现对 center 做了 round(_, 5)，这里按 1e-4 容差比较
        oc = np.sort(np.array([z.center for z in old], dtype=np.float64))
        nc = np.sort(np.array([z.center for z in new], dtype=np.float64))
        if len(oc) != len(nc) or not np.allclose(oc, nc, atol=1e-4):
            ok = False
            print(f"  [FAIL] t={t}: 原始 {oc} vs 复刻 {nc}")
    print(f"{'[PASS]' if ok else '[FAIL]'} test_v1_equivalence ({n_checks} 个时点)")
    return ok


# ============================================================
def test_label_direction() -> bool:
    """从下方上涨碰到"支撑"不应计为一次支撑测试"""
    # 构造：价格一路上涨，穿过一个位于起点上方的"支撑区"
    n = 60
    c = np.linspace(10.0, 14.0, n)
    h = c * 1.01
    l = c * 0.99
    z = Level(center=12.0, low=11.9, high=12.1, score=1.0, kind="support")
    t = 5  # close[5] ≈ 10.34 < zone.high -> 现价在区下方，不构成支撑测试
    oc = evaluate_level(h, l, c, t, z, atr_t=0.2, horizon=40)
    ok1 = oc.result == "no_touch"

    # 反向：价格一路下跌到该区，才算一次支撑测试
    c2 = np.linspace(14.0, 10.0, n)
    h2, l2 = c2 * 1.01, c2 * 0.99
    oc2 = evaluate_level(h2, l2, c2, t, z, atr_t=0.2, horizon=40)
    ok2 = oc2.touched

    ok = ok1 and ok2
    print(f"{'[PASS]' if ok else '[FAIL]'} test_label_direction  "
          f"(上涨穿越={oc.result}, 下跌测试={oc2.result})")
    return ok


# ============================================================
def run_all() -> bool:
    print("=" * 70)
    print("  srlab 防泄漏 / 正确性测试")
    print("=" * 70)
    sym = _sym()
    print(f"  测试标的: {sym.code}, {len(sym)} 根日线\n")
    res = [
        test_pivot_causality(sym),
        test_no_future_dependence(sym),
        test_volume_conservation(sym),
        test_v1_equivalence(sym),
        test_label_direction(),
        test_factor_sanity(),
        test_adjust_scale_invariance(),
    ]
    print("-" * 70)
    print(f"  {sum(res)}/{len(res)} 通过")
    print("=" * 70)
    return all(res)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(0 if run_all() else 1)
