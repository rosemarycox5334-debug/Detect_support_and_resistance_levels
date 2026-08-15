# -*- coding: utf-8 -*-
"""
srlab —— 支撑/阻力位识别的可回测实验室

模块职责：
    data       数据适配（A股 parquet / MT5 parquet）、ATR、标的池与训练/测试切分
    base       Level / Ctx 数据结构，Detector 统一接口
    pivots     因果 ZigZag 枢轴（带"确认时刻"，保证无未来信息）
    profile    体积守恒的成交量-价格剖面、ATR 分箱、时间衰减
    detectors  V1 基线（现有算法）、V2 融合算法、Placebo 对照
    labeling   触及事件去重、前瞻结果标注（守住/突破/未决）
    metrics    命中率、几何误差、提升度、Wilson 区间、概率校准
    walkforward 无泄漏走查回测运行器
    leakage    自动化防泄漏测试

设计原则：
    1. 检测器只能看到 ctx 中截止到 t 的数据，评测只能看到 t 之后的数据；
       两者在代码结构上物理隔离，并有 leakage.py 自动化验证。
    2. 所有阈值以 ATR 为单位，跨标的、跨波动率可比。
    3. 任何"胜率"都给出 Wilson 下界与样本量，禁止裸比例。
"""

__all__ = [
    "data", "base", "pivots", "profile", "detectors",
    "labeling", "metrics", "walkforward", "leakage",
]
