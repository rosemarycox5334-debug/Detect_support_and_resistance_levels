# 密位 · DENSITY SR

> **基于 K 线（蜡烛图）数据自动检测支撑位与阻力位，适用于任意品种：股票、期货、外汇、加密货币等——只要有 K 线就能用。**

`Detect_support_and_resistance_levels` 是一个本地化的技术分析工具，提供 Web UI 和 Python API，用于从 OHLCV K 线中识别支撑/阻力位，并给出每个关键位的历史统计置信度（触及概率 × 守住概率）。

![DENSITY 主界面截图](out/ui_bg.png)

## 核心特性

- **通用市场支持**
  - A股、期货、外汇、加密货币等任意有 K 线的品种
  - 支持日线、周线、小时线、分钟线等多时间框架
- **V3 融合算法**
  - 成交量分布 + 极值结构 + ATR 归一化
  - 多时间框架（MTF）共振验证
  - 历史分位档与样本外统计校准
- **概率输出**
  - 触及概率（价格在未来 N 根 K 线内到达该位）
  - 守住概率（到达后在该位附近获得反应的概率）
  - 综合有效概率与历史分位档实测表现
- **本地优先**
  - 不依赖外部实时行情 API
  - 读取本地 parquet K 线文件，离线分析
- **Web 界面**
  - 选择本地数据文件夹
  - 单品种深度分析 + 全市场批量扫描
  - 交互式 K 线图与关键位可视化

## 支持的文件格式

将 K 线数据保存为 `.parquet` 文件，放入同一个文件夹即可。支持三种命名约定：

| 市场 | 文件名示例 | 周期 |
|------|-----------|------|
| 期货/主连 | `螺纹钢主连_D1.parquet` | `D1`, `W1`, `H1`... |
| MT5/通用 | `EURUSD_H1.parquet` | `D1`, `W1`, `H1`, `M5`... |
| A股 | `000001_daily.parquet` | `daily`, `60min`, `15min`, `5min` |

必需列：`time`（unix 秒或毫秒）、`open`、`high`、`low`、`close`、`tick_volume`。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖：Flask、Pandas、NumPy、SciPy、PyArrow。

### 2. 准备数据

将你本地 parquet 格式的 K 线文件放入一个文件夹，例如：

```
D:\MT5_K线数据\
  ├─ EURUSD_D1.parquet
  ├─ EURUSD_H1.parquet
  ├─ XAUUSD_D1.parquet
  └─ ...
```

### 3. 启动 Web 应用

```bash
python app.py
```

然后在浏览器打开：http://127.0.0.1:5000

点击【选择数据文件夹】加载 K 线数据，选择品种和周期后即可查看支撑/阻力位分析。

### 4. 一键批量扫描

设置好数据文件夹后，点击【全部扫描】可对所有品种进行分析，并按综合有效概率排序，快速定位值得关注的交易机会。

## 主要 API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 主界面 |
| `/api/data_dir` | POST | 设置本地数据文件夹路径 |
| `/api/instruments` | GET | 列出所有品种及可用周期 |
| `/api/analyze` | GET | 单品种分析 |
| `/api/analyze_all` | GET | 一键分析全部品种 |

示例：

```bash
# 设置数据目录
curl -X POST http://127.0.0.1:5000/api/data_dir \
  -H "Content-Type: application/json" \
  -d '{"path": "D:\\\\MT5_K线数据"}'

# 单品种分析
curl "http://127.0.0.1:5000/api/analyze?symbol=EURUSD&tf=D1&n_zones=3"
```

## 回测与验证

项目内置了完整的回测评测框架：

```bash
# 快速冒烟测试（20 只股票）
python scripts/run_eval.py --quick

# 完整评测（200 只股票）
python scripts/run_eval.py
```

输出包含：
- `levels.parquet`：逐关键位明细
- `geo.parquet`：逐决策点几何精度
- `report.txt`：汇总报告

关键算法改进均有实测数据支撑，详情见源码内 `OPTIMIZATION_PLAN.md` 及各模块 docstring。

## 技术栈

- **后端**：Python + Flask
- **核心算法**：Pandas / NumPy / SciPy
- **前端**：原生 HTML + Plotly
- **数据格式**：Parquet

## 开源许可

本项目采用 [GNU General Public License v3.0](LICENSE)。

## 免责声明

本工具输出的支撑/阻力位及概率统计仅用于技术研究和辅助决策，不构成任何投资建议。市场有风险，交易需谨慎。
