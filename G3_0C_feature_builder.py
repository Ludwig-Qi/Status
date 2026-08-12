"""
G3 组 · Stage 0C · B 类特征构造 + state_feature_snapshot
==========================================================
基于 0A 清点 V3，构造 13 项可立即从 COS 落表的 B 类特征。

数据存储:
  COS 数据以 Parquet 格式存储在本地镜像目录中（"COS 清洗层 + 本地镜像"）。
  A股路径: {DATA_ROOT}/ashare/lqtp_data/*
  美股路径: {DATA_ROOT}/us/...
  具体路径结构见下方 DATA_LAYOUT 字典。

  假设可使用 pd.read_parquet() 直接读取——按需拼接分区路径。

关键口径（来自 cos数据清单_20260811.html）:
  - 美股 Ret 系列: 已是小数，禁止再除
  - A股 Return: bp（÷10000 得小数）
  - A股行业: 必填过滤 IndustrySource (固定 sw_l1)
  - 美股 Ticker 列名: us_stock_daily=Ticker, us_etf_daily=Ticker
  - A股 Symbol 列名: ashare_*=Symbol

TODO（待确认后替换）:
  - DATA_ROOT 本地镜像根路径
  - Parquet 分区结构（Hive 分区 date= / 无分区 / 按表名单文件）
  - us_etf_daily / us_stock_daily 的具体文件位置和列名
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import date, datetime
from pathlib import Path
import os

# ============================================================
# 0. 配置
# ============================================================

# TODO: 替换为实际的本地镜像根路径
DATA_ROOT = os.environ.get("COS_DATA_ROOT", "/data/cos_mirror")

# COS 清单中提到的数据路径结构（待确认实际目录名）
DATA_LAYOUT = {
    # A股（数据源: clean_data/ashare/lqtp_data/*）
    "ashare_stock_daily":       "ashare/lqtp_data/stock_daily",
    "ashare_stock_minute":      "ashare/lqtp_data/stock_minute",
    "ashare_stock_valuation_daily": "ashare/lqtp_data/stock_valuation_daily",
    "ashare_stock_industry":    "ashare/lqtp_data/stock_industry",
    "ashare_index_daily":       "ashare/lqtp_data/index_daily",
    "ashare_calendar":          "ashare/lqtp_data/calendar",
    "ashare_universe_daily":    "ashare/lqtp_data/universe_daily",

    # 美股（数据源: massive_data / clean_data）
    "us_stock_daily":           "us/stock_daily",
    "us_etf_daily":             "us/etf_daily",
    "us_calendar":              "us/calendar",
}

# 美股 ETF Ticker → COS 表名（用于覆盖验证）
US_TICKER_TABLE_MAP = {
    "SPY":  "us_etf_daily",
    "QQQ":  "us_etf_daily",
    "SOXX": "us_etf_daily",
    "SMH":  "us_etf_daily",
    "IWM":  "us_etf_daily",
    "EEM":  "us_etf_daily",
    "TLT":  "us_etf_daily",
    "IEF":  "us_etf_daily",
    "SHY":  "us_etf_daily",
    "UUP":  "us_etf_daily",
    "GLD":  "us_etf_daily",
    "USO":  "us_etf_daily",
    "FXI":  "us_etf_daily",
    "KWEB": "us_etf_daily",  # 也可能是 us_stock_daily
}

# ============================================================
# 0. 配置：13 项 B 类特征的构造规格
# ============================================================

@dataclass
class FeatureSpec:
    """单条 B 类特征的构造规格"""
    canonical_id: str          # 特征 ID（与 0A 清点一致）
    name_cn: str               # 中文名
    source_table: str          # COS 表名
    ticker: str                # ETF Ticker 或股票代码
    formula: str               # 构造公式（人类可读）
    unit_note: str             # 单位/口径注意事项
    pit_note: str              # PIT 时区对齐注意事项
    cos_col: str = "Ret"       # COS 中的收益列名
    needs_ashare: bool = False # 是否需要 A 股侧数据
    ashare_table: str = ""     # A 股侧表名
    ashare_filter: str = ""    # A 股侧过滤条件


FEATURE_SPECS: List[FeatureSpec] = [
    # ---- F12 港股离岸（2 项）----
    FeatureSpec(
        canonical_id="china_adr_ret",
        name_cn="中概股组合隔夜收益",
        source_table="us_stock_daily",
        ticker="KWEB",  # 或自建中概股池
        formula="KWEB 等权/市值加权日收益 → 取 Ret_Overnight（隔夜）或 Ret（日频）",
        unit_note="Ret 已是小数；优先用 Ret_Overnight 捕捉隔夜定价",
        pit_note="美股收盘(夏令04:00/冬令05:00)→次一A股交易日对齐",
        cos_col="Ret_Overnight",  # 优先隔夜
    ),
    FeatureSpec(
        canonical_id="fxi_kweb",
        name_cn="互联网−大盘中国ETF差",
        source_table="us_etf_daily",
        ticker="FXI,KWEB",  # 双 ticker
        formula="FXI.Ret − KWEB.Ret",
        unit_note="两者均为小数，直接相减",
        pit_note="需两个 ETF 同一交易日均有数据",
    ),

    # ---- F13 美股全球（8 项）----
    FeatureSpec(
        canonical_id="spx_ret",
        name_cn="标普500收益",
        source_table="us_etf_daily", ticker="SPY",
        formula="SPY.Ret",
        unit_note="已是小数",
        pit_note="美股收盘→次一A股交易日",
    ),
    FeatureSpec(
        canonical_id="ndx_ret",
        name_cn="纳斯达克100收益",
        source_table="us_etf_daily", ticker="QQQ",
        formula="QQQ.Ret",
        unit_note="已是小数",
        pit_note="同上",
    ),
    FeatureSpec(
        canonical_id="sox_ret",
        name_cn="费城半导体指数收益",
        source_table="us_etf_daily", ticker="SOXX",  # SMH 为备选
        formula="SOXX.Ret（备选 SMH.Ret）",
        unit_note="已是小数；先确认 COS 中哪个半导体 ETF 覆盖更完整",
        pit_note="同上",
    ),
    FeatureSpec(
        canonical_id="russell2000_ret",
        name_cn="罗素2000收益",
        source_table="us_etf_daily", ticker="IWM",
        formula="IWM.Ret",
        unit_note="已是小数",
        pit_note="同上",
    ),
    FeatureSpec(
        canonical_id="msci_em_ret",
        name_cn="MSCI新兴市场收益",
        source_table="us_etf_daily", ticker="EEM",
        formula="EEM.Ret",
        unit_note="已是小数",
        pit_note="同上",
    ),
    FeatureSpec(
        canonical_id="ust_yields",
        name_cn="美国国债收益率变化(proxy)",
        source_table="us_etf_daily", ticker="TLT,IEF,SHY",
        formula="Δyield ≈ −(TLT.Ret/16 + IEF.Ret/7 + SHY.Ret/2) / 3",
        unit_note="利率 proxy——ETF 价格变化 ÷ 久期近似；有凸性偏差，只做方向信号",
        pit_note="同上",
    ),
    FeatureSpec(
        canonical_id="dxy_chg",
        name_cn="美元指数变化(proxy)",
        source_table="us_etf_daily", ticker="UUP",
        formula="UUP.Ret",
        unit_note="UUP 跟踪美元指数期货非现货，有小幅基差",
        pit_note="同上",
    ),
    FeatureSpec(
        canonical_id="semiconductor_china_divergence",
        name_cn="SOX−A股电子背离",
        source_table="us_etf_daily",
        ticker="SOXX",
        formula="SOXX.Ret − A股电子行业等权 Ret",
        unit_note="美股小数 − A股 bp/10000；需单位对齐",
        pit_note="美股收盘→次一A股日；A股侧用同交易日",
        needs_ashare=True,
        ashare_table="ashare_stock_industry",
        ashare_filter="IndustrySource=='sw_l1' AND IndustryCode 属于电子行业",
    ),

    # ---- F14 商品航运（3 项）----
    FeatureSpec(
        canonical_id="brent_wti",
        name_cn="原油收益(proxy)",
        source_table="us_etf_daily", ticker="USO",
        formula="USO.Ret",
        unit_note="⚠️ ETF proxy——展期损耗导致 contango 时持续跑输现货; 标注 _proxy",
        pit_note="同上",
    ),
    FeatureSpec(
        canonical_id="gold",
        name_cn="黄金收益(proxy)",
        source_table="us_etf_daily", ticker="GLD",
        formula="GLD.Ret",
        unit_note="GLD 跟踪误差通常 <0.5%",
        pit_note="同上",
    ),
    FeatureSpec(
        canonical_id="gold_risk_off",
        name_cn="黄金避险强度",
        source_table="us_etf_daily",
        ticker="GLD",
        formula="GLD.Ret − A股全市场等权 Ret（小数）",
        unit_note="GLD 小数 − A股 bp/10000；全市场取 ashare_stock_daily.Return 等权均值",
        pit_note="两边对齐到同一A股交易日",
        needs_ashare=True,
        ashare_table="ashare_stock_daily",
    ),
]


# ============================================================
# 1. COS 数据读取（直接读 Parquet 本地镜像）
# ============================================================

def _resolve_path(table_name: str) -> Path:
    """将逻辑表名解析为本地 Parquet 路径。"""
    if table_name not in DATA_LAYOUT:
        raise KeyError(f"未知表名: {table_name}。请在 DATA_LAYOUT 中登记路径。")
    return Path(DATA_ROOT) / DATA_LAYOUT[table_name]


def read_cos_table(
    table_name: str,
    tickers: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
    start_date: str = "2015-01-01",
    end_date: str = "2026-08-11",
) -> pd.DataFrame:
    """
    从 COS 本地镜像读取 Parquet 数据。

    假设:
      - 数据按表名分目录存储
      - 每个目录下是一个或多个 .parquet 文件
      - 可能按 date= 分区（Hive 风格），也可能无分区
      - Ticker/Symbol 列存在于表中，读取后可在内存中过滤

    如果实际存储结构不同（如按 ticker 分子目录、单文件等），
    只需修改此函数内的路径拼接逻辑。
    """
    base_path = _resolve_path(table_name)

    if not base_path.exists():
        raise FileNotFoundError(
            f"数据路径不存在: {base_path}\n"
            f"请确认 DATA_ROOT ('{DATA_ROOT}') 和 DATA_LAYOUT['{table_name}'] 是否正确。"
        )

    # ---- 发现 Parquet 文件 ----
    # 尝试三种常见布局:
    #   A) base_path/*.parquet           (无分区)
    #   B) base_path/date=YYYYMMDD/*.parquet (Hive 分区)
    #   C) base_path/**/*.parquet        (嵌套)
    parquet_files = sorted(base_path.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(base_path.glob("**/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"{base_path} 下无 .parquet 文件。")

    print(f"  [read] {table_name}: {len(parquet_files)} 个 parquet 文件")

    # ---- 读取 ----
    # 小表直接全读；大表可用 filters 下推（如果 Parquet 支持）
    try:
        df = pd.read_parquet(
            parquet_files,
            columns=columns,
            # filters: 如果分区列为 date，可下推过滤
            # filters=[("date", ">=", start_date), ("date", "<=", end_date)],
        )
    except Exception as e:
        # 降级: 逐文件读取再合并（兼容性更好）
        print(f"  [read] 批量读取失败({e})，尝试逐文件合并...")
        parts = []
        for f in parquet_files:
            try:
                parts.append(pd.read_parquet(f, columns=columns))
            except Exception as fe:
                print(f"  [warn] 跳过 {f.name}: {fe}")
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # ---- 日期过滤 ----
    date_col = "TradeDate"  # 美股可能为 trade_date——待确认后改为自动检测
    if date_col not in df.columns:
        # 尝试小写
        date_col_lower = "trade_date"
        if date_col_lower in df.columns:
            date_col = date_col_lower

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

    # ---- Ticker 过滤 ----
    ticker_col = None
    for candidate in ["Ticker", "ticker", "Symbol", "symbol"]:
        if candidate in df.columns:
            ticker_col = candidate
            break

    if tickers and ticker_col:
        df = df[df[ticker_col].isin(tickers)]

    return df


# ============================================================
# 2. COS 覆盖验证
# ============================================================

def verify_cos_coverage(specs: List[FeatureSpec]) -> pd.DataFrame:
    """
    验证 13 项 B 类所需的 ETF Ticker 在 COS 中是否存在。

    对每个 Ticker，查询 us_etf_daily 或 us_stock_daily，
    返回: Ticker / 表名 / 是否存在 / 最早日期 / 最晚日期 / 交易日数
    """
    # 收集所有 unique ticker
    all_tickers: Dict[str, str] = {}  # ticker -> source_table
    for s in specs:
        for t in s.ticker.split(","):
            t = t.strip()
            if t not in all_tickers:
                all_tickers[t] = s.source_table

    results = []
    for ticker, table in all_tickers.items():
        try:
            df = read_cos_table(table, tickers=[ticker])
            if df is not None and len(df) > 0:
                results.append({
                    "ticker": ticker,
                    "table": table,
                    "exists": True,
                    "earliest": df["TradeDate"].min(),
                    "latest": df["TradeDate"].max(),
                    "trading_days": len(df),
                })
            else:
                results.append({
                    "ticker": ticker, "table": table,
                    "exists": False,
                    "earliest": None, "latest": None, "trading_days": 0,
                })
        except NotImplementedError:
            results.append({
                "ticker": ticker, "table": table,
                "exists": "API_PENDING",
                "earliest": None, "latest": None, "trading_days": None,
            })

    return pd.DataFrame(results)


# ============================================================
# 3. 特征构造引擎
# ============================================================

def build_us_etf_feature(spec: FeatureSpec, df: pd.DataFrame) -> pd.DataFrame:
    """
    从美股 ETF 日线构造单个特征。

    df 应包含列: TradeDate, Ticker, Ret (已是小数)
    返回: TradeDate, canonical_id, value
    """
    ticker = spec.ticker.split(",")[0].strip()

    if spec.cos_col in df.columns:
        ret = df[spec.cos_col]
    elif "Ret" in df.columns:
        ret = df["Ret"]
    else:
        raise ValueError(f"表 {spec.source_table} 中无收益列，可用列: {df.columns.tolist()}")

    return pd.DataFrame({
        "TradeDate": df["TradeDate"],
        "canonical_id": spec.canonical_id,
        "value": ret.astype(float),
    })


def build_us_yield_proxy(spec: FeatureSpec) -> pd.DataFrame:
    """
    构造 #66 ust_yields: 从 TLT/IEF/SHY 三个 ETF 价格变化反推利率变化。

    Δyield ≈ −(TLT_ret/16 + IEF_ret/7 + SHY_ret/2) / 3
    久期: TLT≈16y, IEF≈7y, SHY≈2y
    """
    tickers = ["TLT", "IEF", "SHY"]
    durations = {"TLT": 16.0, "IEF": 7.0, "SHY": 2.0}

    dfs = {}
    for t in tickers:
        dfs[t] = read_cos_table("us_etf_daily", tickers=[t], columns=["TradeDate", "Ret"])

    # 取各 ETF 的 TradeDate 交集
    common_dates = dfs["TLT"][["TradeDate"]]
    for t in ["IEF", "SHY"]:
        common_dates = common_dates.merge(dfs[t][["TradeDate"]], on="TradeDate", how="inner")

    # 对齐后计算加权
    aligned = common_dates.copy()
    for t in tickers:
        aligned = aligned.merge(
            dfs[t][["TradeDate", "Ret"]].rename(columns={"Ret": f"ret_{t}"}),
            on="TradeDate", how="left"
        )

    weights = [1.0 / durations[t] for t in tickers]
    weight_sum = sum(weights)
    aligned["value"] = -(
        aligned["ret_TLT"] / durations["TLT"] +
        aligned["ret_IEF"] / durations["IEF"] +
        aligned["ret_SHY"] / durations["SHY"]
    ) / 3.0

    return pd.DataFrame({
        "TradeDate": aligned["TradeDate"],
        "canonical_id": "ust_yields",
        "value": aligned["value"],
    })


def build_cross_market_feature(spec: FeatureSpec) -> pd.DataFrame:
    """
    构造跨市场特征（#78 semiconductor_china_divergence, #92 gold_risk_off）。

    美股侧: 取 ETF Ret（已是小数）
    A股侧: 取 Return（bp, ÷10000 转小数）或行业等权 Ret
    """
    # ---- 美股侧 ----
    us_ticker = spec.ticker.split(",")[0].strip()
    df_us = read_cos_table(spec.source_table, tickers=[us_ticker], columns=["TradeDate", "Ret"])
    df_us["us_ret"] = df_us["Ret"].astype(float)

    # ---- A股侧 ----
    if spec.canonical_id == "semiconductor_china_divergence":
        # 取 sw_l1 电子行业全部股票 → 等权收益
        df_a = read_cos_table(
            "ashare_stock_daily",
            columns=["TradeDate", "Symbol", "Return"],
        )
        df_industry = read_cos_table(
            "ashare_stock_industry",
            columns=["TradeDate", "Symbol", "IndustryCode"],
        )
        # 过滤电子行业 (SW 一级行业代码取决于实际分类，此处用占位)
        # TODO: 确认 SW 电子行业 IndustryCode
        df_elec = df_industry[df_industry["IndustryCode"].isin(["850000"])]  # SW电子占位码
        df_a = df_a.merge(df_elec[["TradeDate", "Symbol"]], on=["TradeDate", "Symbol"])
        df_a["ashare_ret"] = df_a["Return"] / 10000.0  # bp → 小数
        df_a = df_a.groupby("TradeDate")["ashare_ret"].mean().reset_index()

    elif spec.canonical_id == "gold_risk_off":
        df_a = read_cos_table(
            "ashare_stock_daily",
            columns=["TradeDate", "Return"],
        )
        df_a["ashare_ret"] = df_a["Return"] / 10000.0
        df_a = df_a.groupby("TradeDate")["ashare_ret"].mean().reset_index()
    else:
        raise ValueError(f"未处理的跨市场特征: {spec.canonical_id}")

    # ---- 时间对齐 ----
    # 美股 t 日 → A股 t+1 日（美股收盘晚于同日A股收盘，只能用于次一交易日）
    # 创建映射: us.TradeDate → A股下一交易日
    df_a_sorted = df_a.sort_values("TradeDate")
    df_a_sorted["next_trade_date"] = df_a_sorted["TradeDate"].shift(-1)
    # 简化: 暂时用 A股同交易日（TODO: 用 ashare_calendar 精确映射）
    df_us["ashare_date"] = df_us["TradeDate"]  # TODO: → 实际应为次一A股交易日

    merged = df_us.merge(
        df_a_sorted.rename(columns={"TradeDate": "ashare_date"}),
        on="ashare_date", how="inner"
    )
    merged["value"] = merged["us_ret"] - merged["ashare_ret"]

    return pd.DataFrame({
        "TradeDate": merged["ashare_date"],
        "canonical_id": spec.canonical_id,
        "value": merged["value"],
    })


def build_fxi_kweb_spread() -> pd.DataFrame:
    """
    构造 #47 fxi_kweb: FXI.Ret − KWEB.Ret。
    两个 ETF 同表，按 Ticker 分别拉取后做差。
    """
    df_fxi = read_cos_table("us_etf_daily", tickers=["FXI"], columns=["TradeDate", "Ret"])
    df_kweb = read_cos_table("us_etf_daily", tickers=["KWEB"], columns=["TradeDate", "Ret"])

    merged = df_fxi[["TradeDate", "Ret"]].merge(
        df_kweb[["TradeDate", "Ret"]],
        on="TradeDate", suffixes=("_fxi", "_kweb")
    )
    merged["value"] = merged["Ret_fxi"] - merged["Ret_kweb"]

    return pd.DataFrame({
        "TradeDate": merged["TradeDate"],
        "canonical_id": "fxi_kweb",
        "value": merged["value"],
    })


# ---- 构造主循环 ----

def build_all_b_features(
    start_date: str = "2015-01-01",
    end_date: str = "2026-08-11",
) -> pd.DataFrame:
    """
    遍历 13 项 FEATURE_SPECS，逐项构造并拼接为统一 DataFrame。

    返回列: TradeDate, canonical_id, value
    """
    all_features = []

    for spec in FEATURE_SPECS:
        print(f"[build] {spec.canonical_id} ({spec.name_cn}) ...")

        try:
            if spec.canonical_id == "ust_yields":
                df = build_us_yield_proxy(spec)
            elif spec.canonical_id == "fxi_kweb":
                df = build_fxi_kweb_spread()
            elif spec.needs_ashare:
                df = build_cross_market_feature(spec)
            else:
                # 单 ETF 直取
                ticker = spec.ticker.split(",")[0].strip()
                df_raw = read_cos_table(
                    spec.source_table,
                    tickers=[ticker],
                    columns=["TradeDate", spec.cos_col],
                    start_date=start_date,
                    end_date=end_date,
                )
                df = build_us_etf_feature(spec, df_raw)

            all_features.append(df)

        except NotImplementedError:
            print(f"  [skip] data_access API 未就绪，跳过 {spec.canonical_id}")
            continue
        except Exception as e:
            print(f"  [error] {spec.canonical_id}: {e}")
            continue

    if not all_features:
        raise RuntimeError("无任何特征成功构造。请先确认 data_access API。")

    result = pd.concat(all_features, ignore_index=True)
    return result


# ============================================================
# 4. state_feature_snapshot 表结构
# ============================================================

SNAPSHOT_SCHEMA = {
    "canonical_id":    "STRING   -- 特征 ID，与 0A 清点一致，如 'spx_ret'",
    "TradeDate":       "DATE     -- A股交易日（特征可用的交易日）",
    "value":           "DOUBLE   -- 特征值",
    "available_time":  "TIMESTAMP-- PIT 可用时点（对A股决策而言，数据何时可知）",
    "feature_version": "STRING   -- 公式版本号，如 'v1.0'，公式调整时必须递增",
    "data_source":     "STRING   -- 'COS' / 'FRED' / 'CBOE' / 'TBD'",
    "insert_time":     "TIMESTAMP-- 写入时间",
}


def build_snapshot_table(
    df_features: pd.DataFrame,
    feature_version: str = "v1.0",
    data_source: str = "COS",
) -> pd.DataFrame:
    """
    将特征 DataFrame 组装为 state_feature_snapshot 格式。

    输入 df_features 需包含: TradeDate, canonical_id, value
    """
    df = df_features.copy()
    df["available_time"] = pd.NaT  # 初版占位——后续按实际 PIT 规则填充
    df["feature_version"] = feature_version
    df["data_source"] = data_source
    df["insert_time"] = datetime.now()

    return df[["canonical_id", "TradeDate", "value", "available_time",
               "feature_version", "data_source", "insert_time"]]


# ============================================================
# 5. 美股→A股交易日映射
# ============================================================

def map_us_to_ashare_trading_day(
    df_us: pd.DataFrame,
    us_date_col: str = "TradeDate",
) -> pd.DataFrame:
    """
    美股收盘日 → 次一A股交易日。

    规则: 美股 t 日收盘(夏令04:00/冬令05:00北京时间) →
          A股同日已收盘，只能用于 A股 t+1 日（若 t+1 为交易日）。

    TODO: 使用 ashare_calendar 精确映射
    """
    # 占位实现: 直接 +1 自然日（生产需替换为下一交易日映射）
    df = df_us.copy()
    df["ashare_trade_date"] = pd.to_datetime(df[us_date_col]) + pd.Timedelta(days=1)
    return df


# ============================================================
# 6. 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("G3 Stage 0C · B 类特征构造")
    print(f"DATA_ROOT = {DATA_ROOT}")
    print("=" * 60)

    # Step 0: 打印特征清单
    print(f"\n待构造特征: {len(FEATURE_SPECS)} 项\n")
    for i, s in enumerate(FEATURE_SPECS, 1):
        tag = "跨市场" if s.needs_ashare else "单ETF"
        print(f"  {i:2d}. [{tag}] {s.canonical_id:35s} {s.ticker:15s} {s.source_table}")

    # Step 1: 覆盖验证
    print("\n" + "-" * 40)
    print("Step 1: COS 覆盖验证")
    print("-" * 40)
    coverage = verify_cos_coverage(FEATURE_SPECS)
    print(coverage.to_string(index=False))

    # Step 2: 构造
    print("\n" + "-" * 40)
    print("Step 2: 特征构造")
    print("-" * 40)
    try:
        df_all = build_all_b_features()
        print(f"\n成功构造 {df_all['canonical_id'].nunique()} 项特征")
        print(f"日期范围: {df_all['TradeDate'].min()} → {df_all['TradeDate'].max()}")
        print(f"总行数: {len(df_all)}")

        df_snapshot = build_snapshot_table(df_all)
        print(f"\nstate_feature_snapshot: {len(df_snapshot)} 行, "
              f"{df_snapshot['canonical_id'].nunique()} 列")
        print(df_snapshot.head(10).to_string(index=False))

        # TODO: 写入 Parquet
        # out_path = Path(DATA_ROOT) / "features" / "state_feature_snapshot_v1.parquet"
        # df_snapshot.to_parquet(out_path)

    except FileNotFoundError as e:
        print(f"\n[blocked] {e}")
        print(f"请设置正确的 DATA_ROOT 环境变量或修改脚本中的 DATA_ROOT 常量。")
        print(f"当前 DATA_ROOT = '{DATA_ROOT}'")
