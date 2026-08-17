"""
G3 组 · PIT 回放框架
====================
任意历史交易日 as_of 下，只用 ≤ as_of 的数据重建特征快照。

核心约束（手册 PIT 纪律）:
  - 每个特征必须携带 available_time（对 A 股决策时点而言）
  - 美股 t 日收盘数据只能用于 A 股 t+1 日
  - replay 必须能检测未来数据泄露（future shift）
"""

import pandas as pd
import numpy as np
from typing import Callable, List, Optional
from dataclasses import dataclass


# ============================================================
# 1. 美股 → A 股交易日映射
# ============================================================

def map_us_to_ashare_day(
    us_dates: pd.Series,
    ashare_calendar: pd.DataFrame,
) -> pd.Series:
    """
    美股收盘日 → 次一 A 股交易日。

    规则:
        美股 t 日收盘（夏令时 04:00 / 冬令时 05:00 北京时间）
        → A 股同日 15:00 已收盘，只能用于 t+1 日。
        若 t+1 非 A 股交易日（周末/节假日），顺延到下一个交易日。

    参数:
        us_dates:        美股交易日序列（datetime）
        ashare_calendar: A 股日历 DataFrame，需含 TradeDate 与 IsTradeDay 两列

    返回:
        与 us_dates 等长的映射序列（A 股交易日）。
    """
    cal = ashare_calendar.copy()
    cal["TradeDate"] = pd.to_datetime(cal["TradeDate"])
    trade_days = pd.Series(
        cal.loc[cal["IsTradeDay"], "TradeDate"].sort_values().values
    )

    # 对每个美股日期找"严格大于它"的第一个 A 股交易日
    us = pd.to_datetime(us_dates).sort_values()
    mapped = pd.Series(index=us.index, dtype="datetime64[ns]")

    for idx, us_date in us.items():
        # 找第一个 > us_date 的交易日（用 numpy datetime64 保证类型一致）
        key = np.datetime64(us_date)
        pos = np.searchsorted(trade_days.values, key, side="right")
        if pos < len(trade_days):
            mapped[idx] = trade_days.iloc[pos]
        else:
            mapped[idx] = pd.NaT  # 美股日期晚于 A 股日历末端

    return mapped.reindex(us_dates.index)


# ============================================================
# 2. PIT 回放
# ============================================================

@dataclass
class ReplayResult:
    """单日回放结果"""
    as_of: pd.Timestamp
    features: pd.DataFrame          # 该日可用的特征快照（值 + available_time）
    dropped_future: int             # 被未来数据检查丢弃的行数
    passed: bool                    # 是否通过无前视校验


def replay(
    as_of_date: str,
    builder_fn: Callable[[pd.Timestamp], pd.DataFrame],
    calendar: Optional[pd.DataFrame] = None,
) -> ReplayResult:
    """
    对任意历史交易日做 PIT 回放。

    参数:
        as_of_date: A 股交易日（如 "2024-03-15"）
        builder_fn: 特征构造函数，签名 fn(as_of) -> DataFrame，
                    内部必须只读取 available_time ≤ as_of 的数据
        calendar:   可选。传入 A 股日历用于映射校验

    返回:
        ReplayResult
    """
    as_of = pd.Timestamp(as_of_date)

    # 调用特征构造函数——builder 内部负责数据截断
    df = builder_fn(as_of)

    # 前视校验：任何 available_time > as_of 的行都是泄露
    future_mask = pd.Series(False, index=df.index)
    if "available_time" in df.columns:
        future_mask = pd.to_datetime(df["available_time"]) > as_of
    elif calendar is not None:
        # 无 available_time 时退化为基于 TradeDate 的检查
        if "TradeDate" in df.columns:
            future_mask = pd.to_datetime(df["TradeDate"]) > as_of

    dropped = int(future_mask.sum())
    clean = df[~future_mask].copy()

    return ReplayResult(
        as_of=as_of,
        features=clean,
        dropped_future=dropped,
        passed=(dropped == 0),
    )


def validate_no_future_shift(
    df: pd.DataFrame,
    as_of: str,
    available_col: str = "available_time",
) -> bool:
    """
    断言 DataFrame 中不存在未来数据。

    参数:
        df:            待检查的 DataFrame
        as_of:         决策时点
        available_col: 可用时点列名

    返回:
        True = 无前视; False = 存在泄露（同时打印违规行）
    """
    if available_col not in df.columns:
        raise KeyError(f"缺少 {available_col} 列，无法做前视检查。")

    as_of_ts = pd.Timestamp(as_of)
    future = df[pd.to_datetime(df[available_col]) > as_of_ts]

    if len(future) > 0:
        print(f"[PIT FAIL] 发现 {len(future)} 行未来数据:")
        print(future.head(10))
        return False
    return True


# ============================================================
# 3. 批量回放
# ============================================================

def run_replay_suite(
    dates: List[str],
    builder_fn: Callable[[pd.Timestamp], pd.DataFrame],
) -> pd.DataFrame:
    """
    批量回放多个历史交易日，输出一致性报告。

    参数:
        dates:      A 股交易日列表，如 ["2024-01-05", "2024-01-08", ...]
        builder_fn: 特征构造函数（同 replay）

    返回:
        每行一个日期的回放报告:
        as_of / n_features / dropped_future / passed / elapsed_ms
    """
    import time

    rows = []
    for d in dates:
        t0 = time.time()
        try:
            r = replay(d, builder_fn)
            rows.append({
                "as_of": d,
                "n_features": len(r.features),
                "dropped_future": r.dropped_future,
                "passed": r.passed,
                "elapsed_ms": round((time.time() - t0) * 1000, 1),
            })
        except Exception as e:
            rows.append({
                "as_of": d,
                "n_features": 0,
                "dropped_future": -1,
                "passed": False,
                "elapsed_ms": round((time.time() - t0) * 1000, 1),
                "error": str(e)[:80],
            })

    report = pd.DataFrame(rows)
    n_pass = int(report["passed"].sum())
    print(f"PIT 回放: {n_pass}/{len(dates)} 日通过")
    return report
