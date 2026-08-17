"""
G3 组 · 数据清洗与标准化工具库
================================
Stage 0C 通用工具：缩尾 / 标准化 / 缺失处理 / 公式 Hash。

设计约定:
  - 所有函数接受 pd.Series 或 pd.DataFrame，返回同类型
  - fix_pipeline 固定顺序: raw → winsorize → normalize
  - 禁止在 gate 后再次标准化（手册 F06 红线）
"""

import hashlib
import pandas as pd
import numpy as np
from typing import Union, Optional, List


# ============================================================
# 1. 缩尾与标准化
# ============================================================

def winsorize(
    data: Union[pd.Series, pd.DataFrame],
    lower: float = 0.01,
    upper: float = 0.99,
) -> Union[pd.Series, pd.DataFrame]:
    """
    分位数缩尾：低于 lower 分位的值截断到该分位，高于 upper 分位的值截断到该分位。

    参数:
        data:  单列 Series（时序）或多列 DataFrame（截面）
        lower: 下分位（默认 1%）
        upper: 上分位（默认 99%）

    返回:
        缩尾后的同类型对象。
    """
    if isinstance(data, pd.DataFrame):
        # 按列独立缩尾（每列是一个特征/因子）
        return data.apply(lambda col: winsorize(col, lower, upper))

    lo = data.quantile(lower)
    hi = data.quantile(upper)
    return data.clip(lo, hi)


def zscore(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    标准化：(x - mean) / std。
    对 DataFrame 按列独立标准化。
    """
    if isinstance(data, pd.DataFrame):
        return data.apply(zscore)
    mu, sd = data.mean(), data.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=data.index)
    return (data - mu) / sd


def rank_pct(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    百分位排名（0~1），用于截面排序类特征。
    对 DataFrame 按列独立排名。
    """
    if isinstance(data, pd.DataFrame):
        return data.apply(rank_pct)
    return data.rank(pct=True)


# ============================================================
# 2. 缺失值处理
# ============================================================

def handle_missing(
    df: pd.DataFrame,
    method: str = "ffill",
    flag: bool = False,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    统一缺失值处理。

    参数:
        df:     输入 DataFrame（行为日期，列为特征）
        method: "ffill" 前向填充 | "drop" 删除含缺失的行 | "none" 保持原样
        flag:   是否新增 _missing 标记列（0/1）
        limit:  ffill 的最大连续填充次数（超过则保留 NaN）

    返回:
        处理后的 DataFrame。
    """
    df = df.copy()

    if flag:
        for col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    if method == "ffill":
        df = df.ffill(limit=limit)
    elif method == "drop":
        df = df.dropna()
    elif method == "none":
        pass
    else:
        raise ValueError(f"未知缺失处理方法: {method}")

    return df


# ============================================================
# 3. 公式 Hash
# ============================================================

def compute_formula_hash(formula_str: str, version: str = "v1.0") -> str:
    """
    基于公式字符串生成 Hash，用于 state_feature_snapshot 的 feature_version。

    规则:
        hash = sha256(formula_str + "|" + version) 的前 16 位 hex
        公式或版本任一变化 → Hash 变化。

    参数:
        formula_str: 人类可读的公式描述（如 "SPY.Ret"）
        version:     公式版本号

    返回:
        16 位 hex 字符串，如 "a3f8c2d9e1b04a7f"
    """
    payload = f"{formula_str}|{version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ============================================================
# 4. 固定处理管线
# ============================================================

def fix_pipeline(
    raw: pd.DataFrame,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    normalize: str = "zscore",
    missing: str = "ffill",
) -> pd.DataFrame:
    """
    固定顺序特征处理管线：raw → winsorize → normalize。

    手册红线（F06）:
        gate 后禁止再次标准化；本管线是 raw 侧的唯一标准化入口。

    参数:
        raw:          原始特征 DataFrame（行为日期，列为特征）
        winsor_lower: 缩尾下分位
        winsor_upper: 缩尾上分位
        normalize:    "zscore" | "rank" | "none"
        missing:      缺失处理方式（见 handle_missing）

    返回:
        处理后的 DataFrame。
    """
    df = handle_missing(raw, method=missing)
    df = winsorize(df, winsor_lower, winsor_upper)

    if normalize == "zscore":
        df = zscore(df)
    elif normalize == "rank":
        df = rank_pct(df)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"未知标准化方式: {normalize}")

    return df
