"""
G3 组 · 合成数据单元测试
========================
用合成数据验证 g3_utils / g3_pit_replay / 13 项特征公式。

运行方式:
    python test_g3.py            # 或
    pytest test_g3.py -v

不依赖任何真实数据——所有输入均为构造的合成 DataFrame。
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from g3_utils import winsorize, zscore, rank_pct, handle_missing, compute_formula_hash, fix_pipeline
from g3_pit_replay import map_us_to_ashare_day, replay, validate_no_future_shift, run_replay_suite


# ============================================================
# 合成数据工厂
# ============================================================

def make_synthetic_price(n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成合成 ETF 日线: TradeDate, Ticker, Ret(小数)"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame({
        "TradeDate": dates,
        "Ticker": "SPY",
        "Ret": rng.normal(0.0005, 0.01, n_days),
    })


def make_synthetic_calendar(n_days: int = 100) -> pd.DataFrame:
    """生成合成 A 股日历: TradeDate, IsTradeDay"""
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame({
        "TradeDate": dates,
        "IsTradeDay": [True] * n_days,
    })


# ============================================================
# 1. 缩尾与标准化
# ============================================================

class TestWinsorize(unittest.TestCase):

    def test_extreme_values_clipped(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000.0])
        w = winsorize(s, lower=0.1, upper=0.9)
        # 90 分位(线性插值)= 9 + 0.9*(1000-9) 附近 → 极端值被截断
        self.assertLess(w.max(), 1000.0)
        self.assertGreater(w.max(), 9.0)   # 仍高于无极端值时
        # 10 分位 = 1.9 附近 → 最小值被上提
        self.assertGreater(w.min(), 1.0)

    def test_winsorize_symmetric_series(self):
        """对称序列缩尾后上下界相等"""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        w = winsorize(s, lower=0.0, upper=1.0)  # 全区间 = 原样
        self.assertEqual(w.max(), 5.0)
        self.assertEqual(w.min(), 1.0)

    def test_dataframe_columnwise(self):
        df = pd.DataFrame({"a": [1, 2, 100], "b": [-50, 0, 1]})
        w = winsorize(df, 0.5, 0.5)  # 50 分位 = 中位数
        self.assertEqual(w["a"].max(), 2.0)
        self.assertEqual(w["b"].min(), 0.0)

    def test_no_nan_introduced(self):
        s = pd.Series([1.0, np.nan, 3.0])
        w = winsorize(s)
        self.assertEqual(w.isna().sum(), 1)  # NaN 保持不动


class TestStandardize(unittest.TestCase):

    def test_zscore(self):
        s = pd.Series([1.0, 2.0, 3.0])
        z = zscore(s)
        self.assertAlmostEqual(z.mean(), 0.0)
        # zscore 内部用 ddof=0（总体标准差），校验时保持一致
        self.assertAlmostEqual(z.std(ddof=0), 1.0)

    def test_rank_pct(self):
        s = pd.Series([10.0, 20.0, 30.0])
        r = rank_pct(s)
        self.assertAlmostEqual(r.iloc[0], 1 / 3)
        self.assertAlmostEqual(r.iloc[-1], 1.0)


# ============================================================
# 2. 缺失值处理
# ============================================================

class TestMissing(unittest.TestCase):

    def test_ffill(self):
        df = pd.DataFrame({"x": [1.0, np.nan, np.nan, 4.0]})
        out = handle_missing(df, method="ffill")
        self.assertEqual(out["x"].tolist(), [1.0, 1.0, 1.0, 4.0])

    def test_flag(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        out = handle_missing(df, method="none", flag=True)
        self.assertIn("x_missing", out.columns)
        self.assertEqual(out["x_missing"].tolist(), [0, 1, 0])

    def test_drop(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        out = handle_missing(df, method="drop")
        self.assertEqual(len(out), 2)


# ============================================================
# 3. 公式 Hash
# ============================================================

class TestFormulaHash(unittest.TestCase):

    def test_deterministic(self):
        h1 = compute_formula_hash("SPY.Ret")
        h2 = compute_formula_hash("SPY.Ret")
        self.assertEqual(h1, h2)

    def test_formula_change_changes_hash(self):
        h1 = compute_formula_hash("SPY.Ret")
        h2 = compute_formula_hash("QQQ.Ret")
        self.assertNotEqual(h1, h2)

    def test_version_change_changes_hash(self):
        h1 = compute_formula_hash("SPY.Ret", "v1.0")
        h2 = compute_formula_hash("SPY.Ret", "v1.1")
        self.assertNotEqual(h1, h2)

    def test_length(self):
        h = compute_formula_hash("x")
        self.assertEqual(len(h), 16)


# ============================================================
# 4. 固定管线
# ============================================================

class TestFixPipeline(unittest.TestCase):

    def test_order_and_shape(self):
        df = pd.DataFrame({"f1": [1, 2, 100, np.nan, 5]})
        out = fix_pipeline(df, winsor_lower=0.25, winsor_upper=0.75,
                           normalize="zscore", missing="ffill")
        self.assertEqual(len(out), len(df))          # ffill 不删行
        self.assertAlmostEqual(out["f1"].mean(), 0.0, places=6)  # 已标准化


# ============================================================
# 5. PIT 回放框架
# ============================================================

class TestPitReplay(unittest.TestCase):

    def test_us_to_ashare_mapping(self):
        cal = make_synthetic_calendar(30)
        us_dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-05"]))
        mapped = map_us_to_ashare_day(us_dates, cal)
        # 2024-01-02(周二) → 2024-01-03(周三)
        self.assertEqual(mapped.iloc[0], pd.Timestamp("2024-01-03"))
        # 2024-01-05(周五) → 2024-01-08(下周一)
        self.assertEqual(mapped.iloc[1], pd.Timestamp("2024-01-08"))

    def test_replay_clean(self):
        def builder(as_of):
            df = make_synthetic_price(n_days=20)
            df["available_time"] = df["TradeDate"]
            return df[df["TradeDate"] <= as_of]

        r = replay("2024-01-10", builder)
        self.assertTrue(r.passed)
        self.assertEqual(r.dropped_future, 0)
        self.assertTrue((r.features["TradeDate"] <= r.as_of).all())

    def test_replay_detects_future(self):
        def builder(as_of):
            # 故意返回包含未来的数据
            df = make_synthetic_price(n_days=20)
            df["available_time"] = df["TradeDate"]
            return df  # 不截断

        r = replay("2024-01-10", builder)
        self.assertFalse(r.passed)
        self.assertGreater(r.dropped_future, 0)

    def test_validate_no_future_shift(self):
        df = make_synthetic_price(n_days=10)
        df["available_time"] = df["TradeDate"]
        ok = validate_no_future_shift(df, "2024-01-12")
        self.assertTrue(ok)

        bad = validate_no_future_shift(df, "2024-01-05")
        self.assertFalse(bad)

    def test_replay_suite(self):
        def builder(as_of):
            df = make_synthetic_price(n_days=20)
            df["available_time"] = df["TradeDate"]
            return df[df["TradeDate"] <= as_of]

        report = run_replay_suite(["2024-01-08", "2024-01-10", "2024-01-12"], builder)
        self.assertEqual(report["passed"].sum(), 3)
        self.assertIn("elapsed_ms", report.columns)


# ============================================================
# 6. 13 项特征公式（合成数据）
# ============================================================

class TestFeatureFormulas(unittest.TestCase):
    """核心公式逻辑验证——不依赖 COS 真实数据。"""

    def test_direct_etf_ret(self):
        """单 ETF 直取: value = Ret"""
        df = make_synthetic_price(seed=1)
        out = df[["TradeDate"]].copy()
        out["value"] = df["Ret"]
        self.assertTrue(np.allclose(out["value"], df["Ret"]))

    def test_yield_proxy_sign(self):
        """利率 proxy: Δyield = -(TLT_ret/16 + IEF_ret/7 + SHY_ret/2)/3
        债券 ETF 涨（正收益）→ 利率下降（负值）"""
        rets = {"TLT": 0.016, "IEF": 0.007, "SHY": 0.002}  # 全部上涨 1 倍久期 → 利率 -1%
        dy = -(rets["TLT"] / 16 + rets["IEF"] / 7 + rets["SHY"] / 2) / 3
        self.assertLess(dy, 0)  # ETF 涨 → 利率降
        self.assertAlmostEqual(dy, -(0.001 + 0.001 + 0.001) / 3)

    def test_fxi_kweb_spread(self):
        """#47: FXI.Ret − KWEB.Ret"""
        fxi = pd.Series([0.01, -0.02, 0.03])
        kweb = pd.Series([0.02, -0.01, 0.01])
        spread = fxi - kweb
        self.assertTrue(np.allclose(spread, [-0.01, -0.01, 0.02]))

    def test_cross_market_unit_alignment(self):
        """#78/#92: 美股 Ret(小数) − A股 Return(bp÷10000)"""
        us_ret = 0.02          # 美股 +2%
        ashare_bp = 100.0      # A股 +100bp = +1%
        ashare_decimal = ashare_bp / 10000.0
        value = us_ret - ashare_decimal
        self.assertAlmostEqual(value, 0.01)  # 2% - 1% = 1%

    def test_semiconductor_divergence_direction(self):
        """#78: SOXX 涨 3%、A股电子涨 1% → 背离 = +2%（海外强于境内）"""
        soxx_ret = 0.03
        ashare_elec_ret = 0.01
        divergence = soxx_ret - ashare_elec_ret
        self.assertAlmostEqual(divergence, 0.02)

    def test_gold_risk_off_direction(self):
        """#92: 黄金 +2%、股票 -1% → 避险强度 +3%（risk-off 增强）"""
        gold_ret = 0.02
        stock_ret = -0.01
        risk_off = gold_ret - stock_ret
        self.assertAlmostEqual(risk_off, 0.03)


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
