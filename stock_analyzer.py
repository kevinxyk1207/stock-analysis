"""
单股分析封装 + 星级评定 + 操作建议
纯 Python 逻辑，不依赖 Streamlit，可独立测试
"""
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LOCAL_DIR)

from enhanced_fetcher import EnhancedStockFetcher, load_stock_name_map
from b1_selector import B1Selector, B1Config
from deep_analyzer import deep_analyze


# ── B1 配置 ──

B1_CONFIG = B1Config(
    j_threshold=10.0, j_q_threshold=0.30,
    kdj_n=9,
    zx_m1=5, zx_m2=20, zx_m3=40, zx_m4=60,
    zxdq_span=10,
    wma_short=5, wma_mid=10, wma_long=15,
    max_vol_lookback=20,
)


# ── 单股分析器 ──

class SingleStockAnalyzer:
    """封装单只股票的完整分析流程"""

    def __init__(self):
        self.fetcher = EnhancedStockFetcher()
        self.selector = B1Selector(B1_CONFIG)

    def analyze(self, code: str, fundamentals: dict = None,
                deep_insights: dict = None, market_state: dict = None) -> dict:
        """
        主入口：分析单只股票

        Args:
            code: 6位股票代码
            fundamentals: 基本面数据 {revenue, revenue_growth, ...}
            deep_insights: 深度洞察 {verdict_override, deep_highlights, deep_risks}
            market_state: 市场状态 {is_bull, status, note}

        Returns:
            完整分析结果 dict，含 error 键表示失败
        """
        code = str(code).strip().zfill(6)
        fundamentals = fundamentals or {}
        deep_insights = deep_insights or {}
        market_state = market_state or {}

        # 1. 获取数据
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")

        try:
            df = self.fetcher.get_daily_data(code, start_date, end_date, adjust="qfq")
        except Exception as e:
            return {"error": f"数据获取失败: {e}"}

        if df is None or df.empty:
            return {"error": f"未找到 {code} 的数据，请检查代码是否正确"}
        if len(df) < 60:
            return {"error": f"{code} 数据不足（仅 {len(df)} 个交易日，需要至少 60 日）"}

        # 2. 获取股票名称
        name_map = load_stock_name_map()
        stock_name = name_map.get(code, code)

        # 3. 准备数据 + 检查条件
        try:
            prepared = self.selector.prepare_data(df)
            cond = self.selector.check_b1_conditions(prepared, date_idx=-1)
        except Exception as e:
            return {"error": f"指标计算失败: {e}"}

        if not cond:
            return {"error": "指标计算结果为空"}

        # 4. 评分
        score_10d = round(self.selector._calculate_score(cond, "10d"), 1)
        score_60d = round(self.selector._calculate_score_60d(cond), 1)

        # 5. 基础价格信息
        close = float(prepared["close"].iloc[-1])
        prev_close = float(prepared["close"].iloc[-2]) if len(prepared) >= 2 else close
        change_pct = round((close - prev_close) / prev_close * 100, 2)
        last_date = str(prepared.index[-1])[:10]

        # 6. 均线
        mas = {}
        for p in [5, 10, 20, 60, 120]:
            if len(prepared) >= p:
                mas[p] = round(float(prepared["close"].rolling(p).mean().iloc[-1]), 2)
            else:
                mas[p] = None

        zxdkx_val = cond.get("zxdkx", round((mas.get(5, close) + mas.get(20, close) +
                                              mas.get(60, close) + mas.get(40, mas.get(20, close))) / 4, 2))

        # 7. ATR
        tr = pd.concat([
            (prepared["high"] - prepared["low"]).abs(),
            (prepared["high"] - prepared["close"].shift(1)).abs(),
            (prepared["low"] - prepared["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = round(float(tr.rolling(14).mean().iloc[-1]), 2)

        # 8. 区间收益
        rets = {}
        for period, label in [(5, "5d"), (10, "10d"), (20, "20d"), (60, "60d")]:
            if len(prepared) > period:
                rets[label] = round((close - prepared["close"].iloc[-period - 1]) /
                                    prepared["close"].iloc[-period - 1] * 100, 1)
            else:
                rets[label] = None

        # 9. 高低点
        high_20 = round(float(prepared["close"].iloc[-20:].max()), 2) if len(prepared) >= 20 else close
        low_20 = round(float(prepared["close"].iloc[-20:].min()), 2) if len(prepared) >= 20 else close
        high_60 = round(float(prepared["close"].iloc[-60:].max()), 2) if len(prepared) >= 60 else close
        at_high = close >= high_60 * 0.98

        # 10. 星级评定
        star_result = compute_star_rating(cond, score_60d, fundamentals, deep_insights)

        # 11. 操作建议
        is_bull = market_state.get("is_bull", True)
        operation = get_operation_suggestion(star_result["stars"], is_bull, star_result["risks"])

        # 12. 深度分析（公司画像+风险扫描+阶段判断）
        deep = deep_analyze(code, fundamentals, market_state.get("market_cap"))

        # 13. K线图数据（最近60日OHLCV + 均线 + 知行线）
        chart_df = prepared.iloc[-60:].copy()
        chart_data = []
        for idx, row in chart_df.iterrows():
            chart_data.append({
                "date": str(idx)[:10],
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]),
                "ma5": round(float(row.get("ma_5", row["close"])), 2),
                "ma20": round(float(row.get("ma_20", row["close"])), 2),
                "ma60": round(float(row.get("ma_60", row["close"])), 2),
                "zxdkx": round(float(row.get("zxdkx", row["close"])), 2),
            })

        # 14. 组装返回
        return {
            "meta": {
                "code": code,
                "name": stock_name,
                "last_date": last_date,
            },
            "price": {
                "close": close,
                "prev_close": prev_close,
                "change_pct": change_pct,
            },
            "market": {
                "is_bull": is_bull,
                "status": market_state.get("status", "bull" if is_bull else "bear"),
                "note": market_state.get("note", ""),
                "avg_price": market_state.get("avg_price"),
                "ma60": market_state.get("ma60"),
            },
            "star": star_result,
            "operation": operation,
            "scores": {
                "score_10d": score_10d,
                "score_60d": score_60d,
            },
            "b1_conditions": {
                "kdj_low": cond.get("kdj_low", False),
                "j_value": round(cond.get("j_value", 50), 1),
                "j_percentile": round(cond.get("j_percentile", 0.5), 2),
                "close_gt_zxdkx": cond.get("close_gt_zxdkx", False),
                "zxdq_gt_zxdkx": cond.get("zxdq_gt_zxdkx", False),
                "weekly_bull": cond.get("weekly_bull", False),
                "vol_check": cond.get("vol_check", True),
                "all_conditions": cond.get("all_conditions", False),
            },
            "indicators": {
                "rsi": round(cond.get("rsi", 50), 1),
                "macd_quality": round(cond.get("macd_quality", 0), 1),
                "trend_strength": round(cond.get("trend_strength", 0), 1),
                "ma_alignment": int(cond.get("ma_alignment", 0)),
                "vol_trend_ratio": round(cond.get("vol_trend_ratio", 1.0), 3),
                "vol_health": round(cond.get("vol_health", 0.5), 2),
                "zxdkx_ratio": round(cond.get("zxdkx_ratio", 1.0), 3),
                "zxdq_zxdkx_ratio": round(cond.get("zxdq_zxdkx_ratio", 1.0), 3),
                "pullback_setup": round(cond.get("pullback_setup", 0), 1),
                "drawdown_ratio": round(cond.get("drawdown_ratio", 1.0), 3),
                "volume_surge": round(cond.get("volume_surge", 1.0), 2),
                "price_accel_5": round(cond.get("price_accel_5", 0), 2),
                "kdj_rebound": round(cond.get("kdj_rebound", 0), 1),
                "main_force_flow": round(cond.get("main_force_flow", 0), 3),
                "flow_strength": round(cond.get("flow_strength", 0), 2),
                "north_bound_proxy": round(cond.get("north_bound_proxy", 0), 1),
                "ma60_slope_20d": round(cond.get("ma60_slope_20d", 0), 2),
                "vol_health_chg_5d": round(cond.get("vol_health_chg_5d", 0), 3),
                "zxdkx_ratio_chg_5d": round(cond.get("zxdkx_ratio_chg_5d", 0), 3),
            },
            "price_levels": {
                "ma5": mas.get(5),
                "ma10": mas.get(10),
                "ma20": mas.get(20),
                "ma60": mas.get(60),
                "ma120": mas.get(120),
                "zxdkx": zxdkx_val,
                "high_20": high_20,
                "low_20": low_20,
                "high_60": high_60,
                "at_high": at_high,
                "atr": atr,
                "atr_pct": round(atr / close * 100, 2),
            },
            "returns": rets,
            "fundamentals": fundamentals,
            "insights": deep_insights,
            "risks": star_result["risks"],
            "deep": deep,
            "chart_data": chart_data,
            "peers": [str(p) for p in deep.get("peers", [])] if isinstance(deep, dict) else [],
        }


# ── 市场状态检测 ──

def detect_market_state(fetcher: EnhancedStockFetcher = None) -> dict:
    """
    检测当前市场牛熊状态

    三层策略：
    1. 用上证指数 sh.000001 的 close vs MA60
    2. 用本地缓存股票均价 vs MA60
    3. 未知

    Returns:
        {is_bull, status, note, avg_price, ma60}
    """
    if fetcher is None:
        fetcher = EnhancedStockFetcher()

    # 策略1: 上证指数
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
        df_idx = fetcher.get_daily_data("sh.000001", start, end, adjust="qfq")
        if df_idx is not None and len(df_idx) >= 60:
            close = df_idx["close"]
            ma60 = close.rolling(60).mean()
            if pd.notna(ma60.iloc[-1]):
                is_bull = close.iloc[-1] > ma60.iloc[-1]
                return {
                    "is_bull": bool(is_bull),
                    "status": "bull" if is_bull else "bear",
                    "note": "基于上证指数",
                    "avg_price": round(float(close.iloc[-1]), 2),
                    "ma60": round(float(ma60.iloc[-1]), 2),
                }
    except Exception:
        pass

    # 策略2: 本地缓存股票均价
    try:
        from enhanced_fetcher import load_cache_data
        raw = load_cache_data(min_rows=60, common_range=False)
        if raw and len(raw) >= 10:
            close_dict = {s: pd.Series(df["close"].values) for s, df in raw.items()}
            mc = pd.DataFrame(close_dict).mean(axis=1)
            mma60 = mc.rolling(60, min_periods=60).mean()
            if len(mma60) > 0 and pd.notna(mma60.iloc[-1]):
                is_bull = mc.iloc[-1] > mma60.iloc[-1]
                return {
                    "is_bull": bool(is_bull),
                    "status": "bull" if is_bull else "bear",
                    "note": f"基于 {len(raw)} 只缓存股票均价",
                    "avg_price": round(float(mc.iloc[-1]), 2),
                    "ma60": round(float(mma60.iloc[-1]), 2),
                }
    except Exception:
        pass

    # 策略3: 未知
    return {
        "is_bull": True,
        "status": "unknown",
        "note": "市场数据不可用，默认按牛市处理",
        "avg_price": None,
        "ma60": None,
    }


# ── 基本面数据获取 ──

def fetch_fundamentals_all() -> dict:
    """
    获取全 A 股最新季报基本面数据

    Returns:
        {code: {revenue, revenue_growth, net_profit, profit_growth, gross_margin, roe, eps}}
    """
    try:
        import akshare as ak
        today = datetime.now()
        quarter_month = ((today.month - 1) // 3) * 3 + 1
        quarter_end = datetime(today.year, quarter_month, 1) - timedelta(days=1)
        date_str = quarter_end.strftime("%Y%m%d")

        df = ak.stock_yjbb_em(date=date_str)
        result = {}
        for _, row in df.iterrows():
            code = str(row["股票代码"]).zfill(6)
            result[code] = {
                "revenue": row.get("营业总收入-营业总收入"),
                "revenue_growth": row.get("营业总收入-同比增长"),
                "net_profit": row.get("净利润-净利润"),
                "profit_growth": row.get("净利润-同比增长"),
                "gross_margin": row.get("销售毛利率"),
                "roe": row.get("净资产收益率"),
                "eps": row.get("每股收益"),
            }
        return result
    except Exception:
        return {}


# ── 深度洞察加载 ──

def load_deep_insights() -> dict:
    """加载 fundamental_insights.json"""
    _here = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(_here, "fundamental_insights.json"),
        os.path.join(_here, "data_cache", "fundamental_insights.json"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("stocks", {})
            except Exception:
                pass
    return {}


# ── 星级评定 ──

def compute_star_rating(cond: dict, score_60d: float,
                         fundamentals: dict, deep_insights: dict) -> dict:
    """
    完全复用 daily_report.py:analyze_stock 的星级算法

    评分组成：
    1) 60d 技术得分 (0-30)
    2) 风险扣分 (-20~0)
    3) MACD 动量 (0-20)
    4) 均线结构 (0-15)
    5) 趋势强度 (0-10)
    6) RSI 位置 (0-5)
    7) 量能配合 (0-10)
    8) 基本面加分 (-10~+25)
    9) 深度洞察调整 (±3)
    """
    rsi = round(cond.get("rsi", 50), 1)
    zr = round(cond.get("zxdkx_ratio", 1.0), 3)
    trend = round(cond.get("trend_strength", 0), 1)
    macd_q = round(cond.get("macd_quality", 0), 1)
    ma_align = int(cond.get("ma_alignment", 0))
    vol_trend = round(cond.get("vol_trend_ratio", 1.0), 3)

    # 风险信号
    risks = []
    if rsi < 30:
        risks.append("RSI超卖")
    if zr < 0.98:
        risks.append("跌破知行线")
    if trend < -3:
        risks.append("趋势恶化")
    # 60日跌幅过大检查（需要从外部传入 returns）
    # 这里在调用方检查

    star_score = 0

    # 1) 60d 技术得分 (0-30)
    if score_60d >= 90: star_score += 30
    elif score_60d >= 85: star_score += 25
    elif score_60d >= 80: star_score += 20
    elif score_60d >= 75: star_score += 15
    elif score_60d >= 70: star_score += 10
    else: star_score += 5

    # 2) 风险扣分
    star_score -= len(risks) * 8

    # 3) MACD 动量 (0-20)
    if macd_q > 150: star_score += 20
    elif macd_q > 100: star_score += 15
    elif macd_q > 50: star_score += 10
    elif macd_q > 0: star_score += 5

    # 4) 均线结构 (0-15)
    if ma_align == 3: star_score += 15
    elif ma_align == 2: star_score += 8

    # 5) 趋势强度 (0-10)
    if trend > 8: star_score += 10
    elif trend > 5: star_score += 7
    elif trend > 2: star_score += 4
    elif trend > 0: star_score += 2

    # 6) RSI 位置 (0-5)
    if 50 <= rsi <= 70: star_score += 5
    elif 70 < rsi <= 80: star_score += 3

    # 7) 量能配合 (0-10)
    if vol_trend > 1.3: star_score += 10
    elif vol_trend > 1.1: star_score += 7
    elif vol_trend > 0.9: star_score += 3

    # 8) 基本面加分 (-10~+25)
    fund_score = 0
    fund_details = []
    rev_g = fundamentals.get("revenue_growth", 0) or 0
    prf_g = fundamentals.get("profit_growth", 0) or 0
    margin = fundamentals.get("gross_margin", 0) or 0
    roe = fundamentals.get("roe", 0) or 0

    if rev_g and rev_g > 50: fund_score += 6; fund_details.append(f"营收增{rev_g:.0f}%")
    elif rev_g and rev_g > 30: fund_score += 4; fund_details.append(f"营收增{rev_g:.0f}%")
    elif rev_g and rev_g > 20: fund_score += 2
    elif rev_g and rev_g > 10: fund_score += 1

    if margin and margin > 50: fund_score += 5; fund_details.append(f"毛利{margin:.0f}%")
    elif margin and margin > 40: fund_score += 3
    elif margin and margin > 30: fund_score += 1

    if roe and roe > 15: fund_score += 4
    elif roe and roe > 10: fund_score += 2

    if rev_g and prf_g and prf_g > rev_g * 5 and rev_g > 0:
        fund_score -= 3; fund_details.append("利润质量存疑")

    # 9) 深度洞察调整
    if deep_insights:
        verdict_ov = deep_insights.get("verdict_override", "")
        if "核心标的" in verdict_ov or "首选" in verdict_ov:
            fund_score += 3
        if "风险最多" in verdict_ov or "不宜重仓" in verdict_ov:
            fund_score -= 3

    total_score = star_score + max(-10, min(25, fund_score))

    if total_score >= 90: stars = 5
    elif total_score >= 75: stars = 4
    elif total_score >= 60: stars = 3
    elif total_score >= 45: stars = 2
    else: stars = 1

    return {
        "stars": stars,
        "score": total_score,
        "stars_text": "★" * stars + "☆" * (5 - stars),
        "risks": risks,
        "fund_score": fund_score,
        "fund_details": fund_details,
    }


# ── 操作建议 ──

def get_operation_suggestion(stars: int, is_bull: bool, risks: list) -> str:
    """根据星级和市场状态生成操作建议"""
    if stars >= 5:
        base = "强烈建议持有/加仓，核心配置标的" if is_bull else "标的优秀但熊市中建议轻仓持有"
    elif stars >= 4:
        base = "建议买入或继续持有，良好入场时机" if is_bull else "标的良好，熊市建议等待或极轻仓"
    elif stars >= 3:
        base = "建议观望，等待更好的入场信号"
    elif stars >= 2:
        base = "建议减仓或避免新开仓，风险偏高"
    else:
        base = "不建议参与，风险过高"

    if risks:
        base += f"\n\n风险提示：{'、'.join(risks)}"
    return base
