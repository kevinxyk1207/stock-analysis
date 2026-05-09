"""
三档差异化信号引擎
短线(爆发力)、中线(趋势持续)、长线(结构稳定) — 各自独立信号
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, NamedTuple


# ── 信号定义 ──────────────────────────────────────────────

class SignalDef(NamedTuple):
    """单个信号定义"""
    key: str                 # 字典键名（对应 conditions 中的字段）
    weight: float            # 在本期限内的权重
    desc: str                # 描述


# 三档信号注册表
HORIZON_SIGNALS = {
    'short': [   # 短线 — 动量爆发
        SignalDef('volume_surge',       25, '量比: volume/MA_vol_20'),
        SignalDef('price_accel_5',      25, '5日加速度: close/close(5)-1'),
        SignalDef('range_expand_5',     15, '波动扩张: ATR(5)/close'),
        SignalDef('rel_strength_20',    20, '20日相对强度'),
        SignalDef('kdj_rebound',        15, 'J值低位反弹强度'),
    ],
    'long': [    # 长线 — 趋势稳定/回撤控制（原60d评分用错信号，修复为真正的结构信号）
        SignalDef('ma60_slope_20d',      25, 'MA60 20日斜率: 长期趋势方向'),
        SignalDef('drawdown_ratio',      20, '回撤控制: close/60日高'),
        SignalDef('low_volatility',      20, '低波动: 20日收益率标准差'),
        SignalDef('price_vs_ma120',      20, '相对MA120位置: 结构位'),
        SignalDef('weekly_bull_strength', 15, '周线多头强度: 周线确认'),
    ],
}

HORIZON_LABELS_NEW = {
    'short': '短线爆发', 'long': '长线结构',
}


# ── 新增数据列计算 ──────────────────────────────────────────

def compute_horizon_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算三档信号所需的附加数据列
    在 B1Selector.prepare_data() 末尾调用

    Args:
        df: 已包含 close, volume, J, rsi, macd_hist, weekly_bull_strength 的 DataFrame

    Returns:
        添加了新列的 DataFrame
    """
    result = df.copy()
    close = result["close"].astype(float)
    volume = result["volume"].astype(float)

    # ── 均线（多个信号复用） ──
    if 'ma_5' not in result.columns:
        result['ma_5'] = close.rolling(5, min_periods=1).mean()
    if 'ma_10' not in result.columns:
        result['ma_10'] = close.rolling(10, min_periods=1).mean()
    if 'ma_20' not in result.columns:
        result['ma_20'] = close.rolling(20, min_periods=1).mean()
    if 'ma_40' not in result.columns:
        result['ma_40'] = close.rolling(40, min_periods=1).mean()
    if 'ma_60' not in result.columns:
        result['ma_60'] = close.rolling(60, min_periods=1).mean()
    if 'ma_120' not in result.columns:
        result['ma_120'] = close.rolling(120, min_periods=1).mean()

    # ── 成交量均线 ──
    if 'vol_ma_5' not in result.columns:
        result['vol_ma_5'] = volume.rolling(5, min_periods=1).mean()
    if 'vol_ma_20' not in result.columns:
        result['vol_ma_20'] = volume.rolling(20, min_periods=1).mean()

    # ── 短线信号原始值 ──
    if 'volume_surge' not in result.columns:
        result['volume_surge'] = volume / result['vol_ma_20'].replace(0, np.nan)

    if 'price_accel_5' not in result.columns:
        result['price_accel_5'] = close.pct_change(5)

    if 'atr_5' not in result.columns:
        tr = pd.concat([
            (result['high'] - result['low']).abs(),
            (result['high'] - close.shift(1)).abs(),
            (result['low'] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        result['atr_5'] = tr.rolling(5, min_periods=1).mean()

    if 'range_expand_5' not in result.columns:
        result['range_expand_5'] = result['atr_5'] / close.replace(0, np.nan) * 100

    if 'rel_strength_20' not in result.columns:
        low_20 = close.rolling(20, min_periods=1).min()
        high_20 = close.rolling(20, min_periods=1).max()
        result['rel_strength_20'] = (close - low_20) / (high_20 - low_20 + 1e-9)

    if 'kdj_rebound' not in result.columns and 'J' in result.columns:
        j = result['J']
        j_low_5 = j.rolling(5, min_periods=1).min()
        result['kdj_rebound'] = ((j - j_low_5) / (j_low_5.abs() + 1e-9) * 100).clip(0, 100)

    # ── 反转潜力: 低位 × 缩量 (IC=0.061) ──
    if 'reversal_potential' not in result.columns:
        high_60 = close.rolling(60, min_periods=1).max()
        low_60 = close.rolling(60, min_periods=1).min()
        pos_60 = (close - low_60) / (high_60 - low_60 + 1e-9)
        # (1-位置) × (20日均量/今日量) = 低位 × 缩量
        result['reversal_potential'] = ((1 - pos_60) * result['vol_ma_20'] / volume).clip(0, 10)

    # ── 中线信号原始值 ──
    if 'ma_alignment' not in result.columns:
        alignment = (
            (result['ma_5'] > result['ma_10']).astype(int) +
            (result['ma_10'] > result['ma_20']).astype(int) +
            (result['ma_20'] > result['ma_40']).astype(int)
        )
        result['ma_alignment'] = alignment

    if 'trend_strength' not in result.columns:
        result['trend_strength'] = (result['ma_20'] - result['ma_60']) / close.replace(0, np.nan) * 100

    if 'macd_quality' not in result.columns and 'macd_hist' in result.columns:
        hist = result['macd_hist']
        rising = hist > hist.shift(1)
        result['macd_quality'] = (hist * rising).clip(lower=0) / close.replace(0, np.nan) * 10000

    if 'rsi_trending_zone' not in result.columns:
        rsi_col = result.get('rsi', pd.Series(50.0, index=result.index))
        result['rsi_trending_zone'] = np.where(
            (rsi_col >= 50) & (rsi_col <= 70), 100.0,
            np.where((rsi_col > 70) & (rsi_col <= 80),
                     100.0 - (rsi_col - 70.0) / 10.0 * 50.0,
                     np.where((rsi_col >= 40) & (rsi_col < 50),
                              (rsi_col - 40.0) / 10.0 * 50.0, 0.0))
        )

    if 'vol_trend_ratio' not in result.columns:
        result['vol_trend_ratio'] = result['vol_ma_5'] / result['vol_ma_20'].replace(0, np.nan)

    # ── 长线信号原始值 ──
    if 'ma60_slope_20d' not in result.columns:
        result['ma60_slope_20d'] = result['ma_60'].pct_change(20) * 100

    if 'drawdown_ratio' not in result.columns:
        high_60 = close.rolling(60, min_periods=1).max()
        result['drawdown_ratio'] = close / high_60.replace(0, np.nan)

    if 'low_volatility' not in result.columns:
        daily_ret = close.pct_change()
        result['low_volatility'] = daily_ret.rolling(20, min_periods=1).std() * 100

    if 'price_vs_ma120' not in result.columns:
        result['price_vs_ma120'] = close / result['ma_120'].replace(0, np.nan)

    # ── 回调买点信号 ──
    if 'high_20d' not in result.columns:
        result['high_20d'] = close.rolling(20, min_periods=1).max()

    if 'drawdown_20d' not in result.columns:
        result['drawdown_20d'] = close / result['high_20d'].replace(0, np.nan)

    if 'pullback_setup' not in result.columns:
        # 1) 长期趋势分（0-30）
        pv120 = result['price_vs_ma120'].fillna(1.0)
        trend_score = np.where(
            pv120 > 1.03, 30.0,
            np.where(pv120 > 1.0, 15.0, 0.0)
        )
        # 2) 回调到位分（0-30）
        dd20 = result['drawdown_20d'].fillna(1.0)
        pullback_score = np.where(
            (dd20 >= 0.90) & (dd20 <= 0.97), 30.0,
            np.where((dd20 >= 0.80) & (dd20 < 0.90), 15.0,
            np.where((dd20 > 0.97) & (dd20 < 1.0), 10.0, 0.0))
        )
        # 3) 缩量验证分（0-20）
        vtr = result['vol_trend_ratio'].fillna(1.0)
        vol_score = np.where(vtr < 0.85, 20.0, np.where(vtr < 1.0, 10.0, 0.0))
        # 4) 反弹信号分（0-20）
        kdj_r = result.get('kdj_rebound', pd.Series(0.0, index=result.index)).fillna(0.0)
        reversal_score = np.where(kdj_r > 20, 20.0, np.where(kdj_r > 5, 10.0, 0.0))

        result['pullback_setup'] = (trend_score + pullback_score
                                     + vol_score + reversal_score)

    # ── 资金流向信号（统一计算，始终覆盖）──
    # 主力资金流向 = (价格变化 * 成交量变化)的5日累积
    price_change = close.pct_change()
    main_flow = (price_change * 100) * (volume / result['vol_ma_20'])
    result['main_force_flow'] = main_flow.rolling(5, min_periods=1).mean()

    # 资金强度 = 成交量相对强度 * 价格趋势强度
    vol_strength = result['vol_ma_5'] / result['vol_ma_20']
    price_trend = result['ma_5'] / result['ma_20']
    result['flow_strength'] = (vol_strength * price_trend - 1.0) * 100

    # 北向资金代理：基于大盘相对强度和外资偏好特征
    market_cap_proxy = np.where(close > 20, 5.0, 2.0)  # 价格作为市值代理
    stability = 10.0 - result['low_volatility'].clip(0, 10)  # 低波动偏好
    growth = np.where(result['trend_strength'] > 0, 3.0, 0.0)  # 趋势偏好
    result['north_bound_proxy'] = market_cap_proxy + stability + growth

    return result


# ── 归一化函数（原始值 → 0-100） ─────────────────────────

def norm_volume_surge(v: float) -> float:
    """量比: 1.0=0, 1.3=30, 1.5=50, 2.0=80, 3.0=100"""
    if v < 1.0:
        return 0.0
    if v < 1.5:
        return (v - 1.0) / 0.5 * 50.0
    if v < 2.0:
        return 50.0 + (v - 1.5) / 0.5 * 30.0
    if v < 3.0:
        return 80.0 + (v - 2.0) / 1.0 * 20.0
    return 100.0


def norm_price_accel(v: float) -> float:
    """5日涨幅: <0%=0, 3%=30, 5%=50, 10%=80, 20%+=100"""
    if v < 0.0:
        return 0.0
    if v < 0.05:
        return v / 0.05 * 50.0
    if v < 0.10:
        return 50.0 + (v - 0.05) / 0.05 * 30.0
    if v < 0.20:
        return 80.0 + (v - 0.10) / 0.10 * 20.0
    return 100.0


def norm_range_expand(v: float) -> float:
    """ATR(5)/close%: <1%=0, 2%=40, 4%=80, 8%+=100"""
    v = min(v, 10.0)
    if v < 1.0:
        return 0.0
    if v < 4.0:
        return (v - 1.0) / 3.0 * 80.0
    if v < 8.0:
        return 80.0 + (v - 4.0) / 4.0 * 20.0
    return 100.0


def norm_rel_strength(v: float) -> float:
    """20日相对强度: 0-1 直接映射到 0-100"""
    return max(0.0, min(100.0, v * 100.0))


def norm_reversal_potential(v: float) -> float:
    """反转潜力: 0-10 → 0-100, 越高越好"""
    v = max(0.0, min(10.0, v))
    if v < 0.5: return v / 0.5 * 20.0       # 0-0.5 → 0-20
    if v < 1.5: return 20.0 + (v-0.5)/1.0*40.0  # 0.5-1.5 → 20-60
    if v < 3.0: return 60.0 + (v-1.5)/1.5*40.0  # 1.5-3.0 → 60-100
    return 100.0

def norm_kdj_rebound(v: float) -> float:
    """J值反弹: 直接 clip 0-100"""
    return max(0.0, min(100.0, v))


def norm_ma_alignment(v: float) -> float:
    """均线排列 0-3 → 0-100"""
    return v / 3.0 * 100.0


def norm_trend_strength(v: float) -> float:
    """趋势强度: <0=0, 3%=60, 5%+=100"""
    v = max(v, 0.0)
    if v < 3.0:
        return v / 3.0 * 60.0
    if v < 5.0:
        return 60.0 + (v - 3.0) / 2.0 * 40.0
    return 100.0


def norm_macd_quality(v: float) -> float:
    """MACD动量质量: clip 0-100"""
    return max(0.0, min(100.0, v))


def norm_rsi_trending_zone(v: float) -> float:
    """RSI在50-70=100, 70-80递减, 40-50递增, 其他=0"""
    return max(0.0, min(100.0, v))


def norm_vol_trend(v: float) -> float:
    """量能趋势: <0.9=0, 1.0=50, 1.3+=100"""
    if v < 0.9:
        return 0.0
    if v < 1.0:
        return (v - 0.9) / 0.1 * 50.0
    if v < 1.3:
        return 50.0 + (v - 1.0) / 0.3 * 50.0
    return 100.0


def norm_ma60_slope(v: float) -> float:
    """MA60斜率: <0=0, 5%=60, 10%+=100"""
    v = max(v, 0.0)
    if v < 5.0:
        return v / 5.0 * 60.0
    if v < 10.0:
        return 60.0 + (v - 5.0) / 5.0 * 40.0
    return 100.0


def norm_weekly_bull(v: float) -> float:
    """周线多头强度: clip 0-100"""
    return max(0.0, min(100.0, v * 1000.0))


def norm_drawdown(v: float) -> float:
    """回撤控制: <0.90=0, 0.95=50, 1.0=100"""
    if v < 0.90:
        return 0.0
    if v < 1.0:
        return (v - 0.90) / 0.10 * 100.0
    return 100.0


def norm_low_volatility(v: float) -> float:
    """低波动: <1%=100, 2%=60, 4%=20, 5%+=0"""
    if v < 1.0:
        return 100.0
    if v < 2.0:
        return 100.0 - (v - 1.0) / 1.0 * 40.0
    if v < 4.0:
        return 60.0 - (v - 2.0) / 2.0 * 40.0
    if v < 5.0:
        return 20.0 - (v - 4.0) / 1.0 * 20.0
    return 0.0


def norm_price_vs_ma120(v: float) -> float:
    """close/MA120: <0.95=0, 1.0=50, 1.15=100"""
    if v < 0.95:
        return 0.0
    if v < 1.0:
        return (v - 0.95) / 0.05 * 50.0
    if v < 1.15:
        return 50.0 + (v - 1.0) / 0.15 * 50.0
    return 100.0


def norm_zxdkx_ratio_chg(v: float) -> float:
    """zxdkx比值5日变化: [-0.05, +0.05] → [0, 100]"""
    v = max(-0.05, min(0.05, v))
    return (v + 0.05) / 0.10 * 100.0


def norm_zxdq_ratio_chg(v: float) -> float:
    """zxdq比值5日变化: [-0.05, +0.05] → [0, 100]"""
    v = max(-0.05, min(0.05, v))
    return (v + 0.05) / 0.10 * 100.0


def norm_vol_health_chg(v: float) -> float:
    """vol_health 5日变化: [-0.10, +0.10] → [0, 100]"""
    v = max(-0.10, min(0.10, v))
    return (v + 0.10) / 0.20 * 100.0


def norm_pullback_setup(v: float) -> float:
    """回调买点信号（恒等映射，已在0-100）"""
    return min(100.0, max(0.0, v))


def norm_main_force_flow(v: float) -> float:
    """主力资金流向：[-0.5, +0.5] → [0, 100]"""
    v = max(-0.5, min(0.5, v))
    return (v + 0.5) / 1.0 * 100.0


def norm_flow_strength(v: float) -> float:
    """资金强度：[-0.1, +0.1] → [0, 100]"""
    v = max(-0.1, min(0.1, v))
    return (v + 0.1) / 0.2 * 100.0


def norm_north_bound_proxy(v: float) -> float:
    """北向资金代理：[0, 10] → [0, 100]"""
    v = max(0.0, min(10.0, v))
    return v / 10.0 * 100.0


# 归一化函数注册表
NORM_FUNCTIONS = {
    'volume_surge': norm_volume_surge,
    'price_accel_5': norm_price_accel,
    'range_expand_5': norm_range_expand,
    'rel_strength_20': norm_rel_strength,
    'kdj_rebound': norm_kdj_rebound,
    'reversal_potential': norm_reversal_potential,
    'ma_alignment': norm_ma_alignment,
    'trend_strength': norm_trend_strength,
    'macd_quality': norm_macd_quality,
    'rsi_trending_zone': norm_rsi_trending_zone,
    'vol_trend_ratio': norm_vol_trend,
    'ma60_slope_20d': norm_ma60_slope,
    'weekly_bull_strength': norm_weekly_bull,
    'drawdown_ratio': norm_drawdown,
    'low_volatility': norm_low_volatility,
    'price_vs_ma120': norm_price_vs_ma120,
    'zxdkx_ratio_chg_5d': norm_zxdkx_ratio_chg,
    'zxdq_ratio_chg_5d': norm_zxdq_ratio_chg,
    'vol_health_chg_5d': norm_vol_health_chg,
    'pullback_setup': norm_pullback_setup,
    'main_force_flow': norm_main_force_flow,
    'flow_strength': norm_flow_strength,
    'north_bound_proxy': norm_north_bound_proxy,
}


# ── HorizonSignalEngine ─────────────────────────────────

class HorizonSignalEngine:
    """
    三档差异化信号评分引擎

    Usage:
        engine = HorizonSignalEngine()
        scores = engine.score_all(conditions_dict)
        # -> {'short': 56.2, 'medium': 71.3, 'long': 42.8}
    """

    def __init__(self, config: Dict[str, List[SignalDef]] = None):
        self.config = config or HORIZON_SIGNALS

    def score_one(self, conditions: Dict[str, float], horizon: str) -> float:
        """
        对单个期限评分

        Args:
            conditions: 字典，键为信号名，值为原始值
            horizon: 'short', 'medium', 'long'

        Returns:
            0-100 综合评分
        """
        signals = self.config.get(horizon, [])
        if not signals:
            return 0.0

        total = 0.0
        total_w = 0.0
        for sd in signals:
            raw = conditions.get(sd.key, 0.0)
            norm_fn = NORM_FUNCTIONS.get(sd.key)
            if norm_fn is None:
                continue
            score = norm_fn(raw)
            total += score * sd.weight
            total_w += sd.weight

        return min(total / total_w, 100.0) if total_w > 0 else 0.0

    def score_all(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """对所有三档期限评分"""
        return {h: self.score_one(conditions, h) for h in self.config}
