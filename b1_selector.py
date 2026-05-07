"""
B1选股策略模块
基于参考项目StockTradebyZ的B1Selector实现
包含KDJ低位、知行线、周线多头排列等核心逻辑
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 三档差异化信号引擎
from horizon_signal_engine import HorizonSignalEngine, compute_horizon_columns

# 尝试导入numba加速，如果不可用则使用纯Python
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.warning("numba未安装，使用纯Python计算，性能可能较低")


@dataclass
class B1Config:
    """B1选股策略配置"""
    # KDJ参数
    j_threshold: float = -5.0  # J值阈值
    j_q_threshold: float = 0.10  # J值历史分位阈值
    kdj_n: int = 9  # KDJ周期

    # 知行线参数
    zx_m1: int = 10  # 短期均线
    zx_m2: int = 50  # 中期均线
    zx_m3: int = 200  # 长期均线
    zx_m4: int = 300  # 超长期均线
    zxdq_span: int = 10  # 双EMA跨度

    # 周线多头排列参数
    wma_short: int = 10  # 周线短期均线
    wma_mid: int = 20  # 周线中期均线
    wma_long: int = 30  # 周线长期均线

    # 成交量验证
    max_vol_lookback: int = 20  # 成交量回溯天数


# 按持有期的评分权重（优化自网格搜索 + 权重扫描回测验证）
# 格式: {zxdkx, zxdq, vol, macd, rsi, vol_surge, ma_alignment, trend_strength, pullback_setup}
# v5: 新增 pullback_setup（回调买点信号）
# v4权重（无pullback_setup）
TIME_HORIZON_WEIGHTS_V4 = {
    '10d': {'zxdkx': 0,  'zxdq': 28, 'vol': 14, 'macd': 14, 'rsi': 14,
            'vol_surge': 0, 'ma_alignment': 22, 'trend_strength': 8, 'pullback_setup': 0},
    '60d': {'zxdkx': 28, 'zxdq': 28, 'vol': 7,  'macd': 0,  'rsi': 7,
            'vol_surge': 15, 'ma_alignment': 0, 'trend_strength': 15, 'pullback_setup': 0},
}

# v5权重（含pullback_setup）- 原始版本
TIME_HORIZON_WEIGHTS_V5_ORIGINAL = {
    '10d': {'zxdkx': 0,  'zxdq': 28, 'vol': 14, 'macd': 14, 'rsi': 14,
            'vol_surge': 0, 'ma_alignment': 19, 'trend_strength': 6,
            'pullback_setup': 5},
    '60d': {'zxdkx': 28, 'zxdq': 28, 'vol': 7,  'macd': 0,  'rsi': 7,
            'vol_surge': 15, 'ma_alignment': 0, 'trend_strength': 0,
            'pullback_setup': 15},
}

# v9权重（10d用legacy因子, 60d独立由HorizonSignalEngine评分）
TIME_HORIZON_WEIGHTS = {
    '10d': {'zxdkx': 0,  'zxdq': 28, 'vol': 14, 'macd': 14, 'rsi': 14,
            'vol_surge': 0, 'ma_alignment': 22, 'trend_strength': 8},
    # 60d不再使用legacy权重——由 HorizonSignalEngine.score_one(conditions, 'long') 评分
}

HORIZON_LABELS = {'10d': '短线', '60d': '长线'}


class B1Selector:
    """
    B1选股器
    基于KDJ低位、知行线、周线多头排列的选股策略
    """

    def __init__(self, config: Optional[B1Config] = None):
        """
        初始化B1选股器

        Args:
            config: B1配置，如果为None则使用默认配置
        """
        self.config = config or B1Config()
        self.signal_engine = HorizonSignalEngine()
        logger.info(f"B1Selector初始化完成，配置: {self.config}")

    def compute_kdj(self, df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
        """
        计算KDJ指标

        Args:
            df: 包含high, low, close的DataFrame
            n: KDJ周期

        Returns:
            添加了K, D, J列的DataFrame
        """
        if df.empty:
            return df.assign(K=np.nan, D=np.nan, J=np.nan)

        # 计算RSV
        low_n = df["low"].rolling(window=n, min_periods=1).min()
        high_n = df["high"].rolling(window=n, min_periods=1).max()
        rsv = ((df["close"] - low_n) / (high_n - low_n + 1e-9) * 100).values

        # 计算K, D, J
        K = np.empty(len(rsv))
        D = np.empty(len(rsv))
        K[0] = D[0] = 50.0

        for i in range(1, len(rsv)):
            K[i] = 2.0 / 3.0 * K[i - 1] + 1.0 / 3.0 * rsv[i]
            D[i] = 2.0 / 3.0 * D[i - 1] + 1.0 / 3.0 * K[i]

        J = 3.0 * K - 2.0 * D

        return df.assign(K=K, D=D, J=J)

    def compute_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        计算MACD指标

        Returns:
            添加了macd, macd_signal, macd_hist列的DataFrame
        """
        close = df["close"].astype(float)
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal_line

        return df.assign(macd=macd_line, macd_signal=macd_signal_line, macd_hist=macd_hist)

    def compute_rsi(self, df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
        """
        计算RSI指标

        Returns:
            添加了rsi列的DataFrame
        """
        close = df["close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=n, min_periods=1).mean()
        avg_loss = loss.rolling(window=n, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return df.assign(rsi=rsi)

    def compute_zx_lines(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        计算知行线（zxdq和zxdkx）

        Args:
            df: 包含close的DataFrame

        Returns:
            (zxdq, zxdkx) 元组
        """
        close = df["close"].astype(float)

        # zxdq: 双EMA
        zxdq = close.ewm(span=self.config.zxdq_span, adjust=False).mean()
        zxdq = zxdq.ewm(span=self.config.zxdq_span, adjust=False).mean()

        # zxdkx: 四条均线的平均值
        # 使用min_periods=1确保即使数据不足也能计算
        ma1 = close.rolling(self.config.zx_m1, min_periods=1).mean()
        ma2 = close.rolling(self.config.zx_m2, min_periods=1).mean()
        ma3 = close.rolling(self.config.zx_m3, min_periods=1).mean()
        ma4 = close.rolling(self.config.zx_m4, min_periods=1).mean()

        zxdkx = (ma1 + ma2 + ma3 + ma4) / 4.0

        return zxdq, zxdkx

    def compute_weekly_ma_bull(self, df: pd.DataFrame) -> pd.Series:
        """
        计算周线多头排列标志

        Args:
            df: 日线数据DataFrame

        Returns:
            布尔序列，True表示周线多头排列
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('date') if 'date' in df.columns else df

        close = df["close"].astype(float)

        # 按周分组，取每周最后一个交易日
        # 使用ISO周编号
        idx = close.index
        year_week = idx.isocalendar().year.astype(str) + "-" + idx.isocalendar().week.astype(str).str.zfill(2)

        # 获取每周最后一个交易日的收盘价
        weekly_close = close.groupby(year_week).last()

        # 计算周线均线
        # 使用min_periods=1确保即使数据不足也能计算
        ma_short = weekly_close.rolling(window=self.config.wma_short, min_periods=1).mean()
        ma_mid = weekly_close.rolling(window=self.config.wma_mid, min_periods=1).mean()
        ma_long = weekly_close.rolling(window=self.config.wma_long, min_periods=1).mean()

        # 多头排列条件：短期 > 中期 > 长期
        bull_condition = (ma_short > ma_mid) & (ma_mid > ma_long)

        # 将周线信号映射回日线
        # 每周的所有交易日都使用该周最后一个交易日的信号
        daily_bull = pd.Series(False, index=df.index)

        for week_id, week_dates in close.groupby(year_week).groups.items():
            if week_id in bull_condition.index and bull_condition[week_id]:
                daily_bull.loc[week_dates] = True

        return daily_bull

    def _compute_weekly_bull_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        计算周线多头排列强度（均线间距之和）

        Returns:
            强度序列，正值表示多头排列强度，0表示非多头
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('date') if 'date' in df.columns else df

        close = df["close"].astype(float)
        idx = close.index
        year_week = idx.isocalendar().year.astype(str) + "-" + idx.isocalendar().week.astype(str).str.zfill(2)
        weekly_close = close.groupby(year_week).last()

        ma_short = weekly_close.rolling(window=self.config.wma_short, min_periods=1).mean()
        ma_mid = weekly_close.rolling(window=self.config.wma_mid, min_periods=1).mean()
        ma_long = weekly_close.rolling(window=self.config.wma_long, min_periods=1).mean()

        # 多头强度 = (MA短-MA中)/close + (MA中-MA长)/close（归一化到价格水平）
        weekly_strength = ((ma_short - ma_mid) + (ma_mid - ma_long)) / weekly_close
        weekly_strength = weekly_strength.clip(lower=0)  # 非多头时为0

        # 映射回日线
        daily_strength = pd.Series(0.0, index=df.index)
        for week_id, week_dates in close.groupby(year_week).groups.items():
            if week_id in weekly_strength.index:
                daily_strength.loc[week_dates] = weekly_strength[week_id]

        return daily_strength

    def _compute_vol_health(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        计算成交量健康度 = 上涨日成交量 / 总成交量（滚动窗口）

        Returns:
            0~1之间的比率，越高越健康
        """
        if df.empty:
            return pd.Series()

        close = df["close"].values
        open_p = df["open"].values
        volume = df["volume"].values
        is_up = close >= open_p  # 阳线

        result = np.zeros(len(df))
        for i in range(len(df)):
            start = max(0, i - lookback + 1)
            end = i + 1
            up_vol = np.sum(volume[start:end] * is_up[start:end])
            total_vol = np.sum(volume[start:end])
            result[i] = up_vol / total_vol if total_vol > 0 else 0.5

        return pd.Series(result, index=df.index)

    def check_max_volume_not_bearish(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        检查最近N日内最大成交量日是否为阴线

        Args:
            df: 包含open, close, volume的DataFrame
            lookback: 回溯天数

        Returns:
            布尔序列，True表示最大成交量日不是阴线
        """
        if df.empty:
            return pd.Series()

        open_prices = df["open"].values
        close_prices = df["close"].values
        volumes = df["volume"].values

        result = np.zeros(len(df), dtype=bool)

        for i in range(len(df)):
            start = max(0, i - lookback + 1)
            end = i + 1

            # 找到窗口内成交量最大的索引
            window_volumes = volumes[start:end]
            if len(window_volumes) == 0:
                result[i] = True
                continue

            max_vol_idx = np.argmax(window_volumes) + start

            # 检查最大成交量日是否为阴线（收盘价 < 开盘价）
            if max_vol_idx < len(close_prices) and max_vol_idx < len(open_prices):
                result[i] = close_prices[max_vol_idx] >= open_prices[max_vol_idx]
            else:
                result[i] = True

        return pd.Series(result, index=df.index)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预计算所有需要的指标

        Args:
            df: 原始日线数据，需包含open, high, low, close, volume

        Returns:
            添加了所有计算指标的DataFrame
        """
        if df.empty:
            return df

        # 复制数据避免修改原数据
        result = df.copy()

        # 计算KDJ
        kdj_df = self.compute_kdj(result, n=self.config.kdj_n)
        result["K"] = kdj_df["K"]
        result["D"] = kdj_df["D"]
        result["J"] = kdj_df["J"]

        # 计算知行线
        zxdq, zxdkx = self.compute_zx_lines(result)
        result["zxdq"] = zxdq
        result["zxdkx"] = zxdkx

        # 连续评分需要的指标
        close = result["close"].astype(float)
        result["zxdkx_ratio"] = close / zxdkx
        result["zxdq_zxdkx_ratio"] = zxdq / zxdkx

        # 计算MACD
        macd_df = self.compute_macd(result)
        result["macd"] = macd_df["macd"]
        result["macd_signal"] = macd_df["macd_signal"]
        result["macd_hist"] = macd_df["macd_hist"]
        result["macd_hist_ratio"] = macd_df["macd_hist"] / close

        # 计算RSI
        rsi_df = self.compute_rsi(result)
        result["rsi"] = rsi_df["rsi"]

        # 计算周线多头排列
        weekly_bull = self.compute_weekly_ma_bull(result)
        result["weekly_bull"] = weekly_bull

        # 周线多头强度
        result["weekly_bull_strength"] = self._compute_weekly_bull_strength(result)

        # 计算成交量验证
        if self.config.max_vol_lookback > 0:
            vol_check = self.check_max_volume_not_bearish(result, self.config.max_vol_lookback)
            result["vol_check"] = vol_check
            result["vol_health"] = self._compute_vol_health(result, self.config.max_vol_lookback)
        else:
            result["vol_check"] = True
            result["vol_health"] = 0.5

        # 计算三档差异化信号所需列
        result = compute_horizon_columns(result)

        # 趋势变化信号（5日变化）
        result['zxdkx_ratio_chg_5d'] = result['zxdkx_ratio'] - result['zxdkx_ratio'].shift(5)
        result['zxdq_ratio_chg_5d'] = result['zxdq_zxdkx_ratio'] - result['zxdq_zxdkx_ratio'].shift(5)
        result['vol_health_chg_5d'] = result['vol_health'] - result['vol_health'].shift(5)

        return result

    def check_b1_conditions(self, prepared_df: pd.DataFrame, date_idx: int = -1) -> Dict[str, bool]:
        """
        检查B1选股条件

        Args:
            prepared_df: 预计算好的DataFrame
            date_idx: 日期索引，-1表示最新日期

        Returns:
            包含各条件检查结果的字典
        """
        if prepared_df.empty:
            return {}

        # 获取指定日期的数据
        if date_idx < 0:
            date_idx = len(prepared_df) + date_idx

        if date_idx < 0 or date_idx >= len(prepared_df):
            return {}

        row = prepared_df.iloc[date_idx]

        results = {}

        # 条件1: KDJ低位
        # J值 < 阈值 或 J值处于历史低分位
        j_value = row.get("J", 50)
        j_history = prepared_df["J"].dropna()

        if len(j_history) > 0:
            j_percentile = (j_history < j_value).sum() / len(j_history)
        else:
            j_percentile = 0.5

        results["kdj_low"] = (j_value < self.config.j_threshold) or (j_percentile < self.config.j_q_threshold)
        results["j_value"] = j_value
        results["j_percentile"] = j_percentile

        # 条件2: 知行线条件
        close = row.get("close", 0)
        zxdq = row.get("zxdq", 0)
        zxdkx = row.get("zxdkx", 0)

        results["close_gt_zxdkx"] = close > zxdkx
        results["zxdq_gt_zxdkx"] = zxdq > zxdkx
        results["close"] = close
        results["zxdq"] = zxdq
        results["zxdkx"] = zxdkx

        # 连续评分需要的指标
        results["zxdkx_ratio"] = row.get("zxdkx_ratio", 1.0)
        results["zxdq_zxdkx_ratio"] = row.get("zxdq_zxdkx_ratio", 1.0)

        # 条件3: 周线多头排列
        results["weekly_bull"] = row.get("weekly_bull", False)
        results["weekly_bull_strength"] = row.get("weekly_bull_strength", 0.0)

        # 条件4: 成交量验证
        results["vol_check"] = row.get("vol_check", True)
        results["vol_health"] = row.get("vol_health", 0.5)

        # MACD动量
        results["macd"] = row.get("macd", 0.0)
        results["macd_signal"] = row.get("macd_signal", 0.0)
        results["macd_hist"] = row.get("macd_hist", 0.0)
        results["macd_hist_ratio"] = row.get("macd_hist_ratio", 0.0)

        # RSI
        results["rsi"] = row.get("rsi", 50.0)

        # 综合条件
        results["all_conditions"] = (
            results.get("kdj_low", False) and
            results.get("close_gt_zxdkx", False) and
            results.get("zxdq_gt_zxdkx", False) and
            results.get("weekly_bull", False) and
            results.get("vol_check", True)
        )

        # 三档差异化信号原始值
        for key in ['volume_surge', 'price_accel_5', 'range_expand_5',
                     'rel_strength_20', 'kdj_rebound',
                     'ma_alignment', 'trend_strength', 'macd_quality',
                     'rsi_trending_zone', 'vol_trend_ratio',
                     'ma60_slope_20d', 'drawdown_ratio',
                     'low_volatility', 'price_vs_ma120',
                     'pullback_setup', 'main_force_flow',
                     'flow_strength', 'north_bound_proxy']:
            results[key] = row.get(key, 0.0)

        # 趋势变化信号
        results['zxdkx_ratio_chg_5d'] = row.get('zxdkx_ratio_chg_5d', 0.0)
        results['zxdq_ratio_chg_5d'] = row.get('zxdq_ratio_chg_5d', 0.0)
        results['vol_health_chg_5d'] = row.get('vol_health_chg_5d', 0.0)

        return results

    def filter_stocks(self, stock_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        批量筛选股票（已取消b1硬门槛，全部评分）

        Args:
            stock_data_dict: 股票代码到DataFrame的字典

        Returns:
            股票信息字典（全部评分，不再硬过滤）
        """
        results = {}

        for symbol, df in stock_data_dict.items():
            try:
                if df.empty or len(df) < max(self.config.zx_m4, 60):
                    continue

                # 预计算指标
                prepared_df = self.prepare_data(df)

                # 检查最新日期的条件
                conditions = self.check_b1_conditions(prepared_df, date_idx=-1)

                # 计算综合得分（不再设硬门槛）
                score = self._calculate_score(conditions)

                results[symbol] = {
                    "symbol": symbol,
                    "score": score,
                    "conditions": conditions,
                    "close": prepared_df["close"].iloc[-1] if len(prepared_df) > 0 else 0,
                    "data_points": len(df)
                }

                logger.debug(f"股票{symbol}评分完成，得分: {score:.2f}")

            except Exception as e:
                logger.error(f"处理股票{symbol}时出错: {e}")
                continue

        return results

    def _get_raw_component_scores(self, conditions: Dict[str, any]) -> Dict[str, float]:
        """
        计算组件的原始得分（各归一化到0-100）

        Returns:
            {zxdkx, zxdq, vol, macd, rsi, vol_surge, ma_alignment, trend_strength}
        """
        scores = {}

        # close/zxdkx 偏离度
        ratio = conditions.get("zxdkx_ratio", 1.0)
        scores['zxdkx'] = min(100.0, (ratio - 1.0) * 2000.0) if ratio > 1.0 else 0.0

        # zxdq/zxdkx 偏离度
        zxdq_ratio = conditions.get("zxdq_zxdkx_ratio", 1.0)
        scores['zxdq'] = min(100.0, (zxdq_ratio - 1.0) * 2000.0) if zxdq_ratio > 1.0 else 0.0

        # 成交量健康度
        vh = conditions.get("vol_health", 0.5)
        scores['vol'] = max(0.0, min(100.0, (vh - 0.5) * 200.0))

        # MACD柱线动量
        macd_hist = conditions.get("macd_hist", 0.0)
        close_price = conditions.get("close", 1.0)
        macd_ratio = macd_hist / max(close_price, 0.01)
        scores['macd'] = min(100.0, macd_ratio * 10000.0) if macd_ratio > 0 else 0.0

        # RSI动量(50-80区间)
        rsi = conditions.get("rsi", 50.0)
        if 50 < rsi <= 80:
            scores['rsi'] = (rsi - 50.0) / 30.0 * 100.0
        elif rsi > 80:
            scores['rsi'] = 100.0
        else:
            scores['rsi'] = 0.0

        # ── 新增因子 ──

        # 放量比: 量比越高越好
        vs = conditions.get("volume_surge", 1.0)
        if vs < 1.0:
            scores['vol_surge'] = 0.0
        elif vs < 1.5:
            scores['vol_surge'] = (vs - 1.0) / 0.5 * 50.0
        elif vs < 2.0:
            scores['vol_surge'] = 50.0 + (vs - 1.5) / 0.5 * 30.0
        elif vs < 3.0:
            scores['vol_surge'] = 80.0 + (vs - 2.0) / 1.0 * 20.0
        else:
            scores['vol_surge'] = 100.0

        # 均线排列得分: 0-3分映射到0-100
        ma_al = conditions.get("ma_alignment", 0.0)
        scores['ma_alignment'] = min(100.0, ma_al / 3.0 * 100.0)

        # 趋势强度: (MA20-MA60)/close 转为0-100
        ts = conditions.get("trend_strength", 0.0)
        ts = max(ts, 0.0)
        if ts < 3.0:
            scores['trend_strength'] = ts / 3.0 * 60.0
        elif ts < 5.0:
            scores['trend_strength'] = 60.0 + (ts - 3.0) / 2.0 * 40.0
        else:
            scores['trend_strength'] = 100.0

        # 回调买点: 取原始值（已在0-100）
        scores['pullback_setup'] = conditions.get('pullback_setup', 0.0)

        # 资金流向因子（使用归一化函数）
        from horizon_signal_engine import norm_main_force_flow, norm_flow_strength, norm_north_bound_proxy

        mf_raw = conditions.get('main_force_flow', 0.0)
        scores['main_force_flow'] = norm_main_force_flow(mf_raw)

        fs_raw = conditions.get('flow_strength', 0.0)
        scores['flow_strength'] = norm_flow_strength(fs_raw)

        nb_raw = conditions.get('north_bound_proxy', 0.0)
        scores['north_bound_proxy'] = norm_north_bound_proxy(nb_raw)

        return scores

    def _calculate_score(self, conditions: Dict[str, any],
                         time_horizon: str = '10d') -> float:
        """
        计算10d短线得分（legacy因子加权，验证夏普0.95）

        Args:
            conditions: 条件字典
            time_horizon: 保留参数，仅支持'10d'
        """
        weights = TIME_HORIZON_WEIGHTS['10d']
        comp = self._get_raw_component_scores(conditions)
        total_w = sum(weights.values())

        score = sum(comp[k] * w for k, w in weights.items()) / total_w
        return min(score, 100.0)

    def _calculate_score_60d(self, conditions: Dict[str, any]) -> float:
        """
        计算60d长线得分——使用 HorizonSignalEngine 的 long 信号池
        （ma60_slope / weekly_bull_strength / drawdown_ratio / low_volatility / price_vs_ma120）
        """
        return self.signal_engine.score_one(conditions, 'long')

    def compute_horizon_scores(self, conditions: Dict[str, any]) -> Dict[str, float]:
        """
        计算双档差异化信号评分（短线爆发/长线结构）

        Returns:
            {'short': 0-100, 'long': 0-100}
        """
        return self.signal_engine.score_all(conditions)

    def batch_rank_score(self, all_conditions: Dict[str, Dict],
                          time_horizon: str = '10d') -> Dict[str, float]:
        """
        横截面排名评分（自适应归一化）

        对全市场每只股票，在核心信号上做排名分（百分位），
        然后按 TIME_HORIZON_WEIGHTS 加权汇总。

        Args:
            all_conditions: {symbol: conditions_dict} 全市场条件字典
            time_horizon: '10d' 或 '60d'

        Returns:
            {symbol: score_0_100}
        """
        if not all_conditions:
            return {}

        weights = TIME_HORIZON_WEIGHTS.get(time_horizon, TIME_HORIZON_WEIGHTS['10d'])
        # 信号键映射: 权重名 → conditions 中的键
        signal_map = {
            'zxdkx': 'zxdkx_ratio',
            'zxdq': 'zxdq_zxdkx_ratio',
            'vol': 'vol_health',
            'macd': 'macd_hist_ratio',
            'rsi': 'rsi',
        }

        # 1. 收集全市场原始值
        raw_values = {}
        for wname, ckey in signal_map.items():
            raw_values[wname] = {}
            for sym, cond in all_conditions.items():
                raw_values[wname][sym] = cond.get(ckey, 0.0)

        # 2. 排名分（百分位 * 100）
        ranked = {}
        for wname, vals in raw_values.items():
            series = pd.Series(vals)
            ranked[wname] = (series.rank(pct=True, ascending=True) * 100).to_dict()

        # 3. 加权汇总
        results = {}
        total_w = sum(weights.values())
        for sym in all_conditions:
            score = sum(ranked[wname].get(sym, 0) * w
                       for wname, w in weights.items()) / total_w
            results[sym] = min(score, 100.0)

        return results

    SOFT_COND_WEIGHTS = {
        'kdj_low': 10,
        'close_gt_zxdkx': 25,
        'zxdq_gt_zxdkx': 25,
        'weekly_bull': 15,
        'vol_check': 25,
    }

    def batch_rank_soft_conditions(self, all_conditions: Dict[str, Dict]) -> Dict[str, float]:
        """
        软化b1条件 — 用排名分替代硬门

        对每只股票的5个b1条件做排名归一化，加权汇总。
        高分组条件好的股票得分高，但不设硬门槛。

        Args:
            all_conditions: {symbol: conditions_dict} 全市场条件字典

        Returns:
            {symbol: score_0_100} 软条件得分
        """
        if not all_conditions:
            return {}

        # 信号键映射: 条件名 → conditions 中的键 → 排名方向(+1正向, -1反向)
        signal_map = {
            'kdj_low': ('j_percentile', -1),       # 越低分越好
            'close_gt_zxdkx': ('zxdkx_ratio', 1),   # 越高分越好
            'zxdq_gt_zxdkx': ('zxdq_zxdkx_ratio', 1),
            'weekly_bull': ('weekly_bull_strength', 1),
            'vol_check': ('vol_health', 1),
        }

        # 1. 收集全市场原始值
        raw_values = {}
        for cname, (ckey, _) in signal_map.items():
            raw_values[cname] = {}
            for sym, cond in all_conditions.items():
                raw_values[cname][sym] = cond.get(ckey, 0.0)

        # 2. 排名分
        ranked = {}
        for cname, (_, direction) in signal_map.items():
            series = pd.Series(raw_values[cname])
            if direction == 1:
                ranked[cname] = (series.rank(pct=True, ascending=True) * 100).to_dict()
            else:
                ranked[cname] = (series.rank(pct=True, ascending=False) * 100).to_dict()

        # 3. 加权汇总
        results = {}
        total_w = sum(self.SOFT_COND_WEIGHTS.values())
        for sym in all_conditions:
            score = sum(ranked[cname].get(sym, 0) * w
                       for cname, w in self.SOFT_COND_WEIGHTS.items()) / total_w
            results[sym] = min(score, 100.0)

        return results

    def get_top_stocks(self, stock_data_dict: Dict[str, pd.DataFrame],
                      top_n: int = 20) -> List[Dict]:
        """
        获取排名前N的股票

        Args:
            stock_data_dict: 股票数据字典
            top_n: 返回前N只股票

        Returns:
            股票信息列表，按得分降序排序
        """
        filtered_stocks = self.filter_stocks(stock_data_dict)

        # 按得分排序
        sorted_stocks = sorted(
            filtered_stocks.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return sorted_stocks[:top_n]


def test_b1_selector():
    """测试B1选股器"""
    import logging
    logging.basicConfig(level=logging.INFO)

    print("测试B1选股器")
    print("=" * 50)

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    test_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(10000, 100000, 100)
    }, index=dates)

    print("测试数据:")
    print(test_data[['open', 'high', 'low', 'close', 'volume']].head())

    # 创建B1选股器
    selector = B1Selector()

    # 预计算数据
    prepared_data = selector.prepare_data(test_data)
    print("\n预计算数据（添加了指标）:")
    print(prepared_data[['close', 'K', 'D', 'J', 'zxdq', 'zxdkx', 'weekly_bull']].tail())

    # 检查条件
    conditions = selector.check_b1_conditions(prepared_data, date_idx=-1)
    print("\nB1条件检查结果:")
    for key, value in conditions.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 测试批量筛选
    stock_dict = {'TEST': test_data}
    filtered = selector.filter_stocks(stock_dict)

    if filtered:
        print(f"\n筛选结果: 找到{len(filtered)}只符合条件的股票")
        for symbol, info in filtered.items():
            print(f"  {symbol}: 得分={info['score']:.2f}, 收盘价={info['close']:.2f}")
    else:
        print("\n未找到符合条件的股票")

    print("\n测试完成")


if __name__ == "__main__":
    test_b1_selector()