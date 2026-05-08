
"""
B1选股策略回测模块
验证评分系统的预测能力——高分股票是否在未来表现更好
"""
import os
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _compute_tp_sl_return(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    entry_idx: int, holding_period: int,
    take_profit: Optional[float], stop_loss: Optional[float]
) -> Optional[float]:
    """
    计算带止盈止损的前向收益

    从 entry_idx+1 到 entry_idx+holding_period 逐日扫描：
      1) 如果盘中最高 >= 入场价*(1+TP) → 止盈退出, 收益=TP
      2) 如果盘中最低 <= 入场价*(1-SL) → 止损退出, 收益=-SL
      3) 到期未触发 → 以收盘价计算收益

    Returns:
        收益率（小数），None表示数据不足
    """
    n = len(close)
    fwd_idx = entry_idx + holding_period
    if fwd_idx >= n and take_profit is None and stop_loss is None:
        return None
    end_idx = min(fwd_idx, n - 1)
    entry_price = close[entry_idx]

    for j in range(entry_idx + 1, end_idx + 1):
        if take_profit is not None and high[j] >= entry_price * (1 + take_profit):
            return take_profit
        if stop_loss is not None and low[j] <= entry_price * (1 - stop_loss):
            return -stop_loss

    if fwd_idx < n:
        return (close[fwd_idx] - entry_price) / entry_price
    return None


def compute_dynamic_exit_return(
    close: np.ndarray,
    zxdkx: np.ndarray,
    zxdq: np.ndarray,
    entry_idx: int,
    max_holding: int,
    exit_mode: str,
) -> Optional[float]:
    """
    计算基于条件退出的前向收益

    从 entry_idx+1 开始逐日检查退出条件，条件满足则当日收盘退出。
    到 max_holding 仍未触发 → 到期收盘退出。

    Args:
        close, zxdkx, zxdq: numpy arrays（需同源，如 prepared_df["close"].values）
        entry_idx: 入场索引
        max_holding: 最长持有天数
        exit_mode:
            'close_lt_zxdkx'       — 收盘 < zxdkx
            'zxdq_lt_zxdkx'        — zxdq < zxdkx
            'both'                  — 两者任一
            'close_lt_zxdkx_995'    — 收盘 < zxdkx * 0.995 (紧止损)

    Returns:
        收益率，None表示数据不足
    """
    n = len(close)
    end_idx = min(entry_idx + max_holding, n - 1)
    entry_price = close[entry_idx]

    for j in range(entry_idx + 1, end_idx + 1):
        if exit_mode == 'close_lt_zxdkx':
            if close[j] < zxdkx[j]:
                return (close[j] - entry_price) / entry_price
        elif exit_mode == 'zxdq_lt_zxdkx':
            if zxdq[j] < zxdkx[j]:
                return (close[j] - entry_price) / entry_price
        elif exit_mode == 'both':
            if close[j] < zxdkx[j] or zxdq[j] < zxdkx[j]:
                return (close[j] - entry_price) / entry_price
        elif exit_mode == 'close_lt_zxdkx_995':
            if close[j] < zxdkx[j] * 0.995:
                return (close[j] - entry_price) / entry_price
        else:
            raise ValueError(f"未知退出模式: {exit_mode}")

    fwd_idx = entry_idx + max_holding
    if fwd_idx < n:
        return (close[fwd_idx] - entry_price) / entry_price
    return None


@dataclass
class BacktestConfig:
    """回测配置"""
    holding_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    score_buckets: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    )
    min_signals_per_bucket: int = 5
    store_all_signals: bool = True


@dataclass
class BacktestSignal:
    """单个回测信号"""
    date: pd.Timestamp
    score: float
    conditions: Dict[str, any]
    forward_returns: Dict[int, Optional[float]]


@dataclass
class BacktestBucketStats:
    """评分区间聚合统计"""
    bucket_name: str
    count: int
    avg_return: float
    win_rate: float
    avg_score: float


@dataclass
class BacktestConditionStats:
    """各条件单独统计"""
    condition_name: str
    signal_count: int
    avg_return: float
    win_rate: float


@dataclass
class BacktestResult:
    """完整回测结果"""
    symbol: str
    data_range: str
    data_days: int
    total_signals: int
    holding_period: int
    bucket_stats: List[BacktestBucketStats]
    condition_stats: List[BacktestConditionStats]
    spearman_corr: Optional[float]
    all_signals: List[BacktestSignal] = field(default_factory=list)


class B1Backtest:
    """B1回测器"""

    def __init__(self, selector, bt_config=None):
        self.selector = selector
        self.config = bt_config or BacktestConfig()

    def compute_warmup(self) -> int:
        """计算需要的预热期"""
        cfg = self.selector.config
        max_period = max(cfg.zx_m4, cfg.wma_long * 5, cfg.zx_m3, cfg.zx_m2)
        return max_period + 50

    def run_single(self, df: pd.DataFrame, symbol: str,
                   take_profit: Optional[float] = None,
                   stop_loss: Optional[float] = None) -> List[BacktestResult]:
        """对一只股票运行回测（对每个holding_period返回一个BacktestResult）

        Args:
            df: 日线数据
            symbol: 股票代码
            take_profit: 止盈阈值(小数)，None表示不使用
            stop_loss: 止损阈值(小数)，None表示不使用
        """
        if df.empty or len(df) < 100:
            logger.warning(f"{symbol}: 数据不足, 跳过回测")
            return []

        warmup = self.compute_warmup()
        prepared_df = self.selector.prepare_data(df)

        if len(prepared_df) <= warmup:
            logger.warning(f"{symbol}: 数据{len(prepared_df)}天不足预热期{warmup}天, 跳过")
            return []

        all_signals = self._walk_forward(prepared_df, warmup,
                                         take_profit=take_profit, stop_loss=stop_loss)
        if not all_signals:
            logger.info(f"{symbol}: 无有效信号")
            return []

        results = []
        for hp in self.config.holding_periods:
            bucket_stats, condition_stats, corr = self._aggregate(all_signals, hp)
            data_range = f"{prepared_df.index[0].strftime('%Y-%m-%d')} ~ {prepared_df.index[-1].strftime('%Y-%m-%d')}"
            result = BacktestResult(
                symbol=symbol,
                data_range=data_range,
                data_days=len(prepared_df),
                total_signals=len(all_signals),
                holding_period=hp,
                bucket_stats=bucket_stats,
                condition_stats=condition_stats,
                spearman_corr=corr,
                all_signals=all_signals if self.config.store_all_signals else []
            )
            results.append(result)
        return results

    def _walk_forward(self, prepared_df: pd.DataFrame, warmup: int,
                      take_profit: Optional[float] = None,
                      stop_loss: Optional[float] = None) -> List[BacktestSignal]:
        """核心：遍历每日计算评分和前向收益"""
        signals = []
        close_col = prepared_df["close"].values
        high_col = prepared_df["high"].values
        low_col = prepared_df["low"].values
        j_values = prepared_df["J"].values
        n_rows = len(prepared_df)

        has_tp_sl = take_profit is not None or stop_loss is not None

        for i in range(warmup, n_rows):
            j_history = j_values[max(0, i - 252):i + 1]
            j_history = j_history[~np.isnan(j_history)]
            j_percentile = np.mean(j_history < j_values[i]) if len(j_history) > 0 else 0.5

            conditions = self.selector.check_b1_conditions(prepared_df, date_idx=i)
            conditions["j_percentile"] = j_percentile
            conditions["j_value"] = j_values[i]

            cfg = self.selector.config
            conditions["kdj_low"] = (j_values[i] < cfg.j_threshold) or (j_percentile < cfg.j_q_threshold)
            conditions["all_conditions"] = (
                conditions.get("kdj_low", False)
                and conditions.get("close_gt_zxdkx", False)
                and conditions.get("zxdq_gt_zxdkx", False)
                and conditions.get("weekly_bull", False)
                and conditions.get("vol_check", True)
            )

            score = self.selector._calculate_score(conditions, '10d')

            forward_returns = {}
            for hp in self.config.holding_periods:
                if has_tp_sl:
                    forward_returns[hp] = _compute_tp_sl_return(
                        high_col, low_col, close_col, i, hp, take_profit, stop_loss)
                else:
                    fwd_idx = i + hp
                    if fwd_idx < n_rows:
                        forward_returns[hp] = (close_col[fwd_idx] - close_col[i]) / close_col[i]
                    else:
                        forward_returns[hp] = None

            signals.append(BacktestSignal(
                date=prepared_df.index[i], score=score,
                conditions=conditions, forward_returns=forward_returns
            ))
        return signals

    def _aggregate(self, signals, holding_period):
        """按评分区间聚合"""
        valid = [s for s in signals if s.forward_returns.get(holding_period) is not None]
        if not valid:
            return [], [], None

        scores = np.array([s.score for s in valid])
        returns = np.array([s.forward_returns[holding_period] for s in valid])

        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(scores, returns)
            corr = corr if not np.isnan(corr) else None
        except ImportError:
            corr = None

        bucket_stats = []
        for lo, hi in self.config.score_buckets:
            bucket_signals = [s for s in valid if lo <= s.score < hi]
            if len(bucket_signals) < self.config.min_signals_per_bucket:
                continue
            rets = [s.forward_returns[holding_period] for s in bucket_signals]
            bucket_stats.append(BacktestBucketStats(
                bucket_name=f"{int(lo):>3}~{int(hi):<3}",
                count=len(bucket_signals),
                avg_return=np.mean(rets) * 100,
                win_rate=np.mean([r > 0 for r in rets]) * 100,
                avg_score=np.mean([s.score for s in bucket_signals])
            ))

        all_rets = [s.forward_returns[holding_period] for s in valid]
        bucket_stats.append(BacktestBucketStats(
            bucket_name="  全部", count=len(valid),
            avg_return=np.mean(all_rets) * 100,
            win_rate=np.mean([r > 0 for r in all_rets]) * 100,
            avg_score=np.mean([s.score for s in valid])
        ))

        condition_names = ["kdj_low", "close_gt_zxdkx", "zxdq_gt_zxdkx", "weekly_bull", "vol_check"]
        condition_stats = []
        for cname in condition_names:
            cond_signals = [s for s in valid if s.conditions.get(cname, False)]
            if len(cond_signals) < self.config.min_signals_per_bucket:
                continue
            crets = [s.forward_returns[holding_period] for s in cond_signals]
            condition_stats.append(BacktestConditionStats(
                condition_name=cname, signal_count=len(cond_signals),
                avg_return=np.mean(crets) * 100,
                win_rate=np.mean([r > 0 for r in crets]) * 100
            ))

        return bucket_stats, condition_stats, corr

    def export_dataframe(self, results):
        """导出信号明细到DataFrame"""
        rows = []
        for result in results:
            for sig in result.all_signals:
                row = {"symbol": result.symbol, "date": sig.date, "score": sig.score}
                for k, v in sig.conditions.items():
                    if isinstance(v, (int, float)):
                        row[k] = v
                    else:
                        row[k] = int(v)
                for hp, ret in sig.forward_returns.items():
                    row[f"return_{hp}d"] = ret
                rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()


class BacktestReport:
    """回测报告格式化输出"""

    @staticmethod
    def print_report(results):
        """打印完整回测报告"""
        if not results:
            print("\n无回测结果")
            return

        r = results[0]
        print("\n" + "=" * 72)
        print(f"  B1选股回测报告")
        print(f"  股票: {r.symbol}")
        print(f"  数据范围: {r.data_range} ({r.data_days}个交易日)")
        print(f"  信号总数: {r.total_signals}")
        print("=" * 72)

        for result in results:
            BacktestReport._print_single_result(result)

        print("\n评分-收益相关性 (Spearman):")
        for result in results:
            corr_str = f"{result.spearman_corr:.4f}" if result.spearman_corr is not None else "N/A"
            print(f"  持仓{result.holding_period:>2}d: {corr_str}")
        print("=" * 72 + "\n")

    @staticmethod
    def _print_single_result(result):
        hp = result.holding_period
        print(f"\n--- 持仓{hp:>2}d ---")
        if not result.bucket_stats:
            print("  无有效信号")
            return

        print(f"  {'评分区间':<12} {'信号数':<8} {'平均收益%':<12} {'胜率%':<10} {'平均评分':<10}")
        print(f"  {'-'*50}")
        for bs in result.bucket_stats:
            print(f"  {bs.bucket_name:<12} {bs.count:<8} {bs.avg_return:<+12.2f} {bs.win_rate:<10.1f} {bs.avg_score:<10.2f}")

        if result.condition_stats:
            print(f"\n  各条件表现:")
            print(f"  {'条件名':<20} {'信号数':<8} {'平均收益%':<12} {'胜率%':<10}")
            print(f"  {'-'*48}")
            for cs in result.condition_stats:
                print(f"  {cs.condition_name:<20} {cs.signal_count:<8} {cs.avg_return:<+12.2f} {cs.win_rate:<10.1f}")

    @staticmethod
    def plot_results(results, save_path=None):
        """生成图表（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib未安装，跳过图表")
            return

        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, result in zip(axes, results):
            buckets = [bs for bs in result.bucket_stats if bs.count > 0 and bs.bucket_name != "  全部"]
            if not buckets:
                continue
            names = [b.bucket_name for b in buckets]
            avg_rets = [b.avg_return for b in buckets]
            win_rates = [b.win_rate for b in buckets]
            x = np.arange(len(names))
            ax.bar(x - 0.175, avg_rets, 0.35, label="平均收益%", color="steelblue")
            ax2 = ax.twinx()
            ax2.bar(x + 0.175, win_rates, 0.35, label="胜率%", color="coral", alpha=0.7)
            ax.set_xlabel("评分区间")
            ax.set_ylabel("平均收益%")
            ax2.set_ylabel("胜率%")
            ax.set_title(f"持仓{result.holding_period}d - {result.symbol}")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45)
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图表已保存: {save_path}")
        else:
            plt.show()


class BacktestMockDataGenerator:
    """回测模拟数据生成"""

    @staticmethod
    def _make_dates(n_days):
        """生成连续交易日日期序列，确保长度精确"""
        start = pd.Timestamp("2000-01-01")
        # 生成足够多的日历日，然后过滤为交易日，确保有n_days个
        dates = pd.date_range(start=start, periods=int(n_days * 1.4), freq="B")
        return dates[:n_days]

    @staticmethod
    def random_walk(n_days=1250, base_price=50.0, volatility=0.015, seed=42):
        np.random.seed(seed)
        dates = BacktestMockDataGenerator._make_dates(n_days)
        returns = np.random.randn(n_days) * volatility
        price = base_price * np.exp(np.cumsum(returns))
        open_p = price * (1 + np.random.randn(n_days) * volatility * 0.3)
        high_p = np.maximum(open_p, price) * (1 + np.abs(np.random.randn(n_days)) * volatility * 0.5)
        low_p = np.minimum(open_p, price) * (1 - np.abs(np.random.randn(n_days)) * volatility * 0.5)
        volume = np.random.randint(1000000, 10000000, n_days)
        return pd.DataFrame({
            "open": open_p, "high": high_p, "low": low_p,
            "close": price, "volume": volume
        }, index=dates)

    @staticmethod
    def trend_step(n_days=1250, base_price=50.0, volatility=0.012, seed=42):
        np.random.seed(seed)
        dates = BacktestMockDataGenerator._make_dates(n_days)
        seg_len = n_days // 6
        trends = [0.001, 0.002, -0.001, 0.003, 0.0, -0.002]
        trend_returns = np.repeat(trends, seg_len)
        if len(trend_returns) < n_days:
            trend_returns = np.pad(trend_returns, (0, n_days - len(trend_returns)), "edge")
        noise = np.random.randn(n_days) * volatility
        daily_returns = trend_returns + noise
        price = base_price * np.exp(np.cumsum(daily_returns))
        open_p = price * (1 + np.random.randn(n_days) * volatility * 0.3)
        high_p = np.maximum(open_p, price) * (1 + np.abs(np.random.randn(n_days)) * volatility * 0.5)
        low_p = np.minimum(open_p, price) * (1 - np.abs(np.random.randn(n_days)) * volatility * 0.5)
        volume = np.random.randint(1000000, 10000000, n_days)
        return pd.DataFrame({
            "open": open_p, "high": high_p, "low": low_p,
            "close": price, "volume": volume
        }, index=dates)

    @staticmethod
    def mean_reverting(n_days=1250, base_price=50.0, theta=0.01, sigma=0.02, seed=42):
        np.random.seed(seed)
        dates = BacktestMockDataGenerator._make_dates(n_days)
        price = np.zeros(n_days)
        price[0] = base_price
        for i in range(1, n_days):
            price[i] = price[i-1] + theta * (base_price - price[i-1]) + sigma * np.random.randn()
        price = np.abs(price) + 1
        open_p = price * (1 + np.random.randn(n_days) * sigma * 0.3)
        high_p = np.maximum(open_p, price) * (1 + np.abs(np.random.randn(n_days)) * sigma * 0.5)
        low_p = np.minimum(open_p, price) * (1 - np.abs(np.random.randn(n_days)) * sigma * 0.5)
        volume = np.random.randint(1000000, 10000000, n_days)
        return pd.DataFrame({
            "open": open_p, "high": high_p, "low": low_p,
            "close": price, "volume": volume
        }, index=dates)


def test_backtest():
    """测试回测模块"""
    from b1_selector import B1Selector, B1Config

    print("=" * 60)
    print("B1回测模块测试")
    print("=" * 60)

    config = B1Config(
        j_threshold=10.0, j_q_threshold=0.30,
        zx_m1=5, zx_m2=20, zx_m3=40, zx_m4=60,
        wma_short=5, wma_mid=10, wma_long=15,
    )
    selector = B1Selector(config)
    bt_config = BacktestConfig(holding_periods=[5, 10, 20])

    # 测试1: 趋势分段数据
    print("\n测试1: 趋势分段数据 (预期: 评分-收益正相关)")
    df = BacktestMockDataGenerator.trend_step(1250, seed=42)
    backtest = B1Backtest(selector, bt_config)
    results = backtest.run_single(df, "TEST_TREND")
    if results:
        print(f"  信号数: {results[0].total_signals}")
        for r in results:
            print(f"  持仓{r.holding_period:>2}d: 相关性={r.spearman_corr:.4f}")
    BacktestReport.print_report(results)

    # 测试2: 随机游走
    print("\n测试2: 随机游走数据")
    df2 = BacktestMockDataGenerator.random_walk(1250, seed=42)
    results2 = backtest.run_single(df2, "TEST_RANDOM")
    if results2:
        print(f"  信号数: {results2[0].total_signals}")

    # 测试3: 数据不足
    print("\n测试3: 数据不足 (预期: 0结果)")
    small_df = BacktestMockDataGenerator.random_walk(50, seed=42)
    results3 = backtest.run_single(small_df, "TEST_SMALL")
    print(f"  结果数: {len(results3)}")

    # 测试4: 导出DataFrame
    print("\n测试4: 导出DataFrame")
    if results:
        export_df = backtest.export_dataframe(results)
        print(f"  行数: {len(export_df)}, 列: {list(export_df.columns)}")

    print("\n测试完成")


@dataclass
class CrossSectionalEval:
    """单次横截面评估结果"""
    date: pd.Timestamp
    n_stocks: int
    score_groups: Dict[int, List[float]]  # group_id -> [forward_returns]


@dataclass
class CrossSectionalResult:
    """横截面回测结果"""
    holding_period: int
    total_evaluations: int
    group_stats: List["GroupStats"]


@dataclass
class GroupStats:
    """分组统计"""
    group_name: str
    avg_score: float
    avg_return: float
    win_rate: float
    n_samples: int
    n_positive: int
    n_negative: int


class CrossSectionalBacktest:
    """
    横截面回测

    核心思想：在每个时间点，对所有可评估的股票评分，
    按得分分组，看高分组未来收益是否跑赢低分组。
    这才是B1选股策略正确的验证方式。
    """

    # 自适应模式下三档的评分混合权重 (base, soft, trend)
    ADAPTIVE_WEIGHTS = {
        'short':  (0.80, 0.00, 0.20),
        'long':   (0.60, 0.25, 0.15),
    }

    # mapping: horizon名 → batch_rank_score 的 time_horizon 参数
    HORIZON_TO_TIME = {'short': '10d', 'long': '60d'}

    def __init__(self, selector, bt_config=None):
        self.selector = selector
        self.config = bt_config or BacktestConfig()

    def run(self, stock_data_dict: Dict[str, pd.DataFrame],
            eval_step: int = 20, n_groups: int = 5,
            warmup_extra: int = 0,
            take_profit: Optional[float] = None,
            stop_loss: Optional[float] = None,
            exit_mode: Optional[str] = None,
            scoring_mode: str = 'legacy',
            horizon: str = 'short') -> List[CrossSectionalResult]:
        """
        运行横截面回测

        Args:
            stock_data_dict: {symbol: DataFrame(ohlcv)}
            eval_step: 每N个交易日评估一次
            n_groups: 分成几组（5=五分组）
            warmup_extra: 额外预热期
            take_profit: 止盈阈值(小数)，None=不使用
            stop_loss: 止损阈值(小数)，None=不使用
            exit_mode: 动态退出模式，如'zxdq_lt_zxdkx'。
                      设置后holding_periods被解释为max_holding，且TP/SL被忽略。
            scoring_mode: 'legacy'(旧权重) | 'horizon'(新三档信号) | 'adaptive'(自适应排名)
            horizon: 'short'|'long'

        Returns:
            每个holding_period对应一个CrossSectionalResult
        """
        # 1. 预计算所有股票指标
        logger.info(f"预计算 {len(stock_data_dict)} 只股票指标...")
        prepared = {}
        for sym, df in stock_data_dict.items():
            if df.empty or len(df) < 100:
                continue
            pdf = self.selector.prepare_data(df)
            if len(pdf) > 0:
                prepared[sym] = pdf

        if not prepared:
            logger.warning("无有效股票数据")
            return []

        # 2. 找到所有股票共同的最小长度（取所有股票长度的最小值作为上限）
        min_len = min(len(pdf) for pdf in prepared.values())
        cfg = self.selector.config
        warmup = max(cfg.zx_m4, cfg.wma_long * 5, cfg.zx_m3, cfg.zx_m2) + 50
        warmup = max(warmup, 100)

        logger.info(f"预热期: {warmup}, 最小数据长度: {min_len}")

        # 3. 遍历时间点
        eval_dates = list(range(warmup, min_len - max(self.config.holding_periods), eval_step))

        results_by_hp = {hp: [] for hp in self.config.holding_periods}

        for eval_idx in eval_dates:
            if scoring_mode == 'adaptive':
                # ── 自适应模式: 两步走 ──
                # 先收集全市场条件, 再排名打分
                all_conditions = {}
                all_pdf_data = {}
                for sym, pdf in prepared.items():
                    if eval_idx >= len(pdf):
                        continue
                    try:
                        conditions = self.selector.check_b1_conditions(pdf, date_idx=eval_idx)
                        # 修正j_percentile避免未来函数
                        j_values = pdf["J"].values
                        j_hist = j_values[max(0, eval_idx - 252):eval_idx + 1]
                        j_hist = j_hist[~np.isnan(j_hist)]
                        if len(j_hist) > 0:
                            conditions["j_percentile"] = np.mean(j_hist < j_values[eval_idx])
                        conditions["kdj_low"] = (
                            j_values[eval_idx] < cfg.j_threshold
                            or conditions["j_percentile"] < cfg.j_q_threshold
                        )
                        all_conditions[sym] = conditions
                        all_pdf_data[sym] = (pdf, eval_idx)
                    except Exception:
                        continue

                if len(all_conditions) < 10:
                    continue

                # long用独立引擎评分，short用batch_rank_score
                if horizon == 'long':
                    base_scores = {}
                    for sym, cond in all_conditions.items():
                        base_scores[sym] = self.selector._calculate_score_60d(cond)
                    # 百分位化
                    base_series = pd.Series(base_scores)
                    base_scores = (base_series.rank(pct=True) * 100).to_dict()
                else:
                    time_h = self.HORIZON_TO_TIME.get(horizon, '10d')
                    base_scores = self.selector.batch_rank_score(all_conditions, time_h)
                soft_scores = {}
                if horizon == 'long':
                    soft_scores = self.selector.batch_rank_soft_conditions(all_conditions)

                # 趋势信号排名（3个趋势信号简单平均）
                trend_scores = {}
                trend_keys = ['zxdkx_ratio_chg_5d', 'zxdq_ratio_chg_5d', 'vol_health_chg_5d']
                trend_raw = {k: {} for k in trend_keys}
                for sym, cond in all_conditions.items():
                    for k in trend_keys:
                        trend_raw[k][sym] = cond.get(k, 0.0)
                for k, vals in trend_raw.items():
                    series = pd.Series(vals)
                    ranked = (series.rank(pct=True, ascending=True) * 100).to_dict()
                    for sym in all_conditions:
                        if sym not in trend_scores:
                            trend_scores[sym] = 0.0
                        trend_scores[sym] += ranked.get(sym, 0) / len(trend_keys)

                # 混合权重
                base_w, soft_w, trend_w = self.ADAPTIVE_WEIGHTS.get(
                    horizon, self.ADAPTIVE_WEIGHTS['short'])

                stock_scores = []
                for sym in all_conditions:
                    score = base_scores.get(sym, 0) * base_w
                    if soft_scores and sym in soft_scores:
                        score += soft_scores[sym] * soft_w
                    if sym in trend_scores:
                        score += trend_scores[sym] * trend_w
                    pdf, idx = all_pdf_data[sym]
                    stock_scores.append((sym, score, pdf, idx))
            else:
                # ── 旧模式: 逐只评分 ──
                stock_scores = []
                for sym, pdf in prepared.items():
                    if eval_idx >= len(pdf):
                        continue
                    try:
                        conditions = self.selector.check_b1_conditions(pdf, date_idx=eval_idx)
                        # 修正j_percentile避免未来函数
                        j_values = pdf["J"].values
                        j_hist = j_values[max(0, eval_idx - 252):eval_idx + 1]
                        j_hist = j_hist[~np.isnan(j_hist)]
                        if len(j_hist) > 0:
                            conditions["j_percentile"] = np.mean(j_hist < j_values[eval_idx])
                        conditions["kdj_low"] = (
                            j_values[eval_idx] < cfg.j_threshold
                            or conditions["j_percentile"] < cfg.j_q_threshold
                        )
                        if scoring_mode == 'horizon':
                            horizon_scores = self.selector.compute_horizon_scores(conditions)
                            score = horizon_scores.get(horizon, 0.0)
                        elif scoring_mode == 'legacy' and horizon == 'long':
                            score = self.selector._calculate_score_60d(conditions)
                        else:
                            score = self.selector._calculate_score(conditions, '10d')
                        stock_scores.append((sym, score, pdf, eval_idx))
                    except Exception:
                        continue

            if len(stock_scores) < 10:
                continue

            # 按评分排序
            stock_scores.sort(key=lambda x: x[1])

            for hp in self.config.holding_periods:
                # 对每个持仓周期，计算各组的收益
                groups = {i: [] for i in range(n_groups)}
                scores = {i: [] for i in range(n_groups)}

                for i, (sym, score, pdf, idx) in enumerate(stock_scores):
                    g = min(i * n_groups // len(stock_scores), n_groups - 1)

                    if exit_mode:
                        ret = compute_dynamic_exit_return(
                            pdf["close"].values, pdf["zxdkx"].values, pdf["zxdq"].values,
                            idx, hp, exit_mode)
                    elif take_profit is not None or stop_loss is not None:
                        ret = _compute_tp_sl_return(
                            pdf["high"].values, pdf["low"].values, pdf["close"].values,
                            idx, hp, take_profit, stop_loss)
                    else:
                        fwd_idx = idx + hp
                        if fwd_idx < len(pdf):
                            ret = (pdf["close"].iloc[fwd_idx] - pdf["close"].iloc[idx]) / pdf["close"].iloc[idx]
                        else:
                            ret = None
                    if ret is not None:
                        groups[g].append(ret)
                        scores[g].append(score)

                # 存储评估结果
                for g in range(n_groups):
                    if groups[g]:
                        results_by_hp[hp].append({
                            "date": pdf.index[eval_idx],
                            "group": g,
                            "returns": groups[g],
                            "scores": scores[g]
                        })

        # 4. 聚合结果
        final_results = []
        for hp in self.config.holding_periods:
            evals = results_by_hp[hp]
            if not evals:
                continue

            # 按分组聚合
            group_data = {i: {"returns": [], "scores": []} for i in range(n_groups)}
            for e in evals:
                g = e["group"]
                group_data[g]["returns"].extend(e["returns"])
                group_data[g]["scores"].extend(e["scores"])

            group_stats = []
            for g in range(n_groups):
                rets = group_data[g]["returns"]
                scrs = group_data[g]["scores"]
                if not rets:
                    continue
                group_stats.append(GroupStats(
                    group_name=f"Q{g + 1} (低)" if g == 0 else
                               f"Q{g + 1}" if g < n_groups - 1 else
                               f"Q{g + 1} (高)",
                    avg_score=np.mean(scrs),
                    avg_return=np.mean(rets) * 100,
                    win_rate=np.mean([r > 0 for r in rets]) * 100,
                    n_samples=len(rets),
                    n_positive=sum(1 for r in rets if r > 0),
                    n_negative=sum(1 for r in rets if r < 0)
                ))

            # 高分组 - 低分组
            if len(group_stats) >= 2:
                high = group_stats[-1]
                low = group_stats[0]
                spread = GroupStats(
                    group_name=f"Q{n_groups}-Q1 差",
                    avg_score=high.avg_score - low.avg_score,
                    avg_return=high.avg_return - low.avg_return,
                    win_rate=0,
                    n_samples=0,
                    n_positive=0,
                    n_negative=0
                )
                group_stats.append(spread)

            final_results.append(CrossSectionalResult(
                holding_period=hp,
                total_evaluations=len(set(e["date"] for e in evals)),
                group_stats=group_stats
            ))

        # 打印摘要
        for r in final_results:
            print(f"\n横截面回测 持仓{r.holding_period}d:")
            print(f"  评估次数: {r.total_evaluations}")
            print(f"  {'分组':<14} {'平均评分':<10} {'平均收益%':<12} {'胜率%':<10} {'样本数':<8}")
            print(f"  {'-'*54}")
            for gs in r.group_stats:
                print(f"  {gs.group_name:<14} {gs.avg_score:<+10.2f} {gs.avg_return:<+12.2f} {gs.win_rate:<10.1f} {gs.n_samples:<8}")

        return final_results

    def plot_results(self, results: List[CrossSectionalResult], save_path: str = None):
        """
        生成横截面回测图表

        Args:
            results: CrossSectionalBacktest.run() 的返回值
            save_path: 图片保存路径，None时显示
        """
        try:
            import matplotlib.pyplot as plt
            # 尝试设置中文字体
            for font_name in ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']:
                try:
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    # 验证字体是否可用
                    fig_test, ax_test = plt.subplots()
                    ax_test.set_title('测试')
                    fig_test.savefig(os.devnull, format='png')
                    plt.close(fig_test)
                    break
                except Exception:
                    continue
        except ImportError:
            logger.warning("matplotlib未安装，跳过图表")
            return

        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, r in zip(axes, results):
            # Q1-Q5 分组柱状图（不含spread行）
            stats = [gs for gs in r.group_stats if not gs.group_name.endswith("差")]
            if not stats:
                continue

            names = [s.group_name for s in stats]
            returns = [s.avg_return for s in stats]
            win_rates = [s.win_rate for s in stats]

            x = np.arange(len(names))
            ax.bar(x - 0.175, returns, 0.35, label="平均收益%", color="steelblue")
            ax.bar(x + 0.175, win_rates, 0.35, label="胜率%", color="coral")

            # 标注具体数值
            for i, (ret, wr) in enumerate(zip(returns, win_rates)):
                ax.annotate(f"{ret:+.1f}%", (x[i] - 0.175, ret),
                           ha="center", va="bottom" if ret >= 0 else "top", fontsize=8)
                ax.annotate(f"{wr:.1f}%", (x[i] + 0.175, wr),
                           ha="center", va="bottom", fontsize=8)

            # 显示Q5-Q1 spread
            spread = next((gs for gs in r.group_stats if gs.group_name.endswith("差")), None)
            title = f"持仓 {r.holding_period}d"
            if spread:
                title += f"\nQ5-Q1 差: {spread.avg_return:+.2f}%"

            ax.set_title(title, fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30)
            ax.axhline(y=0, color="gray", linewidth=0.5)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("横截面回测：分组收益对比", fontsize=15)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"图表已保存: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    @staticmethod
    def export_results(results: List[CrossSectionalResult],
                       csv_path: str = None) -> pd.DataFrame:
        """
        导出横截面回测结果到DataFrame/CSV

        Returns:
            包含各组统计数据的DataFrame
        """
        rows = []
        for r in results:
            for gs in r.group_stats:
                rows.append({
                    "holding_period": r.holding_period,
                    "total_evaluations": r.total_evaluations,
                    "group": gs.group_name,
                    "avg_score": round(gs.avg_score, 2),
                    "avg_return": round(gs.avg_return, 2),
                    "win_rate": round(gs.win_rate, 1),
                    "n_samples": gs.n_samples,
                    "n_positive": gs.n_positive,
                    "n_negative": gs.n_negative,
                })

        df = pd.DataFrame(rows)
        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.info(f"结果已保存: {csv_path}")

        return df


# ═══════════════════════════════════════════════════════════════
# Top N 持仓回测 — 模拟真实买卖：筛选→买入TopN→持有→卖出
# ═══════════════════════════════════════════════════════════════


def _top_n_hold_eval_date(
    selector, prepared: Dict[str, pd.DataFrame],
    eval_idx: int, top_n: int, hold_period: int,
    scoring_mode: str, horizon: str,
    index_df: Optional[pd.DataFrame] = None,
    is_bear_market: bool = False,
    exit_mode: Optional[str] = None,
    max_per_industry: int = 0,
    industry_map: Dict[str, str] = None,
    mine_filter: bool = False,
    stop_loss: Optional[float] = None,
    sector_momentum_weight: float = 0.0,
) -> Optional[Dict]:
    """
    在单个评估日期执行：评分 → 选Top N → 买入持有 → 计算收益

    Args:
        exit_mode: 动态退出模式。'close_lt_zxdkx'=跌破知行线即卖。
                   None=固定持有hold_period天。

    Returns:
        dict with keys: entry_date, exit_date, stock_returns, avg_return,
                        benchmark_return, index_return, is_bear_market
        None if insufficient data
    """
    if is_bear_market:
        return None  # 熊市空仓

    stock_scores = []
    mine_rejects = 0
    for sym, pdf in prepared.items():
        if eval_idx >= len(pdf) or eval_idx + hold_period >= len(pdf):
            continue
        try:
            # ── 排雷过滤（入场前排除问题股）──
            if mine_filter:
                row = pdf.iloc[eval_idx]
                # 1) J值超买
                j_val = row.get('J', 50)
                if j_val > 100:
                    mine_rejects += 1; continue
                # 2) 连续3天缩量
                if eval_idx >= 3:
                    vol = pdf['volume'].values
                    if vol[eval_idx] < vol[eval_idx-1] < vol[eval_idx-2]:
                        mine_rejects += 1; continue
                # 3) 远离知行线 (>30%)
                zxdkx_ratio = row.get('zxdkx_ratio', 1.0)
                if zxdkx_ratio > 1.3:
                    mine_rejects += 1; continue

            conditions = selector.check_b1_conditions(pdf, date_idx=eval_idx)
            # j_percentile 避免未来函数
            j_values = pdf["J"].values
            j_hist = j_values[max(0, eval_idx - 252):eval_idx + 1]
            j_hist = j_hist[~np.isnan(j_hist)]
            if len(j_hist) > 0:
                conditions["j_percentile"] = np.mean(j_hist < j_values[eval_idx])
            conditions["kdj_low"] = (
                j_values[eval_idx] < selector.config.j_threshold
                or conditions["j_percentile"] < selector.config.j_q_threshold
            )

            if scoring_mode == 'adaptive':
                stock_scores.append((sym, 0.0, pdf, eval_idx))  # score filled later
            elif scoring_mode == 'horizon':
                hs = selector.compute_horizon_scores(conditions)
                score = hs.get(horizon, 0.0)
                stock_scores.append((sym, score, pdf, eval_idx))
            elif scoring_mode == 'legacy' and horizon == 'long':
                score = selector._calculate_score_60d(conditions)
                stock_scores.append((sym, score, pdf, eval_idx))
            elif scoring_mode == 'resonance':
                # 双分计算，后续统一排名取百分位
                s10 = selector._calculate_score(conditions, '10d')
                s60 = selector._calculate_score_60d(conditions)
                stock_scores.append((sym, s10, s60, pdf, eval_idx))
            else:
                score = selector._calculate_score(conditions, '10d')
                stock_scores.append((sym, score, pdf, eval_idx))
        except Exception:
            continue

    if len(stock_scores) < top_n:
        return None

    # 共振模式：10d和60d百分位平均
    if scoring_mode == 'resonance':
        scores_10d = np.array([s[1] for s in stock_scores])
        scores_60d = np.array([s[2] for s in stock_scores])
        pct_10d = pd.Series(scores_10d).rank(pct=True) * 100
        pct_60d = pd.Series(scores_60d).rank(pct=True) * 100
        resonance = ((pct_10d + pct_60d) / 2).values
        for i, s in enumerate(stock_scores):
            sym, _, _, pdf, idx = s
            stock_scores[i] = (sym, resonance[i], pdf, idx)

    # 自适应模式：先收集全市场条件再排名
    if scoring_mode == 'adaptive':
        all_cond = {}
        for sym, _, pdf, idx in stock_scores:
            all_cond[sym] = selector.check_b1_conditions(pdf, date_idx=idx)
        time_h = CrossSectionalBacktest.HORIZON_TO_TIME.get(horizon, '10d')
        base_scores = selector.batch_rank_score(all_cond, time_h)
        rescored = []
        for sym, _, pdf, idx in stock_scores:
            rescored.append((sym, base_scores.get(sym, 0), pdf, idx))
        stock_scores = rescored

    # 行业动量调整：强势行业加分，弱势行业减分
    if sector_momentum_weight > 0 and industry_map:
        stock_ret60 = {}
        for sym, score, pdf, idx in stock_scores:
            if idx >= 60:
                stock_ret60[sym] = (pdf['close'].iloc[idx] - pdf['close'].iloc[idx-60]) / pdf['close'].iloc[idx-60]
        if stock_ret60:
            ind_rets = {}
            for sym, ret in stock_ret60.items():
                ind = industry_map.get(str(sym).zfill(6), '??')
                ind_rets.setdefault(ind, []).append(ret)
            ind_momentum = {ind: np.mean(rets) for ind, rets in ind_rets.items()}
            ind_series = pd.Series(ind_momentum)
            ind_pct = (ind_series.rank(pct=True) * 100).to_dict()
            adjusted = []
            for sym, score, pdf, idx in stock_scores:
                ind = industry_map.get(str(sym).zfill(6), '??')
                sec = ind_pct.get(ind, 50.0)
                adj_score = score * (1 - sector_momentum_weight) + sec * sector_momentum_weight
                adjusted.append((sym, adj_score, pdf, idx))
            stock_scores = adjusted

    # 按评分降序排列
    stock_scores.sort(key=lambda x: x[1], reverse=True)

    # 选出 Top N（可限制同行业数量）
    if max_per_industry:
        industry_count = {}
        top_stocks = []
        for item in stock_scores:
            sym = item[0]
            ind = industry_map.get(str(sym).zfill(6), '??')
            if industry_count.get(ind, 0) >= max_per_industry:
                continue
            top_stocks.append(item)
            industry_count[ind] = industry_count.get(ind, 0) + 1
            if len(top_stocks) >= top_n:
                break
    else:
        top_stocks = stock_scores[:top_n]

    # 计算收益
    if stop_loss is not None:
        # ── 止损模式：逐日检查盘中最低价 ──
        stock_returns = {}
        exit_dates = {}
        for sym, score, pdf, idx in top_stocks:
            ret = _compute_tp_sl_return(
                pdf["high"].values, pdf["low"].values, pdf["close"].values,
                idx, hold_period, None, stop_loss)
            if ret is not None:
                stock_returns[sym] = ret
                exit_dates[sym] = pdf.index[min(idx + hold_period, len(pdf) - 1)]

    elif exit_mode:
        # ── 动态退出：逐日检查，跌破知行线即卖 ──
        stock_returns = {}
        exit_dates = {}
        for sym, score, pdf, idx in top_stocks:
            ret = compute_dynamic_exit_return(
                pdf["close"].values, pdf["zxdkx"].values, pdf["zxdq"].values,
                idx, hold_period, exit_mode)
            if ret is not None:
                stock_returns[sym] = ret
                # 找实际退出日：扫描找到第一个触发退出的日期
                close_arr = pdf["close"].values
                zxdkx_arr = pdf["zxdkx"].values
                exit_day = idx + hold_period  # 默认到期日
                for j in range(idx + 1, min(idx + hold_period, len(close_arr))):
                    if exit_mode == 'close_lt_zxdkx' and close_arr[j] < zxdkx_arr[j]:
                        exit_day = j; break
                    elif exit_mode == 'zxdq_lt_zxdkx' and pdf["zxdq"].values[j] < zxdkx_arr[j]:
                        exit_day = j; break
                    elif exit_mode == 'both' and (close_arr[j] < zxdkx_arr[j] or pdf["zxdq"].values[j] < zxdkx_arr[j]):
                        exit_day = j; break
                exit_dates[sym] = pdf.index[min(exit_day, len(pdf) - 1)]
    else:
        # ── 固定持有 ──
        entry_price = {}
        exit_prices = {}
        for sym, score, pdf, idx in top_stocks:
            entry_price[sym] = pdf["close"].iloc[idx]
            exit_prices[sym] = pdf["close"].iloc[idx + hold_period]

        stock_returns = {}
        for sym in entry_price:
            if entry_price[sym] > 0:
                stock_returns[sym] = (exit_prices[sym] - entry_price[sym]) / entry_price[sym]
            else:
                stock_returns[sym] = 0.0

    avg_return = np.mean(list(stock_returns.values()))

    # 基准收益：全市场平均
    benchmark_rets = []
    for sym, pdf in prepared.items():
        if eval_idx + hold_period < len(pdf):
            ret = (pdf["close"].iloc[eval_idx + hold_period] - pdf["close"].iloc[eval_idx]) / pdf["close"].iloc[eval_idx]
            benchmark_rets.append(ret)
    benchmark_return = np.mean(benchmark_rets) if benchmark_rets else 0.0

    # 大盘指数收益（沪深300）
    index_return = None
    if index_df is not None:
        entry_date_stamp = next(iter(prepared.values())).index[eval_idx]
        exit_date_stamp = next(iter(prepared.values())).index[
            min(eval_idx + hold_period, len(next(iter(prepared.values()))) - 1)
        ]
        # 找到指数在入场日和出场日最近的行情
        entry_idx = index_df.index.get_indexer([entry_date_stamp], method='ffill')[0]
        exit_idx = index_df.index.get_indexer([exit_date_stamp], method='ffill')[0]
        if entry_idx != -1 and exit_idx != -1 and entry_idx < len(index_df) and exit_idx < len(index_df):
            idx_entry = index_df["close"].iloc[entry_idx]
            idx_exit = index_df["close"].iloc[exit_idx]
            if idx_entry > 0:
                index_return = (idx_exit - idx_entry) / idx_entry

    entry_date = next(iter(prepared.values())).index[eval_idx]
    if exit_mode and exit_dates:
        exit_date = max(exit_dates.values())  # 最晚退出日
    else:
        exit_date = next(iter(prepared.values())).index[min(eval_idx + hold_period, len(next(iter(prepared.values()))) - 1)]

    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "exit_mode": exit_mode,
        "n_stocks": len(stock_returns),
        "stock_returns": stock_returns,
        "avg_return": avg_return,
        "benchmark_return": benchmark_return,
        "index_return": index_return,
        "top_score": top_stocks[0][1],
        "bottom_score": top_stocks[-1][1],
    }


class TopNHoldingBacktest:
    """
    Top N 持仓回测

    模拟真实交易行为：
    1. 每 N 个交易日评估一次全市场股票
    2. 按评分选出得分最高的 Top N 只
    3. 等权重买入，持有 H 个交易日
    4. 到期卖出，记录收益

    最终输出：所有历史期次的汇总表现
    """

    def __init__(self, selector):
        self.selector = selector

    def run(self, stock_data_dict: Dict[str, pd.DataFrame],
            top_n: int = 10,
            hold_period: int = 20,
            eval_step: int = 20,
            scoring_mode: str = 'legacy',
            horizon: str = 'short',
            max_periods: int = 0,
            index_df: Optional[pd.DataFrame] = None,
            market_filter: bool = False,
            exit_mode: Optional[str] = None,
            max_per_industry: int = 0,
            industry_map: Dict[str, str] = None,
            mine_filter: bool = False,
            stop_loss: Optional[float] = None,
            sector_momentum_weight: float = 0.0,
            entry_signal: bool = False) -> "TopNHoldingSummary":
        """
        运行 Top N 持仓回测

        Args:
            stock_data_dict: {symbol: DataFrame(ohlcv)}
            top_n: 每期选前几只
            hold_period: 持有期（交易日）
            eval_step: 每隔多少个交易日评估一次
            scoring_mode: 'legacy' | 'horizon' | 'adaptive'
            horizon: 'short' | 'long'
            max_periods: 最多评估期数，0=全部
            index_df: 大盘指数DataFrame（如沪深300）
            market_filter: 是否启用市场环境过滤（等权均价<MA60时空仓）
            exit_mode: 动态退出模式。None=固定持有
            max_per_industry: 同行业最大持仓数
            industry_map: 股票代码→行业名称映射
            mine_filter: 是否启用排雷过滤
            stop_loss: 止损阈值（小数），如0.08=跌8%止损。None=不启用

        Returns:
            TopNHoldingSummary
        """
        # 1. 预计算
        logger.info(f"预计算 {len(stock_data_dict)} 只股票指标...")
        prepared = {}
        for sym, df in stock_data_dict.items():
            if df.empty or len(df) < 100:
                continue
            pdf = self.selector.prepare_data(df)
            if len(pdf) > 0:
                prepared[sym] = pdf

        if not prepared:
            logger.warning("无有效股票数据")
            return TopNHoldingSummary.empty()

        # 2. 确定评估时间点
        min_len = min(len(pdf) for pdf in prepared.values())
        cfg = self.selector.config
        warmup = max(cfg.zx_m4, cfg.wma_long * 5, cfg.zx_m3, cfg.zx_m2) + 50
        warmup = max(warmup, 100)

        eval_indices = list(range(warmup, min_len - hold_period, eval_step))
        if max_periods > 0:
            eval_indices = eval_indices[-max_periods:]  # 取最近 N 期

        logger.info(f"预热期: {warmup}, 最小数据长度: {min_len}, 评估期数: {len(eval_indices)}")

        # 2.5 市场环境代理（等权均价 + MA60）
        is_bear_by_idx = {}
        if market_filter:
            logger.info("计算市场代理（等权均价MA60）...")
            # 取所有股票的close对齐为DataFrame
            close_dict = {}
            for sym, pdf in prepared.items():
                close_dict[sym] = pd.Series(pdf['close'].values)
            close_df = pd.DataFrame(close_dict)
            market_close = close_df.mean(axis=1)
            market_ma60 = market_close.rolling(60, min_periods=60).mean()
            # 预先计算每个评估点的市场状态
            for ei in eval_indices:
                if ei < len(market_ma60) and pd.notna(market_ma60.iloc[ei]):
                    is_bear_by_idx[ei] = market_close.iloc[ei] < market_ma60.iloc[ei]
                else:
                    is_bear_by_idx[ei] = False

        # 2.6 买点信号（回调后放量反弹日才入场）
        entry_signal_by_idx = {}
        if entry_signal:
            close_dict = {}
            vol_dict = {}
            for sym, pdf in prepared.items():
                close_dict[sym] = pd.Series(pdf['close'].values)
                vol_dict[sym] = pd.Series(pdf['volume'].values)
            close_df = pd.DataFrame(close_dict)
            vol_df = pd.DataFrame(vol_dict)
            mc = close_df.mean(axis=1)          # 市场均价
            mma60 = mc.rolling(60, min_periods=60).mean()
            m_high20 = mc.rolling(20, min_periods=1).max()  # 20日最高
            mv = vol_df.mean(axis=1)            # 市场均量
            mv_ma5 = mv.rolling(5, min_periods=1).mean()
            mv_ma20 = mv.rolling(20, min_periods=1).mean()
            mma20 = mc.rolling(20, min_periods=1).mean()
            for ei in range(60, min_len - hold_period):
                if ei >= len(mc):
                    break
                bull = mc.iloc[ei] > mma60.iloc[ei]            # 中长期牛市
                breakout = mc.iloc[ei] >= m_high20.iloc[ei]     # 创20日新高
                entry_signal_by_idx[ei] = bull and breakout

        # 3. 逐期评估（买点模式用信号日替换评估点）
        evals = []
        cash_periods = 0
        entry_signal_count = 0
        if entry_signal:
            # 用所有买点信号日作为入场日（每天检查，不以eval_step为准）
            actual_eval_indices = [ei for ei in range(warmup, min_len - hold_period)
                                   if entry_signal_by_idx.get(ei, False)]
            entry_signal_count = len(actual_eval_indices)
            logger.info(f"买点信号: {entry_signal_count} 次入场")
        else:
            actual_eval_indices = eval_indices

        for i, eval_idx in enumerate(actual_eval_indices):
            if (i + 1) % 10 == 0:
                logger.info(f"  评估进度: {i+1}/{len(actual_eval_indices)}")

            is_bear = is_bear_by_idx.get(eval_idx, False) if market_filter else False
            if is_bear:
                cash_periods += 1

            result = _top_n_hold_eval_date(
                self.selector, prepared, eval_idx,
                top_n, hold_period, scoring_mode, horizon,
                index_df=index_df, is_bear_market=is_bear,
                exit_mode=exit_mode,
                max_per_industry=max_per_industry,
                industry_map=industry_map,
                mine_filter=mine_filter,
                stop_loss=stop_loss,
                sector_momentum_weight=sector_momentum_weight,
            )
            if result is not None:
                evals.append(result)

        if not evals:
            logger.warning("无有效评估结果")
            return TopNHoldingSummary.empty()

        # 4. 汇总
        returns = np.array([e["avg_return"] for e in evals])
        benchmark_returns = np.array([e["benchmark_return"] for e in evals])

        overall_avg = np.mean(returns) * 100
        overall_win_rate = np.mean(returns > 0) * 100
        benchmark_avg = np.mean(benchmark_returns) * 100
        excess = overall_avg - benchmark_avg

        # 大盘指数收益（沪深300）
        index_returns = np.array([e.get("index_return") for e in evals if e.get("index_return") is not None])
        index_avg = np.mean(index_returns) * 100 if len(index_returns) > 0 else None
        index_excess = overall_avg - index_avg if index_avg is not None else None
        index_nav_val = np.cumprod(1 + index_returns)[-1] if len(index_returns) > 0 else None

        # 连续亏损
        max_consec_loss = 0
        curr_loss = 0
        for r in returns:
            if r < 0:
                curr_loss += 1
                max_consec_loss = max(max_consec_loss, curr_loss)
            else:
                curr_loss = 0

        # 累计净值
        cum_nav = np.cumprod(1 + returns)
        benchmark_nav = np.cumprod(1 + benchmark_returns)

        # 最大回撤
        peak = np.maximum.accumulate(cum_nav)
        drawdown = (cum_nav - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        # 夏普（简化，用期数收益率当无风险=0）
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / hold_period) if np.std(returns) > 0 else 0

        return TopNHoldingSummary(
            holding_period=hold_period,
            top_n=top_n,
            total_periods=len(evals),
            cash_periods=cash_periods,
            entry_signal_count=entry_signal_count,
            evals=evals,
            overall_avg_return=overall_avg,
            overall_win_rate=overall_win_rate,
            max_consecutive_losses=max_consec_loss,
            best_period_return=np.max(returns) * 100,
            worst_period_return=np.min(returns) * 100,
            benchmark_avg_return=benchmark_avg,
            excess_return=excess,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            cum_nav=cum_nav[-1] if len(cum_nav) > 0 else 1.0,
            benchmark_nav=benchmark_nav[-1] if len(benchmark_nav) > 0 else 1.0,
            index_avg_return=index_avg,
            index_excess_return=index_excess,
            index_nav=index_nav_val,
        )


@dataclass
class TopNHoldingSummary:
    """Top N 持仓回测汇总"""
    holding_period: int = 0
    top_n: int = 0
    total_periods: int = 0
    cash_periods: int = 0
    entry_signal_count: int = 0
    evals: List[Dict] = field(default_factory=list)
    overall_avg_return: float = 0.0
    overall_win_rate: float = 0.0
    max_consecutive_losses: int = 0
    best_period_return: float = 0.0
    worst_period_return: float = 0.0
    benchmark_avg_return: float = 0.0
    excess_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    cum_nav: float = 1.0
    benchmark_nav: float = 1.0
    index_avg_return: Optional[float] = None
    index_excess_return: Optional[float] = None
    index_nav: Optional[float] = None

    @staticmethod
    def empty():
        return TopNHoldingSummary()

    def print_report(self):
        """打印回测报告"""
        if self.total_periods == 0:
            print("\n无回测结果")
            return

        print("\n" + "=" * 62)
        print(f"  Top {self.top_n} 持仓回测报告")
        print("=" * 62)
        print(f"  持有期: {self.holding_period} 个交易日")
        print(f"  评估期数: {self.total_periods} 期 (空仓 {self.cash_periods} 期)")
        if self.entry_signal_count > 0:
            print(f"  买点信号触发: {self.entry_signal_count} 次")
        print(f"  {'─' * 42}")
        print(f"  平均收益率:    {self.overall_avg_return:>+8.2f}%")
        print(f"  胜率:          {self.overall_win_rate:>8.1f}%")
        print(f"  最大回撤:      {self.max_drawdown:>8.2f}%")
        print(f"  夏普比率:      {self.sharpe_ratio:>8.2f}")
        print(f"  累计净值:      {self.cum_nav:>8.4f}")
        print(f"  {'─' * 42}")
        print(f"  基准平均收益:  {self.benchmark_avg_return:>+8.2f}%")
        print(f"  超额收益:      {self.excess_return:>+8.2f}%")
        print(f"  基准累计净值:  {self.benchmark_nav:>8.4f}")
        if self.index_avg_return is not None:
            print(f"  {'─' * 42}")
            print(f"  大盘(沪深300):  {self.index_avg_return:>+8.2f}%")
            print(f"  跑赢大盘:      {self.index_excess_return:>+8.2f}%")
            print(f"  大盘累计净值:  {self.index_nav:>8.4f}")
        print(f"  {'─' * 42}")
        print(f"  最佳单期收益:  {self.best_period_return:>+8.2f}%")
        print(f"  最差单期收益:  {self.worst_period_return:>+8.2f}%")
        print(f"  最大连续亏损:  {self.max_consecutive_losses} 期")
        print("=" * 62)

        # 最近10期明细
        print(f"\n  最近 {min(10, self.total_periods)} 期明细:")
        if self.index_avg_return is not None:
            print(f"  {'期次':<6} {'入场':<12} {'出场':<12} {'策略%':<8} {'基准%':<8} {'大盘%':<8}")
            print(f"  {'─' * 62}")
            recent = self.evals[-10:] if len(self.evals) > 10 else self.evals
            for i, e in enumerate(recent):
                idx_ret = e.get("index_return")
                idx_str = f"{idx_ret*100:<+8.2f}" if idx_ret is not None else "  N/A    "
                print(f"  {i+1:<6} {str(e['entry_date'].date()):<12} {str(e['exit_date'].date()):<12} "
                      f"{e['avg_return']*100:<+8.2f} {e['benchmark_return']*100:<+8.2f} {idx_str}")

        # 选股质量
        print(f"\n  选股质量（所有期次汇总）:")
        top_scores = [e["top_score"] for e in self.evals]
        bottom_scores = [e["bottom_score"] for e in self.evals]
        print(f"  平均 Top1 评分: {np.mean(top_scores):.1f}")
        print(f"  平均 Top{self.top_n} 评分: {np.mean(bottom_scores):.1f}")

    def to_dataframe(self) -> pd.DataFrame:
        """导出明细到DataFrame"""
        rows = []
        for i, e in enumerate(self.evals):
            idx_ret = e.get("index_return")
            rows.append({
                "period": i + 1,
                "entry_date": e["entry_date"],
                "exit_date": e["exit_date"],
                "n_stocks": e["n_stocks"],
                "avg_return": round(e["avg_return"] * 100, 2),
                "benchmark_return": round(e["benchmark_return"] * 100, 2),
                "index_return": round(idx_ret * 100, 2) if idx_ret is not None else None,
                "excess_return": round((e["avg_return"] - e["benchmark_return"]) * 100, 2),
                "excess_vs_index": round((e["avg_return"] - idx_ret) * 100, 2) if idx_ret is not None else None,
                "top_score": round(e["top_score"], 1),
                "bottom_score": round(e["bottom_score"], 1),
            })
        return pd.DataFrame(rows)

    def save_csv(self, path: str):
        """保存明细到CSV"""
        df = self.to_dataframe()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"明细已保存: {path}")

    def plot_results(self, save_path: str = None):
        """
        生成可视化图表：净值曲线 + 每期收益 + 回撤 + 收益分布

        Args:
            save_path: 图片保存路径（不含扩展名），None时显示
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            # Windows中文字体
            import matplotlib.font_manager as fm
            for fname in ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']:
                try:
                    plt.rcParams['font.sans-serif'] = [fname, 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                    # 验证可用
                    fig_test, ax_test = plt.subplots()
                    ax_test.set_title('测试')
                    fig_test.savefig(os.devnull, format='png')
                    plt.close(fig_test)
                    break
                except Exception:
                    continue
        except ImportError:
            logger.warning("matplotlib未安装，跳过图表")
            return

        if self.total_periods == 0:
            logger.warning("无数据，跳过图表")
            return

        # 准备数据
        returns = np.array([e["avg_return"] for e in self.evals])
        benchmark_rets = np.array([e["benchmark_return"] for e in self.evals])
        has_index = self.evals[0].get("index_return") is not None
        index_rets = np.array([e.get("index_return", 0) for e in self.evals]) if has_index else None

        strat_nav = np.cumprod(1 + returns)
        bench_nav = np.cumprod(1 + benchmark_rets)
        index_nav = np.cumprod(1 + index_rets) if has_index else None

        periods = np.arange(1, len(returns) + 1)

        # 最大回撤
        peak = np.maximum.accumulate(strat_nav)
        drawdown = (strat_nav - peak) / peak * 100

        # 创建图
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Top {self.top_n} 持仓回测 — 持有{self.holding_period}d  ({self.total_periods}期)",
                     fontsize=15, fontweight='bold')

        # 1. 净值曲线
        ax1 = axes[0, 0]
        ax1.plot(periods, strat_nav, 'b-', linewidth=2, label=f'策略 (累计{self.cum_nav:.4f})')
        ax1.plot(periods, bench_nav, 'gray', linewidth=1.5, linestyle='--', label=f'全市场平均 ({self.benchmark_nav:.4f})')
        if index_nav is not None:
            ax1.plot(periods, index_nav, 'orange', linewidth=1.5, linestyle=':', label=f'沪深300 ({self.index_nav:.4f})')
        ax1.axhline(y=1.0, color='black', linewidth=0.5)
        ax1.set_title('累计净值')
        ax1.set_xlabel('期次')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # 2. 每期收益柱状图
        ax2 = axes[0, 1]
        colors = ['green' if r >= 0 else 'red' for r in returns]
        ax2.bar(periods, returns * 100, color=colors, alpha=0.7, width=0.6)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        if index_rets is not None:
            ax2.plot(periods, index_rets * 100, 'orange', marker='o', linewidth=1.5,
                     markersize=3, label='沪深300', alpha=0.7)
        ax2.axhline(y=np.mean(returns) * 100, color='blue', linestyle='--', linewidth=1,
                    label=f'平均 {np.mean(returns)*100:+.2f}%')
        ax2.set_title('每期收益')
        ax2.set_xlabel('期次')
        ax2.set_ylabel('收益率 %')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # 3. 回撤曲线
        ax3 = axes[1, 0]
        ax3.fill_between(periods, 0, drawdown, color='red', alpha=0.3)
        ax3.plot(periods, drawdown, 'r-', linewidth=1)
        ax3.set_title(f'最大回撤 {self.max_drawdown:.2f}%')
        ax3.set_xlabel('期次')
        ax3.set_ylabel('回撤 %')
        ax3.grid(alpha=0.3)
        ax3.set_ylim(min(drawdown) * 1.2, 5)

        # 4. 收益分布直方图
        ax4 = axes[1, 1]
        ax4.hist(returns * 100, bins=min(15, self.total_periods // 2 + 1),
                 color='steelblue', edgecolor='white', alpha=0.7)
        ax4.axvline(x=np.mean(returns) * 100, color='red', linestyle='--', linewidth=2,
                    label=f'平均 {np.mean(returns)*100:+.2f}%')
        ax4.axvline(x=0, color='black', linewidth=1)
        ax4.set_title(f'收益分布  胜率 {self.overall_win_rate:.1f}%  夏普 {self.sharpe_ratio:.2f}')
        ax4.set_xlabel('收益率 %')
        ax4.set_ylabel('频次')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"图表已保存: {save_path}")
        else:
            plt.show()
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 主入口测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from b1_selector import B1Selector, B1Config

    logging.basicConfig(level=logging.WARNING)

    import sys
    if "--topn" in sys.argv:
        # Top N 持仓回测演示
        print("=" * 60)
        print("Top N 持仓回测演示")
        print("=" * 60)

        config = B1Config(
            j_threshold=10.0, j_q_threshold=0.30,
            zx_m1=5, zx_m2=20, zx_m3=40, zx_m4=60,
            wma_short=5, wma_mid=10, wma_long=15,
        )
        selector = B1Selector(config)

        # 生成50只模拟股票
        print("\n生成50只模拟股票数据...")
        stock_dict = {}
        for i in range(50):
            seed = 100 + i
            stock_dict[f"STOCK{i:03d}"] = BacktestMockDataGenerator.trend_step(1500, seed=seed)

        bt = TopNHoldingBacktest(selector)
        summary = bt.run(stock_dict, top_n=10, hold_period=20, eval_step=20)
        summary.print_report()
        summary.save_csv("reports/topn_backtest.csv")
        print("\n演示完成")
    else:
        # 横截面回测测试
        print("=" * 60)
        print("横截面回测测试")
        print("=" * 60)

        config = B1Config(
            j_threshold=10.0, j_q_threshold=0.30,
            zx_m1=5, zx_m2=20, zx_m3=40, zx_m4=60,
            wma_short=5, wma_mid=10, wma_long=15,
        )
        selector = B1Selector(config)
        bt_config = BacktestConfig(holding_periods=[20, 60])

        # 生成50只模拟股票
        print("\n生成50只模拟股票数据...")
        stock_dict = {}
        for i in range(50):
            seed = 100 + i
            sym = f"STOCK{i:03d}"
            df = BacktestMockDataGenerator.trend_step(1500, seed=seed)
            stock_dict[sym] = df

        print(f"共 {len(stock_dict)} 只股票")

        cs_backtest = CrossSectionalBacktest(selector, bt_config)
        results = cs_backtest.run(stock_dict, eval_step=20, n_groups=5)
        print("\n测试完成")
