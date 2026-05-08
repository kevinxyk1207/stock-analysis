"""
新因子权重扫描 — 测试 0%~80% 找出最佳权重
结果保存到 reports/weight_sweep.csv
"""
import sys, os, logging, numpy as np
logging.basicConfig(level=logging.WARNING)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_fetcher import EnhancedStockFetcher, load_cache_data
from b1_selector import B1Selector, B1Config, TIME_HORIZON_WEIGHTS
from b1_backtest import TopNHoldingBacktest
import baostock as bs, pandas as pd

def main():
    # 1. 加载数据
    print("加载HS300数据...")
    fetcher = EnhancedStockFetcher()
    hs300 = fetcher.get_hs300_stocks()
    symbols = set(str(s).zfill(6) for s in hs300['code'].tolist())
    stock_data = load_cache_data(min_rows=500, common_range=False)
    cfg = B1Config()
    warmup = max(cfg.zx_m4, cfg.wma_long * 5, cfg.zx_m3, cfg.zx_m2) + 50
    filtered = {k: v for k, v in stock_data.items()
                if str(k).zfill(6) in symbols and len(v) >= warmup + 200}
    print(f"HS300数据充足: {len(filtered)} 只")

    # 2. 指数数据
    print("获取指数数据...")
    lg = bs.login()
    rs = bs.query_history_k_data_plus('sh.000300','date,close',
        start_date='2020-01-01', end_date='2025-12-31', frequency='d', adjustflag='3')
    data = []
    while rs.next(): data.append(rs.get_row_data())
    bs.logout()
    index_df = pd.DataFrame(data, columns=['date','close'])
    index_df['close'] = index_df['close'].replace('', np.nan).astype(float).dropna()
    index_df['date'] = pd.to_datetime(index_df['date'])
    index_df = index_df.set_index('date')

    # 3. 权重生成函数
    def make_weights(pct, old_base, new_ratio):
        """pct: 0-100, 新因子占比"""
        w = {}
        old_sum = sum(old_base.values())
        for k, v in old_base.items():
            w[k] = round(v / old_sum * (100 - pct), 1)
        new_sum = sum(new_ratio.values())
        for k, v in new_ratio.items():
            w[k] = round(v / new_sum * pct, 1)
        d = round(100 - sum(w.values()), 1)
        if d:
            w[max(w, key=w.get)] += d
        return w

    # 4. 各周期的基线和新因子比例
    HORIZONS = {
        '10d': {
            'old': {'zxdkx': 0, 'zxdq': 40, 'vol': 20, 'macd': 20, 'rsi': 20},
            'new': {'vol_surge': 28, 'ma_alignment': 20, 'trend_strength': 10},
            'hp': 10, 'step': 10,
        },
        '60d': {
            'old': {'zxdkx': 40, 'zxdq': 40, 'vol': 10, 'rsi': 10},
            'new': {'vol_surge': 10, 'ma_alignment': 25, 'trend_strength': 27},
            'hp': 60, 'step': 20,
        },
    }

    # 5. 扫描
    W = TIME_HORIZON_WEIGHTS
    bt = TopNHoldingBacktest(B1Selector(B1Config()))

    all_rows = []
    levels = list(range(0, 81, 10))  # 0,10,20,...,80

    print(f"\n扫描开始: {len(levels)} 个权重 × {len(HORIZONS)} 个周期")
    for pct in levels:
        for hkey, hcfg in HORIZONS.items():
            W[hkey] = make_weights(pct, hcfg['old'], hcfg['new'])

        row = {'new_factor_pct': pct}
        for hkey, hcfg in HORIZONS.items():
            s = bt.run(filtered, top_n=10,
                       hold_period=hcfg['hp'], eval_step=hcfg['step'],
                       index_df=index_df)
            row[f'{hkey}_periods'] = s.total_periods
            row[f'{hkey}_return'] = round(s.overall_avg_return, 2)
            row[f'{hkey}_winrate'] = round(s.overall_win_rate, 1)
            row[f'{hkey}_sharpe'] = round(s.sharpe_ratio, 2)
            row[f'{hkey}_excess'] = round(s.excess_return, 2)
            row[f'{hkey}_dd'] = round(s.max_drawdown, 2)

        all_rows.append(row)
        print(f"  {pct:>2}%: "
              f"10d {row['10d_return']:+.1f}%/{row['10d_sharpe']:.2f}  "
              f"60d {row['60d_return']:+.1f}%/{row['60d_sharpe']:.2f}")

    # 6. 保存结果
    df = pd.DataFrame(all_rows)
    out_path = 'reports/weight_sweep.csv'
    os.makedirs('reports', exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {out_path}")

    # 7. 找峰值
    for hkey in HORIZONS:
        print(f"\n{hkey} 最佳夏普: pct={df.loc[df[f'{hkey}_sharpe'].idxmax(), 'new_factor_pct']}%")
        print(f"  最佳收益: pct={df.loc[df[f'{hkey}_return'].idxmax(), 'new_factor_pct']}%")
        print(f"  最佳胜率: pct={df.loc[df[f'{hkey}_winrate'].idxmax(), 'new_factor_pct']}%")

    # 恢复权重为V1
    v1 = {
        '10d': {'zxdkx':0,'zxdq':35,'vol':15,'macd':15,'rsi':15,'vol_surge':10,'ma_alignment':5,'trend_strength':5},
        '20d': {'zxdkx':8,'zxdq':35,'vol':22,'macd':0,'rsi':15,'vol_surge':5,'ma_alignment':10,'trend_strength':5},
        '60d': {'zxdkx':35,'zxdq':35,'vol':8,'macd':0,'rsi':8,'vol_surge':2,'ma_alignment':5,'trend_strength':7},
    }
    for k in v1: W[k] = v1[k]
    print("\n权重已恢复为V1(15%)")


if __name__ == '__main__':
    main()
