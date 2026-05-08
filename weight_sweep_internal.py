"""
新因子内部分配扫描 — 固定总占比30%，测试vol_surge/ma_alignment/trend_strength比例
结果保存到 reports/weight_sweep_internal.csv
"""
import sys, os, logging, numpy as np
logging.basicConfig(level=logging.WARNING)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_fetcher import EnhancedStockFetcher, load_cache_data
from b1_selector import B1Selector, B1Config, TIME_HORIZON_WEIGHTS
from b1_backtest import TopNHoldingBacktest
import baostock as bs, pandas as pd

def main():
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

    print("获取指数数据...")
    lg = bs.login()
    rs = bs.query_history_k_data_plus('sh.000300','date,close',
        start_date='2020-01-01', end_date='2025-12-31', frequency='d', adjustflag='3')
    data = []
    while rs.next(): data.append(rs.get_row_data())
    bs.logout()
    index_df = pd.DataFrame(data, columns=['date','close'])
    index_df['close'] = index_df['close'].replace('', np.nan).astype(float)
    index_df = index_df.dropna()
    index_df['date'] = pd.to_datetime(index_df['date'])
    index_df = index_df.set_index('date')

    def make_weights(pct, old_base, new_ratio):
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

    # 内部分配组合: (vol_surge, ma_alignment, trend_strength) 占30%的比例
    COMBOS = [
        (100, 0, 0),
        (75, 25, 0),
        (75, 0, 25),
        (67, 33, 0),
        (60, 20, 20),
        (50, 50, 0),
        (50, 25, 25),
        (50, 0, 50),
        (40, 40, 20),
        (40, 20, 40),
        (34, 33, 33),
        (33, 33, 33),  # 等权
        (25, 75, 0),
        (25, 50, 25),
        (25, 25, 50),
        (25, 0, 75),
        (20, 60, 20),
        (20, 40, 40),
        (20, 20, 60),
        (0, 100, 0),
        (0, 75, 25),
        (0, 67, 33),
        (0, 50, 50),
        (0, 33, 67),
        (0, 25, 75),
        (0, 0, 100),
    ]

    OLD_BASE = {
        '10d': {'zxdkx': 0, 'zxdq': 40, 'vol': 20, 'macd': 20, 'rsi': 20},
        '60d': {'zxdkx': 40, 'zxdq': 40, 'vol': 10, 'rsi': 10},
    }
    HP = {'10d': 10, '60d': 60}
    STEP = {'10d': 10, '60d': 20}

    W = TIME_HORIZON_WEIGHTS
    bt = TopNHoldingBacktest(B1Selector(B1Config()))

    all_rows = []
    print(f"\n扫描: {len(COMBOS)} 种内部分配 x 2 个周期 = {len(COMBOS)*2} 组")
    for vs, ma, ts in COMBOS:
        ratio = {'vol_surge': vs, 'ma_alignment': ma, 'trend_strength': ts}
        row = {'vol_surge_pct': vs, 'ma_alignment_pct': ma, 'trend_strength_pct': ts}

        for hkey in ['10d', '60d']:
            W[hkey] = make_weights(30, OLD_BASE[hkey], ratio)
            s = bt.run(filtered, top_n=10,
                       hold_period=HP[hkey], eval_step=STEP[hkey],
                       index_df=index_df)
            row[f'{hkey}_periods'] = s.total_periods
            row[f'{hkey}_return'] = round(s.overall_avg_return, 2)
            row[f'{hkey}_winrate'] = round(s.overall_win_rate, 1)
            row[f'{hkey}_sharpe'] = round(s.sharpe_ratio, 2)
            row[f'{hkey}_excess'] = round(s.excess_return, 2)
            row[f'{hkey}_dd'] = round(s.max_drawdown, 2)

        all_rows.append(row)
        print(f"  [{vs:>3}/{ma:>3}/{ts:>3}]  "
              f"10d {row['10d_return']:+.1f}%/{row['10d_sharpe']:.2f}  "
              f"60d {row['60d_return']:+.1f}%/{row['60d_sharpe']:.2f}")

    df = pd.DataFrame(all_rows)
    out_path = 'reports/weight_sweep_internal.csv'
    os.makedirs('reports', exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {out_path}")

    # 找最佳
    for hkey in ['10d', '60d']:
        print(f"\n{'='*50}")
        print(f"{hkey} 最佳夏普:")
        best = df.loc[df[f'{hkey}_sharpe'].idxmax()]
        print(f"  V:{best['vol_surge_pct']:.0f}%  M:{best['ma_alignment_pct']:.0f}%  T:{best['trend_strength_pct']:.0f}%  "
              f"夏普={best[f'{hkey}_sharpe']}  收益={best[f'{hkey}_return']}%  胜率={best[f'{hkey}_winrate']}%  回撤={best[f'{hkey}_dd']}%")
        print(f"  最佳收益:")
        best_r = df.loc[df[f'{hkey}_return'].idxmax()]
        print(f"  V:{best_r['vol_surge_pct']:.0f}%  M:{best_r['ma_alignment_pct']:.0f}%  T:{best_r['trend_strength_pct']:.0f}%  "
              f"夏普={best_r[f'{hkey}_sharpe']}  收益={best_r[f'{hkey}_return']}%  胜率={best_r[f'{hkey}_winrate']}%  回撤={best_r[f'{hkey}_dd']}%")
        print(f"  最佳胜率:")
        best_w = df.loc[df[f'{hkey}_winrate'].idxmax()]
        print(f"  V:{best_w['vol_surge_pct']:.0f}%  M:{best_w['ma_alignment_pct']:.0f}%  T:{best_w['trend_strength_pct']:.0f}%  "
              f"夏普={best_w[f'{hkey}_sharpe']}  收益={best_w[f'{hkey}_return']}%  胜率={best_w[f'{hkey}_winrate']}%  回撤={best_w[f'{hkey}_dd']}%")

    print("\n扫描完成")

if __name__ == '__main__':
    main()
