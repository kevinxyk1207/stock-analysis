"""
沪深300 B1选股评分 — 双档持有期评分（短线/长线）+ 重点监控
选股范围: 沪深300（6年全周期夏普0.98，唯一有效范围；扩至中证500/1000均导致衰退）
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime

from enhanced_fetcher import load_cache_data, load_stock_name_map
from b1_selector import B1Selector, B1Config, HORIZON_LABELS


# 重点监控股票列表
WATCH_LIST = ['301611', '600869', '605358']


def main():
    config = B1Config(
        j_threshold=10.0, j_q_threshold=0.30,
        kdj_n=9,
        zx_m1=5, zx_m2=20, zx_m3=40, zx_m4=60,
        zxdq_span=10,
        wma_short=5, wma_mid=10, wma_long=15,
        max_vol_lookback=20
    )
    selector = B1Selector(config)

    print("加载沪深300缓存数据...")
    stock_data_raw = load_cache_data(min_rows=200, common_range=False)
    print(f"缓存共 {len(stock_data_raw)} 只")

    # 只保留沪深300（6年全周期夏普0.98，扩至中证500/1000均导致衰退）
    from enhanced_fetcher import EnhancedStockFetcher
    fetcher = EnhancedStockFetcher()
    hs300 = fetcher.get_hs300_stocks()
    hs300_set = set(str(s).zfill(6) for s in hs300['code'].tolist())
    stock_data = {k: v for k, v in stock_data_raw.items() if str(k).zfill(6) in hs300_set}
    print(f"过滤后(仅HS300): {len(stock_data)} 只")

    last_date = max(df.index[-1] for df in stock_data.values())
    print(f"最新数据日期: {last_date.date()}")

    # ── 市场环境判断 ──
    close_dict = {}
    for sym, df in stock_data.items():
        close_dict[sym] = pd.Series(df['close'].values)
    market_close = pd.DataFrame(close_dict).mean(axis=1)
    market_ma60 = market_close.rolling(60, min_periods=60).mean()
    is_bear = market_close.iloc[-1] < market_ma60.iloc[-1] if len(market_ma60) > 0 and pd.notna(market_ma60.iloc[-1]) else False
    regime = "!! 熊市 (均价<MA60) 建议空仓 !!" if is_bear else "OK 牛市 (均价>MA60)"
    print(f"市场环境: {regime}")

    print("加载股票名称...")
    name_map = load_stock_name_map()
    print(f"共 {len(name_map)} 只股票有名称信息")

    # 预计算并评分（三档持有期）
    results = []
    total = len(stock_data)
    horizons = ['10d', '60d']

    # 收集信号分布数据（用于诊断）
    signal_collector = {k: [] for k in [
        'zxdkx_ratio', 'zxdq_zxdkx_ratio', 'vol_health',
        'macd_hist_ratio', 'rsi', 'j_value', 'j_percentile',
        'weekly_bull_strength', 'volume_surge', 'price_accel_5',
        'range_expand_5', 'trend_strength', 'ma_alignment',
        'zxdkx_ratio_chg_5d', 'zxdq_ratio_chg_5d', 'vol_health_chg_5d',
        'main_force_flow', 'flow_strength', 'north_bound_proxy',
    ]}

    for i, (symbol, df) in enumerate(stock_data.items(), 1):
        symbol = str(symbol).zfill(6)  # 确保股票代码6位（前导零不丢失）
        if i % 50 == 0:
            print(f"  进度: {i}/{total}")
        try:
            if df.empty or len(df) < 60:
                continue

            prepared = selector.prepare_data(df)
            conditions = selector.check_b1_conditions(prepared, date_idx=-1)

            # 双档评分（10d用legacy因子, 60d用独立引擎）
            scores = {}
            for h in horizons:
                if h == '60d':
                    scores[h] = selector._calculate_score_60d(conditions)
                else:
                    scores[h] = selector._calculate_score(conditions, '10d')

            close = float(prepared["close"].iloc[-1])

            row = {
                "symbol": symbol,
                "name": name_map.get(symbol, ""),
                "close": close,
                "J": round(conditions.get("j_value", 0), 1),
                "zxdkx_ratio": round(conditions.get("zxdkx_ratio", 1.0), 3),
            }
            for h in horizons:
                row[f"score_{h}"] = round(scores[h], 1)

            results.append(row)

            # 收集信号值
            for k in signal_collector:
                v = conditions.get(k)
                if v is not None and not (isinstance(v, float) and (v != v)):
                    signal_collector[k].append(v)

        except Exception:
            continue

    # 按各期限分别输出前十名
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\nB1 选股推荐 (沪深300)  —  {today_str}  共 {len(results)} 只\n")

    for h in horizons:
        sorted_by_h = sorted(results, key=lambda x: x[f"score_{h}"], reverse=True)
        top = sorted_by_h[:10]
        print(f"【{HORIZON_LABELS[h]} Top 10】")
        print(f"  代码    名称         收盘    J值   zxdkx比  得分")
        for r in top:
            print(f"  {r['symbol']:<6} {r['name']:<6} {r['close']:<8.2f} {r['J']:<5} {r['zxdkx_ratio']:<8.3f} {r[f'score_{h}']:<.1f}")
        print()

    # 重点监控股票排名及评分
    print(f"\n【重点监控股票】")
    print(f"  代码    名称         收盘   ")
    for h in horizons:
        sorted_by_h = sorted(results, key=lambda x: x[f"score_{h}"], reverse=True)
        total_n = len(sorted_by_h)
        print(f"  {'—'*40}")
        print(f"  {HORIZON_LABELS[h]}排名:")
        for sym in WATCH_LIST:
            for r in sorted_by_h:
                if r['symbol'] == sym:
                    rank = sorted_by_h.index(r) + 1
                    print(f"  {r['symbol']:<6} {r['name']:<6} {r['close']:<8.2f}  "
                          f"评分={r[f'score_{h}']:<.1f}  排名={rank}/{total_n}")
                    break
    print()

    # 保存三档前十名到Excel（代码列文本格式，避免前导零丢失）
    rows = []
    for h in horizons:
        sorted_by_h = sorted(results, key=lambda x: x[f"score_{h}"], reverse=True)
        for rank, r in enumerate(sorted_by_h[:10], 1):
            rows.append({
                "期限": HORIZON_LABELS[h], "排名": rank,
                "代码": r["symbol"], "名称": r["name"],
                "收盘": r["close"],
                "J值": r["J"], "zxdkx比": r["zxdkx_ratio"],
                "得分": r[f"score_{h}"]
            })
    df = pd.DataFrame(rows)
    xlsx_path = f"reports/screening_{datetime.now().strftime('%Y%m%d')}.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='B1选股推荐', index=False)
            ws = writer.sheets['B1选股推荐']
            # 代码列设为文本格式
            from openpyxl.styles import numbers
            for cell in ws['C'][1:]:  # C列是代码（0索引第2列，Excel第3列）
                cell.number_format = '@'
    except PermissionError:
        print(f"[警告] xlsx文件被占用，跳过xlsx写入，仅保存CSV")
    # 同时保存CSV备用
    # CSV已由daily_report覆盖，不再生成重复文件

    # ── 信号诊断 ──
    print(f"\n【信号诊断】")
    diag_signals = [
        ('核心信号', [
            ('zxdkx_ratio', 'zxdkx比', '↑越高越好'),
            ('zxdq_zxdkx_ratio', 'zxdq比', '↑越高越好'),
            ('vol_health', '量健康', '↑越高越好'),
            ('macd_hist_ratio', 'MACD', '↑越高越好'),
            ('rsi', 'RSI', '50-70最佳'),
        ]),
        ('短线信号', [
            ('volume_surge', '量比', '↑>1.5放量'),
            ('price_accel_5', '5日涨幅%', '↑加速'),
            ('range_expand_5', '波动扩张', '↑波动放大'),
        ]),
        ('长线信号', [
            ('weekly_bull_strength', '周线多头', '↑多头强'),
            ('j_percentile', 'J值分位', '↓低位好'),
        ]),
        ('资金流向信号', [
            ('main_force_flow', '主力资金', '↑流入为正'),
            ('flow_strength', '资金强度', '↑强度越高越好'),
            ('north_bound_proxy', '北向代理', '↑偏好度越高越好'),
        ]),
    ]

    for group_name, sig_list in diag_signals:
        print(f"  [{group_name}]")
        for key, label, _ in sig_list:
            vals = signal_collector.get(key, [])
            if vals:
                s = pd.Series(vals)
                p10, p50, p90 = s.quantile(0.1), s.quantile(0.5), s.quantile(0.9)
                if key == 'rsi':
                    status = '中性'
                    if p50 > 65: status = '偏热'
                    elif p50 < 40: status = '偏冷'
                elif key in ('zxdkx_ratio', 'zxdq_zxdkx_ratio'):
                    status = '偏强' if p50 > 1.03 else ('偏弱' if p50 < 0.97 else '中性')
                elif key == 'vol_health':
                    status = '量能足' if p50 > 0.55 else ('量能弱' if p50 < 0.45 else '中性')
                elif key == 'ma_alignment':
                    status = '排列好' if p50 > 1.5 else '排列差'
                elif key == 'j_percentile':
                    status = '低位' if p50 < 0.3 else ('高位' if p50 > 0.7 else '中性')
                elif key == 'main_force_flow':
                    status = '流入强' if p50 > 0.1 else ('流出' if p50 < -0.1 else '中性')
                elif key == 'flow_strength':
                    status = '强度高' if p50 > 0.05 else ('弱势' if p50 < -0.05 else '中性')
                elif key == 'north_bound_proxy':
                    status = '偏好高' if p50 > 8 else ('低偏好' if p50 < 5 else '中性')
                else:
                    status = ''
                print(f"    {label:<8} P10={p10:<8.3f}  P50={p50:<8.3f}  P90={p90:<8.3f}  {status}")

    # 核心信号 + 资金流向信号 + 最终评分 相关性矩阵
    core_keys = ['zxdkx_ratio', 'zxdq_zxdkx_ratio', 'vol_health', 'macd_hist_ratio', 'rsi',
                 'main_force_flow', 'flow_strength', 'north_bound_proxy']
    score_keys = [f'score_{h}' for h in horizons]
    corr_data = {}
    for k in core_keys:
        corr_data[k] = signal_collector.get(k, [])
    for h in horizons:
        corr_data[f'score_{h}'] = [r[f'score_{h}'] for r in results[:len(next(iter(corr_data.values()), []))]]

    if all(len(v) > 0 for v in corr_data.values()):
        corr_df = pd.DataFrame(corr_data)
        corr_df = corr_df.dropna()
        if len(corr_df) > 100:
            print(f"\n  [信号相关性]")
            c = corr_df.corr()
            cols = corr_df.columns.tolist()
            print(f"    {'信号':<20}  {'  '.join(f'{k:<6}' for k in cols)}")
            for name in cols:
                print(f"    {name:<20}  {'  '.join(f'{c.loc[name, k]:.2f}' for k in cols)}")



if __name__ == "__main__":
    main()
