"""
纸上跟踪脚本 — 每日运行，管理虚拟投资组合
首次运行: 买入20d Top 5 (周期扫频验证夏普1.48最优)
后续运行: 展示持仓收益 / 20交易日到期后自动换仓
"""
import sys, os, json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from enhanced_fetcher import load_cache_data, load_stock_name_map, EnhancedStockFetcher
from b1_selector import B1Selector, B1Config

TRACKING_FILE = "reports/paper_portfolio.json"


def load_portfolio():
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_portfolio(pf):
    os.makedirs("reports", exist_ok=True)
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)


def get_market_regime(stock_data):
    close_dict = {}
    for sym, df in stock_data.items():
        close_dict[sym] = pd.Series(df["close"].values)
    mc = pd.DataFrame(close_dict).mean(axis=1)
    mma60 = mc.rolling(60, min_periods=60).mean()
    if len(mma60) > 0 and pd.notna(mma60.iloc[-1]):
        return mc.iloc[-1] > mma60.iloc[-1], mc.iloc[-1], mma60.iloc[-1]
    return True, 0, 0


def count_trading_days(data_dict, from_date):
    """估算从from_date至今的交易日数"""
    for df in data_dict.values():
        if not df.empty:
            dt = pd.to_datetime(from_date)
            return int(sum(1 for d in df.index if d > dt))
    return 0


def main():
    print("=" * 50)
    print(f"  B1 纸上跟踪 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    # 加载数据
    from enhanced_fetcher import EnhancedStockFetcher
    fetcher = EnhancedStockFetcher()
    hs300 = fetcher.get_hs300_stocks()
    hs300_set = set(str(s).zfill(6) for s in hs300["code"].tolist())

    stock_data_raw = load_cache_data(min_rows=200, common_range=False)
    stock_data = {k: v for k, v in stock_data_raw.items() if str(k).zfill(6) in hs300_set}

    last_date = max(df.index[-1] for df in stock_data.values())
    print(f"最新数据: {last_date.date()}")

    # 市场环境
    is_bull, mc, ma = get_market_regime(stock_data)
    regime = "牛市" if is_bull else "熊市"
    print(f"市场环境: {regime} (均价={mc:.2f}, MA60={ma:.2f})")

    config = B1Config(
        j_threshold=10.0, j_q_threshold=0.30, kdj_n=9,
        zx_m1=5, zx_m2=20, zx_m3=40, zx_m4=60, zxdq_span=10,
        wma_short=5, wma_mid=10, wma_long=15, max_vol_lookback=20,
    )
    selector = B1Selector(config)
    name_map = load_stock_name_map()

    # 计算60d评分
    results = []
    for sym, df in stock_data.items():
        sym = str(sym).zfill(6)
        try:
            if df.empty or len(df) < 60:
                continue
            prepared = selector.prepare_data(df)
            conditions = selector.check_b1_conditions(prepared, date_idx=-1)
            score = selector._calculate_score_60d(conditions)
            close = float(prepared["close"].iloc[-1])
            results.append(
                {"symbol": sym, "name": name_map.get(sym, ""),
                 "close": close, "score": round(score, 1)}
            )
        except Exception:
            continue

    all_ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    top5 = all_ranked[:5]
    reserve = all_ranked[5:8]  # 备选池 6-8名
    score_rank = {r["symbol"]: i+1 for i, r in enumerate(all_ranked)}  # 每只股票的当前排名
    today_str = last_date.strftime("%Y-%m-%d")

    # ── 持仓管理 ──
    pf = load_portfolio()

    if pf is None:
        # 首次运行：建仓
        if not is_bull:
            print("\n⚠ 熊市 — 不建议建仓。等待牛市信号。")
            print("   (手动覆盖: python paper_tracker.py --force)")
            if "--force" in sys.argv:
                print("   --force 已激活，继续建仓...")
            else:
                return

        pf = {
            "entry_date": today_str,
            "stocks": [],
            "entry_market_close": round(float(mc), 2),
        }
        for r in top5:
            pf["stocks"].append(
                {"symbol": r["symbol"], "name": r["name"],
                 "entry_price": round(r["close"], 2), "score": r["score"]}
            )
        save_portfolio(pf)

        print(f"\n✓ 纸上建仓 ({today_str})")
        print(f"  入场市场价: {pf['entry_market_close']}")
        print(f"  {'代码':<8} {'名称':<8} {'入场价':>10} {'得分':>6}")
        for s in pf["stocks"]:
            print(f"  {s['symbol']:<8} {s['name']:<8} {s['entry_price']:>10.2f} {s['score']:>6.1f}")

    else:
        # 已有持仓：检查是否到期
        days_held = count_trading_days(stock_data, pf["entry_date"])
        print(f"\n持仓天数: {days_held} 个交易日 (入场日 {pf['entry_date']})")

        # 获取当前价格
        current_prices = {}
        for f in os.listdir(fetcher.cache_dir):
            if not f.endswith(".csv"):
                continue
            sym = f[:-4]
            current_prices[sym] = None
            try:
                df = pd.read_csv(
                    os.path.join(fetcher.cache_dir, f), index_col=0, parse_dates=True
                )
                if not df.empty:
                    current_prices[sym] = float(df["close"].iloc[-1])
            except Exception:
                pass

        # 展示当前持仓
        total_pnl = 0
        print(f"\n  {'代码':<8} {'名称':<8} {'入场价':>10} {'现价':>10} {'盈亏':>10} {'当前排名':>8}")
        print(f"  {'─' * 60}")
        alerts = []
        for s in pf["stocks"]:
            cp = current_prices.get(s["symbol"].lstrip("0"), current_prices.get(s["symbol"]))
            rank = score_rank.get(s["symbol"], "?")
            if cp is None:
                cp_str = "无数据"
                pnl_str = "—"
            else:
                pnl = (cp - s["entry_price"]) / s["entry_price"] * 100
                total_pnl += pnl
                cp_str = f"{cp:.2f}"
                pnl_str = f"{pnl:+.2f}%"
            rank_str = f"#{rank}"
            if isinstance(rank, int) and rank > 10:
                rank_str += " ⚠"
                alerts.append(f"{s['symbol']} {s['name']} 排名#{rank} (跌出前10)")
            print(f"  {s['symbol']:<8} {s['name']:<8} {s['entry_price']:>10.2f} {cp_str:>10} {pnl_str:>10} {rank_str:>8}")

        avg_pnl = total_pnl / len(pf["stocks"]) if pf["stocks"] else 0
        print(f"  {'─' * 60}")
        print(f"  组合平均盈亏: {avg_pnl:+.2f}%")

        # 备选池
        print(f"\n  备选池 (当前6-8名):")
        for r in reserve:
            print(f"  #{score_rank.get(r['symbol'],'?')} {r['symbol']} {r['name']} {r['score']:.1f}分")

        # 排名预警
        if alerts:
            print(f"\n  排名预警:")
            for a in alerts:
                print(f"  {a}")

        # 市场基准对比
        if "entry_market_close" in pf:
            market_chg = (mc - pf["entry_market_close"]) / pf["entry_market_close"] * 100
            print(f"  同期HS300均价: {market_chg:+.2f}%")

        # 到期换仓
        if days_held >= 20:
            print(f"\n⏰ 已持有20个交易日，到期换仓！")
            # 记录旧持仓表现
            old_entry = pf["entry_date"]
            old_return = avg_pnl
            print(f"  旧持仓 ({old_entry} → {today_str}): {old_return:+.2f}%")

            # 建新仓
            if not is_bull:
                print(f"  当前熊市 — 清仓等待，不建新仓")
                pf = {
                    "entry_date": today_str,
                    "stocks": [],
                    "entry_market_close": round(float(mc), 2),
                    "note": "熊市空仓",
                }
            else:
                pf = {
                    "entry_date": today_str,
                    "stocks": [],
                    "entry_market_close": round(float(mc), 2),
                }
                for r in top5:
                    pf["stocks"].append(
                        {"symbol": r["symbol"], "name": r["name"],
                         "entry_price": round(r["close"], 2), "score": r["score"]}
                    )

            # 保存交易记录
            history_file = "reports/paper_trade_history.csv"
            hist_row = {
                "entry_date": old_entry,
                "exit_date": today_str,
                "return_pct": round(old_return, 2),
                "market_return_pct": round(
                    (mc - pf["entry_market_close"]) / pf["entry_market_close"] * 100
                    if pf["entry_market_close"] > 0
                    else 0,
                    2,
                ),
                "stocks": ",".join(s["symbol"] for s in pf["stocks"]),
            }
            if os.path.exists(history_file):
                hist_df = pd.read_csv(history_file)
                hist_df = pd.concat([hist_df, pd.DataFrame([hist_row])], ignore_index=True)
            else:
                hist_df = pd.DataFrame([hist_row])
            hist_df.to_csv(history_file, index=False, encoding="utf-8-sig")

            save_portfolio(pf)

            if pf["stocks"]:
                print(f"\n✓ 新仓建仓 ({today_str})")
                for s in pf["stocks"]:
                    print(f"  {s['symbol']} {s['name']} @{s['entry_price']:.2f}")
            else:
                print(f"\n  空仓等待牛市信号")

        else:
            remaining = 20 - days_held
            print(f"\n  [*] 距换仓还有 {remaining} 个交易日")

    # ── 历史统计 ──
    history_file = "reports/paper_trade_history.csv"
    if os.path.exists(history_file):
        h = pd.read_csv(history_file)
        if len(h) > 0:
            print(f"\n  ── 纸上跟踪历史 ({len(h)}笔交易) ──")
            print(f"  平均收益: {h['return_pct'].mean():+.2f}%")
            print(f"  胜率: {np.mean(h['return_pct']>0)*100:.1f}%")
            cum = np.prod(1 + h["return_pct"].values / 100)
            print(f"  累计净值: {cum:.4f}")

    print()


if __name__ == "__main__":
    main()
