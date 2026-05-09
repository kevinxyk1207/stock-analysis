"""
买卖点组合回测
日线潜伏放量(入场信号) + 周线v3确认(买入) + 周线v3优化(卖出)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
from datetime import datetime


def detect_lurking_signal(df_daily, idx):
    """日线潜伏放量信号——量先动、价未动、位置低"""
    if idx < 60:
        return False
    close = df_daily["close"].iloc[:idx+1]
    vol = df_daily["volume"].iloc[:idx+1]

    vol_ma5 = vol.iloc[-5:].mean()
    vol_ma20 = vol.iloc[-20:].mean()
    high_60 = close.iloc[-60:].max()
    low_60 = close.iloc[-60:].min()
    if high_60 == low_60 or vol_ma20 == 0:
        return False
    pos_60 = (close.iloc[-1] - low_60) / (high_60 - low_60) * 100
    ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
    ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0

    # 量比1.5x, 位置<30%, 5日没涨(<5%), 20日没飞(<20%)
    return (vol_ma5 > vol_ma20 * 1.5 and pos_60 < 30 and ret_5d < 5 and ret_20d < 20)


def detect_weekly_entry(weekly_close, weekly_high, weekly_vol, i):
    """周线 v3 买入确认"""
    if i < 26:
        return False, ""
    wc = weekly_close.iloc[i]
    wv = weekly_vol.iloc[i] * wc / 1e8
    avg_vol_10w = weekly_vol.iloc[max(0,i-10):i].mean() * weekly_close.iloc[max(0,i-10):i].mean() / 1e8
    ma5w = weekly_close.iloc[max(0,i-4):i+1].mean()
    delta = weekly_close.iloc[i-13:i+1].diff()
    gain = delta.clip(lower=0).mean()
    loss = (-delta).clip(lower=0).mean()
    rsi = 100 - 100/(1+gain/(loss+1e-9)) if loss > 0 else 100
    low_26w = weekly_close.iloc[max(0,i-26):i+1].min()
    ema5 = weekly_close.iloc[max(0,i-4):i+1].ewm(span=5).mean().iloc[-1]
    ema15 = weekly_close.iloc[max(0,i-14):i+1].ewm(span=15).mean().iloc[-1]

    if not (wv > avg_vol_10w * 2 and 55 <= rsi <= 78 and wc > ma5w):
        return False, ""
    if wc < low_26w * 1.30:
        return False, ""
    if ema5 <= ema15:
        return False, ""

    reason = f"放量{wv/avg_vol_10w:.1f}x RSI{rsi:.0f} 距低{ (wc/low_26w-1)*100:.0f}%"
    return True, reason


def detect_weekly_exit(weekly_close, weekly_high, weekly_vol, i, entry_idx):
    """周线 v3 优化卖出（去掉滞后信号，加持仓限制）"""
    wc = weekly_close.iloc[i]
    wv = weekly_vol.iloc[i] * wc / 1e8
    avg_vol_10w = weekly_vol.iloc[max(0,i-10):i].mean() * weekly_close.iloc[max(0,i-10):i].mean() / 1e8
    ma5w = weekly_close.iloc[max(0,i-4):i+1].mean()
    delta = weekly_close.iloc[i-13:i+1].diff()
    gain = delta.clip(lower=0).mean()
    loss = (-delta).clip(lower=0).mean()
    rsi = 100 - 100/(1+gain/(loss+1e-9)) if loss > 0 else 100
    rh = weekly_high.iloc[max(0,i-12):i+1].max()
    pv = weekly_vol.iloc[max(0,i-12):i+1].max() * weekly_close.iloc[max(0,i-12):i+1].max() / 1e8
    near_peak = wc >= rh * 0.97
    wv_prev = weekly_vol.iloc[i-1] * weekly_close.iloc[i-1] / 1e8 if i > 0 else wv
    two_week_div = (near_peak and wv < pv * 0.5 and wv_prev < pv * 0.5)
    rsi_not_confirm = near_peak and rsi < 75

    if two_week_div:
        return True, "连续2周量价背离"
    elif rsi_not_confirm and wc < ma5w * 0.97:
        return True, "RSI背离+跌破MA5"
    return False, ""


def run_combined_backtest(df, name="", mc=None, mma60=None):
    """运行完整买卖点回测"""
    weekly_close = df["close"].resample("W-FRI").last().dropna()
    weekly_high = df["high"].resample("W-FRI").max()
    weekly_vol = df["volume"].resample("W-FRI").sum()
    daily_close = df["close"]

    trades = []
    position = None
    entry_idx_w = None
    entry_price = 0

    for wi in range(26, len(weekly_close)):
        w_date = weekly_close.index[wi]
        wc = weekly_close.iloc[wi]

        if position is None:
            # 个股趋势过滤：MA20 > MA60（中长期向上）
            if len(weekly_close) >= 60:
                ma20w = weekly_close.iloc[max(0,wi-20):wi+1].mean()
                ma60w = weekly_close.iloc[max(0,wi-60):wi+1].mean()
                if ma20w <= ma60w:
                    continue

            di = None
            for d in range(len(daily_close)-1, 0, -1):
                if daily_close.index[d] <= w_date:
                    di = d
                    break
            if di is None or di < 60:
                continue

            lurking = detect_lurking_signal(df, di)
            if lurking:
                low_26w = weekly_close.iloc[max(0,wi-26):wi+1].min()
                if wc >= low_26w * 1.05:
                    position = "long"
                    entry_idx_w = wi
                    entry_price = wc
                    trades.append({
                        "entry_date": str(weekly_close.index[wi])[:10],
                        "entry_price": round(wc, 2),
                        "entry_reason": f"潜伏放量 距低{(wc/low_26w-1)*100:.0f}%",
                        "code": name,
                    })

        elif position == "long":
            exit_ok, reason = detect_weekly_exit(weekly_close, weekly_high, weekly_vol, wi, entry_idx_w)
            if exit_ok or wi == len(weekly_close) - 1:  # 最后一周强制平仓
                ret = (wc - entry_price) / entry_price * 100
                weeks = wi - entry_idx_w
                trades[-1].update({
                    "exit_date": str(weekly_close.index[wi])[:10],
                    "exit_price": round(wc, 2),
                    "return": round(ret, 1),
                    "weeks": weeks,
                    "exit_reason": reason if exit_ok else "持仓到期",
                })
                position = None
    return trades


def detect_current_signals(code: str, df) -> dict:
    """
    检测当前最后一根 K 线的买卖信号。
    返回 {"buy": [...], "sell": [...], "neutral": bool}
    供 Web 调用，不依赖周线。
    """
    if df is None or len(df) < 60:
        return {"buy": [], "sell": [], "neutral": True}

    close = df["close"]
    vol = df["volume"]
    high = df["high"]
    idx = len(df) - 1

    high_60 = close.iloc[-60:].max()
    low_60 = close.iloc[-60:].min()
    pos_60 = (close.iloc[-1] - low_60) / (high_60 - low_60) * 100 if high_60 != low_60 else 50

    vol_ma5 = vol.iloc[-5:].mean()
    vol_ma20 = vol.iloc[-20:].mean()
    vol_ratio = vol.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1

    ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
    ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0

    buy_signals = []
    sell_signals = []

    # ── 买点：极紧反转 ──
    if pos_60 < 15 and vol_ratio < 0.5:
        buy_signals.append({
            "type": "极紧反转",
            "detail": f"位置{pos_60:.0f}% 缩量{vol_ratio:.1f}x",
            "strength": "strong",
        })

    # ── 买点：潜伏放量 ──
    if vol_ratio > 1.5 and pos_60 < 30 and ret_5d < 5 and ret_20d < 20:
        strength = "medium"
        if vol_ratio > 2.0 and pos_60 < 20:
            strength = "strong"
        buy_signals.append({
            "type": "潜伏放量",
            "detail": f"量比{vol_ratio:.1f}x 位置{pos_60:.0f}%",
            "strength": strength,
        })

    # ── 卖点：量价背离 ──
    lookback_amt = min(30, 240)  # 约1年
    rh = high.iloc[-lookback_amt:].max()
    pv = (vol * close).iloc[-lookback_amt:].max() / 1e8
    today_amt = vol.iloc[-1] * close.iloc[-1] / 1e8
    prev_amt = vol.iloc[-2] * close.iloc[-2] / 1e8 if len(vol) >= 2 else today_amt
    near_peak = close.iloc[-1] >= rh * 0.97

    if near_peak and today_amt < pv * 0.5 and prev_amt < pv * 0.5:
        sell_signals.append({
            "type": "量价背离",
            "detail": f"近{rh:.1f}高点 量缩至{pv*1e8/1e8:.0f}亿",
            "strength": "strong",
        })

    # ── 卖点：RSI 背离 ──
    delta = close.pct_change().iloc[-14:]
    gain = delta.clip(lower=0).mean()
    loss = (-delta).clip(lower=0).mean()
    rsi = 100 - 100/(1+gain/(loss+1e-9)) if loss > 0 else 100
    ma20 = close.iloc[-20:].mean()
    if near_peak and rsi < 75 and close.iloc[-1] < ma20 * 0.97:
        sell_signals.append({
            "type": "RSI背离",
            "detail": f"RSI{rsi:.0f} 破MA20",
            "strength": "medium",
        })

    return {
        "buy": buy_signals,
        "sell": sell_signals,
        "neutral": len(buy_signals) == 0 and len(sell_signals) == 0,
        "pos_60": round(pos_60, 0),
        "vol_ratio": round(vol_ratio, 1),
        "ret_20d": round(ret_20d, 1),
    }


def main():
    from enhanced_fetcher import load_cache_data
    data = load_cache_data(min_rows=500, common_range=False)
    eligible = [(k, len(v)) for k, v in data.items() if len(v) >= 1000]
    eligible.sort(key=lambda x: x[1], reverse=True)
    stocks = {k: data[k] for k, _ in eligible[:100]}

    all_trades = []
    stock_stats = []
    for code, df in stocks.items():
        trades = run_combined_backtest(df, code)
        if trades:
            completed = [t for t in trades if "return" in t]
            if len(completed) >= 2:
                rets = [t["return"] for t in completed]
                all_trades.extend(completed)
                stock_stats.append({
                    "code": code,
                    "trades": len(completed),
                    "avg_ret": np.mean(rets),
                    "win_rate": np.mean([r > 0 for r in rets]) * 100,
                    "total_ret": np.prod([1 + r/100 for r in rets]) * 100 - 100,
                })

    print(f"\n有交易的股票: {len(stock_stats)} 只, 总交易: {len(all_trades)} 次")
    if not stock_stats:
        print("没有找到足够交易信号")
        return

    print(f"\n--- 总体统计 ---")
    avg_r = np.mean([s["avg_ret"] for s in stock_stats])
    avg_w = np.mean([s["win_rate"] for s in stock_stats])
    avg_tr = np.mean([s["total_ret"] for s in stock_stats])
    print(f"每只股票平均收益: {avg_r:+.1f}%")
    print(f"平均胜率: {avg_w:.1f}%")
    print(f"平均累计收益: {avg_tr:+.0f}%")
    print(f"总交易次数: {len(all_trades)}")

    # 按收益排序
    rets_sorted = sorted([t["return"] for t in all_trades])
    print(f"\n--- 单次交易分布 ---")
    print(f"总交易: {len(rets_sorted)}, 均值: {np.mean(rets_sorted):+.1f}%, 中位: {np.median(rets_sorted):+.1f}%")
    print(f"胜率: {np.mean([r>0 for r in rets_sorted])*100:.0f}%")
    print(f"Top5: {rets_sorted[-5:]}")
    print(f"Bot5: {rets_sorted[:5]}")

    # 最近交易
    recent = sorted(all_trades, key=lambda t: t["entry_date"], reverse=True)[:5]
    print(f"\n--- 最近 5 笔 ---")
    for t in recent:
        r = t.get("return", "持仓中")
        print(f"  {t['code']} {t['entry_date']}→{t.get('exit_date','?')} {t['entry_price']}→{t.get('exit_price','?')} 收益={r}%" if isinstance(r, float) else f"  {t['code']} {t['entry_date']} 持仓中")


if __name__ == "__main__":
    main()
