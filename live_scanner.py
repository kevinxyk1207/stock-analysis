"""
实时异动扫描器
全在线：实时行情 → 量价信号 → 基本面验证
不依赖本地缓存
"""
import sys, os, json, logging, time
# 绕过 Windows 系统代理
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["no_proxy"] = "*"
import requests as _requests
_original_get = _requests.Session.get
def _patched_get(self, url, **kwargs):
    kwargs.setdefault("proxies", {"http": None, "https": None})
    kwargs.setdefault("timeout", 10)
    return _original_get(self, url, **kwargs)
_requests.Session.get = _patched_get
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 东方财富 API
_SPOT_URL = "https://push2.eastmoney.com/api/qt/clist/get"
_HIST_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def fetch_all_spot() -> pd.DataFrame:
    """拉全市场实时行情（akshare，已验证 ~2s for 5850 stocks）"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return pd.DataFrame()
        col_map = {
            "代码": "code", "名称": "name", "最新价": "price",
            "涨跌幅": "pct", "涨跌额": "change_amt",
            "成交量": "volume", "成交额": "turnover",
            "换手率": "turnover_rate",
            "最高": "high", "最低": "low", "今开": "open", "昨收": "prev_close",
            "市盈率-动态": "pe", "市净率": "pb",
            "总市值": "total_mv",
            "60日涨跌幅": "change_60d",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["code"] = df["code"].astype(str).str.zfill(6)
        for col in ["price", "pct", "volume", "turnover", "turnover_rate",
                     "high", "low", "open", "prev_close", "change_60d"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        logger.error(f"实时行情获取失败: {e}")
        return pd.DataFrame()


def fetch_daily_hist(code: str, days: int = 120) -> pd.DataFrame:
    """拉单只股票日线（akshare，~0.5s/只）"""
    try:
        import akshare as ak
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                                 "最高": "high", "最低": "low", "成交量": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()
    except Exception:
        return pd.DataFrame()


def scan_signals(spot: pd.DataFrame) -> pd.DataFrame:
    """
    Layer 1: 实时行情信号扫描（全市场 5850 只，秒出）
    """
    df = spot.copy()
    if df.empty:
        return df

    # 过滤：有成交、有换手率、非 ST
    df = df[df["turnover"] > 1e6]  # 成交额 > 100万
    df = df[df["turnover_rate"] > 0.1]  # 换手率 > 0.1%
    df = df[~df["name"].str.contains("ST|退", na=False)]
    df = df[df["price"] > 2]  # 排除 2 元以下仙股

    if df.empty:
        return df

    # ── 信号 1: 涨幅 Top（市场在投票）──
    df["_signal_momentum"] = 0
    top_pct = df["pct"].quantile(0.95)
    df.loc[df["pct"] >= top_pct, "_signal_momentum"] = 2
    df.loc[(df["pct"] >= top_pct * 0.6) & (df["pct"] < top_pct), "_signal_momentum"] = 1

    # ── 信号 2: 有效放量（换手率高 + 成交额大）──
    df["_signal_volume"] = 0
    high_tr = df["turnover_rate"].quantile(0.90)
    high_to = df["turnover"].quantile(0.90)
    df.loc[(df["turnover_rate"] >= high_tr) & (df["turnover"] >= high_to), "_signal_volume"] = 2
    df.loc[(df["turnover_rate"] >= high_tr * 0.5) & (df["turnover"] >= high_to * 0.5), "_signal_volume"] = 1

    # ── 信号 3: 突破（今日高点 > 昨日收盘 × 1.02）──
    df["_signal_breakout"] = 0
    df.loc[(df["high"] >= df["prev_close"] * 1.03) & (df["pct"] >= 2), "_signal_breakout"] = 2
    df.loc[(df["high"] >= df["prev_close"] * 1.02) & (df["pct"] >= 1), "_signal_breakout"] = 1

    # ── 综合 ──
    df["_total_signal"] = df["_signal_momentum"] + df["_signal_volume"] + df["_signal_breakout"]
    df = df.sort_values("_total_signal", ascending=False)

    return df


def verify_volume_pattern(code: str, df_spot_row) -> dict:
    """
    Layer 2: 拉日线验证量价形态
    """
    df = fetch_daily_hist(code, days=120)
    if df.empty or len(df) < 20:
        return {"verified": False, "reason": "数据不足"}

    close = df["close"]
    vol = df["volume"]

    today_vol = df_spot_row.get("volume", 0)
    turnover = df_spot_row.get("turnover", 0)

    signals = []

    # 1. 放量判断：今日量 vs 20日均量
    avg_vol_20 = vol.tail(21).iloc[:-1].mean()
    vol_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 1
    if vol_ratio >= 2:
        signals.append(f"爆量 {vol_ratio:.1f}x")
    elif vol_ratio >= 1.5:
        signals.append(f"放量 {vol_ratio:.1f}x")

    # 2. 位置判断：价格在 60 日中的位置
    high_60 = close.tail(60).max()
    low_60 = close.tail(60).min()
    latest = close.iloc[-1]
    pos_60 = (latest - low_60) / (high_60 - low_60) * 100 if high_60 > low_60 else 50

    if pos_60 < 30:
        signals.append(f"低位({pos_60:.0f}%)")
    elif pos_60 > 80:
        signals.append(f"高位({pos_60:.0f}%)")

    # 3. 均线排列
    ma5 = close.tail(5).mean()
    ma20 = close.tail(20).mean()
    ma60 = close.tail(60).mean() if len(close) >= 60 else ma20
    if ma5 > ma20 > ma60:
        signals.append("多头排列")
    elif ma5 < ma20 < ma60:
        signals.append("空头排列")

    # 4. 近期涨幅
    ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
    ret_60d = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) >= 61 else 0

    verified = len(signals) >= 2 or vol_ratio >= 1.5

    return {
        "verified": verified,
        "vol_ratio": round(vol_ratio, 1),
        "position_60": round(pos_60, 0),
        "ret_20d": round(ret_20d, 1),
        "ret_60d": round(ret_60d, 1),
        "ma_trend": "多头" if ma5 > ma20 > ma60 else ("空头" if ma5 < ma20 < ma60 else "交叉"),
        "signals": signals,
        "turnover": turnover,
    }


def run_live_scan(top_n: int = 20) -> dict:
    """
    主入口：实时异动扫描全流程
    """
    t0 = time.time()
    logger.info("实时异动扫描开始")

    # Layer 1: 全市场实时信号
    logger.info("[1/3] 拉全市场实时行情...")
    spot = fetch_all_spot()
    if spot.empty:
        return {"error": "实时行情获取失败", "hits": []}
    logger.info(f"  获取 {len(spot)} 只")

    signals = scan_signals(spot)
    logger.info(f"  信号筛选后: {len(signals)} 只 (信号≥1)")

    # Top hits for Layer 2
    hits_df = signals[signals["_total_signal"] >= 3].head(top_n * 3)
    if hits_df.empty:
        hits_df = signals[signals["_total_signal"] >= 2].head(top_n * 2)
    logger.info(f"  待验证: {len(hits_df)} 只")

    # Layer 2: 量价验证
    logger.info("[2/3] 日线量价验证...")
    verified = []
    for i, (_, row) in enumerate(hits_df.iterrows()):
        code = row["code"]
        try:
            result = verify_volume_pattern(code, row)
            if result["verified"]:
                verified.append({
                    "code": code,
                    "name": str(row["name"]),
                    "price": float(row["price"]),
                    "pct": float(row["pct"]),
                    "turnover": result["turnover"],
                    "vol_ratio": result["vol_ratio"],
                    "position_60": result["position_60"],
                    "ret_20d": result["ret_20d"],
                    "ret_60d": result["ret_60d"],
                    "ma_trend": result["ma_trend"],
                    "signals": result["signals"],
                    "total_signal": int(row["_total_signal"]),
                })
            if len(verified) >= top_n:
                break
        except Exception:
            continue
        if (i + 1) % 10 == 0:
            logger.info(f"  ... {i+1}/{len(hits_df)}, 已确认 {len(verified)}")

    logger.info(f"  确认 {len(verified)} 只异动股")

    # Layer 3: 基本面交叉
    logger.info("[3/3] 基本面验证...")
    try:
        from stock_analyzer import fetch_fundamentals_all
        fund_all = fetch_fundamentals_all()
    except Exception:
        fund_all = {}

    for v in verified:
        code = v["code"]
        fund = fund_all.get(code, {})
        v["profit_yoy"] = _fv(fund.get("profit_growth"))
        v["margin"] = _fv(fund.get("gross_margin"))
        v["fund_ok"] = (v["profit_yoy"] > 0 and v["margin"] > 15)

    logger.info(f"  总耗时: {time.time()-t0:.0f}s")
    return {"time": datetime.now().strftime("%H:%M:%S"), "hits": verified}


def _fv(val):
    try: return float(val)
    except: return 0


if __name__ == "__main__":
    result = run_live_scan(top_n=15)
    if "error" in result:
        print(f"ERROR: {result['error']}")
        exit(1)

    print(f"\n实时异动扫描 — {result['time']}")
    print(f"{'='*70}")
    print(f"{'代码':<8} {'名称':<8} {'价格':>7} {'涨幅':>7} {'量比':>5} {'位置':>5} {'20d':>7} {'均线':>4} {'基本面':>6}")
    print(f"{'-'*70}")
    for h in result["hits"]:
        fund_tag = "OK" if h.get("fund_ok") else ("Q{}".format(round(h.get("profit_yoy", 0), 0)) if h.get("profit_yoy") else "??")
        print(f"{h['code']:<8} {h['name']:<8} {h['price']:>7.2f} {h['pct']:>+6.2f}% {h['vol_ratio']:>4.1f}x {h['position_60']:>4.0f}% {h['ret_20d']:>+6.1f}% {h['ma_trend']:>4} {fund_tag:>6}")
