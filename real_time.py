"""Real-time intraday data via East Money API (single-stock, instant)"""
import requests
import akshare as ak
from datetime import datetime

# 东方财富单股实时行情 API，<1 秒返回
_EM_SPOT_URL = "https://push2.eastmoney.com/api/qt/stock/get"
_EM_FIELDS = "f43,f44,f45,f46,f47,f48,f50,f51,f52,f55,f57,f58,f60,f116,f117,f162,f167,f168,f169,f170,f171"


def _make_secid(code: str) -> str:
    """将 6 位代码转为东方财富 secid（深交所 0.xxx，上交所 1.xxx）"""
    if code.startswith(("0", "3")):
        return f"0.{code}"
    return f"1.{code}"


def get_realtime_quote(code: str):
    """Get real-time spot quote for a single stock. Instant (~200ms)."""
    try:
        resp = requests.get(_EM_SPOT_URL, params={
            "secid": _make_secid(code),
            "fields": _EM_FIELDS,
        }, timeout=5, proxies={"http": None, "https": None})
        data = resp.json()
        d = data.get("data", {})
        if not d:
            return None

        price = d.get("f43", None)
        if price is None:
            return None
        price = price / 100 if price else None

        return {
            "price": price,
            "change_pct": _d(d, "f170") / 100 if _d(d, "f170") else None,  # 涨跌幅
            "change_amt": _d(d, "f169") / 100 if _d(d, "f169") else None,  # 涨跌额
            "high": _d(d, "f44") / 100 if _d(d, "f44") else None,
            "low": _d(d, "f45") / 100 if _d(d, "f45") else None,
            "open": _d(d, "f46") / 100 if _d(d, "f46") else None,
            "prev_close": _d(d, "f60") / 100 if _d(d, "f60") else None,
            "volume": int(_d(d, "f47") or 0),
            "turnover": _d(d, "f48"),
            "turnover_rate": _d(d, "f168") / 100 if _d(d, "f168") else None,
            "pe": _d(d, "f162") / 100 if _d(d, "f162") else None,
            "pb": _d(d, "f167") / 100 if _d(d, "f167") else None,
            "total_mv": _d(d, "f116"),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    except Exception:
        return None


def _d(d, key):
    """Safe dict value, returns None if missing or 0"""
    v = d.get(key)
    if v is None or v == "-" or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def get_intraday_chart(code: str, period: str = "1"):
    """Get today's intraday minute candles as list of dicts."""
    try:
        df = ak.stock_zh_a_hist_min_em(symbol=code, period=period, adjust="")
        if df is None or df.empty:
            return []
        today = datetime.now().strftime("%Y-%m-%d")
        df_today = df[df["时间"].astype(str).str[:10] == today]
        if df_today.empty:
            df_today = df.tail(240)
        candles = []
        for _, r in df_today.iterrows():
            candles.append({
                "time": str(r["时间"])[-8:],
                "open": float(r["开盘"]),
                "close": float(r["收盘"]),
                "high": float(r["最高"]),
                "low": float(r["最低"]),
                "volume": int(r["成交量"]),
            })
        return candles
    except Exception:
        return []


