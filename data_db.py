"""
SQLite 数据层 — 选股系统核心数据汇总
每只股票一行：B1评分 + 深研评级 + 信号状态 + 收益 + 基本面
"""
import sys, os, sqlite3, json, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", "stock_db.sqlite")


def rebuild(limit: int = 0):
    """
    重建数据库：遍历缓存中所有股票，计算当前快照指标。
    limit=0 表示全部，否则只处理前 N 只（用于快速测试）。
    """
    t0 = time.time()
    logger.info("重建数据库...")

    from enhanced_fetcher import EnhancedStockFetcher, load_stock_name_map
    from b1_selector import B1Selector, B1Config
    from stock_analyzer import fetch_fundamentals_all
    from combined_backtest import detect_current_signals

    fetcher = EnhancedStockFetcher()
    name_map = load_stock_name_map()
    selector = B1Selector(B1Config())
    fund_all = fetch_fundamentals_all()

    # 获取缓存中的所有股票代码
    cache_dir = fetcher.cache_dir
    codes = [f.replace(".csv", "") for f in os.listdir(cache_dir) if f.endswith(".csv")]
    if limit > 0:
        codes = codes[:limit]

    logger.info(f"  处理 {len(codes)} 只股票...")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    # 建表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            code TEXT PRIMARY KEY,
            name TEXT,
            last_price REAL,
            last_date TEXT,
            score_short REAL,
            score_long REAL,
            dr_overall TEXT,
            profit_yoy REAL,
            revenue_yoy REAL,
            gross_margin REAL,
            roe REAL,
            ret_20d REAL,
            ret_60d REAL,
            ret_120d REAL,
            pos_60 REAL,
            vol_ratio REAL,
            ma_trend TEXT,
            signal_buy TEXT,
            signal_sell TEXT,
            industry TEXT,
            market_cap REAL,
            updated TEXT
        )
    """)
    conn.execute("DELETE FROM stocks")

    rows = []
    for i, code in enumerate(codes):
        try:
            # 日线数据
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
            df = fetcher._get_cached_data(code, start, end)
            if df is None or len(df) < 60:
                continue

            close = df["close"]
            vol = df["volume"]
            last_price = float(close.iloc[-1])
            last_date = str(df.index[-1])[:10]

            # 收益
            ret_20d = round((close.iloc[-1] / close.iloc[-21] - 1) * 100, 1) if len(close) >= 21 else None
            ret_60d = round((close.iloc[-1] / close.iloc[-61] - 1) * 100, 1) if len(close) >= 61 else None
            ret_120d = round((close.iloc[-1] / close.iloc[-121] - 1) * 100, 1) if len(close) >= 121 else None

            # B1 评分
            try:
                prepared = selector.prepare_data(df)
                cond = selector.check_b1_conditions(prepared, date_idx=-1)
                s_short = round(selector._calculate_score(cond, "short"), 1)
                s_long = round(selector._calculate_score_long(cond), 1)
            except Exception:
                s_short = s_long = None

            # 信号
            sigs = detect_current_signals(code, df)
            buy_sigs = ",".join(s["type"] for s in sigs.get("buy", [])) or None
            sell_sigs = ",".join(s["type"] for s in sigs.get("sell", [])) or None
            pos_60 = sigs.get("pos_60")
            vol_ratio = sigs.get("vol_ratio")

            # 均线趋势
            ma5 = close.iloc[-5:].mean()
            ma20 = close.iloc[-20:].mean()
            ma60 = close.iloc[-60:].mean() if len(close) >= 60 else ma20
            if ma5 > ma20 > ma60:
                ma_trend = "多头"
            elif ma5 < ma20 < ma60:
                ma_trend = "空头"
            else:
                ma_trend = "交叉"

            # 基本面
            fund = fund_all.get(code, {})
            profit_yoy = _fv(fund.get("profit_growth"))
            revenue_yoy = _fv(fund.get("revenue_growth"))
            gross_margin = _fv(fund.get("gross_margin"))
            roe = _fv(fund.get("roe"))

            # 深研评级（market_context 重构中，暂时跳过）
            dr_overall = None

            name = name_map.get(code, code)
            rows.append((
                code, name, last_price, last_date,
                s_short, s_long, dr_overall,
                profit_yoy, revenue_yoy, gross_margin, roe,
                ret_20d, ret_60d, ret_120d,
                pos_60, vol_ratio, ma_trend,
                buy_sigs, sell_sigs,
                str(fund.get("industry", "")) if fund.get("industry") else None,
                round(_fv(fund.get("net_profit")) * 4 * 25 / 1e8, 0) if _fv(fund.get("net_profit")) > 0 else None,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ))
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            logger.info(f"    ... {i+1}/{len(codes)}")

    conn.executemany("""
        INSERT INTO stocks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()

    # 索引
    conn.execute("CREATE INDEX IF NOT EXISTS idx_score_long ON stocks(score_long DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dr ON stocks(dr_overall)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signal ON stocks(signal_buy)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_margin ON stocks(gross_margin DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ret60 ON stocks(ret_60d DESC)")

    n = len(rows)
    conn.close()
    logger.info(f"  完成: {n} 只, 耗时 {time.time()-t0:.0f}s")
    return n


def query(sql: str, params=None) -> list:
    """执行查询并返回 dict 列表"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, params or [])
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── 常用查询 ──

def top_by_score(n=10, min_margin=0):
    """B1 长线评分最高的 N 只"""
    return query(
        "SELECT code, name, score_long, score_short, dr_overall, ret_60d, gross_margin "
        "FROM stocks WHERE score_long IS NOT NULL AND gross_margin >= ? "
        "ORDER BY score_long DESC LIMIT ?", (min_margin, n))

def top_by_return(n=10, days=60):
    """近期收益最高的 N 只"""
    col = f"ret_{days}d"
    return query(f"SELECT code, name, {col}, score_long, dr_overall "
                 f"FROM stocks WHERE {col} IS NOT NULL ORDER BY {col} DESC LIMIT ?", (n,))

def with_buy_signal():
    """当前有买入信号的股票"""
    return query("SELECT code, name, signal_buy, score_long, ret_60d, pos_60, vol_ratio "
                 "FROM stocks WHERE signal_buy IS NOT NULL ORDER BY score_long DESC")

def green_high_margin(min_margin=40):
    """深研 green + 高毛利"""
    return query("SELECT code, name, gross_margin, score_long, ret_60d "
                 "FROM stocks WHERE dr_overall='green' AND gross_margin>=? "
                 "ORDER BY score_long DESC", (min_margin,))

def by_industry(industry_keyword: str, n=20):
    """按行业关键词筛选"""
    return query("SELECT code, name, industry, score_long, profit_yoy, gross_margin "
                 "FROM stocks WHERE industry LIKE ? ORDER BY score_long DESC LIMIT ?",
                 (f"%{industry_keyword}%", n))


def _fv(val):
    try: return float(val)
    except: return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    n = rebuild(limit=100)
    print(f"\n数据库: {n} 只")
    if n > 0:
        print("\nTop 10 B1 长线:")
        for r in top_by_score(10):
            print(f"  {r['code']} {r['name']}: B1={r['score_long']} {r['dr_overall'] or '?'}")
        print("\n有买入信号:")
        for r in with_buy_signal():
            print(f"  {r['code']} {r['name']}: {r['signal_buy']}")
