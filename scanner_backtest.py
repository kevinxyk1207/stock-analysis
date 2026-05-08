"""
扫描器回测：验证双轨过滤的历史有效性
回测2023Q1-2025Q4每个季报日，选股后跟踪60个交易日表现
"""
import sys, os, json, logging, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

QUARTERS = ["20200331","20200630","20200930","20201231",
            "20210331","20210630","20210930","20211231",
            "20220331","20220630","20220930","20221231",
            "20230331","20230630","20230930","20231231",
            "20240331","20240630","20240930","20241231",
            "20250331"]

TRACK_LABELS = {"杠杆释放": "TrackA", "暴增": "TrackB"}


def load_price_cache():
    """加载本地价格缓存"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
    prices = {}
    for f in os.listdir(cache_dir):
        if not f.endswith(".csv"):
            continue
        sym = f[:-4]
        try:
            df = pd.read_csv(os.path.join(cache_dir, f), index_col=0, parse_dates=True)
            if len(df) >= 60:
                prices[sym] = df["close"]
        except Exception:
            continue
    logger.info(f"价格缓存: {len(prices)} 只")
    return prices


def get_q1_data(date_str):
    """拉取指定季度数据"""
    try:
        import akshare as ak
        df = ak.stock_yjbb_em(date=date_str)
        if df is None or df.empty:
            return None

        col_map = {
            "股票代码": "code", "股票简称": "name",
            "营业总收入-营业总收入": "revenue",
            "营业总收入-同比增长": "revenue_yoy",
            "净利润-净利润": "net_profit",
            "净利润-同比增长": "profit_yoy",
            "销售毛利率": "gross_margin",
            "净资产收益率": "roe",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["code"] = df["code"].astype(str).str.zfill(6)
        for col in ["revenue", "net_profit", "revenue_yoy", "profit_yoy", "gross_margin", "roe"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
    except Exception as e:
        logger.error(f"获取{date_str}数据失败: {e}")
        return None


def filter_tracks(df):
    """双轨过滤"""
    c = df.copy()
    c = c[c["net_profit"].notna() & (c["net_profit"] > 0)]
    c = c[c["revenue_yoy"].notna() & (c["revenue_yoy"] > 0)]
    c = c[c["gross_margin"].notna() & (c["gross_margin"] > 10)]

    # Track A: 杠杆释放
    lever = c[(c["profit_yoy"] > c["revenue_yoy"] * 1.5) &
              (c["revenue_yoy"] > 15) &
              (c["net_profit"] > 0.5e8)].copy()
    lever["_track"] = "杠杆释放"

    # Track B: 暴增
    surge = c[(c["net_profit"] > 0.5e8) & (c["profit_yoy"] > 200)].copy()
    surge["_track"] = "暴增"

    result = pd.concat([lever, surge]).drop_duplicates(subset=["code"])
    return result


def calc_forward_return(code, eval_date, prices, days=60):
    """计算未来60个交易日收益"""
    sym_raw = code.lstrip("0") or "0"
    for s in [code, sym_raw]:
        if s in prices:
            px = prices[s]
            try:
                dt = pd.to_datetime(eval_date)
                # 找到评估日或之后最近的交易日
                future = px[px.index >= dt]
                if len(future) >= days + 1:
                    entry = future.iloc[0]
                    exit_px = future.iloc[days]
                    return (exit_px - entry) / entry
            except Exception:
                pass
    return None


def main():
    print("=" * 60)
    print(f"  扫描器回测 — {len(QUARTERS)} 个季度")
    print("=" * 60)

    prices = load_price_cache()
    if not prices:
        print("无价格数据")
        return

    all_results = {}

    for date_str in QUARTERS:
        print(f"\n{date_str}...", end=" ", flush=True)

        df = get_q1_data(date_str)
        if df is None:
            print("数据缺失")
            continue

        candidates = filter_tracks(df)

        # 只保留有价格数据的标的
        track_returns = {"杠杆释放": [], "暴增": []}
        for _, row in candidates.iterrows():
            code = row["code"]
            track = row["_track"]
            ret = calc_forward_return(code, date_str, prices, 60)
            if ret is not None:
                track_returns[track].append(ret)

        summary = {}
        for track, rets in track_returns.items():
            if rets:
                avg = np.mean(rets) * 100
                win = np.mean(np.array(rets) > 0) * 100
                n = len(rets)
                summary[track] = {"avg_ret": round(avg, 2), "win_rate": round(win, 1), "n": n}

        all_results[date_str] = {
            "total_filtered": len(candidates),
            "tracks": summary,
        }

        lr = summary.get('杠杆释放', {})
        sr = summary.get('暴增', {})
        print(f"杠杆{lr.get('n',0)}只/均+{lr.get('avg_ret',0):.1f}%/胜{lr.get('win_rate',0):.0f}%  "
              f"暴增{sr.get('n',0)}只/均+{sr.get('avg_ret',0):.1f}%/胜{sr.get('win_rate',0):.0f}%")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'季度':>10} {'杠杆均收益':>10} {'杠杆胜率':>8} {'暴增均收益':>10} {'暴增胜率':>8}")

    all_lever = []; all_surge = []
    for date_str in QUARTERS:
        r = all_results.get(date_str, {}).get("tracks", {})
        la = r.get("杠杆释放", {}); sb = r.get("暴增", {})
        print(f"{date_str:>10} {la.get('avg_ret',0):>+10.2f}% {la.get('win_rate',0):>7.1f}% {sb.get('avg_ret',0):>+10.2f}% {sb.get('win_rate',0):>7.1f}%")

    # 保存
    os.makedirs("reports", exist_ok=True)
    path = f"reports/scanner_backtest_{datetime.now().strftime('%Y%m%d')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {path}")


if __name__ == "__main__":
    main()
