"""
完整扫瞄管道: scanner → refine → deep_research_v2 → 输出精筛候选
运行: python run_pipeline.py
"""
import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 跨目录引用 stock_selection 的模块
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from deep_research_v2 import deep_research
from stock_analyzer import fetch_fundamentals_all


def load_refined():
    """读取 refine_filter 产出的精筛候选"""
    path = os.path.join(HERE, "data_cache", "refined_candidates.json")
    if not os.path.exists(path):
        print("精筛候选文件不存在，先运行 refine_filter.py")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)["candidates"]


def load_b1_cache():
    """加载 stock_selection 的 OHLCV 缓存用于 B1 评分"""
    from enhanced_fetcher import EnhancedStockFetcher
    from b1_selector import B1Selector, B1Config

    fetcher = EnhancedStockFetcher()
    hs300 = set(str(s).zfill(6) for s in fetcher.get_hs300_stocks()["code"].tolist())

    # 加载全量数据
    cache_dir = os.path.join(HERE, "data_cache")
    config = B1Config(j_threshold=10.0, j_q_threshold=0.30, kdj_n=9,
        zx_m1=5, zx_m2=20, zx_m3=40, zx_m4=60, zxdq_span=10,
        wma_short=5, wma_mid=10, wma_long=15, max_vol_lookback=20)
    selector = B1Selector(config)

    return selector, cache_dir


def _sf(val):
    try: return float(val)
    except: return None


def get_b1_score(code, selector, cache_dir):
    """获取单只股票的B1评分"""
    for fname in [f"{code}.csv", f"{code.lstrip('0')}.csv", f"{code.zfill(6)}.csv"]:
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path):
            try:
                tdf = pd.read_csv(path, index_col=0, parse_dates=True)
                if len(tdf) >= 60:
                    prepared = selector.prepare_data(tdf)
                    cond = selector.check_b1_conditions(prepared, date_idx=-1)
                    s60 = selector._calculate_score_long(cond)
                    s10 = selector._calculate_score(cond, "short")
                    close = float(prepared["close"].iloc[-1])
                    rsi = cond.get("rsi", 50)
                    ret_60d = (close - prepared["close"].iloc[-61]) / prepared["close"].iloc[-61] * 100 if len(prepared) > 60 else 0
                    return {
                        "scores": {"score_long": round(s60, 1), "score_short": round(s10, 1)},
                        "returns": {"60d": round(ret_60d, 1)},
                        "price": {"close": round(close, 2), "rsi": round(rsi, 1)},
                    }
            except Exception:
                pass
    return {"scores": {}, "returns": {}, "price": {}}


def main():
    print("=" * 60)
    print(f"  扫瞄管道 v1 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 加载
    candidates = load_refined()
    if not candidates:
        return
    print(f"精筛候选: {len(candidates)} 只")

    all_fund = fetch_fundamentals_all()
    print(f"基本面数据: {len(all_fund)} 只")

    selector, cache_dir = load_b1_cache()
    print(f"B1引擎就绪")

    # 逐只分析
    results = []
    for i, r in enumerate(candidates):
        code = r["代码"]
        name = r.get("名称", "?")
        fund = all_fund.get(code, {})
        if not fund:
            continue

        # B1评分
        b1 = get_b1_score(code, selector, cache_dir)

        # 五维分析
        research = deep_research(code, fundamentals=fund, b1_result=b1)

        results.append({
            "code": code,
            "name": name,
            "profit_yoy": r.get("利润增速%"),
            "margin": r.get("毛利率%"),
            "pe_consensus": r.get("研报共识PE"),
            "score_long": b1.get("scores", {}).get("score_long", 0),
            "overall_level": research["overall_level"],
            "verdict": research["verdict"],
            "insights": research["insights"][:5],
            "warnings": research["warnings"][:3],
        })

        level_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(research["overall_level"], "?")
        print(f"  {i+1:>2}. {code} {name:<8} B1={b1.get('scores',{}).get('score_60d',0):.0f}  {level_icon} {research['verdict'][:60]}")

    # 按B1评分排序，输出Top
    results.sort(key=lambda x: x["score_long"], reverse=True)

    print(f"\n{'='*60}")
    print(f"  精筛结果 (按B1评分排序)")
    print(f"  {'代码':<8} {'名称':<8} {'利润增速':>8} {'B1':>6} {'PE':>6} {'评级'}")
    print(f"  {'─'*50}")
    for r in results:
        pe = r.get("pe_consensus") or "—"
        print(f"  {r['code']:<8} {r['name']:<8} {r['profit_yoy']:>+8.0f}% {r['score_60d']:>6.0f} {str(pe):>6} {r['overall_level']}")

    # 保存
    out_path = os.path.join(HERE, "data_cache", "pipeline_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d"), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
