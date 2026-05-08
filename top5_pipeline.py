"""
Top 5 选股流水线
5500 → Q1过滤(~100) → B1评分+五维度 → 综合排序 → Top 5
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

from enhanced_fetcher import load_stock_name_map
from stock_analyzer import SingleStockAnalyzer, fetch_fundamentals_all
from deep_research_v2 import deep_research
from auto_scanner import run_scan


def combined_score(b1_60d: float, dr_overall: str, industry_pct: float,
                   profit_yoy: float, margin: float, revenue_yoy: float,
                   analyst_cv: float | None) -> float:
    """
    综合评分。权重基于 6 年回测数据。

    重点惩罚：利润质量差（利润增速远高于营收 = 可能非经常性损益）
    """
    score = 0.0

    # B1 60d 评分 → 权重 15%（仅在HS300内验证过夏普0.98，中小盘参考价值有限）
    score += b1_60d * 0.15

    # 深度研究 → 权重 40%（五维度基于全市场数据，不依赖HS300范围）
    dr_map = {"green": 85, "yellow": 50, "red": 15}
    score += dr_map.get(dr_overall, 50) * 0.40

    # 行业排位 → 权重 15%
    score += industry_pct * 0.15

    # 毛利率（有护城河） → 权重 10%
    score += min(margin, 80) * 0.10

    # 分析师共识 → 权重 10%
    if analyst_cv is not None and analyst_cv > 0:
        cv_score = max(0, 100 - analyst_cv * 200)
    else:
        cv_score = 50
    score += cv_score * 0.10

    # 利润质量惩罚 → 权重 10%，营收不涨利润暴增 = 虚胖
    quality_score = 50  # 中性
    if revenue_yoy > 0 and profit_yoy > 100:
        ratio = profit_yoy / revenue_yoy
        if ratio > 10:      # 利润增速 > 10× 营收增速 → 几乎肯定靠非经常性损益
            quality_score = -50
        elif ratio > 5:
            quality_score = 0
        elif ratio > 3:
            quality_score = 20
    if revenue_yoy < 0 and profit_yoy > 100:  # 营收降但利润暴增 → 卖资产/砍研发
        quality_score = -50
    if revenue_yoy > 0 and revenue_yoy < 30 and profit_yoy > 300:  # 营收微增利润暴增
        quality_score = min(quality_score, -20)
    score += quality_score * 0.10

    return round(score, 1)


def main():
    print("=" * 64)
    print("  B1 Top 5 选股流水线")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 64)

    # ── Step 1: 全市场扫描 ──
    print("\n[1/4] 全市场 Q1 过滤...")
    scan = run_scan(top_n=0)  # 全部候选，不截断
    if "error" in scan:
        print(f"  失败: {scan['error']}")
        return
    pool = scan["candidates"]
    print(f"  5500 → {len(pool)} 只")

    # ── Step 2: 全量 B1 + 深度研究 ──
    print(f"\n[2/4] 对 {len(pool)} 只候选股运行 B1 + 五维度...")
    analyzer = SingleStockAnalyzer()
    fund_all = fetch_fundamentals_all()
    name_map = load_stock_name_map()

    results = []
    t0 = time.time()
    for i, c in enumerate(pool):
        code = c["代码"]
        try:
            b1 = analyzer.analyze(code)
            if "error" in b1:
                continue
            fund = fund_all.get(code, {})
            dr = deep_research(code, fundamentals=fund, b1_result=b1)

            # 提取行业综合排名
            ind_pct = 50
            for dim in dr["dimensions"]:
                if dim["dimension"] == "行业位置":
                    rankings = dim.get("data", {}).get("rankings", {})
                    if rankings:
                        ind_pct = np.mean(list(rankings.values()))
                    break

            # 提取分析师 CV
            analyst_cv = None
            for dim in dr["dimensions"]:
                if dim["dimension"] == "分析师共识":
                    analyst_cv = dim.get("data", {}).get("profit_cv")
                    break

            score = combined_score(
                b1_60d=b1["scores"]["score_60d"],
                dr_overall=dr["overall_level"],
                industry_pct=ind_pct,
                profit_yoy=_fv(fund.get("profit_growth")),
                margin=_fv(fund.get("gross_margin")),
                revenue_yoy=_fv(fund.get("revenue_growth")),
                analyst_cv=analyst_cv,
            )

            results.append({
                "code": code,
                "name": b1["meta"]["name"],
                "score": score,
                "b1_60d": b1["scores"]["score_60d"],
                "b1_10d": b1["scores"]["score_10d"],
                "stars": b1["star"]["stars"],
                "dr_overall": dr["overall_level"],
                "profit_yoy": _fv(fund.get("profit_growth")),
                "revenue_yoy": _fv(fund.get("revenue_growth")),
                "margin": _fv(fund.get("gross_margin")),
                "roe": _fv(fund.get("roe")),
                "industry_pct": round(ind_pct, 0),
                "analyst_cv": round(analyst_cv, 3) if analyst_cv else None,
                "dr_summary": dr["verdict"],
            })
        except Exception as e:
            continue
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(pool)} ({time.time()-t0:.0f}s)")

    print(f"  完成 {len(results)} 只, 耗时 {time.time()-t0:.0f}s")

    # ── Step 3: 排序 ──
    df = pd.DataFrame(results).sort_values("score", ascending=False)

    # ── Step 4: 输出 Top 5 ──
    print(f"\n[3/4] 综合排序完成")
    print(f"\n{'='*64}")
    print(f"  TOP 5 候选")
    print(f"{'='*64}")

    for i, (_, r) in enumerate(df.head(5).iterrows()):
        print(f"\n  #{i+1}  {r['code']} {r['name']}  — 综合得分 {r['score']:.0f}")
        print(f"  {'─'*50}")
        print(f"  B1: 60d={r['b1_60d']:.0f}, 10d={r['b1_10d']:.0f}, 星级={'★'*r['stars']}")
        print(f"  深度: {r['dr_overall']}")
        print(f"  利润+{r['profit_yoy']:.0f}% | 营收+{r['revenue_yoy']:.0f}% | 毛利率{r['margin']:.1f}% | ROE{r['roe']:.1f}%")
        print(f"  行业排位: top{r['industry_pct']:.0f}% | 分析师CV: {r['analyst_cv'] or 'N/A'}")
        print(f"  {r['dr_summary'][:120]}")

    # ── 完整候选池 ──
    print(f"\n\n[4/4] 完整候选池 Top 20")
    print(f"  {'排名':<4} {'代码':<8} {'名称':<8} {'综合':>5} {'B1':>5} {'深度':>6} {'利润%':>8} {'毛利率':>6} {'行业%':>6} {'CV':>6}")
    print(f"  {'-'*70}")
    for i, (_, r) in enumerate(df.head(20).iterrows()):
        print(f"  {i+1:<4} {r['code']:<8} {r['name']:<8} {r['score']:>5.0f} {r['b1_60d']:>5.0f} {r['dr_overall']:>6} {r['profit_yoy']:>+7.0f}% {r['margin']:>5.1f}% {r['industry_pct']:>5.0f}% {r['analyst_cv'] or '—':>6}")

    # ── 保存 ──
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "top5": df.head(5)[["code", "name", "score", "b1_60d", "dr_overall", "profit_yoy", "margin"]].to_dict("records"),
        "top20": df.head(20)[["code", "name", "score", "b1_60d", "dr_overall"]].to_dict("records"),
    }
    out_path = os.path.join("data_cache", "top5_picks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


def _fv(val):
    try: return float(val)
    except: return 0


if __name__ == "__main__":
    main()
