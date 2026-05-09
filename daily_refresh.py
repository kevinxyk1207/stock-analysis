"""每日自动刷新：扫描器 + Top5 流水线 → 缓存文件（供 Web 读取）"""
import sys, os, json, logging, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")


def main():
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("每日自动刷新开始")
    logger.info("=" * 50)

    # 1. 全市场扫描
    logger.info("[1/3] 全市场扫描...")
    from auto_scanner import run_scan
    scan = run_scan(top_n=0)
    if "error" not in scan:
        scan_path = os.path.join(CACHE_DIR, "scanner_candidates.json")
        with open(scan_path, "w", encoding="utf-8") as f:
            json.dump(scan, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"  扫描完成: {scan['total_filtered']} 只候选 → {scan_path}")
    else:
        logger.error(f"  扫描失败: {scan['error']}")
        return

    # 2. Top5 综合评分
    logger.info("[2/3] Top5 综合评分...")
    from top5_pipeline import combined_score
    from stock_analyzer import SingleStockAnalyzer, fetch_fundamentals_all
    from deep_research_v2 import deep_research
    import pandas as pd
    import numpy as np

    analyzer = SingleStockAnalyzer()
    fund_all = fetch_fundamentals_all()

    pool = scan["candidates"]
    results = []
    for i, c in enumerate(pool):
        code = c["代码"]
        try:
            b1 = analyzer.analyze(code)
            if "error" in b1:
                continue
            fund = fund_all.get(code, {})
            dr = deep_research(code, fundamentals=fund, b1_result=b1)

            ind_pct = 50
            for dim in dr["dimensions"]:
                if dim["dimension"] == "行业位置":
                    rankings = dim.get("data", {}).get("rankings", {})
                    if rankings:
                        ind_pct = np.mean(list(rankings.values()))
                    break
            cv = None
            for dim in dr["dimensions"]:
                if dim["dimension"] == "分析师共识":
                    cv = dim.get("data", {}).get("profit_cv")
                    break

            def _fv(v):
                try: return float(v)
                except: return 0

            score = combined_score(
                b1_60d=b1["scores"]["score_60d"],
                dr_overall=dr["overall_level"],
                industry_pct=ind_pct,
                profit_yoy=_fv(fund.get("profit_growth")),
                margin=_fv(fund.get("gross_margin")),
                revenue_yoy=_fv(fund.get("revenue_growth")),
                analyst_cv=cv,
            )
            results.append({
                "code": code, "name": b1["meta"]["name"],
                "score": score, "b1_60d": b1["scores"]["score_60d"],
                "dr_overall": dr["overall_level"],
            })
        except Exception:
            continue
        if (i + 1) % 30 == 0:
            logger.info(f"  ... {i+1}/{len(pool)}")

    df = pd.DataFrame(results).sort_values("score", ascending=False)
    top5_path = os.path.join(CACHE_DIR, "top5_picks.json")
    top5 = df.head(5)[["code", "name", "score", "b1_60d", "dr_overall"]].to_dict("records")
    with open(top5_path, "w", encoding="utf-8") as f:
        json.dump({"date": scan["date"], "top5": top5, "total": len(results)}, f, ensure_ascii=False, indent=2)
    logger.info(f"  Top5: {[(t['code'], t['name']) for t in top5]}")

    # 3. 提交到 Git（可选）
    logger.info("[3/3] 刷新完成, 耗时 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
