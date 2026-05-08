"""
全市场自动扫描器 v1
Step 1: 财务过滤 (akshare Q1数据)
Step 2: 研报聚合 (盈利预测 + PE共识)
Step 3: 财务异常检测 (拐点/改善/风险)
输出候选池 → 待 Claude 深度验证
"""
import sys, os, json, logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = "data_cache"
OUTPUT_PATH = os.path.join(CACHE_DIR, "scanner_candidates.json")


def fetch_q1_data():
    """拉取全市场2026Q1业绩快报"""
    logger.info("拉取全市场Q1数据...")
    try:
        import akshare as ak
        df = ak.stock_yjbb_em(date="20260331")
        if df is None or df.empty:
            logger.error("未获取到Q1数据")
            return None

        # 重命名关键列
        col_map = {
            "股票代码": "code", "股票简称": "name",
            "营业总收入-营业总收入": "revenue",
            "营业总收入-同比增长": "revenue_yoy",
            "净利润-净利润": "net_profit",
            "净利润-同比增长": "profit_yoy",
            "销售毛利率": "gross_margin",
            "净资产收益率": "roe",
            "所处行业": "industry",
            "每股收益": "eps",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # 补全代码
        df["code"] = df["code"].astype(str).str.zfill(6)

        # 转换成数值
        for col in ["revenue", "net_profit", "revenue_yoy", "profit_yoy",
                     "gross_margin", "roe", "eps"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"获取到 {len(df)} 只股票的Q1数据")
        return df
    except Exception as e:
        logger.error(f"获取Q1数据失败: {e}")
        return None


def filter_financial(df):
    """Step 1: 财务过滤"""
    if df is None or df.empty:
        return pd.DataFrame()

    candidates = df.copy()

    # 基础条件
    candidates = candidates[candidates["net_profit"].notna() & (candidates["net_profit"] > 1e8)]
    candidates = candidates[candidates["profit_yoy"].notna() & (candidates["profit_yoy"] > 200)]
    candidates = candidates[candidates["gross_margin"].notna() & (candidates["gross_margin"] > 15)]
    candidates = candidates[candidates["revenue_yoy"].notna() & (candidates["revenue_yoy"] > 10)]

    # 排除金融/地产（适用不同估值逻辑）
    if "industry" in candidates.columns:
        exclude_keywords = ["银行", "保险", "券商", "房地产", "多元金融"]
        for kw in exclude_keywords:
            candidates = candidates[~candidates["industry"].str.contains(kw, na=False)]

    # 计算复合得分用于排序
    candidates["_score"] = (
        candidates["profit_yoy"].fillna(0).clip(upper=2000) * 0.5 +
        candidates["gross_margin"].fillna(0) * 0.3 +
        candidates["revenue_yoy"].fillna(0).clip(upper=500) * 0.2
    )
    candidates = candidates.sort_values("_score", ascending=False)

    logger.info(f"财务过滤后剩余 {len(candidates)} 只")
    return candidates


def add_analyst_consensus(df):
    """Step 2: 研报聚合（前30只，控制请求量）"""
    logger.info("拉取研报数据（前30只）...")
    top_codes = df.head(30)["code"].tolist()
    consensus = {}

    try:
        import akshare as ak
    except Exception:
        return df, {}

    for i, code in enumerate(top_codes):
        try:
            reports_df = ak.stock_research_report_em(symbol=code)
            if reports_df is None or reports_df.empty:
                continue

            cols = list(reports_df.columns)
            profits_2026, pes_2026, profits_2027 = [], [], []
            ratings_list = []

            for _, row in reports_df.head(20).iterrows():
                if len(cols) > 7:
                    p26 = _safe_float(row.iloc[7])
                    if p26 and p26 > 0:
                        profits_2026.append(p26)
                if len(cols) > 8:
                    pe26 = _safe_float(row.iloc[8])
                    if pe26 and 0 < pe26 < 500:
                        pes_2026.append(pe26)
                if len(cols) > 9:
                    p27 = _safe_float(row.iloc[9])
                    if p27 and p27 > 0:
                        profits_2027.append(p27)
                if len(cols) > 4:
                    ratings_list.append(str(row.iloc[4]))

            if profits_2026:
                consensus[code] = {
                    "profit_2026_mean": round(np.mean(profits_2026), 2),
                    "profit_2026_range": f"{min(profits_2026):.1f}-{max(profits_2026):.1f}",
                    "pe_2026_mean": round(np.mean(pes_2026), 1) if pes_2026 else None,
                    "n_reports": len(profits_2026),
                    "top_rating": max(set(ratings_list), key=ratings_list.count) if ratings_list else "?",
                }
        except Exception:
            continue

    logger.info(f"获取到 {len(consensus)} 只股票的研报共识")
    return df, consensus


def detect_anomalies(df):
    """Step 3: 财务异常检测"""

    # 经营杠杆释放: 利润增速 > 营收增速 × 2
    df["_leverage"] = (df["profit_yoy"].fillna(0) > df["revenue_yoy"].fillna(1) * 2).astype(int)

    # 利润质量: 高利润增速 + 合理ROE (>10%)
    df["_quality"] = (df["roe"].fillna(0) > 10).astype(int)

    # 市值估计: net_profit * PE(保守25x)
    df["_est_mcap"] = df["net_profit"].fillna(0) * 4 * 25 / 1e8

    # 综合信号分
    df["_signal"] = df["_leverage"] + df["_quality"]
    return df


def build_candidate_pool(df, consensus, top_n: int = 30):
    """构建候选池输出。top_n=0 返回全部。"""
    candidates = df if top_n == 0 else df.head(top_n).copy()

    rows = []
    for _, row in candidates.iterrows():
        code = row["code"]
        cons = consensus.get(code, {})

        rows.append({
            "代码": code,
            "名称": row.get("name", "?"),
            "行业": row.get("industry", "??"),
            "Q1营收(亿)": round(row.get("revenue", 0) / 1e8, 1) if row.get("revenue") else None,
            "营收增速%": round(row.get("revenue_yoy", 0), 1) if row.get("revenue_yoy") else None,
            "Q1净利(亿)": round(row.get("net_profit", 0) / 1e8, 1) if row.get("net_profit") else None,
            "利润增速%": round(row.get("profit_yoy", 0), 1) if row.get("profit_yoy") else None,
            "毛利率%": round(row.get("gross_margin", 0), 1) if row.get("gross_margin") else None,
            "ROE%": round(row.get("roe", 0), 1) if row.get("roe") else None,
            "估市值(亿)": round(row.get("_est_mcap", 0), 0) if row.get("_est_mcap") else None,
            "研报共识PE": cons.get("pe_2026_mean"),
            "研报数": cons.get("n_reports", 0),
            "评级": cons.get("top_rating", "?"),
            "经营杠杆": "是" if row["_leverage"] else "否",
            "信号分": int(row["_signal"]),
        })

    return rows


def _safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def main():
    print("=" * 60)
    print(f"  全市场自动扫描 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Step 1: 财务过滤
    df = fetch_q1_data()
    if df is None:
        return
    candidates = filter_financial(df)
    if candidates.empty:
        print("无符合条件的标的")
        return

    # Step 2: 研报聚合
    candidates, consensus = add_analyst_consensus(candidates)

    # Step 3: 异常检测
    candidates = detect_anomalies(candidates)

    # 构建输出
    pool = build_candidate_pool(candidates, consensus)

    # 保存
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_filtered": len(candidates),
        "candidates": pool,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 打印
    print(f"\n过滤出 {len(candidates)} 只 → 候选池 Top 20:")
    print(f"{'':>3} {'代码':<8} {'名称':<8} {'利润增速':>8} {'毛利率':>6} {'信号':>4} {'研报PE':>8}")
    print("-" * 55)
    for i, r in enumerate(pool[:20]):
        print(f"{i+1:>3} {r['代码']:<8} {r['名称']:<8} {r['利润增速%']:>+8.0f}% {r['毛利率%']:>6.1f}% {r['信号分']:>4} {r['研报共识PE'] or '—':>8}")

    print(f"\n候选池已保存: {OUTPUT_PATH}")
    print(f"下一步: Claude 对候选池新增标的做全网深度搜索")


def run_scan(top_n: int = 0) -> dict:
    """执行扫描并返回结果 dict。top_n=0 返回全部候选。"""
    df = fetch_q1_data()
    if df is None:
        return {"error": "Q1数据获取失败", "candidates": [], "total_filtered": 0}
    candidates = filter_financial(df)
    if candidates.empty:
        return {"error": "无符合条件的标的", "candidates": [], "total_filtered": 0}
    candidates, consensus = add_analyst_consensus(candidates)
    candidates = detect_anomalies(candidates)
    pool = build_candidate_pool(candidates, consensus, top_n=top_n)
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_filtered": len(candidates),
        "candidates": pool,
    }


if __name__ == "__main__":
    main()
