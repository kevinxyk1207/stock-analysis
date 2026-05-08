"""
全自动深度挖掘引擎
对任意股票，利用 akshare 研报数据 + 主营业务 + 财务数据 + 盈利预测
自动合成研报级深度洞察，无需手动搜索。
"""
import sys
import os
import json
import re
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_LOCAL_DIR, "enhanced_fetcher.py")):
    sys.path.insert(0, _LOCAL_DIR)


def get_research_reports(code: str, limit: int = 10) -> list:
    """获取个股最新研报"""
    try:
        import akshare as ak
        df = ak.stock_research_report_em(symbol=code)
        if df is None or df.empty:
            return []
        # 列顺序固定: 序号 代码 名称 报告名称 评级 券商 上一月报告数
        #   2026利润 2026PE 2027利润 2027PE 2028利润 2028PE 行业 日期 PDF
        cols = list(df.columns)
        reports = []
        for _, row in df.head(limit).iterrows():
            reports.append({
                "title": str(row.iloc[3]) if len(cols) > 3 else "",
                "rating": str(row.iloc[4]) if len(cols) > 4 else "",
                "broker": str(row.iloc[5]) if len(cols) > 5 else "",
                "profit_2026": _safe_float(row.iloc[7]) if len(cols) > 7 else None,
                "pe_2026": _safe_float(row.iloc[8]) if len(cols) > 8 else None,
                "profit_2027": _safe_float(row.iloc[9]) if len(cols) > 9 else None,
                "pe_2027": _safe_float(row.iloc[10]) if len(cols) > 10 else None,
                "profit_2028": _safe_float(row.iloc[11]) if len(cols) > 11 else None,
                "pe_2028": _safe_float(row.iloc[12]) if len(cols) > 12 else None,
                "industry": str(row.iloc[13]) if len(cols) > 13 else "",
                "date": str(row.iloc[14]) if len(cols) > 14 else "",
                "pdf_url": str(row.iloc[15]) if len(cols) > 15 else "",
            })
        return reports
    except Exception:
        return []


def get_business_description(code: str) -> dict:
    """获取主营业务和产品描述"""
    try:
        import akshare as ak
        df = ak.stock_zyjs_ths(symbol=code)
        if df is None or df.empty:
            return {}
        row = df.iloc[0]
        return {
            "main_business": str(row.get("主营业务", "")),
            "products": str(row.get("产品名称", "")),
            "scope": str(row.get("经营范围", ""))[:500],
        }
    except Exception:
        return {}


def get_industry_peers(code: str) -> list:
    """获取同行业可比公司（通过个股研报的行业标签，不拉全市场数据）"""
    try:
        import akshare as ak
        df = ak.stock_research_report_em(symbol=code)
        if df is None or df.empty:
            return []
        # 直接从个股研报中提取行业，然后找同行业股票
        cols = list(df.columns)
        industry = str(df.iloc[0, 13]) if len(cols) > 13 else ""
        if not industry:
            return []

        # 仅用最近一批研报中的同行业股票（不去拉全市场数据）
        peers = set()
        for _, row in df.head(20).iterrows():
            if len(cols) > 13 and str(row.iloc[13]) == industry:
                code_from_report = str(row.iloc[1]).zfill(6)
                if code_from_report != code:
                    peers.add(code_from_report)
        return list(peers)[:5]
    except Exception:
        return []


def synthesize_insights(code: str, reports: list, business: dict,
                        fundamentals: dict, deep_data: dict) -> dict:
    """
    从研报数据 + 主营业务 + 财务数据 合成深度洞察

    不依赖 LLM，用规则 + 模板从结构化数据中提取
    """
    highlights = []
    risks = []

    # ── 从研报中提炼 ──
    ratings = [r["rating"] for r in reports if r["rating"]]
    brokers = [r["broker"] for r in reports if r["broker"]]
    rating_counter = Counter(ratings)
    top_rating = rating_counter.most_common(1)[0] if rating_counter else ("无评级", 0)

    # 盈利预测共识
    profits_2026 = [r["profit_2026"] for r in reports if r["profit_2026"] and r["profit_2026"] > 0]
    pes_2026 = [r["pe_2026"] for r in reports if r["pe_2026"] and r["pe_2026"] > 0]
    profits_2027 = [r["profit_2027"] for r in reports if r["profit_2027"] and r["profit_2027"] > 0]

    if profits_2026:
        avg_profit = np.mean(profits_2026)
        profit_growth_26 = (avg_profit / (profits_2026[0] if len(profits_2026) == 1
                                          else np.median(profits_2026))) * 100 - 100
        highlights.append(
            f"机构一致预期 2026 年净利润 {avg_profit:.1f} 亿元"
            + (f"（{len(profits_2026)} 家券商）" if len(profits_2026) > 1 else "")
        )

    if pes_2026:
        avg_pe = np.mean(pes_2026)
        highlights.append(f"2026 年一致预期 PE {avg_pe:.0f}x（{len(pes_2026)} 家预测）")

    if profits_2026 and profits_2027:
        avg_26 = np.mean(profits_2026)
        avg_27 = np.mean(profits_2027)
        growth = (avg_27 / avg_26 - 1) * 100 if avg_26 > 0 else 0
        if growth > 10:
            highlights.append(f"2027 年预期利润增速 {growth:.0f}%，增长动能持续")

    # 评级分布
    buy_ratings = sum(1 for r in ratings if "买入" in r or "推荐" in r or "强推" in r or "增持" in r)
    if len(ratings) >= 3 and buy_ratings / len(ratings) >= 0.7:
        highlights.append(f"近 {len(ratings)} 份研报中 {buy_ratings} 份为买入/增持，券商共识偏多")

    # 从研报标题中提取关键词
    title_words = " ".join(r["title"] for r in reports[:5])
    hot_keywords = []
    for kw, label in [
        ("AI", "AI/算力"), ("数据中心", "数据中心"), ("AIDC", "AIDC电源"),
        ("新能源", "新能源"), ("出口", "出海/出口"), ("增长", "高增长"),
        ("反转", "困境反转"), ("拐点", "业绩拐点"), ("估值", "估值修复"),
        ("放量", "产品放量"), ("产能", "产能扩张"), ("认证", "技术认证"),
        ("龙头", "行业龙头"), ("突破", "技术突破"),
    ]:
        if kw in title_words:
            hot_keywords.append(label)
    if hot_keywords:
        highlights.append(f"研报核心主题: {', '.join(hot_keywords[:5])}")

    # 研报标题中找风险词
    risk_keywords_in_titles = []
    for kw in ["承压", "不及预期", "下滑", "风险", "挑战", "放缓"]:
        if kw in title_words:
            risk_keywords_in_titles.append(kw)
    if risk_keywords_in_titles:
        risks.append(f"近期研报提及: {'、'.join(risk_keywords_in_titles)}")

    # ── 从主营业务提炼 ──
    if business.get("main_business"):
        bus = business["main_business"]
        if len(bus) > 200:
            bus = bus[:200] + "..."
        highlights.append(f"主营业务: {bus}")

    # ── 从财务数据提炼风险 ──
    rev_g = fundamentals.get("revenue_growth", 0) or 0
    prf_g = fundamentals.get("profit_growth", 0) or 0
    margin = fundamentals.get("gross_margin", 0) or 0
    roe = fundamentals.get("roe", 0) or 0

    try: rev_g = float(rev_g)
    except: rev_g = 0
    try: prf_g = float(prf_g)
    except: prf_g = 0
    try: margin = float(margin)
    except: margin = 0
    try: roe = float(roe)
    except: roe = 0

    if rev_g < -10:
        risks.append(f"营收同比下滑 {rev_g:.1f}%")
    if prf_g < -20:
        risks.append(f"利润同比大幅下滑 {prf_g:.1f}%")
    if margin > 0 and margin < 15:
        risks.append(f"毛利率偏低 ({margin:.1f}%)，产品或服务附加值低或竞争激烈")
    if rev_g > 0 and prf_g > rev_g * 5:
        risks.append(f"利润增速({prf_g:.0f}%)远超营收({rev_g:.0f}%)，增长质量需关注")

    # ── 综合判断 ──
    verdict_parts = []
    if business.get("main_business"):
        verdict_parts.append(business["main_business"][:80])
    if hot_keywords:
        verdict_parts.append(f"当前核心关注: {'、'.join(hot_keywords[:3])}")
    if top_rating[0] != "无评级":
        verdict_parts.append(f"机构评级: {top_rating[0]} ({top_rating[1]}家)")
    if profits_2026:
        verdict_parts.append(f"2026E 净利 {np.mean(profits_2026):.0f}亿")
    verdict = "。".join(verdict_parts) + "。"

    # ── 同行公司 ──
    peers = get_industry_peers(code)
    if peers:
        highlights.append(f"同行业可比公司: {', '.join(peers[:5])}")

    return {
        "verdict_override": verdict,
        "deep_highlights": highlights[:8],
        "deep_risks": risks[:6],
        "reports_count": len(reports),
        "brokers": list(set(brokers))[:10],
        "consensus_profit_2026": round(float(np.mean(profits_2026)), 2) if profits_2026 else None,
        "consensus_pe_2026": round(float(np.mean(pes_2026)), 1) if pes_2026 else None,
        "peers": peers,
    }


def deep_research_stock(code: str, fundamentals: dict = None) -> dict:
    """
    单只股票全自动深度挖掘

    Returns: deep_insights dict (与 fundamental_insights.json 格式兼容)
    """
    if fundamentals is None:
        from stock_analyzer import fetch_fundamentals_all
        all_fund = fetch_fundamentals_all()
        fundamentals = all_fund.get(code, {})

    reports = get_research_reports(code, limit=8)
    business = get_business_description(code)
    deep_data = {}  # placeholder for additional data

    insights = synthesize_insights(code, reports, business, fundamentals, deep_data)

    return insights


def _safe_float(val) -> Optional[float]:
    """安全转为 float"""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── 缓存管理层 ──

def load_all_insights() -> dict:
    """加载已有全部洞察"""
    _here = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(_here, "fundamental_insights.json"),
        os.path.join(_here, "data_cache", "fundamental_insights.json"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("stocks", {})
            except Exception:
                pass
    return {}


def save_all_insights(insights: dict):
    """保存全部洞察（容错：云端只读文件系统时静默跳过）"""
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(_here, "fundamental_insights.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"updated": datetime.now().strftime("%Y-%m-%d"), "stocks": insights},
                      f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 云端文件系统只读时跳过


def get_or_research(code: str) -> dict:
    """
    获取深度洞察：已有则返回，没有则自动挖掘并缓存
    所有错误都会静默处理，确保不影响主分析流程
    """
    all_insights = load_all_insights()

    # 已有高质量洞察（人工撰写或自动挖掘过的）直接返回
    if code in all_insights:
        entry = all_insights[code]
        if isinstance(entry, dict) and entry.get("deep_highlights"):
            return entry

    # 自动挖掘（每个 API 调用独立容错）
    try:
        new_insight = deep_research_stock(code)
        if new_insight and new_insight.get("deep_highlights"):
            all_insights[code] = new_insight
            save_all_insights(all_insights)
            return new_insight
    except Exception:
        pass

    # 回退：已有数据或空模板
    return all_insights.get(code, {
        "verdict_override": "",
        "deep_highlights": [],
        "deep_risks": [],
    })
