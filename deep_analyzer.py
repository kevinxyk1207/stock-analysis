"""
深度分析引擎 — 对任意股票自动生成：
1. 公司画像（行业、市值、上市时间）
2. 财务健康扫描（8项风险检查）
3. 增长质量评估
4. 发展阶段判断
纯 Python 逻辑，不依赖 Streamlit
"""
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))


def get_company_profile(code: str) -> dict:
    """
    获取公司基本信息：行业、市值、股本、上市时间
    """
    try:
        import akshare as ak
        df = ak.stock_individual_info_em(symbol=code)
        info = {}
        for _, row in df.iterrows():
            key = str(row["item"])
            val = row["value"]
            if "行业" in key:
                info["industry"] = str(val)
            elif "总市值" in key:
                try:
                    info["market_cap"] = float(val)
                except (ValueError, TypeError):
                    info["market_cap"] = None
            elif "流通市值" in key:
                try:
                    info["circulating_cap"] = float(val)
                except (ValueError, TypeError):
                    info["circulating_cap"] = None
            elif "总股本" in key:
                try:
                    info["total_shares"] = float(val)
                except (ValueError, TypeError):
                    info["total_shares"] = None
            elif "上市时间" in key:
                info["listed_date"] = str(val)
            elif "股票简称" in key:
                info["short_name"] = str(val)
        return info
    except Exception:
        return {}


def analyze_financial_health(fundamentals: dict, market_cap: float = None) -> dict:
    """
    基于财务数据 + 风险框架 进行健康扫描

    8 项风险检查（来自 stock-analysis-framework）：
    1. 净利润率 > 5%
    2. 利润增速 / 营收增速 一致性（利润质量）
    3. 毛利率水平
    4. ROE 水平
    5. 营收是否正增长
    6. 利润是否正增长
    7. 市值规模合理性
    8. EPS 是否为正
    """
    checks = []
    warnings = []
    highlights = []
    score = 0  # 满分 100

    rev_g = fundamentals.get("revenue_growth") or 0
    prf_g = fundamentals.get("profit_growth") or 0
    margin = fundamentals.get("gross_margin") or 0
    roe = fundamentals.get("roe") or 0
    revenue = fundamentals.get("revenue") or 0
    profit = fundamentals.get("net_profit") or 0
    eps = fundamentals.get("eps") or 0

    # 预处理
    try:
        rev_g = float(rev_g)
    except (ValueError, TypeError):
        rev_g = 0
    try:
        prf_g = float(prf_g)
    except (ValueError, TypeError):
        prf_g = 0
    try:
        margin = float(margin)
    except (ValueError, TypeError):
        margin = 0
    try:
        roe = float(roe)
    except (ValueError, TypeError):
        roe = 0
    try:
        eps = float(eps)
    except (ValueError, TypeError):
        eps = 0

    # 1) 净利润率（需从利润/营收推算）
    if revenue and profit:
        try:
            net_margin = float(profit) / float(revenue) * 100
        except (ValueError, TypeError, ZeroDivisionError):
            net_margin = 0
    else:
        net_margin = 0

    if net_margin > 20:
        checks.append({"item": "净利润率", "status": "优秀", "detail": f"{net_margin:.1f}%", "good": True})
        score += 15
    elif net_margin > 10:
        checks.append({"item": "净利润率", "status": "良好", "detail": f"{net_margin:.1f}%", "good": True})
        score += 10
    elif net_margin > 5:
        checks.append({"item": "净利润率", "status": "及格", "detail": f"{net_margin:.1f}%", "good": True})
        score += 5
    elif net_margin > 0:
        checks.append({"item": "净利润率", "status": "偏低", "detail": f"{net_margin:.1f}%", "good": False})
        warnings.append(f"净利润率仅 {net_margin:.1f}%，低于 5% 及格线")
    else:
        checks.append({"item": "净利润率", "status": "亏损", "detail": f"{net_margin:.1f}%", "good": False})
        warnings.append("公司处于亏损状态")

    # 2) 利润质量：利润增速不应远超营收增速 5 倍
    if rev_g > 0 and prf_g > rev_g * 5:
        checks.append({"item": "利润质量", "status": "存疑", "detail": f"利润增{prf_g:.0f}% vs 营收增{rev_g:.0f}%", "good": False})
        warnings.append(f"利润增速({prf_g:.0f}%)远超营收增速({rev_g:.0f}%)，增长可能不可持续")
        score -= 10
    elif rev_g > 0 and prf_g > rev_g * 2:
        checks.append({"item": "利润质量", "status": "注意", "detail": f"利润增{prf_g:.0f}% vs 营收增{rev_g:.0f}%", "good": True})
        score += 3
    elif rev_g > 0 and prf_g > 0:
        checks.append({"item": "利润质量", "status": "正常", "detail": f"利润增{prf_g:.0f}% vs 营收增{rev_g:.0f}%", "good": True})
        score += 8
    elif rev_g < 0 and prf_g > 0:
        checks.append({"item": "利润质量", "status": "注意", "detail": "营收下滑但利润增长（可能是降本或非经常性收益）", "good": False})
        warnings.append("营收下滑但利润增长，需确认利润来源是否为经常性")
    else:
        checks.append({"item": "利润质量", "status": "双降", "detail": "营收与利润均下滑", "good": False})
        warnings.append("营收与利润双降，公司处于收缩期")

    # 3) 毛利率
    if margin > 60:
        checks.append({"item": "毛利率", "status": "优秀", "detail": f"{margin:.1f}%", "good": True})
        score += 15
        highlights.append(f"高毛利率 {margin:.1f}%，具有较强定价权或技术壁垒")
    elif margin > 40:
        checks.append({"item": "毛利率", "status": "良好", "detail": f"{margin:.1f}%", "good": True})
        score += 10
    elif margin > 20:
        checks.append({"item": "毛利率", "status": "中等", "detail": f"{margin:.1f}%", "good": True})
        score += 5
    elif margin > 0:
        checks.append({"item": "毛利率", "status": "偏低", "detail": f"{margin:.1f}%", "good": False})
        warnings.append(f"毛利率仅 {margin:.1f}%，产品或服务附加值低")
    else:
        checks.append({"item": "毛利率", "status": "异常", "detail": "数据缺失或为负", "good": False})

    # 4) ROE
    if roe > 20:
        checks.append({"item": "ROE", "status": "优秀", "detail": f"{roe:.1f}%", "good": True})
        score += 15
        highlights.append(f"高 ROE {roe:.1f}%，资本运用效率优秀")
    elif roe > 10:
        checks.append({"item": "ROE", "status": "良好", "detail": f"{roe:.1f}%", "good": True})
        score += 10
    elif roe > 5:
        checks.append({"item": "ROE", "status": "中等", "detail": f"{roe:.1f}%", "good": True})
        score += 5
    elif roe > 0:
        checks.append({"item": "ROE", "status": "偏低", "detail": f"{roe:.1f}%", "good": False})
    else:
        checks.append({"item": "ROE", "status": "异常", "detail": "数据缺失或为负", "good": False})

    # 5) 营收增速
    if rev_g > 50:
        checks.append({"item": "营收增速", "status": "爆发", "detail": f"{rev_g:.1f}%", "good": True})
        score += 15
        highlights.append(f"营收爆发式增长 {rev_g:.0f}%")
    elif rev_g > 20:
        checks.append({"item": "营收增速", "status": "高增", "detail": f"{rev_g:.1f}%", "good": True})
        score += 10
    elif rev_g > 10:
        checks.append({"item": "营收增速", "status": "稳增", "detail": f"{rev_g:.1f}%", "good": True})
        score += 6
    elif rev_g > 0:
        checks.append({"item": "营收增速", "status": "微增", "detail": f"{rev_g:.1f}%", "good": True})
        score += 3
    else:
        checks.append({"item": "营收增速", "status": "下滑", "detail": f"{rev_g:.1f}%", "good": False})
        warnings.append(f"营收同比下滑 {rev_g:.1f}%")

    # 6) EPS
    if eps > 2:
        checks.append({"item": "每股收益", "status": "优秀", "detail": f"{eps:.2f}元", "good": True})
        score += 5
    elif eps > 0.5:
        checks.append({"item": "每股收益", "status": "良好", "detail": f"{eps:.2f}元", "good": True})
        score += 3
    elif eps > 0:
        checks.append({"item": "每股收益", "status": "微利", "detail": f"{eps:.2f}元", "good": True})
        score += 1
    else:
        checks.append({"item": "每股收益", "status": "亏损", "detail": f"{eps:.2f}元", "good": False})
        warnings.append("每股收益为负，公司整体亏损")

    # 7) 市值评估
    if market_cap:
        cap_yi = market_cap / 1e8  # 转为亿
        if cap_yi > 1000:
            cap_label = f"大盘股 ({cap_yi:.0f}亿)"
            cap_note = "流动性好，波动相对小"
        elif cap_yi > 300:
            cap_label = f"中盘股 ({cap_yi:.0f}亿)"
            cap_note = "有一定弹性空间"
        elif cap_yi > 100:
            cap_label = f"中小盘 ({cap_yi:.0f}亿)"
            cap_note = "弹性较好，适合 B1 策略关注区间"
        else:
            cap_label = f"小盘股 ({cap_yi:.0f}亿)"
            cap_note = "弹性大但风险也高"
        checks.append({"item": "市值规模", "status": cap_label.split(" ")[0], "detail": f"{cap_yi:.0f}亿", "good": True})
    else:
        cap_note = None

    # 8) 综合得分
    score = max(0, min(100, score))

    if score >= 80:
        health = "优秀"
        health_desc = "财务基本面扎实，各项指标健康"
    elif score >= 60:
        health = "良好"
        health_desc = "财务整体健康，个别指标需关注"
    elif score >= 40:
        health = "中等"
        health_desc = "财务状况一般，存在一些风险点"
    elif score >= 20:
        health = "偏弱"
        health_desc = "财务风险较多，建议谨慎"
    else:
        health = "差"
        health_desc = "财务基本面较差，不建议参与"

    return {
        "health": health,
        "health_score": score,
        "health_desc": health_desc,
        "checks": checks,
        "warnings": warnings,
        "highlights": highlights,
        "cap_note": cap_note,
    }


def evaluate_stage(profile: dict, fundamentals: dict, financial_health: dict) -> dict:
    """
    判断公司所处发展阶段（故事→验证→确认）

    基于：
    - 上市时间长短
    - 利润增速是否验证了故事
    - 是否有新业务在报表中体现
    """
    listed = profile.get("listed_date", "")
    rev_g = fundamentals.get("revenue_growth", 0) or 0
    prf_g = fundamentals.get("profit_growth", 0) or 0

    try:
        rev_g = float(rev_g)
    except (ValueError, TypeError):
        rev_g = 0
    try:
        prf_g = float(prf_g)
    except (ValueError, TypeError):
        prf_g = 0

    # 上市年限
    try:
        if listed and len(str(listed)) == 8:
            list_year = int(str(listed)[:4])
            age = datetime.now().year - list_year
        else:
            age = None
    except (ValueError, TypeError):
        age = None

    # 判断逻辑
    if rev_g > 30 and prf_g > 30:
        stage = "确认期"
        stage_desc = "高增长已被财报验证，新业务可能已成为主要利润来源"
    elif rev_g > 10 and prf_g > 20:
        stage = "验证期"
        stage_desc = "增长逻辑已被财报部分验证，可持续性仍需观察"
    elif rev_g > 0 and prf_g > rev_g:
        stage = "验证初期"
        stage_desc = "利润增速快于营收，效率提升或新业务开始贡献"
    elif rev_g > 20:
        stage = "故事期"
        stage_desc = "营收高增但利润尚未兑现，处于投入期或扩张期"
    elif rev_g < -10 or prf_g < -20:
        stage = "收缩期"
        stage_desc = "业务处于收缩或调整阶段，需等待拐点信号"
    else:
        stage = "成熟期"
        stage_desc = "增长平稳，关注分红和估值"

    return {
        "stage": stage,
        "stage_desc": stage_desc,
        "list_age": f"{age}年" if age else "未知",
    }


def deep_analyze(code: str, fundamentals: dict, market_cap: float = None) -> dict:
    """
    单只股票深度分析主入口

    Returns:
        {
            profile: {industry, market_cap, listed_date, ...},
            financial_health: {health, score, checks, warnings, highlights},
            stage: {stage, stage_desc},
            verdict: str  # 一句话总结
        }
    """
    profile = get_company_profile(code)
    if not market_cap and profile.get("market_cap"):
        market_cap = profile["market_cap"]

    health = analyze_financial_health(fundamentals, market_cap)
    stage = evaluate_stage(profile, fundamentals, health)

    # 综合判断
    verdict_parts = []
    industry = profile.get("industry", "")
    if industry:
        verdict_parts.append(f"所属{industry}行业")

    stage_name = stage.get("stage", "")
    if stage_name:
        verdict_parts.append(f"处于{stage_name}")

    health_score = health.get("health_score", 0)
    if health_score >= 60:
        verdict_parts.append(f"财务健康度{health['health']}({health_score}分)")
    else:
        verdict_parts.append(f"财务健康度{health['health']}({health_score}分)⚠️")

    if health.get("highlights"):
        verdict_parts.append(f"亮点: {'; '.join(health['highlights'][:2])}")

    verdict = "。".join(verdict_parts) + "。"

    return {
        "profile": profile,
        "financial_health": health,
        "stage": stage,
        "verdict": verdict,
    }
