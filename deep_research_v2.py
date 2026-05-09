"""
深度研究引擎 v2 — 不抄研报，发现矛盾
五个维度：分析师分歧 | 基本面×技术面背离 | 盈利质量 | 行业位置 | 风险模式扫描

每个维度输出结构化信号，综合给出评估。
"""
import sys, os, json, logging
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LOCAL_DIR)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
# 维度 1：分析师分歧度
# ═══════════════════════════════════════════════════

def _fetch_reports(code: str) -> list:
    """获取个股研报列表"""
    try:
        import akshare as ak
        df = ak.stock_research_report_em(symbol=code)
        if df is None or df.empty:
            return []
        cols = list(df.columns)
        reports = []
        for _, row in df.head(30).iterrows():
            reports.append({
                "title": str(row.iloc[3]) if len(cols) > 3 else "",
                "rating": str(row.iloc[4]) if len(cols) > 4 else "",
                "broker": str(row.iloc[5]) if len(cols) > 5 else "",
                "profit_2026": _sf(row.iloc[7]) if len(cols) > 7 else None,
                "pe_2026": _sf(row.iloc[8]) if len(cols) > 8 else None,
                "profit_2027": _sf(row.iloc[9]) if len(cols) > 9 else None,
                "date": str(row.iloc[14]) if len(cols) > 14 else "",
            })
        return reports
    except Exception:
        return []


def analyst_check(code: str) -> dict:
    """
    分析师分歧度分析
    - 利润预测离散度（CV = std/mean）→ 分歧大 = 不确定性高
    - 评级分布 → 过度一致看多反而是风险
    - 目标 PE 区间
    """
    reports = _fetch_reports(code)
    signals = []
    level = "green"

    if len(reports) < 3:
        return {
            "dimension": "分析师共识",
            "signals": [f"仅 {len(reports)} 份研报覆盖，机构关注度低"],
            "level": "yellow",
            "summary": "机构覆盖不足",
            "data": {"reports_count": len(reports)},
        }

    profits_26 = [r["profit_2026"] for r in reports if r["profit_2026"] and r["profit_2026"] > 0]
    pes_26 = [r["pe_2026"] for r in reports if r["pe_2026"] and r["pe_2026"] > 0]
    ratings = [r["rating"] for r in reports if r["rating"]]

    data = {"reports_count": len(reports), "brokers": list(set(r["broker"] for r in reports))[:10]}

    # 利润预测离散度
    if len(profits_26) >= 5:
        avg_p = np.mean(profits_26)
        std_p = np.std(profits_26)
        cv = std_p / avg_p if avg_p > 0 else 999
        data["profit_consensus"] = round(avg_p, 2)
        data["profit_range"] = f"{min(profits_26):.1f} - {max(profits_26):.1f}"
        data["profit_cv"] = round(cv, 3)

        if cv > 0.30:
            level = "red"
            signals.append(f"利润预测分歧极大 (CV={cv:.2f})，机构对盈利前景缺乏共识")
        elif cv > 0.15:
            level = "yellow"
            signals.append(f"利润预测存在分歧 (CV={cv:.2f})，需关注不确定性")
        else:
            signals.append(f"利润预测共识度较高 (CV={cv:.2f})，{avg_p:.1f}亿 ({len(profits_26)}家)")

    # PE 区间
    if len(pes_26) >= 3:
        data["pe_consensus"] = round(np.mean(pes_26), 1)
        data["pe_range"] = f"{min(pes_26):.0f} - {max(pes_26):.0f}"
        signals.append(f"一致预期 PE {np.mean(pes_26):.0f}x (区间 {min(pes_26):.0f}-{max(pes_26):.0f})")

    # 评级分布
    if ratings:
        rc = Counter(ratings)
        data["rating_dist"] = dict(rc.most_common(5))
        buy_count = sum(1 for r in ratings if "买入" in r or "推荐" in r or "强推" in r or "增持" in r)
        buy_ratio = buy_count / len(ratings)
        data["buy_ratio"] = round(buy_ratio, 2)

        if buy_ratio > 0.95 and len(ratings) >= 5:
            signals.append(f"过度一致看多 ({buy_count}/{len(ratings)} 买入)，警惕回声室效应")
            if level == "green":
                level = "yellow"
        elif buy_ratio > 0.7:
            signals.append(f"券商共识偏多 ({buy_count}/{len(ratings)} 买入/增持)")
        else:
            signals.append(f"评级分歧明显，买入占比仅 {buy_ratio:.0%}")

    return {
        "dimension": "分析师共识",
        "signals": signals,
        "level": level,
        "summary": f"{len(reports)}份研报, "
                   + (f"利润CV={cv:.2f}" if len(profits_26) >= 5 else "数据不足"),
        "data": data,
    }


# ═══════════════════════════════════════════════════
# 维度 2：基本面 × 技术面背离
# ═══════════════════════════════════════════════════

def fundamental_technical_divergence(code: str, fundamentals: dict,
                                     b1_scores: dict, returns: dict) -> dict:
    """
    检测基本面和技术面的矛盾
    - 利润暴增 + 技术面弱 → 市场在怀疑什么？
    - 技术面强 + 基本面平庸 → 纯资金驱动
    - 双双强劲 → 确认
    """
    signals = []
    level = "green"

    profit_yoy = _fv(fundamentals.get("profit_growth"))
    revenue_yoy = _fv(fundamentals.get("revenue_growth"))
    margin = _fv(fundamentals.get("gross_margin"))
    score_60d = _fv(b1_scores.get("score_long"))
    score_short = _fv(b1_scores.get("score_short"))
    ret_60d = _fv(returns.get("60d"))

    fund_strong = profit_yoy > 200 or (margin > 30 and profit_yoy > 100)
    fund_ok = profit_yoy > 50
    tech_strong = score_60d >= 80
    tech_ok = score_60d >= 60
    momentum_strong = ret_60d is not None and ret_60d > 15

    data = {
        "profit_yoy": profit_yoy, "margin": margin,
        "score_long": score_60d, "ret_60d": ret_60d,
    }

    # 背离检测
    if fund_strong and not tech_ok:
        level = "yellow"
        signals.append(f"基本面强劲 (利润+{profit_yoy:.0f}%, 毛利率{margin:.0f}%) 但技术面评分仅 {score_60d:.0f}")
        signals.append("可能原因：市场尚未定价 / 基本面数据有水分 / 行业逆风压制估值")
    elif tech_strong and not fund_ok and margin < 20:
        level = "red"
        signals.append(f"技术面极强 ({score_60d:.0f}分) 但基本面平庸 (利润+{profit_yoy:.0f}%, 毛利率{margin:.0f}%)")
        signals.append("纯资金驱动，基本面不支撑，追高风险极大")
    elif tech_strong and fund_strong:
        signals.append(f"基本面与技术面共振：利润+{profit_yoy:.0f}% + 评分{score_60d:.0f}")
    elif not fund_ok and not tech_ok:
        level = "yellow"
        signals.append("基本面和技术面均无亮点")

    # 价格与基本面
    if momentum_strong and fund_strong:
        signals.append(f"60日涨幅+{ret_60d:.0f}% 有基本面支撑")
    elif momentum_strong and not fund_ok:
        signals.append(f"60日涨幅+{ret_60d:.0f}% 缺乏基本面支撑，警惕回调")

    return {
        "dimension": "基本面×技术面",
        "signals": signals,
        "level": level,
        "summary": f"利润+{profit_yoy:.0f}% | 评分{score_60d:.0f} | "
                   + ("共振" if fund_strong and tech_strong
                      else "背离" if (fund_strong and not tech_ok) or (tech_strong and not fund_ok)
                      else "中性"),
        "data": data,
    }


# ═══════════════════════════════════════════════════
# 维度 3：盈利质量
# ═══════════════════════════════════════════════════

def earnings_quality(code: str, fundamentals: dict) -> dict:
    """
    盈利质量检查
    - 利润增速 vs 营收增速：利润远超营收 → 可能靠非经常性损益
    - 毛利率水平
    - ROE
    """
    signals = []
    level = "green"

    profit_yoy = _fv(fundamentals.get("profit_growth"))
    revenue_yoy = _fv(fundamentals.get("revenue_growth"))
    margin = _fv(fundamentals.get("gross_margin"))
    roe = _fv(fundamentals.get("roe"))

    data = {"profit_yoy": profit_yoy, "revenue_yoy": revenue_yoy,
            "gross_margin": margin, "roe": roe}

    # 利润质量：利润增速明显高于营收增速 2 倍以上 → 经营杠杆释放或非经常性损益
    if profit_yoy > 100 and revenue_yoy > 0 and profit_yoy > revenue_yoy * 3:
        signals.append(f"利润增速(+{profit_yoy:.0f}%)远超营收(+{revenue_yoy:.0f}%)，关注利润来源")
        signals.append("可能原因：一次性收益 / 费用压缩 / 确实经营杠杆释放 → 需看扣非")
        level = "yellow"

    if revenue_yoy < -10 and profit_yoy > 50:
        signals.append(f"营收下滑({revenue_yoy:.0f}%)但利润暴增(+{profit_yoy:.0f}%)，可能靠变卖资产或削减研发")
        level = "red"

    # 毛利率
    if margin > 0 and margin < 10:
        signals.append(f"毛利率仅 {margin:.1f}%，产品附加值低，抗风险能力弱")
        level = "red" if level == "green" else level
    elif margin > 0 and margin < 20:
        if level == "green":
            level = "yellow"
        signals.append(f"毛利率偏低 ({margin:.1f}%)，竞争激烈或商业模式有问题")
    elif margin >= 40:
        signals.append(f"毛利率 {margin:.1f}%，产品有定价权或技术壁垒")

    # ROE (Q1 单季，年化 = Q1*4)
    if roe > 0 and roe < 2:
        signals.append(f"ROE 仅 {roe:.1f}%(年化约{roe*4:.1f}%)，资本回报率低")
        if level == "green":
            level = "yellow"

    if not signals:
        signals.append("盈利质量指标无明显异常")

    return {
        "dimension": "盈利质量",
        "signals": signals,
        "level": level,
        "summary": f"毛利率{margin:.1f}% | ROE {roe:.1f}% | "
                   + ("利润质量待验证" if level != "green" else "正常"),
        "data": data,
    }


# ═══════════════════════════════════════════════════
# 维度 4：行业位置
# ═══════════════════════════════════════════════════

# 全局缓存，避免重复拉取全市场数据
_industry_data: pd.DataFrame | None = None
_industry_rankings: dict = {}


def _load_industry_data() -> pd.DataFrame:
    """加载全市场 Q1 数据用于行业对比（缓存）"""
    global _industry_data
    if _industry_data is not None:
        return _industry_data
    try:
        import akshare as ak
        from datetime import datetime
        m = datetime.now().month
        y = datetime.now().year
        q_end = ((m-1)//3)*3
        if q_end == 0:
            q_end = 12; y -= 1
        q_date = f"{y}{q_end:02d}31" if q_end in (3, 12) else f"{y}{q_end:02d}30"
        df = ak.stock_yjbb_em(date=q_date)
        if df is None or df.empty:
            return pd.DataFrame()
        col_map = {
            "股票代码": "code", "股票简称": "name",
            "营业总收入-同比增长": "revenue_yoy",
            "净利润-同比增长": "profit_yoy",
            "销售毛利率": "gross_margin",
            "净资产收益率": "roe",
            "所处行业": "industry",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)
        for col in ["profit_yoy", "gross_margin", "roe"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        _industry_data = df
        return df
    except Exception:
        return pd.DataFrame()


def industry_positioning(code: str, fundamentals: dict) -> dict:
    """
    行业内部定位
    - 在同行业中利润增速、毛利率、ROE 的百分位排名
    - 找出行业最佳标的
    """
    signals = []
    level = "green"

    profit_yoy = _fv(fundamentals.get("profit_growth"))
    margin = _fv(fundamentals.get("gross_margin"))
    roe = _fv(fundamentals.get("roe"))

    data = {"profit_yoy": profit_yoy, "margin": margin, "roe": roe}

    try:
        df = _load_industry_data()
        if df.empty or "industry" not in df.columns:
            return {"dimension": "行业位置", "signals": ["行业数据不可用"],
                    "level": "green", "summary": "数据不足", "data": data}

        # 找到该股票的行业
        row = df[df["code"] == code]
        if row.empty:
            return {"dimension": "行业位置", "signals": ["未找到行业分类"],
                    "level": "green", "summary": "行业未知", "data": data}

        industry = row.iloc[0].get("industry", "")
        if not industry or pd.isna(industry):
            return {"dimension": "行业位置", "signals": ["行业分类为空"],
                    "level": "green", "summary": "行业未知", "data": data}

        data["industry"] = str(industry)

        # 同行业筛选
        peers = df[df["industry"] == industry].copy()
        if len(peers) < 3:
            return {"dimension": "行业位置", "signals": [f"行业'{industry}'样本不足({len(peers)}家)"],
                    "level": "green", "summary": f"行业({len(peers)}家)", "data": data}

        data["peer_count"] = len(peers)

        # 计算排名
        rankings = {}
        for metric, label in [("profit_yoy", "利润增速"), ("gross_margin", "毛利率"), ("roe", "ROE")]:
            col = peers[metric].dropna()
            if len(col) < 3 or pd.isna(profit_yoy if metric == "profit_yoy"
                                       else margin if metric == "gross_margin" else roe):
                continue
            my_val = profit_yoy if metric == "profit_yoy" else margin if metric == "gross_margin" else roe
            if my_val is None:
                continue
            pct = (col < my_val).sum() / len(col) * 100
            rankings[label] = round(pct, 0)

        data["rankings"] = rankings

        # 找出行业最优（在利润增速和毛利率上）
        if "profit_yoy" in peers.columns:
            best_profit = peers.nlargest(3, "profit_yoy")
            top_profit_codes = best_profit["code"].tolist()
            if code in top_profit_codes:
                signals.append(f"行业利润增速排名前 {len(peers)} 家中的第 {list(top_profit_codes).index(code)+1} 位")
            elif rankings:
                signals.append(f"行业最高利润增速: {best_profit.iloc[0].get('name', best_profit.iloc[0].get('code', '?'))} "
                             f"({best_profit.iloc[0].get('profit_yoy', '?'):.0f}%)")

        # 综合判断
        if rankings:
            rank_str = " | ".join(f"{k}: top{v:.0f}%" for k, v in rankings.items())
            data["rank_str"] = rank_str
            avg_rank = np.mean(list(rankings.values()))
            if avg_rank >= 70:
                signals.append(f"行业综合排名靠前 — {rank_str}")
            elif avg_rank < 30:
                signals.append(f"行业综合排名靠后 — {rank_str}")
                level = "yellow"
            else:
                signals.append(f"行业位置中等 — {rank_str}")

    except Exception as e:
        signals.append(f"行业分析异常: {e}")

    if not signals:
        signals.append("行业数据不足")

    return {
        "dimension": "行业位置",
        "signals": signals,
        "level": level,
        "summary": f"{data.get('industry', '未知')} ({data.get('peer_count', '?')}家)"
                   + (f" | 排名: {data.get('rank_str', '?')}" if "rank_str" in data else ""),
        "data": data,
    }


# ═══════════════════════════════════════════════════
# 维度 5：风险模式扫描
# ═══════════════════════════════════════════════════

def risk_scan(code: str, fundamentals: dict, price_data: dict,
              b1_scores: dict) -> dict:
    """
    硬规则风险扫描
    """
    signals = []
    risks = []
    level = "green"

    profit_yoy = _fv(fundamentals.get("profit_growth"))
    revenue_yoy = _fv(fundamentals.get("revenue_growth"))
    margin = _fv(fundamentals.get("gross_margin"))
    score_60d = _fv(b1_scores.get("score_long"))
    score_short = _fv(b1_scores.get("score_short"))

    # 1. 利润质量风险
    if profit_yoy > 200 and revenue_yoy < 10:
        risks.append({
            "type": "利润来源可疑",
            "detail": f"利润暴增+{profit_yoy:.0f}%但营收仅+{revenue_yoy:.0f}%",
            "severity": "high",
        })
        level = "red"

    # 2. 毛利率陷阱
    if margin > 0 and margin < 10 and profit_yoy > 100 and revenue_yoy < 0:
        risks.append({
            "type": "低毛利高增长不可持续",
            "detail": f"毛利率{margin:.1f}%靠压缩成本实现利润+{profit_yoy:.0f}%",
            "severity": "high",
        })
        level = "red"

    # 3. 技术面过热
    if score_60d >= 85 and margin < 20 and profit_yoy < 50:
        risks.append({
            "type": "技术面过热无基本面支撑",
            "detail": f"评分{score_60d:.0f}但毛利率{margin:.1f}%、利润+{profit_yoy:.0f}%",
            "severity": "medium",
        })
        if level == "green":
            level = "yellow"

    # 4. 营收下滑
    if revenue_yoy < -20 and profit_yoy < -20:
        risks.append({
            "type": "营收利润双降",
            "detail": f"营收{revenue_yoy:.0f}%、利润{profit_yoy:.0f}%",
            "severity": "high",
        })
        level = "red"

    # 5. 高分低能
    if score_60d >= 70 and profit_yoy < -50:
        risks.append({
            "type": "技术面强但业绩暴跌",
            "detail": f"评分{score_60d:.0f}但利润{profit_yoy:.0f}%，可能超跌反弹陷阱",
            "severity": "medium",
        })
        if level == "green":
            level = "yellow"

    for r in risks:
        signals.append(f"[{r['severity'].upper()}] {r['type']}: {r['detail']}")

    if not risks:
        signals.append("未触发已知风险模式")

    return {
        "dimension": "风险扫描",
        "signals": signals,
        "level": level,
        "summary": f"{len(risks)} 个风险信号" if risks else "无风险信号",
        "data": {"risks": risks, "risk_count": len(risks)},
    }


# ═══════════════════════════════════════════════════
# 综合合成
# ═══════════════════════════════════════════════════

def deep_research(code: str, fundamentals: dict = None,
                  b1_result: dict = None) -> dict:
    """
    主入口：对单只股票进行五维度深度研究

    Args:
        code: 股票代码
        fundamentals: Q1 基本面数据（可选，不传则自动获取）
        b1_result: B1 分析结果（可选，包含 scores/returns/price 等）

    Returns:
        {
            "dimensions": [...],       # 五个维度的完整结果
            "insights": [...],         # 关键洞察列表
            "warnings": [...],         # 警告列表
            "overall_level": "green|yellow|red",
            "verdict": str,            # 一句话总结
            "has_analyst_data": bool,
        }
    """
    # 获取基本面数据
    if fundamentals is None:
        from stock_analyzer import fetch_fundamentals_all
        all_fund = fetch_fundamentals_all()
        fundamentals = all_fund.get(code, {})

    # 提取 B1 数据
    scores = b1_result.get("scores", {}) if b1_result else {}
    returns = b1_result.get("returns", {}) if b1_result else {}
    price = b1_result.get("price", {}) if b1_result else {}

    # ── 并行执行五维度（均独立容错）──
    dimensions = []

    # 1. 分析师
    try:
        dim1 = analyst_check(code)
        dimensions.append(dim1)
    except Exception as e:
        dimensions.append({"dimension": "分析师共识", "signals": [f"分析失败: {e}"],
                           "level": "green", "summary": "数据不可用", "data": {}})

    # 2. 基本面×技术面
    try:
        dim2 = fundamental_technical_divergence(code, fundamentals, scores, returns)
        dimensions.append(dim2)
    except Exception as e:
        dimensions.append({"dimension": "基本面×技术面", "signals": [f"分析失败: {e}"],
                           "level": "green", "summary": "数据不可用", "data": {}})

    # 3. 盈利质量
    try:
        dim3 = earnings_quality(code, fundamentals)
        dimensions.append(dim3)
    except Exception as e:
        dimensions.append({"dimension": "盈利质量", "signals": [f"分析失败: {e}"],
                           "level": "green", "summary": "数据不可用", "data": {}})

    # 4. 行业位置
    try:
        dim4 = industry_positioning(code, fundamentals)
        dimensions.append(dim4)
    except Exception as e:
        dimensions.append({"dimension": "行业位置", "signals": [f"分析失败: {e}"],
                           "level": "green", "summary": "数据不可用", "data": {}})

    # 5. 风险扫描
    try:
        dim5 = risk_scan(code, fundamentals, price, scores)
        dimensions.append(dim5)
    except Exception as e:
        dimensions.append({"dimension": "风险扫描", "signals": [f"分析失败: {e}"],
                           "level": "green", "summary": "数据不可用", "data": {}})

    # ── 汇总 ──
    insights = []
    warnings = []
    red_count = 0
    yellow_count = 0

    for dim in dimensions:
        for sig in dim.get("signals", []):
            if dim["level"] == "red":
                warnings.append(f"[{dim['dimension']}] {sig}")
            elif dim["level"] == "yellow" and ("风险" in dim["dimension"] or "警告" in sig):
                warnings.append(f"[{dim['dimension']}] {sig}")
            else:
                insights.append(f"[{dim['dimension']}] {sig}")

        if dim["level"] == "red":
            red_count += 1
        elif dim["level"] == "yellow":
            yellow_count += 1

    # 综合评定
    if red_count >= 2:
        overall = "red"
        verdict = "多个维度亮红灯，建议回避或轻仓观察"
    elif red_count == 1:
        overall = "yellow"
        verdict = "存在显著风险点，需进一步核实后再决策"
    elif yellow_count >= 3:
        overall = "yellow"
        verdict = "多个维度存在关注点，性价比一般"
    elif yellow_count >= 1:
        overall = "green"
        verdict = "整体健康，有少量关注点"
    else:
        overall = "green"
        verdict = "五维度均无明显异常，基本面和市场共识健康"

    has_analyst = any(d.get("dimension") == "分析师共识"
                      and d.get("data", {}).get("reports_count", 0) >= 3
                      for d in dimensions)

    return {
        "dimensions": dimensions,
        "insights": insights[:12],
        "warnings": warnings[:8],
        "overall_level": overall,
        "verdict": verdict,
        "has_analyst_data": has_analyst,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ═══════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════

def _sf(val):
    """安全转为 float"""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fv(val):
    """安全转 float，失败返回 0"""
    if val is None:
        return 0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0
