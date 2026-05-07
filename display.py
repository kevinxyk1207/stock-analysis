"""
Streamlit 渲染函数
每个展示区块一个函数，按顺序调用
"""
import streamlit as st


# ── 颜色常量 ──

STAR_COLORS = {5: "#2e7d32", 4: "#43a047", 3: "#f9a825", 2: "#ef6c00", 1: "#c62828"}
STAR_BG = {5: "#e8f5e9", 4: "#e8f5e9", 3: "#fff8e1", 2: "#fff3e0", 1: "#ffebee"}
STAR_LABELS = {5: "强烈推荐", 4: "推荐", 3: "观望", 2: "偏弱", 1: "回避"}


def score_label(val: float) -> str:
    if val >= 80: return "强"
    if val >= 60: return "良好"
    if val >= 40: return "中等"
    return "弱"


def score_color(val: float) -> str:
    if val >= 60: return "green"
    if val >= 40: return "orange"
    return "red"


def fmt_val(v):
    """安全格式化数值"""
    if v is None:
        return "N/A"
    return v


# ── 渲染函数 ──

def render_price_card(meta: dict, price: dict, market: dict):
    """股票概要卡片"""
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"{meta['name']} ({meta['code']})")
    with col2:
        change = price["change_pct"]
        delta_str = f"{change:+.2f}%"
        st.metric("现价", f"{price['close']:.2f}", delta=delta_str)
    with col3:
        if market["is_bull"]:
            st.success("牛 市")
        else:
            st.error("熊 市")

    st.caption(f"数据日期: {meta['last_date']} | {market.get('note', '')}")


def render_star_verdict(star: dict, operation: str):
    """星级判定 + 操作建议"""
    stars = star["stars"]
    color = STAR_COLORS[stars]
    bg = STAR_BG[stars]

    st.markdown(f"""
    <div style="background:{bg}; border-left:6px solid {color}; padding:16px;
                border-radius:8px; margin:16px 0;">
        <div style="font-size:32px; color:{color}; font-weight:bold;">
            {star['stars_text']}
            <span style="font-size:18px; color:#666;">{star['score']}分 · {STAR_LABELS[stars]}</span>
        </div>
        <div style="margin-top:8px; font-size:15px; color:#333; white-space:pre-line;">
            {operation.replace(chr(10), '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_warnings(risks: list, ret60=None):
    """风险警告"""
    all_risks = list(risks)
    if ret60 is not None and ret60 < -20:
        all_risks.append(f"60日跌幅过大 ({ret60:.1f}%)")

    if not all_risks:
        return

    items = "".join(f"<li>{r}</li>" for r in all_risks)
    st.markdown(f"""
    <div style="background:#fff3e0; border:1px solid #ff9800; padding:12px;
                border-radius:8px; margin:8px 0;">
        <strong style="color:#e65100;">风险提示</strong>
        <ul style="margin:4px 0; color:#bf360c;">{items}</ul>
    </div>
    """, unsafe_allow_html=True)


def render_scores(scores: dict):
    """双档评分条"""
    c1, c2 = st.columns(2)
    with c1:
        s10 = scores["score_10d"]
        st.metric("10日短线得分", f"{s10:.0f}", delta=score_label(s10),
                  delta_color="normal" if s10 >= 60 else "off")
        st.progress(s10 / 100)
    with c2:
        s60 = scores["score_60d"]
        st.metric("60日长线得分", f"{s60:.0f}", delta=score_label(s60),
                  delta_color="normal" if s60 >= 60 else "off")
        st.progress(s60 / 100)


def render_b1_conditions(cond: dict):
    """B1 核心五条件"""
    checks = [
        ("KDJ 低位", cond.get("kdj_low", False),
         f"J值={cond.get('j_value', '?')}, 分位={cond.get('j_percentile', '?')}"),
        ("价格在知行底线之上", cond.get("close_gt_zxdkx", False),
         "收盘价 > ZXDKX 均线组合"),
        ("知行线多头排列", cond.get("zxdq_gt_zxdkx", False),
         "ZXDQ(短线) > ZXDKX(底线)"),
        ("周线多头排列", cond.get("weekly_bull", False),
         "周线级别均线向上发散"),
        ("量能验证通过", cond.get("vol_check", True),
         "近期最大量日非阴线"),
    ]
    passed = sum(1 for _, ok, _ in checks if ok)
    st.caption(f"B1 条件通过 {passed}/5")

    for name, ok, desc in checks:
        icon = "✅" if ok else "❌"
        color = "#2e7d32" if ok else "#c62828"
        st.markdown(f"<span style='color:{color}'>{icon} **{name}**</span> — {desc}",
                    unsafe_allow_html=True)


def render_indicators_table(ind: dict):
    """关键技术指标表"""
    st.caption("—— 核心技术指标 ——")

    rows = [
        ("RSI (14)", ind.get("rsi"), "50-70 最佳区间", None),
        ("MACD 动量", ind.get("macd_quality"), ">50 为强", None),
        ("趋势强度", f"{ind.get('trend_strength', 0):.1f}%", ">5% 为强趋势", None),
        ("均线排列", f"{ind.get('ma_alignment', 0)}/3", "3=完全多头", ind.get("ma_alignment") == 3),
        ("量能趋势", ind.get("vol_trend_ratio"), ">1.1 放量", ind.get("vol_trend_ratio", 0) > 1.1),
        ("量能健康度", f"{ind.get('vol_health', 0):.0%}", "上涨量占比", None),
        ("知行线位置", ind.get("zxdkx_ratio"), ">1.0 = 线上", ind.get("zxdkx_ratio", 0) > 1.0),
        ("回调买点", ind.get("pullback_setup"), ">50 有回调机会", ind.get("pullback_setup", 0) > 50),
        ("回撤幅度", ind.get("drawdown_ratio"), ">0.95 接近前高", None),
        ("量能激增", ind.get("volume_surge"), ">1.25 明显放量", None),
        ("5日价格加速", f"{ind.get('price_accel_5', 0):.2f}%", "正数=短期强势", None),
        ("KDJ 反弹", ind.get("kdj_rebound"), ">50 J值回升中", None),
        ("主力资金流", ind.get("main_force_flow"), "正值=资金流入", ind.get("main_force_flow", 0) > 0),
        ("资金强度", ind.get("flow_strength"), "正值=多头", None),
        ("北向代理信号", ind.get("north_bound_proxy"), ">50 外资偏多", None),
        ("MA60 斜率(20日)", f"{ind.get('ma60_slope_20d', 0):.2f}%", "正值=中期向上", None),
    ]

    c1, c2 = st.columns(2)
    for i, (label, val, hint, good) in enumerate(rows):
        col = c1 if i % 2 == 0 else c2
        with col:
            v_str = str(val) if val is not None else "N/A"
            if good is True:
                st.markdown(f"**{label}**: <span style='color:#2e7d32'>{v_str}</span> — {hint}",
                            unsafe_allow_html=True)
            elif good is False:
                st.markdown(f"**{label}**: <span style='color:#c62828'>{v_str}</span> — {hint}",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"**{label}**: {v_str} — {hint}")


def render_price_levels(levels: dict, current_price: float):
    """关键价位"""
    st.caption("—— 关键价位 ——")

    items = [
        ("MA120", levels.get("ma120")),
        ("MA60", levels.get("ma60")),
        ("ZXDKX (知行底线)", levels.get("zxdkx")),
        ("MA20", levels.get("ma20")),
        ("MA10", levels.get("ma10")),
        ("MA5", levels.get("ma5")),
        ("现价", current_price),
        ("20日高", levels.get("high_20")),
        ("60日高", levels.get("high_60")),
    ]

    # 按价格排序（现价放中间）
    prices_with_label = []
    for label, val in items:
        if val is not None:
            prices_with_label.append((label, val))

    prices_with_label.sort(key=lambda x: x[1])
    min_p = prices_with_label[0][1]
    max_p = prices_with_label[-1][1]
    span = max_p - min_p if max_p > min_p else 1

    for label, val in prices_with_label:
        pct = (val - min_p) / span * 100
        is_current = label == "现价"
        bar_color = "#1a73e8" if is_current else "#e0e0e0"
        text_style = "font-weight:bold;" if is_current else ""

        st.markdown(f"""
        <div style="display:flex; align-items:center; margin:2px 0;">
            <div style="width:80px; text-align:right; font-size:12px; {text_style}">{label}</div>
            <div style="flex:1; margin:0 8px; background:#eee; height:10px; border-radius:5px; position:relative;">
                <div style="background:{bar_color}; width:{pct}%; height:10px; border-radius:5px;"></div>
            </div>
            <div style="width:70px; font-size:12px; {text_style}">{val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    atr = levels.get("atr")
    atr_pct = levels.get("atr_pct")
    if atr and atr_pct:
        st.caption(f"ATR(14): {atr:.2f} ({atr_pct:.2f}%)")


def render_returns(rets: dict):
    """区间收益"""
    st.caption("—— 区间涨跌幅 ——")
    cols = st.columns(4)
    labels = [("5d", "5日"), ("10d", "10日"), ("20d", "20日"), ("60d", "60日")]
    for i, (key, label) in enumerate(labels):
        with cols[i]:
            val = rets.get(key)
            if val is None:
                st.metric(label, "N/A")
            else:
                st.metric(label, f"{val:+.1f}%",
                          delta_color="normal" if val >= 0 else "inverse")


def render_fundamentals(fin: dict):
    """基本面数据"""
    if not fin or all(v is None for v in fin.values()):
        st.info("暂无基本面数据")
        return

    c1, c2, c3 = st.columns(3)

    def _show(label, val, unit=""):
        if val is not None:
            try:
                v = float(val)
                if unit == "亿":
                    return f"{v/1e8:.2f}亿" if abs(v) > 1e8 else f"{v:.2f}元"
                elif unit == "%":
                    return f"{v:+.2f}%"
                else:
                    return f"{v:.2f}"
            except (ValueError, TypeError):
                return str(val)
        return "N/A"

    with c1:
        st.metric("营收", _show(fin.get("revenue"), fin.get("revenue"), "亿"))
        st.metric("营收增速", _show(fin.get("revenue_growth"), fin.get("revenue_growth"), "%"))
    with c2:
        st.metric("净利润", _show(fin.get("net_profit"), fin.get("net_profit"), "亿"))
        st.metric("利润增速", _show(fin.get("profit_growth"), fin.get("profit_growth"), "%"))
    with c3:
        st.metric("毛利率", _show(fin.get("gross_margin"), fin.get("gross_margin"), "%"))
        st.metric("ROE", _show(fin.get("roe"), fin.get("roe"), "%"))

    eps = fin.get("eps")
    if eps is not None:
        st.caption(f"EPS: {_show(eps, eps)}")


def render_insights(insights: dict):
    """深度洞察"""
    verdict = insights.get("verdict_override", "")
    highlights = insights.get("deep_highlights", [])
    risks = insights.get("deep_risks", [])

    if not verdict and not highlights and not risks:
        st.info("暂无深度洞察数据")
        return

    if verdict:
        st.markdown(f"**综合判断**: {verdict}")

    if highlights:
        st.markdown("**核心亮点**")
        for h in highlights:
            st.markdown(f"- {h}")

    if risks:
        st.markdown("**隐蔽风险**")
        for r in risks:
            st.markdown(f"- {r}")


def render_footer(meta: dict):
    """页脚"""
    st.divider()
    st.caption(
        f"数据日期: {meta['last_date']} | "
        f"数据源: baostock / akshare | "
        f"仅供参考，不构成投资建议"
    )
