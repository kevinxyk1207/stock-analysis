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

def render_price_card(meta: dict, price: dict, market: dict, realtime: dict | None = None):
    """股票概要卡片。realtime 为实时行情 dict 时优先展示。"""
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.subheader(f"{meta['name']} ({meta['code']})")
    with col2:
        if realtime and realtime.get("price"):
            rt_price = realtime["price"]
            rt_pct = realtime.get("change_pct", 0) or 0
            st.metric(
                "实时价",
                f"{rt_price:.2f}",
                delta=f"{rt_pct:+.2f}%",
                delta_color="normal",
            )
        else:
            pct = price.get("change_pct", 0) or 0
            st.metric("收盘价", f"{price['close']:.2f}", delta=f"{pct:+.2f}%")
    with col3:
        if realtime and realtime.get("price"):
            rt_vol = realtime.get("turnover", 0) or 0
            vol_str = f"{rt_vol/1e8:.2f}亿" if rt_vol > 1e8 else f"{rt_vol/1e4:.0f}万"
            st.metric("成交额", vol_str)
        else:
            vol = price.get("volume", 0) or 0
            vol_str = f"{vol/1e8:.2f}亿" if vol > 1e8 else f"{vol/1e4:.0f}万"
            st.metric("成交量", vol_str)
    with col4:
        if market["is_bull"]:
            st.success("牛市")
        else:
            st.error("熊市")

    if realtime and realtime.get("price"):
        rt_data_note = f"实时数据 {realtime.get('time', '')}"
        hist_note = f"昨收 {price['close']:.2f} | {market.get('note', '')}"
        st.caption(f"{rt_data_note} | {hist_note}")
    else:
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
        s10 = scores["score_short"]
        st.metric("10日短线得分", f"{s10:.0f}", delta=score_label(s10),
                  delta_color="normal" if s10 >= 60 else "off")
        st.progress(s10 / 100)
    with c2:
        s60 = scores["score_long"]
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

    def _show(label, val, unit=""):  # label kept for caller compatibility
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


def render_deep_analysis(deep: dict):
    """深度分析：公司画像 + 财务健康扫描 + 发展阶段"""
    if not deep:
        st.info("暂无深度分析数据")
        return

    profile = deep.get("profile", {})
    health = deep.get("financial_health", {})
    stage = deep.get("stage", {})
    verdict = deep.get("verdict", "")

    # 综合判断
    if verdict:
        st.markdown(f"""
        <div style="background:#e3f2fd; border-left:4px solid #1a73e8; padding:12px;
                    border-radius:8px; margin:8px 0;">
            <strong>综合判断</strong><br>{verdict}
        </div>
        """, unsafe_allow_html=True)

    # 公司画像
    if profile:
        industry = profile.get("industry", "")
        cap = profile.get("market_cap")
        listed = profile.get("listed_date", "")
        cap_str = f"{cap/1e8:.0f}亿" if cap else "N/A"
        listed_str = str(listed)[:4] + "年上市" if listed and len(str(listed)) >= 4 else "N/A"

        st.caption("—— 公司画像 ——")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("行业", industry or "N/A")
        with c2:
            st.metric("市值", cap_str)
        with c3:
            st.metric("上市", listed_str)

        if health.get("cap_note"):
            st.caption(health["cap_note"])

    # 发展阶段
    if stage:
        stage_name = stage.get("stage", "")
        stage_colors = {"确认期": "#2e7d32", "验证期": "#43a047", "验证初期": "#66bb6a",
                        "故事期": "#f9a825", "收缩期": "#c62828", "成熟期": "#1565c0"}
        color = stage_colors.get(stage_name, "#666")
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin:8px 0;">
            <span style="background:{color}; color:white; padding:2px 10px; border-radius:12px;
                         font-size:14px; font-weight:bold;">{stage_name}</span>
            <span style="color:#555;">{stage.get('stage_desc', '')}</span>
        </div>
        """, unsafe_allow_html=True)

    # 财务健康扫描
    if health:
        h_score = health.get("health_score", 0)
        h_label = health.get("health", "")
        h_color = "#2e7d32" if h_score >= 60 else "#f9a825" if h_score >= 40 else "#c62828"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin:8px 0;">
            <span style="font-weight:bold;">财务健康度:</span>
            <span style="background:{h_color}; color:white; padding:2px 10px; border-radius:12px;
                         font-size:14px;">{h_label} {h_score}分</span>
            <span style="color:#555;">{health.get('health_desc', '')}</span>
        </div>
        """, unsafe_allow_html=True)

        # 各项检查
        checks = health.get("checks", [])
        if checks:
            good_count = sum(1 for c in checks if c["good"])
            total = len(checks)
            st.caption(f"财务检查通过 {good_count}/{total}")
            for c in checks:
                icon = "✅" if c["good"] else "⚠️"
                detail = c.get("detail", "")
                st.markdown(f"{icon} **{c['item']}**: {c['status']} ({detail})")

    # 亮点
    highlights = health.get("highlights", [])
    if highlights:
        st.markdown("**亮点**")
        for h in highlights:
            st.markdown(f"✅ {h}")

    # 风险
    warnings = health.get("warnings", [])
    if warnings:
        st.markdown("**风险提示**")
        for w in warnings:
            st.markdown(f"⚠️ {w}")


def render_deep_insights_combined(deep: dict, insights: dict):
    """合并显示预计算洞察和自动深度分析"""
    has_precomputed = bool(
        insights.get("verdict_override") or
        insights.get("deep_highlights") or
        insights.get("deep_risks")
    )

    if has_precomputed:
        # 有研报级深度洞察 → 重点展示
        render_insights(insights)
        # 自动基础分析折叠到底部，不抢焦点
        with st.expander("基础财务分析（自动生成）", expanded=False):
            render_deep_analysis(deep)
    else:
        # 无研报 → 显示自动分析 + 提示
        st.info("深度研报撰写中，以下为基础财务分析")
        render_deep_analysis(deep)


def render_insights(insights: dict):
    """深度洞察（研报级，预计算）"""
    verdict = insights.get("verdict_override", "")
    highlights = insights.get("deep_highlights", [])
    risks = insights.get("deep_risks", [])

    if not verdict and not highlights and not risks:
        return

    # 综合判断 — 最醒目的蓝色卡片
    if verdict:
        st.markdown(f"""
        <div style="background:#e8f0fe; border-left:5px solid #1a73e8; padding:14px;
                    border-radius:8px; margin:8px 0 16px 0;">
            <div style="font-size:13px; color:#1a73e8; font-weight:bold; margin-bottom:4px;">
                综合判断</div>
            <div style="font-size:15px; color:#212121; line-height:1.6;">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)

    # 核心亮点
    if highlights:
        st.markdown("**核心亮点**")
        for h in highlights:
            st.markdown(f"""
            <div style="font-size:13px; margin:3px 0 3px 12px; color:#2e7d32;">
                ▸ {h}
            </div>
            """, unsafe_allow_html=True)

    # 隐蔽风险
    if risks:
        st.markdown("<br>**隐蔽风险**", unsafe_allow_html=True)
        for r in risks:
            st.markdown(f"""
            <div style="font-size:13px; margin:3px 0 3px 12px; color:#c62828;">
                ▸ {r}
            </div>
            """, unsafe_allow_html=True)


def render_kline_chart(chart_data: list, stock_name: str = ""):
    """Plotly K线图 + MA5/MA20/MA60 + ZXDKX"""
    if not chart_data or len(chart_data) < 10:
        st.info("K线数据不足")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.info("Plotly 未安装，无法显示K线图")
        return

    dates = [d["date"] for d in chart_data]
    opens = [d["open"] for d in chart_data]
    highs = [d["high"] for d in chart_data]
    lows = [d["low"] for d in chart_data]
    closes = [d["close"] for d in chart_data]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # K线
    fig.add_trace(
        go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes,
            name="K线",
            increasing_line_color="#ef5350", decreasing_line_color="#26a69a",
        ),
        row=1, col=1,
    )

    # 均线 + 知行线
    ma_configs = [
        ("ma5", "#FFC107", "MA5"),
        ("ma20", "#9C27B0", "MA20"),
        ("ma60", "#2196F3", "MA60"),
        ("zxdkx", "#F44336", "ZXDKX"),
    ]
    for key, color, name in ma_configs:
        vals = [d.get(key) for d in chart_data]
        if any(v is not None for v in vals):
            fig.add_trace(
                go.Scatter(x=dates, y=vals, name=name,
                           line=dict(color=color, width=1.5 if name == "ZXDKX" else 1),
                           legendgroup=name),
                row=1, col=1,
            )

    # 成交量柱
    volumes = [d["volume"] for d in chart_data]
    colors = ["#ef5350" if closes[i] >= opens[i] else "#26a69a" for i in range(len(dates))]
    fig.add_trace(
        go.Bar(x=dates, y=volumes, name="量", marker_color=colors,
               opacity=0.5, showlegend=False),
        row=2, col=1,
    )

    # 布局
    fig.update_layout(
        title=f"{stock_name} 最近 60 日",
        height=420,
        margin=dict(l=0, r=0, t=35, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(size=11),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title_text="价格", row=1, col=1, showgrid=True, gridcolor="#eee")
    fig.update_yaxes(title_text="成交量", row=2, col=1, showgrid=False)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_peer_comparison(peers: list, current_code: str, market_state: dict, fund_all: dict):
    """同行业对比表"""
    if not peers:
        return

    st.caption("—— 同行业对比 ——")

    rows = []
    codes_to_fetch = [current_code] + [p for p in peers if p != current_code][:4]

    for code in codes_to_fetch:
        fin = fund_all.get(code, {}) if fund_all else {}
        rev_g = _safe_float(fin.get("revenue_growth"))
        margin = _safe_float(fin.get("gross_margin"))
        roe = _safe_float(fin.get("roe"))
        prf_g = _safe_float(fin.get("profit_growth"))

        rows.append({
            "代码": code,
            "营收增速": f"{rev_g:+.1f}%" if rev_g is not None else "N/A",
            "毛利率": f"{margin:.1f}%" if margin is not None else "N/A",
            "ROE": f"{roe:.1f}%" if roe is not None else "N/A",
            "利润增速": f"{prf_g:+.1f}%" if prf_g is not None else "N/A",
        })

    if len(rows) <= 1:
        return

    import pandas as pd
    df = pd.DataFrame(rows)

    # 高亮当前股票行
    def highlight_current(row):
        if row["代码"] == current_code:
            return ["background-color: #e3f2fd; font-weight: bold"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(highlight_current, axis=1)
    st.dataframe(styled, hide_index=True, use_container_width=True)


def render_deep_research_v2(result: dict):
    """五维度深度研究结果"""
    if not result or not result.get("dimensions"):
        return

    overall = result["overall_level"]
    colors = {"green": ("#2e7d32", "#e8f5e9"), "yellow": ("#f57f17", "#fff8e1"),
              "red": ("#c62828", "#ffebee")}
    border, bg = colors.get(overall, colors["green"])

    st.markdown(f"""
    <div style="background:{bg}; border-left:5px solid {border}; padding:12px;
                border-radius:8px; margin:8px 0 16px 0;">
        <div style="font-size:13px; color:{border}; font-weight:bold;">
            {overall.upper()} | {result['verdict']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Warnings first
    warnings = result.get("warnings", [])
    if warnings:
        with st.expander(f"风险信号 ({len(warnings)})", expanded=True):
            for w in warnings:
                st.markdown(f"""
                <div style="font-size:13px; margin:2px 0; color:#c62828;">{w}</div>
                """, unsafe_allow_html=True)

    # Dimensions
    cols = st.columns(2)
    for i, dim in enumerate(result["dimensions"]):
        level = dim["level"]
        icons = {"green": "OK", "yellow": "--", "red": "!!"}
        icon = icons.get(level, "--")
        with cols[i % 2]:
            with st.container(border=True):
                st.caption(f"[{icon}] {dim['dimension']}: {dim['summary']}")
                for sig in dim.get("signals", [])[:3]:
                    st.caption(f"  {sig}")


def render_intraday_chart(code: str, stock_name: str = ""):
    """Plotly 日内分时图（1分钟K线）"""
    from real_time import get_intraday_chart
    candles = get_intraday_chart(code, period="1")
    if not candles or len(candles) < 5:
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    times = [c["time"] for c in candles]
    closes = [c["close"] for c in candles]
    prev_close = candles[0]["open"] if candles else closes[0]

    fig = go.Figure()

    # 分时线
    fig.add_trace(go.Scatter(
        x=times, y=closes, name="价格",
        line=dict(color="#e53935", width=1.5),
        mode="lines",
    ))

    # 昨收参考线
    fig.add_hline(y=prev_close, line_dash="dash", line_color="#999",
                  annotation_text=f"昨收 {prev_close:.2f}")

    # Y 轴着色：相对昨收红涨绿跌
    y_min = min(closes) * 0.998
    y_max = max(closes) * 1.002

    fig.update_layout(
        title=f"{stock_name} 今日分时图",
        height=300,
        margin=dict(l=0, r=0, t=35, b=0),
        font=dict(size=11),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        xaxis=dict(showgrid=False, dtick=30),
        yaxis=dict(showgrid=True, gridcolor="#eee", range=[y_min, y_max]),
        showlegend=False,
    )

    chg_pct = (closes[-1] - prev_close) / prev_close * 100 if prev_close else 0
    color = "#e53935" if chg_pct >= 0 else "#26a69a"
    fig.add_annotation(
        x=times[-1], y=closes[-1],
        text=f"{closes[-1]:.2f} ({chg_pct:+.2f}%)",
        showarrow=True, arrowhead=0, ax=40, ay=0,
        font=dict(color=color, size=12, family="monospace"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_signal_badge(signals):
    """买卖信号标签"""
    if signals is None:
        return

    buy = signals.get("buy", [])
    sell = signals.get("sell", [])

    if signals.get("neutral", True):
        return

    if sell:
        st.markdown("### 卖点信号")
        for s in sell:
            color = "#c62828" if s["strength"] == "strong" else "#ef5350"
            bg = "#ffebee" if s["strength"] == "strong" else "#fff5f5"
            st.markdown(f"""
            <div style='background:{bg}; border-left:4px solid {color}; padding:8px 12px;
                        border-radius:6px; margin:4px 0;'>
                <span style='color:{color}; font-weight:bold; font-size:14px;'>{s['type']}</span>
                <span style='color:#666; font-size:12px; margin-left:8px;'>{s['detail']}</span>
            </div>
            """, unsafe_allow_html=True)

    if buy:
        st.markdown("### 买点信号")
        for s in buy:
            color = "#2e7d32" if s["strength"] == "strong" else "#66bb6a"
            bg = "#e8f5e9" if s["strength"] == "strong" else "#f1f8e9"
            st.markdown(f"""
            <div style='background:{bg}; border-left:4px solid {color}; padding:8px 12px;
                        border-radius:6px; margin:4px 0;'>
                <span style='color:{color}; font-weight:bold; font-size:14px;'>{s['type']}</span>
                <span style='color:#666; font-size:12px; margin-left:8px;'>{s['detail']}</span>
            </div>
            """, unsafe_allow_html=True)


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def render_footer(meta: dict):
    """页脚"""
    st.divider()
    st.caption(
        f"数据日期: {meta['last_date']} | "
        f"数据源: baostock / akshare | "
        f"仅供参考，不构成投资建议"
    )
