"""
父母选股 Web 应用 — Streamlit 入口
手机优先：智能搜索 → 深度分析
"""
import streamlit as st
import sys
import os
import json
import traceback

# ── 页面配置（必须第一个 st 调用） ──
st.set_page_config(
    page_title="选股分析",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "B1 选股引擎 · 仅供父母参考使用",
    },
)

# ── 导入引擎路径 ──
_LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LOCAL_PATH)


# ═══════════════════════════════════════════
# 缓存数据层
# ═══════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner="正在加载股票列表...")
def build_search_index():
    """构建全 A 股搜索索引（从本地文件，秒出）"""
    import json
    from collections import defaultdict

    # 从本地 JSON 加载（不依赖网络）
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_names.json")
    try:
        with open(json_path, encoding="utf-8") as f:
            name_map = json.load(f)
    except Exception:
        return {}

    index = defaultdict(list)
    for code, name in name_map.items():
        entry = (code, name)
        # 代码匹配
        index[code].append(entry)
        # 名称全匹配
        index[name].append(entry)
        # 名称单字拆解（搜"茅"→ 找到茅台）
        for char in name:
            if char.strip():
                index[char].append(entry)

    return {k: list(set(v)) for k, v in index.items()}


@st.cache_data(ttl=5, show_spinner=False)
def cached_realtime_quote(code: str):
    """实时行情，短 TTL（5 秒）"""
    from real_time import get_realtime_quote
    return get_realtime_quote(code)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_fundamentals():
    from stock_analyzer import fetch_fundamentals_all
    return fetch_fundamentals_all()


@st.cache_data(ttl=86400, show_spinner=False)
def cached_deep_research(code: str):
    """深度洞察：已有返回缓存，无则自动挖掘研报+主营+财务"""
    from deep_research import get_or_research
    return get_or_research(code)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_market_state():
    from stock_analyzer import detect_market_state
    return detect_market_state()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analyze(code: str, _market: str, _fund_key: str, _insight_key: str):
    from stock_analyzer import SingleStockAnalyzer, fetch_fundamentals_all, detect_market_state
    import gc

    analyzer = SingleStockAnalyzer()
    fundamentals_all = fetch_fundamentals_all()
    market = detect_market_state(analyzer.fetcher)

    result = analyzer.analyze(
        code,
        fundamentals=fundamentals_all.get(code, {}),
        deep_insights={},  # 深度洞察独立获取，不阻塞主分析
        market_state=market,
    )
    del analyzer
    gc.collect()
    return result


# ═══════════════════════════════════════════
# 全市场发现页
# ═══════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="正在扫描全市场...")
def _cached_scanner():
    from auto_scanner import run_scan
    return run_scan()


def render_discovery_page():
    st.caption("全市场 Q1 暴增筛选 → 研报聚合 → 异常检测")
    st.divider()

    # 自动加载（首次 2-3 分钟，缓存 1 小时后秒出）
    data = _cached_scanner()

    if "error" in data:
        st.warning(data["error"])
        return

    st.success(f"扫描完成 — {data['date']} — 全市场过滤出 {data['total_filtered']} 只")

    pool = data.get("candidates", [])
    if not pool:
        st.info("无符合条件标的")
        return

    # 表格展示 Top 30
    import pandas as pd
    df = pd.DataFrame(pool[:30])
    cols = ["代码", "名称", "行业", "利润增速%", "毛利率%", "ROE%", "估市值(亿)", "研报共识PE", "研报数", "信号分"]
    df_show = df[[c for c in cols if c in df.columns]].copy()
    if "利润增速%" in df_show.columns:
        df_show["利润增速%"] = df_show["利润增速%"].round(0).astype(int)
    if "毛利率%" in df_show.columns:
        df_show["毛利率%"] = df_show["毛利率%"].round(1)
    if "信号分" in df_show.columns:
        df_show["信号分"] = df_show["信号分"].astype(int)

    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        height=800,
        column_config={
            "代码": st.column_config.TextColumn(width="small"),
            "名称": st.column_config.TextColumn(width="small"),
        },
    )

    st.caption(f"候选池已缓存 1 小时，{len(pool)} 只股票 | 下一步：点击代码复制，到搜索页查看深度分析")


# ═══════════════════════════════════════════
# 会话状态
# ═══════════════════════════════════════════

def init_session():
    """初始化 st.session_state"""
    defaults = {
        "favorites": [],      # [(code, name), ...]
        "history": [],        # [(code, name), ...]  最近 10 条
        "analyze_code": "",   # 触发分析的代码
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_to_history(code: str, name: str):
    """记录查询历史（最近 10 条，去重）"""
    history = st.session_state.history
    # 移除旧记录
    history = [(c, n) for c, n in history if c != code]
    history.insert(0, (code, name))
    st.session_state.history = history[:10]


def toggle_favorite(code: str, name: str):
    """切换自选状态"""
    favs = st.session_state.favorites
    existing = [c for c, _ in favs if c == code]
    if existing:
        st.session_state.favorites = [(c, n) for c, n in favs if c != code]
    else:
        favs.append((code, name))
        st.session_state.favorites = favs


def is_favorite(code: str) -> bool:
    return any(c == code for c, _ in st.session_state.favorites)


# ═══════════════════════════════════════════
# 搜索组件
# ═══════════════════════════════════════════

def render_search_box():
    """智能搜索框 + 自选股 + 历史记录"""
    # 自选股快捷栏
    favs = st.session_state.favorites
    if favs:
        fav_opts = [f"{c} {n}" for c, n in favs]
        picked = st.pills(
            "自选股", options=fav_opts,
            label_visibility="collapsed",
            key="fav_pills",
        )
        if picked:
            st.session_state.analyze_code = picked.split()[0]

    # 搜索输入
    query = st.text_input(
        "搜索股票",
        placeholder="输入代码、名称或关键字，如 茅台 / 000001",
        label_visibility="collapsed",
        key="search_query",
    )

    analyze_clicked = False
    code_to_analyze = ""

    if query:
        query = query.strip()
        # 纯数字 → 直接当作代码
        if query.isdigit() and len(query) == 6:
            code_to_analyze = query
        elif query.isdigit() and len(query) < 6:
            code_to_analyze = query.zfill(6)
        else:
            # 搜索匹配
            idx = build_search_index()
            matches = idx.get(query, [])
            # 也尝试部分匹配
            if not matches:
                all_matches = []
                for k, v in idx.items():
                    if query.lower() in k.lower():
                        all_matches.extend(v)
                # 去重
                seen = set()
                matches = []
                for c, n in all_matches:
                    if c not in seen:
                        seen.add(c)
                        matches.append((c, n))
                matches = matches[:20]

            if len(matches) == 1:
                code_to_analyze = matches[0][0]
            elif len(matches) > 1:
                # 显示匹配列表供选择
                st.caption(f"找到 {len(matches)} 个匹配")
                cols = st.columns(3)
                for i, (c, n) in enumerate(matches[:15]):
                    with cols[i % 3]:
                        if st.button(f"{c}\n{n}", key=f"match_{c}", use_container_width=True):
                            code_to_analyze = c
            elif matches == [] and query:
                st.caption("无匹配结果")

    # 分析按钮
    _, col2 = st.columns([3, 1])
    with col2:
        if st.button("开始分析", type="primary", use_container_width=True):
            if code_to_analyze:
                analyze_clicked = True
            elif query and query.isdigit():
                code_to_analyze = query.zfill(6)
                analyze_clicked = True

    # 历史记录
    history = st.session_state.history
    if history and not query:
        st.caption("最近查询")
        hist_cols = st.columns(min(len(history), 5))
        for i, (c, n) in enumerate(history[:5]):
            with hist_cols[i]:
                if st.button(f"{n}", key=f"hist_{c}", use_container_width=True, help=c):
                    st.session_state.analyze_code = c

    return analyze_clicked, code_to_analyze


# ═══════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════

def main():
    init_session()

    st.title("B1 选股平台")
    page = st.radio("", ["搜索个股", "全市场发现"], horizontal=True, label_visibility="collapsed")

    if page == "全市场发现":
        render_discovery_page()
        return

    st.caption("搜索股票 → 深度分析")

    # 检查是否有从快捷入口传来的代码
    if st.session_state.analyze_code:
        analyze_clicked = True
        code = st.session_state.analyze_code
        st.session_state.analyze_code = ""
    else:
        analyze_clicked, code = render_search_box()

    if not analyze_clicked or not code:
        return

    # 清洗代码
    code = str(code).strip().zfill(6)
    if not code.isdigit() or len(code) != 6:
        st.error("无效代码")
        return

    # 分析流程
    status = st.status(f"正在分析 {code}...", expanded=True)

    try:
        # 实时行情（优先获取，与主分析并行）
        status.write("获取实时行情...")
        realtime = cached_realtime_quote(code)

        market = cached_market_state()
        status.write("正在分析（首次较慢，再次查询秒出）...")

        result = cached_analyze(
            code,
            _market=market.get("status", ""),
            _fund_key=str(len(cached_fundamentals())),
            _insight_key=str(len(cached_deep_research(code))),
        )

        if result.get("error"):
            status.update(label="分析失败", state="error")
            st.error(result["error"])
            return

        status.update(label="分析完成", state="complete")

        # 独立获取深度洞察（不阻塞主分析，失败不影响展示）
        try:
            deep_insights = cached_deep_research(code)
        except Exception:
            deep_insights = {}

        # 五维度深度研究 v2
        @st.cache_data(ttl=3600, show_spinner=False)
        def _cached_research_v2(code, fund_json, b1_json):
            from deep_research_v2 import deep_research as dr_v2
            fund = json.loads(fund_json) if fund_json else {}
            b1 = json.loads(b1_json) if b1_json else {}
            return dr_v2(code, fundamentals=fund, b1_result=b1)

        try:
            research_v2 = _cached_research_v2(
                code,
                fund_json=json.dumps(cached_fundamentals().get(code, {}), default=str),
                b1_json=json.dumps(result, default=str),
            )
        except Exception:
            research_v2 = None

        # 记录历史
        add_to_history(code, result["meta"]["name"])

        # ── 展示结果 ──
        from display import (
            render_price_card, render_star_verdict, render_risk_warnings,
            render_scores, render_b1_conditions, render_indicators_table,
            render_price_levels, render_returns, render_fundamentals,
            render_deep_insights_combined, render_deep_research_v2,
            render_intraday_chart, render_kline_chart,
            render_peer_comparison, render_footer,
        )

        st.divider()

        # 自选按钮 + 概要
        is_fav = is_favorite(code)
        fav_label = "⭐ 取消自选" if is_fav else "☆ 加入自选"
        if st.button(fav_label, key=f"fav_{code}"):
            toggle_favorite(code, result["meta"]["name"])
            st.rerun()

        render_price_card(result["meta"], result["price"], result["market"], realtime=realtime)
        st.divider()

        # 星级判定
        render_star_verdict(result["star"], result["operation"])

        # 风险警告
        render_risk_warnings(result["risks"], ret60=result["returns"].get("60d"))

        # 双档评分
        render_scores(result["scores"])

        st.divider()

        # 深度分析
        st.subheader("深度分析")
        render_deep_insights_combined(result.get("deep", {}), deep_insights)

        # 五维度结构化研究 v2
        if research_v2:
            render_deep_research_v2(research_v2)

        st.divider()

        # B1 核心条件
        with st.expander("B1 核心五条件", expanded=True):
            render_b1_conditions(result["b1_conditions"])

        # 技术指标
        render_indicators_table(result["indicators"])

        st.divider()

        # 关键价位
        render_price_levels(result["price_levels"], result["price"]["close"])

        # 区间收益
        render_returns(result["returns"])

        st.divider()

        # 日内分时图
        render_intraday_chart(code, result["meta"]["name"])

        # K线图
        chart_data = result.get("chart_data", [])
        if chart_data:
            render_kline_chart(chart_data, result["meta"]["name"])

        st.divider()

        # 同行业对比
        peers = result.get("peers", [])
        if peers:
            render_peer_comparison(peers, code, result.get("market", {}), cached_fundamentals())

        # 基本面（折叠）
        with st.expander("基本面数据", expanded=False):
            render_fundamentals(result["fundamentals"])

        # 页脚
        render_footer(result["meta"])

    except Exception as e:
        status.update(label="分析出错", state="error")
        st.error(f"系统错误: {e}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
