"""
父母选股 Web 应用 — Streamlit 入口
手机优先的简洁界面：输入代码 → 深度分析
"""
import streamlit as st
import sys
import os
import traceback
from datetime import datetime

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
_ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stock_selection")
_LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

# 优先当前目录（部署模式），然后 stock_selection（本地开发模式）
if os.path.exists(os.path.join(_LOCAL_PATH, "enhanced_fetcher.py")):
    sys.path.insert(0, _LOCAL_PATH)
elif os.path.isdir(_ENGINE_PATH):
    sys.path.insert(0, _ENGINE_PATH)


# ── 缓存数据 ──

@st.cache_data(ttl=86400, show_spinner=False)
def cached_fundamentals():
    """全 A 股基本面，24小时缓存"""
    from stock_analyzer import fetch_fundamentals_all
    return fetch_fundamentals_all()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_deep_insights():
    """深度洞察，1小时缓存"""
    from stock_analyzer import load_deep_insights
    return load_deep_insights()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_market_state():
    """市场状态，30分钟缓存"""
    from stock_analyzer import detect_market_state
    return detect_market_state()


@st.cache_data(ttl=86400, show_spinner=False)
def cached_analyzer():
    """缓存的 SingleStockAnalyzer 实例"""
    from stock_analyzer import SingleStockAnalyzer
    return SingleStockAnalyzer()


# ── 主页面 ──

def main():
    # 标题区
    st.title("选股分析")
    st.caption("输入 6 位股票代码，获取 B1 引擎深度分析")

    # 输入区
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        code = st.text_input(
            "股票代码",
            max_chars=6,
            placeholder="输入 6 位代码，如 000001",
            label_visibility="collapsed",
        )
        # 预设快捷按钮
    with col_btn:
        analyze_clicked = st.button("开始分析", type="primary", use_container_width=True)

    # 常用代码快捷输入
    quick_codes = st.pills(
        "快捷代码",
        options=["000001", "600519", "300750", "600036", "000858", "002415"],
        format_func=lambda x: f"{x}",
        label_visibility="collapsed",
    )
    if quick_codes and not analyze_clicked:
        code = quick_codes
        analyze_clicked = True

    if not analyze_clicked or not code:
        st.info("输入股票代码后点击「开始分析」")
        return

    # 清洗代码
    code = str(code).strip().zfill(6)

    # 验证
    if not code.isdigit() or len(code) != 6:
        st.error("请输入有效的 6 位数字代码")
        return
    if not code.startswith(("0", "2", "3", "6")):
        st.warning("代码前缀非 0/2/3/6，可能非 A 股")

    # 分析流程
    status = st.status(f"正在分析 {code}...", expanded=True)

    try:
        # 获取缓存数据
        status.write("加载市场数据...")
        market = cached_market_state()
        fundamentals_all = cached_fundamentals()
        deep_insights_all = cached_deep_insights()
        analyzer = cached_analyzer()

        status.write("获取行情数据...")
        result = analyzer.analyze(
            code,
            fundamentals=fundamentals_all.get(code, {}),
            deep_insights=deep_insights_all.get(code, {}),
            market_state=market,
        )

        if result.get("error"):
            status.update(label=f"分析失败", state="error")
            st.error(result["error"])
            return

        status.update(label="分析完成", state="complete")

        # ── 展示结果 ──
        from display import (
            render_price_card, render_star_verdict, render_risk_warnings,
            render_scores, render_b1_conditions, render_indicators_table,
            render_price_levels, render_returns, render_fundamentals,
            render_insights, render_footer,
        )

        st.divider()

        # 1. 概要
        render_price_card(result["meta"], result["price"], result["market"])
        st.divider()

        # 2. 星级判定
        render_star_verdict(result["star"], result["operation"])

        # 3. 风险警告
        render_risk_warnings(result["risks"], ret60=result["returns"].get("60d"))

        # 4. 双档评分
        render_scores(result["scores"])

        st.divider()

        # 5. B1 核心条件（折叠默认展开，因为重要）
        with st.expander("B1 核心五条件", expanded=True):
            render_b1_conditions(result["b1_conditions"])

        # 6. 技术指标
        render_indicators_table(result["indicators"])

        st.divider()

        # 7. 关键价位
        render_price_levels(result["price_levels"], result["price"]["close"])

        # 8. 区间收益
        render_returns(result["returns"])

        st.divider()

        # 9. 基本面（折叠）
        with st.expander("基本面数据", expanded=False):
            render_fundamentals(result["fundamentals"])

        # 10. 深度洞察（折叠）
        insights = result.get("insights", {})
        if insights.get("verdict_override") or insights.get("deep_highlights"):
            with st.expander("深度洞察", expanded=False):
                render_insights(insights)

        # 11. 页脚
        render_footer(result["meta"])

    except Exception as e:
        status.update(label="分析出错", state="error")
        st.error(f"系统错误: {e}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
