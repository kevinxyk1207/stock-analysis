"""
每日收盘自动更新 + 选股输出
下午3点后运行：更新数据缓存 → 跑选股 → 保存结果
"""
import sys, os, logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 50)
    logger.info("每日收盘自动更新开始")
    logger.info("=" * 50)

    # 1. 增量更新缓存
    logger.info("更新数据缓存...")
    from enhanced_fetcher import EnhancedStockFetcher
    fetcher = EnhancedStockFetcher()
    stats = fetcher.update_cache()

    if not stats:
        logger.info("缓存已是最新，无需更新")
    else:
        logger.info(f"更新完成: {len(stats)} 只有新数据")

    # 2. 获取最新成分股列表，补充新纳入的股票
    logger.info("检查成分股变动...")
    try:
        csi800 = fetcher.get_csi800_stocks()
        zz1000 = fetcher.get_zz1000_stocks()
        all_symbols = set(csi800['code'].tolist()) | set(zz1000['code'].tolist())
        # 补充重点监控股
        for w in ['301611', '600869', '605358']:
            all_symbols.add(w)

        cached = set(f.replace('.csv', '') for f in os.listdir(fetcher.cache_dir)
                     if f.endswith('.csv'))
        new_symbols = all_symbols - cached
        if new_symbols:
            logger.info(f"发现 {len(new_symbols)} 只新成分股，开始下载...")
            fetcher.build_cache(symbols=list(new_symbols))
        else:
            logger.info("成分股无变化")
    except Exception as e:
        logger.warning(f"检查成分股失败: {e}")

    # 3. 市场环境判断
    logger.info("检测市场环境...")
    try:
        from enhanced_fetcher import load_cache_data
        import pandas as pd
        stock_data = load_cache_data(min_rows=200, common_range=False)
        if stock_data:
            close_dict = {}
            for sym, df in stock_data.items():
                close_dict[sym] = pd.Series(df['close'].values)
            market_close = pd.DataFrame(close_dict).mean(axis=1)
            market_ma60 = market_close.rolling(60, min_periods=60).mean()
            if len(market_ma60) > 0 and pd.notna(market_ma60.iloc[-1]):
                is_bear = market_close.iloc[-1] < market_ma60.iloc[-1]
                if is_bear:
                    logger.warning("!" * 55)
                    logger.warning("!!!  市场处于熊市 (均价<MA60) — 建议空仓，不买入  !!!")
                    logger.warning("!" * 55)
                else:
                    logger.info("市场处于牛市 (均价>MA60)，正常选股")
    except Exception as e:
        logger.warning(f"市场检测失败: {e}")

    # 4. 跑选股
    logger.info("运行选股程序...")
    from screen_today import main as screen_main
    try:
        screen_main()
        logger.info("选股完成")
    except Exception as e:
        logger.error(f"选股失败: {e}")
        import traceback
        traceback.print_exc()

    # 5. 生成综合日报
    logger.info("生成综合日报...")
    from daily_report import generate_report
    try:
        generate_report()
        logger.info("日报生成完成")
    except Exception as e:
        logger.error(f"日报生成失败: {e}")

    # 5. 纸上跟踪持仓状态
    logger.info("纸上跟踪持仓...")
    from paper_tracker import main as tracker_main
    try:
        tracker_main()
        logger.info("跟踪完成")
    except Exception as e:
        logger.error(f"跟踪失败: {e}")

    logger.info("=" * 50)
    logger.info("每日更新完成")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
