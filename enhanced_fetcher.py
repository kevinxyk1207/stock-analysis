"""增强版数据获取模块"""
import pandas as pd
import numpy as np
import logging
import time
import os
import sys
import json
from typing import Optional, List, Dict, Tuple
import random
import requests
from datetime import datetime, timedelta

# 彻底禁用代理
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

logger = logging.getLogger(__name__)


class EnhancedStockFetcher:
    """增强版股票数据获取器（Baostock + 本地缓存）"""

    def __init__(self, max_retries: int = 3, retry_delay: int = 2,
                 cache_dir: str = None):
        """
        Args:
            cache_dir: 缓存目录，默认 data_cache/
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        # Baostock 登录
        self._bs_logged_in = False
        self._init_baostock()

        # 检测可用数据源（baostock 排最前）
        self.available_sources = self._detect_sources()
        logger.info(f"可用数据源: {self.available_sources}")

    def _init_baostock(self):
        """初始化 baostock 连接"""
        try:
            import baostock as bs
            lg = bs.login()
            if lg.error_code == '0':
                self._bs_logged_in = True
                self._bs = bs
                logger.info("Baostock 登录成功")
            else:
                logger.warning(f"Baostock 登录失败: {lg.error_msg}")
        except Exception as e:
            logger.debug(f"Baostock 初始化失败: {e}")

    def _bs_ensure_login(self):
        """确保 baostock 已登录"""
        if not self._bs_logged_in:
            self._init_baostock()
        return self._bs_logged_in

    def _bs_code(self, symbol: str) -> str:
        """转换股票代码到 baostock 格式: '000001' → 'sz.000001'"""
        if '.' in symbol:  # 已经是 baostock 格式
            return symbol
        if symbol.startswith(('6', '9')):
            return f"sh.{symbol}"
        else:
            return f"sz.{symbol}"

    def _is_stock_code(self, bs_code: str) -> bool:
        """判断是否为A股股票（过滤指数、ETF等）"""
        parts = bs_code.split('.')
        if len(parts) != 2:
            return False
        market, code = parts
        if market not in ('sh', 'sz'):
            return False
        # 上证A股: 6xxxxx (含688科创板)
        # 上证B股: 9xxxxx
        # 深证主板: 00xxxx, 20xxxx
        # 创业板: 30xxxx
        if market == 'sh' and code.startswith('6'):
            return True
        if market == 'sz' and code.startswith(('0', '2', '3')):
            return True
        return False

    def _get_cached_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从本地缓存读取数据

        Returns:
            缓存中指定日期范围的DataFrame，无缓存则返回空DataFrame
        """
        cache_path = os.path.join(self.cache_dir, f"{symbol}.csv")
        if not os.path.exists(cache_path):
            return pd.DataFrame()

        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if df.empty:
                return df

            # 过滤日期范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            return df

        except Exception as e:
            logger.debug(f"读取缓存失败 {symbol}: {e}")
            return pd.DataFrame()

    def _save_cached_data(self, symbol: str, df: pd.DataFrame):
        """保存数据到本地缓存（增量合并）"""
        if df.empty:
            return

        cache_path = os.path.join(self.cache_dir, f"{symbol}.csv")

        try:
            if os.path.exists(cache_path):
                existing = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                # 合并去重：新数据覆盖旧数据
                combined = pd.concat([existing, df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
            else:
                combined = df.copy()

            combined.to_csv(cache_path)
            logger.debug(f"缓存已保存: {symbol} ({len(combined)}条)")

        except Exception as e:
            logger.warning(f"保存缓存失败 {symbol}: {e}")

    def _detect_sources(self) -> List[str]:
        """检测可用的数据源"""
        sources = []

        # 1. baostock
        if self._bs_logged_in:
            sources.append('baostock')

        # 2. akshare
        try:
            import akshare as ak
            sources.append('akshare')
        except ImportError:
            logger.debug("akshare未安装")

        # 3. tushare（需要token）
        try:
            import tushare as ts
            if os.environ.get('TUSHARE_TOKEN'):
                sources.append('tushare')
            else:
                logger.debug("tushare已安装但未设置TUSHARE_TOKEN")
        except ImportError:
            logger.debug("tushare未安装")

        # 4. yfinance
        try:
            import yfinance as yf
            sources.append('yfinance')
        except ImportError:
            logger.debug("yfinance未安装")
        try:
            import akshare as ak
            sources.append('akshare')
        except ImportError:
            logger.debug("akshare未安装")

        return sources

    def get_stock_list(self, market: str = "A") -> pd.DataFrame:
        """
        获取股票列表

        Args:
            market: 市场类型，A表示A股

        Returns:
            股票列表DataFrame
        """
        # 优先使用baostock（最快，无token限制）
        if 'baostock' in self.available_sources:
            df = self._try_baostock_stock_list()
            if not df.empty:
                return df

        # 其次使用tushare
        if 'tushare' in self.available_sources:
            df = self._try_tushare_stock_list(market)
            if not df.empty:
                return df

        # 再其次使用akshare
        if 'akshare' in self.available_sources:
            df = self._try_akshare_stock_list(market)
            if not df.empty:
                return df

        # 最后生成模拟数据
        logger.warning("所有数据源均失败，使用模拟股票列表")
        return self._generate_mock_stock_list()

    def _try_baostock_stock_list(self) -> pd.DataFrame:
        """尝试使用baostock获取A股股票列表"""
        if not self._bs_ensure_login():
            return pd.DataFrame()

        # 从今天开始向前寻找最近的交易日
        for attempt in range(self.max_retries):
            try:
                query_date = datetime.now() - timedelta(days=attempt)
                date_str = query_date.strftime('%Y-%m-%d')
                rs = self._bs.query_all_stock(date_str)

                if rs.error_code != '0':
                    logger.warning(f"Baostock query_all_stock失败: {rs.error_msg}")
                    continue

                rows = []
                while rs.next():
                    code = rs.get_row_data()[0]
                    name = rs.get_row_data()[2]
                    if self._is_stock_code(code):
                        symbol = code.split('.')[1]
                        rows.append({'code': symbol, 'name': name})

                if rows:
                    df = pd.DataFrame(rows)
                    logger.info(f"使用baostock获取到 {len(df)} 只A股 (日期:{date_str})")
                    return df
                else:
                    logger.debug(f"日期{date_str}无股票数据（非交易日），向前尝试")

            except Exception as e:
                logger.debug(f"baostock获取股票列表失败（第{attempt+1}次尝试）: {e}")

        return pd.DataFrame()

    def _try_akshare_stock_list(self, market: str) -> pd.DataFrame:
        """尝试使用akshare获取股票列表，带代理绕过"""
        for attempt in range(self.max_retries):
            try:
                import akshare as ak

                # 设置自定义请求头
                import akshare as ak_module
                if hasattr(ak_module, 'stock_info_a_code_name'):
                    # 直接调用，让akshare处理
                    if market.upper() == "A":
                        stock_list = ak.stock_info_a_code_name()
                        logger.info(f"使用akshare获取到 {len(stock_list)} 只股票")
                        return stock_list

            except Exception as e:
                error_msg = str(e).lower()

                # 检查是否是代理错误
                if 'proxy' in error_msg or 'connection' in error_msg:
                    logger.warning(f"akshare代理错误，尝试绕过代理设置...")
                    # 尝试临时修改requests的session配置
                    try:
                        import akshare as ak
                        import requests

                        # 创建新的session，禁用代理
                        session = requests.Session()
                        session.trust_env = False  # 不读取系统代理设置

                        # 这里需要找到akshare内部使用session的方法
                        # 暂时跳过，使用下一个数据源
                        pass

                    except Exception as inner_e:
                        logger.debug(f"绕过代理失败: {inner_e}")

                if attempt < self.max_retries - 1:
                    logger.warning(f"akshare获取股票列表失败，第{attempt+1}次重试: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.warning(f"akshare获取股票列表失败，已重试{self.max_retries}次: {e}")

        return pd.DataFrame()

    def _try_tushare_stock_list(self, market: str) -> pd.DataFrame:
        """尝试使用tushare获取股票列表"""
        for attempt in range(self.max_retries):
            try:
                import tushare as ts

                token = os.environ.get('TUSHARE_TOKEN')
                if not token:
                    return pd.DataFrame()

                ts.set_token(token)
                pro = ts.pro_api()

                if market.upper() == "A":
                    df = pro.stock_basic(exchange='', list_status='L',
                                        fields='ts_code,symbol,name,area,industry,list_date')
                    # 重命名列
                    df = df.rename(columns={'ts_code': 'code', 'symbol': 'symbol', 'name': 'name'})
                    logger.info(f"使用tushare获取到 {len(df)} 只股票")
                    return df

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"tushare获取股票列表失败，第{attempt+1}次重试: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.warning(f"tushare获取股票列表失败，已重试{self.max_retries}次: {e}")

        return pd.DataFrame()

    def get_daily_data(self, symbol: str, start_date: str, end_date: str,
                       adjust: str = "qfq") -> pd.DataFrame:
        """
        获取股票日线数据

        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            adjust: 复权类型，qfq=前复权

        Returns:
            日线数据DataFrame
        """
        # 尝试从缓存读取
        cached = self._get_cached_data(symbol, start_date, end_date)
        if not cached.empty:
            logger.debug(f"使用缓存数据: {symbol} ({len(cached)}条)")
            return cached

        # 缓存未命中，尝试数据源（东方财富直接API优先，最快最稳定）
        sources_to_try = []
        sources_to_try.append(('em_kline', self._try_em_kline_daily))

        if 'baostock' in self.available_sources:
            sources_to_try.append(('baostock', self._try_baostock_daily))

        if 'akshare' in self.available_sources:
            sources_to_try.append(('akshare', self._try_akshare_daily))

        if 'tushare' in self.available_sources:
            sources_to_try.append(('tushare', self._try_tushare_daily))

        if 'yfinance' in self.available_sources:
            sources_to_try.append(('yfinance', self._try_yfinance_daily))

        for source_name, source_func in sources_to_try:
            df = source_func(symbol, start_date, end_date, adjust)
            if not df.empty:
                logger.info(f"使用{source_name}成功获取股票{symbol}数据，共{len(df)}条")
                # 保存到缓存
                self._save_cached_data(symbol, df)
                return df

        # 所有数据源都失败，生成模拟数据
        logger.warning(f"所有数据源均失败，为股票{symbol}生成模拟数据")
        return self._generate_mock_data(symbol, start_date, end_date)

    def _try_em_kline_daily(self, symbol: str, start_date: str,
                             end_date: str, adjust: str) -> pd.DataFrame:
        """使用东方财富 K 线 API 直接获取日线（最快、不需要登录）"""
        try:
            from real_time import get_daily_kline
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 30
            return get_daily_kline(symbol, days=days)
        except Exception:
            return pd.DataFrame()

    def _try_baostock_daily(self, symbol: str, start_date: str,
                            end_date: str, adjust: str) -> pd.DataFrame:
        """尝试使用baostock获取日线数据"""
        if not self._bs_ensure_login():
            return pd.DataFrame()

        for attempt in range(self.max_retries):
            try:
                bs_code = self._bs_code(symbol)

                start = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                end = pd.to_datetime(end_date).strftime('%Y-%m-%d')

                if adjust == "qfq":
                    adjustflag = "2"
                elif adjust == "hfq":
                    adjustflag = "1"
                else:
                    adjustflag = "3"

                fields = "date,open,high,low,close,volume,amount"
                rs = self._bs.query_history_k_data_plus(
                    bs_code, fields,
                    start_date=start, end_date=end,
                    frequency="d", adjustflag=adjustflag
                )

                if rs.error_code != '0':
                    logger.warning(f"Baostock查询失败: {rs.error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return pd.DataFrame()

                rows = []
                while rs.next():
                    row_data = rs.get_row_data()
                    try:
                        row = {
                            'date': row_data[0],
                            'open': float(row_data[1]) if row_data[1] else 0.0,
                            'high': float(row_data[2]) if row_data[2] else 0.0,
                            'low': float(row_data[3]) if row_data[3] else 0.0,
                            'close': float(row_data[4]) if row_data[4] else 0.0,
                            'volume': float(row_data[5]) if row_data[5] else 0.0,
                            'amount': float(row_data[6]) if row_data[6] else 0.0,
                        }
                        rows.append(row)
                    except ValueError:
                        continue  # 跳过格式异常的行

                if not rows:
                    return pd.DataFrame()

                df = pd.DataFrame(rows)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.astype(float)
                return df

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"baostock获取{symbol}失败，第{attempt+1}次重试: {e}")
                    time.sleep(self.retry_delay)
                else:
                    logger.warning(f"baostock获取{symbol}失败，已重试{self.max_retries}次: {e}")

        return pd.DataFrame()

    def _try_akshare_daily(self, symbol: str, start_date: str,
                          end_date: str, adjust: str) -> pd.DataFrame:
        """尝试使用akshare获取日线数据，带重试和代理处理"""
        for attempt in range(self.max_retries):
            try:
                import akshare as ak

                # 添加随机延迟，避免请求过快
                time.sleep(random.uniform(0.5, 1.5))

                df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                       start_date=start_date, end_date=end_date,
                                       adjust=adjust, timeout=30)

                if not df.empty:
                    # 重命名列
                    column_map = {
                        '日期': 'date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume',
                        '成交额': 'amount',
                        '振幅': 'amplitude',
                        '涨跌幅': 'pct_change',
                        '涨跌额': 'change',
                        '换手率': 'turnover'
                    }
                    df = df.rename(columns=column_map)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    return df
                else:
                    logger.warning(f"股票{symbol}数据为空")
                    return pd.DataFrame()

            except Exception as e:
                error_msg = str(e).lower()

                # 如果是代理错误，尝试特殊处理
                if 'proxy' in error_msg or 'connection' in error_msg:
                    logger.warning(f"akshare代理错误，尝试使用备用方法...")

                    # 尝试使用不同的akshare函数
                    try:
                        import akshare as ak

                        # 尝试使用stock_zh_a_hist_min_em（分钟线数据接口）
                        # 这个接口可能使用不同的后端
                        df = ak.stock_zh_a_hist_min_em(symbol=symbol, period="daily",
                                                      start_date=start_date.replace('-', ''),
                                                      end_date=end_date.replace('-', ''),
                                                      adjust=adjust, timeout=30)

                        if not df.empty:
                            # 处理返回的数据格式
                            column_map = {
                                '时间': 'date',
                                '开盘': 'open',
                                '收盘': 'close',
                                '最高': 'high',
                                '最低': 'low',
                                '成交量': 'volume',
                                '成交额': 'amount'
                            }
                            df = df.rename(columns=column_map)
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date')
                            logger.info(f"使用备用接口获取股票{symbol}数据成功")
                            return df

                    except Exception as alt_e:
                        logger.debug(f"备用接口也失败: {alt_e}")

                if attempt < self.max_retries - 1:
                    logger.warning(f"akshare获取股票{symbol}数据失败，第{attempt+1}次重试: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.warning(f"akshare获取股票{symbol}数据失败，已重试{self.max_retries}次: {e}")

        return pd.DataFrame()

    def _try_yfinance_daily(self, symbol: str, start_date: str,
                           end_date: str, adjust: str) -> pd.DataFrame:
        """尝试使用yfinance获取日线数据"""
        for attempt in range(self.max_retries):
            try:
                import yfinance as yf

                # A股代码转换
                if symbol.startswith('0') or symbol.startswith('3'):
                    yf_symbol = f"{symbol}.SZ"  # 深交所
                elif symbol.startswith('6'):
                    yf_symbol = f"{symbol}.SS"  # 上交所
                else:
                    yf_symbol = symbol

                # 转换日期格式
                start_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                end_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')

                # 下载数据
                stock = yf.download(yf_symbol, start=start_dt, end=end_dt, progress=False)

                if not stock.empty:
                    # 重命名列
                    stock = stock.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    return stock
                else:
                    logger.warning(f"股票{symbol}数据为空")
                    return pd.DataFrame()

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"yfinance获取股票{symbol}数据失败，第{attempt+1}次重试: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.warning(f"yfinance获取股票{symbol}数据失败，已重试{self.max_retries}次: {e}")

        return pd.DataFrame()

    def _try_tushare_daily(self, symbol: str, start_date: str,
                          end_date: str, adjust: str) -> pd.DataFrame:
        """尝试使用tushare获取日线数据"""
        for attempt in range(self.max_retries):
            try:
                import tushare as ts

                token = os.environ.get('TUSHARE_TOKEN')
                if not token:
                    return pd.DataFrame()

                ts.set_token(token)
                pro = ts.pro_api()

                # 添加随机延迟，避免请求过快
                time.sleep(random.uniform(0.5, 1.5))

                # 转换日期格式
                start_dt = pd.to_datetime(start_date).strftime('%Y%m%d')
                end_dt = pd.to_datetime(end_date).strftime('%Y%m%d')

                # 确定交易所后缀
                if symbol.startswith(('0', '3')):
                    ts_code = f"{symbol}.SZ"
                elif symbol.startswith('6'):
                    ts_code = f"{symbol}.SH"
                else:
                    ts_code = symbol

                # 记录复权类型（tushare免费版可能不支持复权）
                if adjust == "qfq":
                    logger.debug(f"请求前复权数据，但tushare免费版可能返回未复权数据")
                elif adjust == "hfq":
                    logger.debug(f"请求后复权数据，但tushare免费版可能返回未复权数据")

                # 获取数据（tushare免费版返回未复权数据）
                df = pro.daily(ts_code=ts_code, start_date=start_dt, end_date=end_dt)

                if not df.empty:
                    # 重命名和格式化
                    df = df.rename(columns={
                        'trade_date': 'date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'vol': 'volume',
                        'amount': 'amount'
                    })
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    df = df.sort_index()
                    return df
                else:
                    logger.warning(f"股票{symbol}数据为空")
                    return pd.DataFrame()

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"tushare获取股票{symbol}数据失败，第{attempt+1}次重试: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.warning(f"tushare获取股票{symbol}数据失败，已重试{self.max_retries}次: {e}")

        return pd.DataFrame()

    def _generate_mock_stock_list(self) -> pd.DataFrame:
        """生成模拟股票列表"""
        stocks = []
        for i in range(1, 101):
            stocks.append({
                'code': f'{i:06d}',
                'name': f'股票{i}'
            })
        return pd.DataFrame(stocks)

    def _generate_mock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟股票数据"""
        try:
            # 解析日期
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            # 生成日期范围
            dates = pd.date_range(start=start, end=end, freq='D')

            if len(dates) == 0:
                dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='D')

            days = len(dates)

            # 生成基础价格序列
            base_price = random.uniform(10, 100)
            price_series = base_price + np.cumsum(np.random.randn(days) * 2)
            price_series = np.abs(price_series)  # 确保价格为正

            # 生成数据
            data = pd.DataFrame({
                'open': price_series * (1 + np.random.randn(days) * 0.01),
                'high': price_series * (1 + np.random.randn(days) * 0.02),
                'low': price_series * (1 - np.random.randn(days) * 0.02),
                'close': price_series,
                'volume': np.random.randint(1000000, 10000000, days)
            }, index=dates)

            logger.info(f"为股票{symbol}生成{len(data)}条模拟数据")
            return data

        except Exception as e:
            logger.error(f"生成模拟数据失败: {e}")
            return pd.DataFrame()

    def get_multiple_stocks_data(self, symbols: List[str], start_date: str,
                                 end_date: str, adjust: str = "") -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型

        Returns:
            股票数据字典
        """
        stocks_data = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"正在获取股票 {symbol} ({i}/{total})...")
                df = self.get_daily_data(symbol, start_date, end_date, adjust)

                if not df.empty:
                    stocks_data[symbol] = df
                else:
                    logger.warning(f"股票{symbol}数据获取失败或为空")

            except Exception as e:
                logger.error(f"处理股票{symbol}时出错: {e}")
                continue

        return stocks_data

    def get_hs300_stocks(self) -> pd.DataFrame:
        """
        获取沪深300成分股列表（从baostock）

        Returns:
            DataFrame with columns ['code', 'name']
        """
        if not self._bs_ensure_login():
            logger.warning("baostock未登录，无法获取沪深300成分股")
            return pd.DataFrame()

        try:
            rs = self._bs.query_hs300_stocks()
            if rs.error_code != '0':
                logger.warning(f"获取沪深300失败: {rs.error_msg}")
                return pd.DataFrame()

            rows = []
            while rs.next():
                row = rs.get_row_data()
                # baostock返回格式: [日期, 代码, 名称]
                raw_code = row[1]
                name = row[2] if len(row) > 2 else ""
                # 去掉 sh./sz. 前缀
                if '.' in raw_code:
                    symbol = raw_code.split('.')[1]
                else:
                    symbol = raw_code
                rows.append({'code': symbol, 'name': name})

            df = pd.DataFrame(rows)
            logger.info(f"获取到 {len(df)} 只沪深300成分股")
            return df

        except Exception as e:
            logger.error(f"获取沪深300成分股失败: {e}")
            return pd.DataFrame()

    def get_zz500_stocks(self) -> pd.DataFrame:
        """
        获取中证500成分股列表（从baostock）

        Returns:
            DataFrame with columns ['code', 'name']
        """
        if not self._bs_ensure_login():
            logger.warning("baostock未登录，无法获取中证500成分股")
            return pd.DataFrame()

        try:
            rs = self._bs.query_zz500_stocks()
            if rs.error_code != '0':
                logger.warning(f"获取中证500失败: {rs.error_msg}")
                return pd.DataFrame()

            rows = []
            while rs.next():
                row = rs.get_row_data()
                raw_code = row[1]
                name = row[2] if len(row) > 2 else ""
                if '.' in raw_code:
                    symbol = raw_code.split('.')[1]
                else:
                    symbol = raw_code
                rows.append({'code': symbol, 'name': name})

            df = pd.DataFrame(rows)
            logger.info(f"获取到 {len(df)} 只中证500成分股")
            return df
        except Exception as e:
            logger.error(f"获取中证500成分股失败: {e}")
            return pd.DataFrame()

    def get_csi800_stocks(self) -> pd.DataFrame:
        """获取中证800成分股 = 沪深300 + 中证500（去重）"""
        hs300 = self.get_hs300_stocks()
        zz500 = self.get_zz500_stocks()
        combined = pd.concat([hs300, zz500], ignore_index=True)
        combined = combined.drop_duplicates(subset='code').reset_index(drop=True)
        logger.info(f"中证800成分股: {len(combined)} 只 (HS300 {len(hs300)} + ZZ500 {len(zz500)})")
        return combined

    def get_zz1000_stocks(self) -> pd.DataFrame:
        """
        获取中证1000成分股（从akshare）

        Returns:
            DataFrame with columns ['code', 'name']
        """
        try:
            import akshare as ak
            df = ak.index_stock_cons('000852')
            rows = []
            for _, row in df.iterrows():
                code = str(row.iloc[0]).zfill(6)
                name = str(row.iloc[1]) if len(row) > 1 else ""
                rows.append({'code': code, 'name': name})
            result = pd.DataFrame(rows)
            logger.info(f"获取到 {len(result)} 只中证1000成分股")
            return result
        except Exception as e:
            logger.error(f"获取中证1000成分股失败: {e}")
            return pd.DataFrame()

    def build_cache(self, symbols: List[str] = None, years: int = 3,
                    adjust: str = "qfq") -> Dict[str, int]:
        """
        批量下载历史数据到本地缓存

        Args:
            symbols: 股票代码列表，None时自动获取中证800
            years: 回溯年数
            adjust: 复权类型

        Returns:
            {symbol: 下载行数} 统计
        """
        if symbols is None:
            csi800 = self.get_csi800_stocks()
            if csi800.empty:
                logger.error("无法获取中证800成分股列表")
                return {}
            symbols = csi800['code'].tolist()
            logger.info(f"使用中证800成分股，共{len(symbols)}只")

        end_date = datetime.now().strftime('%Y%m%d')
        start = datetime.now() - timedelta(days=years * 365 + 30)  # 多给30天缓冲
        start_date = start.strftime('%Y%m%d')

        stats = {}
        total = len(symbols)
        errors = 0

        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"[{i}/{total}] 缓存 {symbol}...")
                df = self.get_daily_data(symbol, start_date, end_date, adjust)

                if not df.empty:
                    stats[symbol] = len(df)
                else:
                    logger.warning(f"{symbol} 数据为空")
                    errors += 1

            except Exception as e:
                logger.error(f"{symbol} 缓存失败: {e}")
                errors += 1

        logger.info(f"缓存完成: {len(stats)} 只成功, {errors} 只失败")
        return stats

    def update_cache(self, symbols: List[str] = None,
                     adjust: str = "qfq") -> Dict[str, int]:
        """
        增量更新本地缓存（只下载缺失的最新交易日数据）

        Args:
            symbols: 要更新的股票列表，None时自动检测缓存目录中已有的股票
            adjust: 复权类型

        Returns:
            {symbol: 新增行数} 统计
        """
        if symbols is None:
            symbols = []
            if os.path.isdir(self.cache_dir):
                for fname in sorted(os.listdir(self.cache_dir)):
                    if fname.endswith('.csv'):
                        symbols.append(fname[:-4])
            if not symbols:
                logger.warning("缓存目录为空，请先运行 build_cache()")
                return {}

        today = datetime.now()
        # 周末不更新
        if today.weekday() >= 5:
            logger.info("今天是周末，跳过缓存更新")
            return {}

        today_str = today.strftime('%Y%m%d')
        stats = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols, 1):
            try:
                cache_path = os.path.join(self.cache_dir, f"{symbol}.csv")

                last_date = None
                if os.path.exists(cache_path):
                    try:
                        existing = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                        if not existing.empty:
                            last_date = existing.index[-1]
                    except Exception:
                        pass

                if last_date is not None:
                    # 如果最后日期距今不到2天，跳过
                    days_since_last = (today - last_date).days
                    if days_since_last <= 2:
                        continue

                    # 只下载缺失的日期范围
                    next_date = last_date + timedelta(days=1)
                    start_str = next_date.strftime('%Y%m%d')
                else:
                    # 无缓存，下载最近1年
                    start = today - timedelta(days=365)
                    start_str = start.strftime('%Y%m%d')

                logger.info(f"[{i}/{total}] 更新 {symbol} ({start_str} → {today_str})")
                df = self._try_baostock_daily(symbol, start_str, today_str, adjust)
                if df.empty:
                    # 可能非交易日，用前复权重新试一次
                    df = self._try_baostock_daily(symbol, start_str, today_str, "qfq")

                if not df.empty:
                    # 只保留新的行（跳过已在缓存中的）
                    if last_date is not None:
                        df = df[df.index > last_date]
                    if not df.empty:
                        self._save_cached_data(symbol, df)
                        stats[symbol] = len(df)
                        logger.info(f"  {symbol}: 新增 {len(df)} 条")
                    else:
                        logger.debug(f"  {symbol}: 无新数据")
                else:
                    logger.debug(f"  {symbol}: baostock未返回新数据")

            except Exception as e:
                logger.error(f"{symbol} 更新失败: {e}")
                continue

        if stats:
            logger.info(f"更新完成: {len(stats)} 只有新数据, 共 {sum(stats.values())} 条")
        else:
            logger.info("所有股票已是最新")
        return stats


def calculate_flow_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于OHLCV数据计算辅助列（资金流向指标已统一由 horizon_signal_engine 计算）
    """
    # 资金流向指标 (main_force_flow, flow_strength) 由 horizon_signal_engine
    # 统一在 B1Selector.prepare_data() 中计算，此处不再重复计算
    return df.copy()


def load_cache_data(cache_dir: str = None, min_rows: int = 200,
                    common_range: bool = True) -> Dict[str, pd.DataFrame]:
    """
    从本地缓存加载所有股票数据

    Args:
        cache_dir: 缓存目录，默认 data_cache/
        min_rows: 最少行数，不足的股票被过滤
        common_range: 是否统一截断到所有股票共同的时间范围

    Returns:
        {symbol: DataFrame} 字典
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

    if not os.path.isdir(cache_dir):
        logger.warning(f"缓存目录不存在: {cache_dir}")
        return {}

    data = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith('.csv'):
            continue
        symbol = fname[:-4]
        try:
            df = pd.read_csv(os.path.join(cache_dir, fname),
                             index_col=0, parse_dates=True)
            if not df.empty and len(df) >= min_rows:
                # 计算资金流向指标
                df = calculate_flow_indicators(df)
                data[symbol] = df
        except Exception as e:
            logger.debug(f"读取缓存 {symbol} 失败: {e}")
            continue

    if not data:
        logger.warning("缓存中没有有效数据")
        return data

    logger.info(f"从缓存加载 {len(data)} 只股票")

    if common_range:
        # 去掉最短20%的起始日期，避免少数短历史股票拖累整体
        start_dates = sorted(df.index[0] for df in data.values())
        cutoff = start_dates[len(start_dates) // 5]
        earliest_end = min(df.index[-1] for df in data.values())

        if earliest_end <= cutoff:
            logger.warning("股票之间没有共同日期范围")
            return data

        for symbol in list(data.keys()):
            df = data[symbol]
            if df.index[0] > cutoff:
                del data[symbol]  # 起始太晚的剔除
            else:
                df = df[(df.index >= cutoff) & (df.index <= earliest_end)]
                if len(df) >= min_rows:
                    data[symbol] = df
                else:
                    del data[symbol]

        logger.info(f"截断到共同范围后: {len(data)} 只股票, "
                    f"{cutoff.date()} ~ {earliest_end.date()}")

    return data


def load_stock_name_map() -> Dict[str, str]:
    """加载A股股票名称映射表。优先从本地 stock_names.json，秒出。"""
    here = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(here, "stock_names.json")
    try:
        if os.path.exists(local_path):
            with open(local_path, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    # 本地文件不存在，回退到 akshare（较慢）
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        name_map = {}
        for _, row in df.iterrows():
            code = str(row.get('code', row.get('symbol', '')))
            name = str(row.get('name', ''))
            if code and name:
                code = code.zfill(6)
                name_map[code] = name
        return name_map
    except Exception as e:
        logger.warning(f"获取股票名称失败: {e}")
        return {}


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description="增强版数据获取模块")
    parser.add_argument("--build-cache", action="store_true",
                       help="构建缓存：下载沪深300近3年历史数据")
    parser.add_argument("--update-cache", action="store_true",
                       help="增量更新缓存：只下载缺失的最新数据")
    parser.add_argument("--list", action="store_true",
                       help="列出缓存状态")
    parser.add_argument("--symbols", nargs="+", default=None,
                       help="指定股票代码列表（空格分隔）")
    parser.add_argument("--years", type=int, default=3,
                       help="历史数据回溯年数（默认3年）")
    args = parser.parse_args()

    fetcher = EnhancedStockFetcher()

    if args.build_cache:
        print(f"开始构建缓存（回溯{args.years}年）...")
        t0 = time.time()
        stats = fetcher.build_cache(symbols=args.symbols, years=args.years)
        elapsed = time.time() - t0
        total_rows = sum(stats.values())
        print(f"\n缓存完成: {len(stats)} 只股票, {total_rows} 条记录, 耗时 {elapsed:.1f}s")
        if stats:
            print(f"平均 {total_rows/len(stats):.0f} 行/只, {elapsed/len(stats):.1f}s/只")

    elif args.update_cache:
        print("开始增量更新缓存...")
        t0 = time.time()
        stats = fetcher.update_cache(symbols=args.symbols)
        elapsed = time.time() - t0
        new_rows = sum(stats.values())
        print(f"更新完成: {len(stats)} 只有新数据, 共 {new_rows} 条, 耗时 {elapsed:.1f}s")

    elif args.list:
        if not os.path.isdir(fetcher.cache_dir):
            print("缓存目录为空")
        else:
            cache_files = sorted(os.listdir(fetcher.cache_dir))
            if not cache_files:
                print("缓存目录为空")
            else:
                print(f"缓存目录: {fetcher.cache_dir}")
                print(f"{'代码':<10} {'行数':<8} {'日期范围':<30}")
                print("-" * 50)
                total_rows = 0
                for fname in cache_files:
                    if not fname.endswith('.csv'):
                        continue
                    path = os.path.join(fetcher.cache_dir, fname)
                    try:
                        df = pd.read_csv(path, index_col=0, parse_dates=True)
                        symbol = fname[:-4]
                        date_range = f"{df.index[0].date()} ~ {df.index[-1].date()}" if not df.empty else "空"
                        print(f"{symbol:<10} {len(df):<8} {date_range:<30}")
                        total_rows += len(df)
                    except Exception:
                        print(f"{fname:<10} {'读取失败':<8}")
                print("-" * 50)
                print(f"共 {len(cache_files)} 个文件, {total_rows} 条记录")

    else:
        # 默认测试
        print("测试增强版数据获取模块")
        print("=" * 50)

        print("1. 测试获取股票列表...")
        stock_list = fetcher.get_stock_list("A")
        print(f"获取到 {len(stock_list)} 只股票")
        print(stock_list.head())

        print("\n2. 测试获取日线数据...")
        test_symbols = ['000001', '000002', '000028']

        for symbol in test_symbols:
            print(f"\n测试股票 {symbol}...")
            df = fetcher.get_daily_data(symbol, "20240101", "20240131", "qfq")
            if not df.empty:
                print(f"  成功获取数据，形状: {df.shape}")
                print(f"  数据列: {list(df.columns)}")
                if len(df) > 0:
                    print(f"  最新数据: 收盘价={df['close'].iloc[-1]:.2f}, 成交量={df['volume'].iloc[-1]:,.0f}")
            else:
                print(f"  获取数据失败")

        print("\n3. 测试沪深300列表...")
        hs300 = fetcher.get_hs300_stocks()
        print(f"获取到 {len(hs300)} 只沪深300成分股")
        print(hs300.head(10))

        print("\n测试完成")