"""
数据获取工具 — 从 akshare / yfinance 获取 A股 ETF 数据

为什么要封装成独立的模块？
- 数据获取和策略逻辑分离，改数据源不改策略
- 统一数据格式，所有策略用同一套数据接口
- 方便加缓存，避免重复下载
"""

import pandas as pd
import numpy as np
from typing import Optional

def get_etf_data(
    etf_code: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    source: str = "akshare"
) -> pd.DataFrame:
    """
    获取 ETF 日线数据

    参数:
        etf_code: ETF代码，如 '510300' (沪深300ETF)
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期，默认到今天
        source: 数据源，目前支持 'akshare'

    返回:
        DataFrame 包含: date, open, high, low, close, volume

    面试追问:
        Q: 为什么用 akshare 而不是 tushare？
        A: akshare 免费且无需注册，适合学习。实盘可换 wind/聚宽
    """
    if source == "akshare":
        return _fetch_from_akshare(etf_code, start_date, end_date)
    else:
        raise ValueError(f"不支持的数据源: {source}")

def _fetch_from_akshare(
    etf_code: str,
    start_date: str,
    end_date: Optional[str]
) -> pd.DataFrame:
    """从 akshare 获取 ETF 数据"""
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请先安装 akshare: pip install akshare")

    # akshare 的 ETF 接口需要加市场后缀
    # sh=上海, sz=深圳
    if etf_code.startswith("51"):
        symbol = f"sh{etf_code}"
    elif etf_code.startswith("15"):
        symbol = f"sz{etf_code}"
    else:
        symbol = etf_code

    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date or pd.Timestamp.today().strftime("%Y-%m-%d"),
        adjust="qfq"  # 前复权
    )

    # 统一列名，屏蔽数据源差异
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df[["date", "open", "high", "low", "close", "volume"]]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加常用技术指标

    为什么放在单独的函数而不是策略里？
    - 多个策略可能共用指标
    - 方便测试不同指标组合
    - 指标计算逻辑可以独立优化
    """
    data = df.copy()

    # 移动均线
    data["ma5"] = data["close"].rolling(5).mean()
    data["ma10"] = data["close"].rolling(10).mean()
    data["ma20"] = data["close"].rolling(20).mean()
    data["ma60"] = data["close"].rolling(60).mean()

    # 日收益率
    data["return"] = data["close"].pct_change()

    # 波动率（20日滚动）
    data["volatility"] = data["return"].rolling(20).std()

    return data


if __name__ == "__main__":
    # 测试：获取沪深300ETF数据
    df = get_etf_data("510300", start_date="2023-01-01")
    df = add_technical_indicators(df)
    print(f"获取到 {len(df)} 条数据")
    print(df[["date", "close", "ma20", "ma60"]].tail())
