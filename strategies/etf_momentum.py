"""
ETF 双均线动量策略 — 你的第一个量化策略

策略逻辑：
  当短期均线上穿长期均线 → 买入（金叉）
  当短期均线下穿长期均线 → 卖出（死叉）

为什么选这个作为第一个策略？
  1. 逻辑简单直观，5 分钟就能理解
  2. 有明确的买卖信号，好 debug
  3. 能跑赢大盘就算成功，有成就感
  4. 面试时能展开讲：参数优化、过拟合、滑点
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.data_fetcher import get_etf_data, add_technical_indicators


def generate_signals(
    df: pd.DataFrame,
    short_ma: int = 5,
    long_ma: int = 20
) -> pd.DataFrame:
    """
    生成买卖信号

    参数:
        df: 含 date, close, ma5, ma20 的 DataFrame
        short_ma: 短期均线周期（默认 5 日）
        long_ma: 长期均线周期（默认 20 日）

    返回:
        含 signal 列的 DataFrame
        signal=1 → 买入, signal=-1 → 卖出, signal=0 → 持有

    核心逻辑（面试必问）:
    1. 金叉 = short_ma 从下方穿过 long_ma = 看涨信号
    2. 死叉 = short_ma 从上方穿过 long_ma = 看跌信号
    3. 用 diff() 检测穿越：-1→1 是金叉，1→-1 是死叉

    面试追问:
        Q: 均线策略的缺点？
        A: 滞后性（趋势反转时反应慢）；震荡市会被反复打脸
        Q: 怎么改进？
        A: 加过滤器（如波动率过滤、ADX 趋势强度过滤）
    """
    data = df.copy()

    # 计算均线
    ma_short_col = f"ma{short_ma}"
    ma_long_col = f"ma{long_ma}"
    if ma_short_col not in data.columns:
        data[ma_short_col] = data["close"].rolling(short_ma).mean()
    if ma_long_col not in data.columns:
        data[ma_long_col] = data["close"].rolling(long_ma).mean()

    # 核心：生成持仓状态（1=持仓，0=空仓）
    data["position"] = np.where(
        data[ma_short_col] > data[ma_long_col], 1, 0
    )

    # 从持仓状态推导买卖信号
    # position 从 0→1 = 买入信号
    # position 从 1→0 = 卖出信号
    data["signal"] = data["position"].diff()

    # 填充 NaN（第一条记录没有 diff）
    data["signal"] = data["signal"].fillna(0)

    return data


def backtest(df: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
    """
    回测策略表现

    参数:
        df: 含信号和价格数据的 DataFrame
        initial_capital: 初始资金（默认 10 万）

    返回:
        含每日持仓、收益的 DataFrame

    回测的核心概念（面试必问）:
    1. 不能有未来信息（look-ahead bias）
       - 今天的信号只能用今天之前的数据
       - 均线用 rolling() 天然避免了未来信息 ✅
    2. 滑点（slippage）— 实际成交价和信号价的差距
       - 简化版忽略，进阶版要加 0.1% 滑点
    3. 手续费 — 每次买卖扣万分之一到万分之三
    """
    data = df.copy()

    # 前一天收盘价买入，所以 shift(1)
    data["strategy_return"] = data["position"].shift(1) * data["return"]

    # 计算净值曲线
    data["equity"] = initial_capital * (1 + data["strategy_return"]).cumprod()
    data["benchmark"] = initial_capital * (1 + data["return"]).cumprod()

    # 标记交易点
    data["buy_signals"] = np.where(data["signal"] == 1, data["close"], np.nan)
    data["sell_signals"] = np.where(data["signal"] == -1, data["close"], np.nan)

    return data


def plot_results(df: pd.DataFrame, save_path: str = None):
    """
    可视化回测结果

    两张图：
    图1: 价格 + 均线 + 买卖点
    图2: 策略净值 vs 基准（买入持有）
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 图1：价格和信号
    ax1.plot(df["date"], df["close"], label="Close", alpha=0.7, linewidth=1)
    ax1.plot(df["date"], df.get("ma5", np.nan), label="MA5", alpha=0.7, linestyle="--")
    ax1.plot(df["date"], df.get("ma20", np.nan), label="MA20", alpha=0.7, linestyle="--")

    if "buy_signals" in df.columns:
        ax1.scatter(df["date"], df["buy_signals"],
                   color="green", marker="^", s=100, label="Buy", zorder=5)
    if "sell_signals" in df.columns:
        ax1.scatter(df["date"], df["sell_signals"],
                   color="red", marker="v", s=100, label="Sell", zorder=5)

    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.set_title("ETF 双均线策略 — 买卖信号")
    ax1.grid(alpha=0.3)

    # 图2：净值曲线
    ax2.plot(df["date"], df["benchmark"], label="Buy & Hold", alpha=0.6, linewidth=1)
    ax2.plot(df["date"], df["equity"], label="Strategy", linewidth=2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Portfolio Value")
    ax2.legend()
    ax2.set_title("策略 vs 基准")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {save_path}")
    plt.show()


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    计算绩效指标（面试必问）

    关键指标:
    - 年化收益率: 衡量赚钱能力
    - 最大回撤: 衡量风险（你最惨的时候亏多少）
    - 夏普比率: 衡量风险调整后收益（越大越好）
    - 胜率: 赚钱的交易占比
    """
    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
    trading_days = len(df)

    # 年化收益率
    annual_return = (1 + total_return) ** (252 / trading_days) - 1

    # 最大回撤
    rolling_max = df["equity"].cummax()
    drawdown = (df["equity"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 夏普比率（假设无风险利率 = 0）
    daily_returns = df["strategy_return"].dropna()
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

    return {
        "总收益率": f"{total_return:.2%}",
        "年化收益率": f"{annual_return:.2%}",
        "最大回撤": f"{max_drawdown:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "交易天数": trading_days,
    }


if __name__ == "__main__":
    # ====== 跑一遍完整的回测 ======

    print("📥 获取数据...")
    df = get_etf_data("510300", start_date="2020-01-01")

    print("📊 计算指标...")
    df = add_technical_indicators(df)

    print("🎯 生成信号...")
    df = generate_signals(df, short_ma=5, long_ma=20)

    print("💰 回测...")
    df = backtest(df, initial_capital=100000)

    print("📈 绩效指标:")
    metrics = calculate_metrics(df)
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    print("📉 生成图表...")
    plot_results(df, save_path="results/momentum_strategy.png")
