"""
数据下载脚本 - 下载股票历史数据

使用 yfinance 库下载股票数据，支持全球主要股票市场
如果没有安装 yfinance，请运行: pip install yfinance
"""

import argparse
from pathlib import Path

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("错误: 需要安装 yfinance 库")
    print("请运行: pip install yfinance")
    exit(1)


def download_stock_data(
    symbol="^GSPC",
    start_date="2020-01-01",
    end_date=None,
    output_file="data/stock_data.csv",
):
    """
    下载股票数据

    Args:
        symbol: 股票代码
            - ^GSPC: S&P 500 指数（美国）
            - ^DJI: 道琼斯指数（美国）
            - AAPL: 苹果公司
            - TSLA: 特斯拉
            - 000001.SS: 上证指数（中国）
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)，默认为今天
        output_file: 输出文件路径
    """
    print("正在下载股票数据...")
    print(f"  股票代码: {symbol}")
    print(f"  开始日期: {start_date}")
    print(f"  结束日期: {end_date if end_date else '今天'}")

    try:
        # 下载数据
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)

        if df.empty:
            print(f"错误: 没有下载到数据，请检查股票代码 '{symbol}' 是否正确")
            return False

        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存数据
        df.to_csv(output_file)

        print("\n✓ 下载成功!")
        print(f"  数据行数: {len(df)}")
        print(
            f"  日期范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}"
        )
        print(f"  保存路径: {output_file}")
        print("\n数据预览:")
        print(df.head())

        return True

    except Exception as e:
        print(f"错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载股票历史数据")

    parser.add_argument(
        "--symbol",
        type=str,
        default="^GSPC",
        help="股票代码，例如: ^GSPC (S&P 500), AAPL (苹果), 000001.SS (上证指数)",
    )

    parser.add_argument(
        "--start", type=str, default="2020-01-01", help="开始日期 (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end", type=str, default=None, help="结束日期 (YYYY-MM-DD)，默认为今天"
    )

    parser.add_argument(
        "--output", type=str, default="data/stock_data.csv", help="输出文件路径"
    )

    args = parser.parse_args()

    # 下载数据
    success = download_stock_data(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        output_file=args.output,
    )

    if success:
        print("\n数据下载完成! 可以运行 stock_predictor.py 进行训练")
    else:
        print("\n数据下载失败，请检查网络连接和股票代码")


if __name__ == "__main__":
    main()
