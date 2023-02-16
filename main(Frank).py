from BenchmarkStrategy import *
from IStrategy import *
from BacktestEngine import *
from MarketUtils import *
from MarketData import *
from Configuration import *
from AnalyticsEngine import *
from Benchmark import BenchmarkSpreadAnalysis


if __name__ == "__main__":
    price_data = get_data(type='index', col_list=['^IXIC', '^RUA'], termDates=['1992-01-02','2011-12-30'])
    benchmark = BenchmarkStrategy(data=price_data)
    spread = benchmark.get_spread()
    benchmark.save_spread(spread)
    df = BenchmarkSpreadAnalysis.clean_data(price_data, spread)
    print(BenchmarkSpreadAnalysis.linear_regression(df))


''' backtest = backtest_walk_forward(price_data)
    backtest.add_strategy(benchmark)
    trades = backtest.run_backtest()
    backtest.save_trades(trades, "test")
    analytics = AnalyticsEngine(trade=trades, data=price_data)
    analytics.save_analytics(filename='Benchmark')'''


