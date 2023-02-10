from BenchmarkStrategy import *
from IStrategy import *
from BacktestEngine import *
from MarketUtils import *
from MarketData import *
from Configuration import *
from AnalyticsEngine import *


class dummy(IStrategy):
    isodd = True

    def __init__(self, name="something"):
        super().__init__(name=name)

    def generate_signal(self, element) -> dict:
        self.isodd = True
        signal = {}
        for asset in element:
            signal[asset] = 1

        return signal


if __name__ == "__main__":
    price_data = get_data(type='index', col_list=['^GSPC', '^IXIC'], termDates=['2010-01-04','2022-12-30'])
    benchmark = BenchmarkStrategy()
    backtest = backtest_walk_forward(price_data)
    backtest.add_strategy(benchmark)
    trades = backtest.run_backtest()
    backtest.save_trades(trades, "test")
    analytics = AnalyticsEngine(trade=trades, data=price_data)
    analytics.save_analytics()
