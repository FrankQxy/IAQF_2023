from BenchmarkStrategy import *
from IStrategy import *
from BacktestEngine import *
from MarketUtils import *
from MarketData import *
from Configuration import *
from AnalyticsEngine import *


###### dummy test strategy #######
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


# End to end check
# Basic end to end is done barring analytics. i think right now you can analyze 
# the csv or try parsing it to your local code SEPARATE FROM STARTEGY class

'''if __name__ == "__main__":
   price_data = get_data(type ='index',col_list = ['^GSPC', '^IXIC'], termDates = ['1992-01-23','1992-06-01'])
   strategy1 = dummy(name = "dummyTest1")
   strategy2 = dummy(name = "dummyTest2")
   backtest = backtest_walk_forward(price_data)
   backtest.add_strategy(strategy1).add_strategy(strategy2)
   trades = backtest.run_backtest()
   backtest.save_trades(trades,"DummyStrategy")'''

# End to end check for the benchmark strategy
if __name__ == "__main__":
    price_data = get_data(type='index', col_list=['^GSPC', '^IXIC'], termDates=['1992-01-23','1992-06-01'])
    benchmark = BenchmarkStrategy(data=price_data)
    backtest = backtest_walk_forward(price_data)
    backtest.add_strategy(benchmark)
    trades = backtest.run_backtest()
    #backtest.save_trades(trades, "BenchmarkStrategy", device='mac')
    path = './DataFiles/BenchmarkStrategy_' + f'{TIMESTAMP()}.csv'
    trades.to_csv(path)
