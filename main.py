from IStrategy import *
from BacktestEngine import *
from MarketUtils import *
from MarketData import *
from Configuration import *
from AnalyticsEngine import *

###### dummy test strategy #######
class dummy(IStrategy):

    isodd = True
    def __init__(self,name="something"):
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
if __name__ == "__main__":
   price_data = get_data(type ='index',col_list = ['^GSPC', '^IXIC'], termDates = ['1992-01-23','1992-06-01'])
   strategy = dummy(name = "dummyTest")
   backtest = backtest_walk_forward(price_data)
   backtest.add_strategy(strategy)
   trades = backtest.run_backtest()
   backtest.save_trades(trades,"DummyStrategy")
