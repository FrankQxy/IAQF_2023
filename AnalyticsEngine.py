import pandas as pd
import pyfolio as pf
import numpy as np
import matplotlib.pyplot as plt
from MarketUtils import *


class AnalyticsEngine():
    def __init__(self, trade=pd.DataFrame(), data=pd.DataFrame()):
        self._trade = trade
        self._data = data
        self._rf = 1

    def absolute_returns(self):
        idx1 = self._data.columns[0]
        idx2 = self._data.columns[1]
        idx1_price = self._data[idx1]
        idx2_price = self._data[idx2]
        idx1_signal = self._trade[self._trade.columns[1]]
        idx2_signal = self._trade[self._trade.columns[2]]
        idx1_factor = 0
        idx2_factor = 0

        n = len(idx1_price)
        state = 'inactive'
        returns = np.zeros(n)

        for i in range(n):
            if state == 'inactive':
                returns[i] = 0
                if idx1_signal[i] > 0:
                    state = 'long'
                    idx1_factor = idx1_signal[i]
                    idx2_factor = idx2_signal[i]
                elif idx1_signal[i] < 0:
                    state = 'short'
                    idx1_factor = idx1_signal[i]
                    idx2_factor = idx2_signal[i]
            elif state == 'long':
                r1 = idx1_price[i] - idx1_price[i - 1]
                r2 = idx2_price[i] - idx2_price[i - 1]
                returns[i] = idx1_factor * r1 + idx2_factor * r2
                if idx1_signal[i] <= 0:
                    status = 'inactive'
            elif state == 'short':
                r1 = idx1_price[i - 1] - idx1_price[i]
                r2 = idx2_price[i - 1] - idx2_price[i]
                returns[i] = idx1_factor * r1 + idx2_factor * r2
                if idx1_signal[i] >= 0:
                    state = 'inactive'

        return returns

    def percentage_return(self):
        abs_rets = self.absolute_returns()
        n = len(abs_rets)
        returns = np.zeros(n)

        for i in range(1, n):
            if abs_rets[i - 1] != 0 and abs_rets[i] != 0:
                returns[i] = 100 * (abs_rets[i] - abs_rets[i - 1]) / abs_rets[i - 1]

        return returns

    def compute(self, abs_rets, per_rets, trade):
        total_return = sum(abs_rets)
        sharpe_ratio = (np.mean(per_rets) - self._rf)/np.std(per_rets)
        capital = trade['capital']
        signal = trade[trade.columns[1]]
        state = 'inactive'

        for i in range(len(capital)):
            if state == 'inactive':
                if signal[i] > 0:
                    state = 'long'
                elif signal[i] < 0:
                    state = 'short'
            elif signal[i] != 0:
                net_profit = capital[i]
                state = 'inactive'

        return [total_return, sharpe_ratio, net_profit]

    def save_analytics(self, filename=''):
        df = pd.DataFrame()
        df['Date'] = self._data.index.array
        df['Absolute_return'] = self.absolute_returns()
        df['Percentage_return'] = self.percentage_return()
        path1 = DATA_PATH / (filename + 'Returns' + f'{TIMESTAMP()}.csv')

        abs_rets, per_rets = self.absolute_returns(), self.percentage_return()
        al = self.compute(abs_rets, per_rets,self._trade)
        df2 = pd.DataFrame([al], columns=['Total_return','Sharpe_ratio','Net_profit'])
        path2 = DATA_PATH / (filename + 'Analytics' + f'{TIMESTAMP()}.csv')

        # Save the files
        df.to_csv(path1)
        df2.to_csv(path2)

    def run_pyfolio(self):
        """
        Generate a number of tear sheets that are useful for analyzing a strategy’s performance
        :param rets: daily returns of the strategy, noncumulative
        :param benchmark_rets: daily returns of the benchmark, noncumulative
        :return: None
        """
        # rets = self._trade
        # pf.create_full_tear_sheet(returns=rets)
        pass

    def get_data(self):
        return self._data.index.array

if __name__ == "__main__":
    benchmark_trade = pd.read_csv('./DataFiles/test20230209.csv')
    price_data = get_data(type='index', col_list=['^GSPC', '^IXIC'], termDates=['2010-01-04', '2022-12-30'])
    analytics = AnalyticsEngine(trade=benchmark_trade, data=price_data)
    #analytics.save_analytics(filename='Benchmark')
    print(analytics.get_data())

