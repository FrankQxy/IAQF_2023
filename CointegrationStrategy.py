import numpy as np
import pandas as pd
import statsmodels.api as sm
from IStrategy import IStrategy


class CointegrationStrategy(IStrategy):
    def __init__(self):
        super().__init__(data=pd.dataframe, name='Cointegration')
        self._tradingRatio = 1
        self._factor = 1  # number of standard deviations as threshold
        self._std = 1   # threshold = factor * std
        self._spread = []

        self._assets = []  # should store two asset names when filled
        self._asset0 = []
        self._asset1 = []

        self._lastPos = 'inactive'

    def cointegration_test(self):
        pass

    def is_ready(self):
        return len(self._asset0) >= 90

    def compute_trading_ratio(self):
        # compute the trading ratio using the recent 90 data
        model = sm.OLS(self._asset0[-90:], self._asset1[-90:])
        model = model.fit()
        self._tradingRatio = model.params[0]
        return self._tradingRatio

    def compute_std(self):
        # compute standard deviation of the spread series using the recent 30 data
        spread = []
        for i in range(-30, 0):
            spread = self._asset0[i] - self._tradingRatio * self._asset1[i]
        self._spread = spread
        self._std = np.std(spread)
        return self._std

    def add_data(self, element):
        if len(self._assets) == 0:
            self._assets = element.keys()

        asset0, asset1 = self._assets
        self._asset0.extend(element[asset0])
        self._asset1.extend(element[asset1])

    def generate_signal(self, element):  # element (dict): format asset:price
        # setup
        self.add_data(element)
        if not self.is_ready():  # return zero strategy
            return {self._assets[0]: 0, self._assets[1]: 0}
        self.compute_trading_ratio()
        self.compute_std()
        threshold = self._factor * self._std

        # conditions
        if self._spread[-1] > threshold or (self._lastPos == 'short0' and self._spread[-1] > 0):
            self._lastPos = 'short0'
            return {self._assets[0]: -1, self._assets[1]: self._tradingRatio}
        elif self._spread[-1] < -threshold or (self._lastPos == 'long0' and self._spread[-1] <0):
            self._lastPos = 'long0'
            return {self._assets[0]: 1, self._assets[1]: -self._tradingRatio}
        else:
            self._lastPos = 'inactive'
            return {self._assets[0]: 0, self._assets[1]: 0}
