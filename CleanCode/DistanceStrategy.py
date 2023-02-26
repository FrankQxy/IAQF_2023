import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IStrategy import *
from collections import deque
import numpy as np
from statistics import stdev, mean


class DistanceStrategy(IStrategy):
    def __init__(self, name='DistanceStrategy', factor=2, formation=252, tradingperiod=126, indices = ['^GSPC', '^IXIC'], capital=100, leverage = 1):
        super().__init__(name=name, capital = capital)
        self._factor = factor  # number of standard deviations as threshold
        self._formation = formation 
        self._tp = tradingperiod
        self._data = deque([],maxlen=self._formation)
        self._normfactor = {indices[0]:0,indices[1]:0}
        self._price_series = pd.DataFrame(columns = [indices[0],indices[1]])
        self._strategy_spread = pd.DataFrame(columns = ['Spread'])
        self._counter = 0
        self._position = {indices[0]:0,indices[1]:0}
        self._indices = indices
        self.sd = 0
        self._sign = 1
        self._leverage = leverage

    def is_ready(self):
        return len(self._data) == self._formation

    def add_price_pair(self, element):
            prices = [e for k,e in element.items()]
            self._data.append(prices)
            self._price_series.loc[len(self._price_series)] = prices 

    def get_spread(self):
        return self._strategy_spread

    def get_spread_full(self):
        init_price = self._price_series.loc[0]
        spread = pd.DataFrame(columns = ['Spread'])
        spread['Spread'] = self._price_series[self._indices[0]]/init_price[self._indices[0]] - self._price_series[self._indices[1]]/init_price[self._indices[1]]
        return spread

    def setSD(self):
        nf = self._data[0]
        spreads = [a[0]/nf[0] - a[1]/nf[1] for a in self._data]
        self.sd = stdev(spreads)
        self.mean = mean(spreads)
        return

    def generate_signal(self, element):

        signal = {self._indices[0]:0,self._indices[1]:0}
        self.add_price_pair(element)
        if not self.is_ready():
           self._normfactor = element
           return signal

        self._counter += 1
        if self._counter == 1: 
            self.setSD()

        if self._counter == self._tp:
           self._normfactor = element
           self.setSD()
           self._sign = 1
           self._counter = 0
           signal[self._indices[0]] = -self._position[self._indices[0]]
           signal[self._indices[1]] = -self._position[self._indices[1]]
           self._state = 'inactive'
           self._position = {self._indices[0]:0,self._indices[1]:0}
           self._strategy_spread.loc[len(self._strategy_spread)] = [0]
           return signal

        normalized_price = {}
        for k in element.keys():
            normalized_price[k] = element[k]/self._normfactor[k]
        
        spread = normalized_price[self._indices[0]] - normalized_price[self._indices[1]]
        spread = (spread - self.mean)
        self._strategy_spread.loc[len(self._strategy_spread)] = [spread]

        if self._state == 'active' and np.sign(spread) != self._sign:
           signal[self._indices[0]] = -self._position[self._indices[0]]
           signal[self._indices[1]] = -self._position[self._indices[1]]
           self._state = 'inactive'
           self._position = {self._indices[0]:0,self._indices[1]:0}
           return signal
        
        if self._state == 'inactive' and spread > self._factor*self.sd:
           self._sign = np.sign(spread)
           self._state = 'active'
           signal[self._indices[0]] = self._leverage*self._capital*1.0/element[self._indices[0]]
           signal[self._indices[1]] = -self._leverage*self._capital*1.0/element[self._indices[1]]
           self._position = signal
           return signal
        
        if self._state == 'inactive' and spread < -self._factor*self.sd:
           self._state = 'active'
           self._sign = np.sign(spread)
           signal[self._indices[0]] = -self._leverage*self._capital*1.0/element[self._indices[0]]
           signal[self._indices[1]] = self._leverage*self._capital*1.0/element[self._indices[1]]
           self._position = signal
           return signal

        return signal




