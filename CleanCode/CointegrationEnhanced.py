import numpy as np
import pandas as pd
import statsmodels.api as sm
from IStrategy import IStrategy
import numpy as np
from collections import deque
from statistics import stdev, mean

class CointegrationStrategy(IStrategy):

    def __init__(self, name='CointegrationStrategy', factor=2, formation=252, tradingperiod=126, indices = ['^GSPC', '^DJI'], capital=100, leverage = 1):
        super().__init__(name=name, capital = capital)
        self._factor = factor  # number of standard deviations as threshold
        self._formation = formation 
        self._tp = tradingperiod
        self._data = deque([],maxlen=self._formation)
        self._normfactor = {indices[0]:0,indices[1]:0}
        self._strategy_spread = pd.DataFrame(columns = ['Spread'])
        self._counter = 0
        self._position = {indices[0]:0,indices[1]:0}
        self._indices = indices
        self.sd = 0
        self._beta = 1
        self._sign = 1
        self._mean = 0
        self._leverage = leverage
        self._alter_data = deque([],maxlen=self._formation)

    def is_ready(self):
        return len(self._data) == self._formation

    def parameters(self):
        nf = self._data[0]
        assets = [[a[0]/nf[0]  for a in self._data],\
                 [a[1]/nf[1] for a in self._data]]
        
        model = sm.RLM(assets[0], assets[1])
        model = model.fit()
        self._beta = model.params[0]
        spreads = [assets[0][i] - assets[1][i]*self._beta for i in range(self._formation)]
        self.sd = stdev(spreads)
        self._mean = mean(spreads)
        self.OUFit(np.array(spreads))

    def add_price_pair(self, element, alter):
        prices = [e for k,e in element.items()]
        self._data.append(prices)
        self._alter_data.append(alter)

    def get_spread(self):
        return self._strategy_spread
    
    def generate_signal(self, element, alter):  # element (dict): format asset:price
        signal = {self._indices[0]:0,self._indices[1]:0}
        self.add_price_pair(element,alter)

        if not self.is_ready():
           self._normfactor = element
           return signal

        self._counter += 1
        if self._counter == 1:
            self.parameters()

        if self._counter == self._tp:
           self._normfactor = element
           self.parameters()
           self._sign = 1
           self._counter = 0
           signal[self._indices[0]] = -self._position[self._indices[0]]
           signal[self._indices[1]] = -self._position[self._indices[1]]
           self._state = 'inactive'
           self._position = {self._indices[0]:0,self._indices[1]:0}
           self._strategy_spread.loc[len(self._strategy_spread)] = [0]
           self._hlf.loc[len(self._hlf)] = [self._half_life]
           return signal

        self._hlf.loc[len(self._hlf)] = [self._half_life]
        normalized_price = {}
        for k in element.keys():
            normalized_price[k] = element[k]/self._normfactor[k]
        
        spread = normalized_price[self._indices[0]] - self._beta*normalized_price[self._indices[1]]
        spread = (spread - self._mean)/self.sd
        self._strategy_spread.loc[len(self._strategy_spread)] = [spread]

        if self._state == 'active' and np.sign(spread) != self._sign:
           signal[self._indices[0]] = -self._position[self._indices[0]]
           signal[self._indices[1]] = -self._position[self._indices[1]]
           self._state = 'inactive'
           self._position = {self._indices[0]:0,self._indices[1]:0}
           return signal
        
        if self._state == 'inactive' and spread < -self._factor:
           self._sign = np.sign(spread)
           self._state = 'active'
           signal[self._indices[0]] = self._leverage*self._capital*1.0/element[self._indices[0]]
           signal[self._indices[1]] = -self._leverage*self._beta*self._capital*1.0/element[self._indices[1]]
           self._position = signal
           return signal
        
        if self._state == 'inactive' and spread > self._factor:
           self._state = 'active'
           self._sign = np.sign(spread)
           signal[self._indices[0]] = -self._leverage*self._capital*1.0/(element[self._indices[0]]*self._beta)
           signal[self._indices[1]] = self._leverage*self._capital*1.0/element[self._indices[1]]
           self._position = signal
           return signal
        
        return signal
