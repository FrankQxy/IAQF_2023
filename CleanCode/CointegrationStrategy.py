import numpy as np
import pandas as pd
import statsmodels.api as sm
from IStrategy import IStrategy
import numpy as np
from collections import deque
from statistics import stdev, mean
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class CointegrationStrategy(IStrategy):

    def __init__(self, name='CointegrationStrategy', factor=2, formation=252, tradingperiod=126, indices = ['^GSPC', '^DJI'], capital=100, leverage = 1):
        super().__init__(name=name, capital = capital)
        self._factor = factor  # number of standard deviations as threshold
        self._formation = formation 
        self._tp = tradingperiod
        self._data = deque([],maxlen=self._formation)
        self._strategy_spread = pd.DataFrame(columns = ['Spread'])
        self._counter = 0
        self._position = {indices[0]:0,indices[1]:0}
        self._indices = indices
        self._alter_data = deque([],maxlen=self._formation)

    def is_ready(self):
        return len(self._data) == self._formation

    def parameters(self):
        assets = [[a[0]  for a in self._data],[a[1] for a in self._data]]
        assets = [np.array(assets[0]), np.array(assets[1]).reshape(-1,1)]
        self._scaler = StandardScaler()
        assets[1] = self._scaler.fit_transform(assets[1])
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(assets[1],assets[0])
        self._beta = self.model.coef_[0]
        spreads = assets[0] - assets[1]*self._beta
        self._scaler_spread = StandardScaler()
        spreads = self._scaler_spread.fit_transform(spreads.reshape(-1,1))
        self.OUFit(spreads)

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
           return signal
        
        self._counter += 1
        if self._counter == 1:
           self.parameters()

        if self._counter == self._tp:
           self._counter = 0   
           signal[self._indices[0]] = -self._position[self._indices[0]]
           signal[self._indices[1]] = -self._position[self._indices[1]]
           self._state = 'inactive'
           return signal 
           
        self._hlf.loc[len(self._hlf)] = [self._half_life]
        y = np.array(element[self._indices[0]]).reshape(1,-1)
        x = np.array(element[self._indices[1]]).reshape(1,-1)
        x = self._scaler.transform(x)
        spread = y - self.model.predict(x)
        spread = self._scaler_spread.transform(spread)
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
           signal[self._indices[0]] = self._capital*0.5
           signal[self._indices[1]] = -self._beta*self._capital*0.5
           self._position = signal
           return signal
        
        if self._state == 'inactive' and spread > self._factor:
           self._state = 'active'
           self._sign = np.sign(spread)
           signal[self._indices[0]] = -self._capital*0.5/self._beta
           signal[self._indices[1]] = self._capital*0.5
           self._position = signal
           return signal
        
        return signal
