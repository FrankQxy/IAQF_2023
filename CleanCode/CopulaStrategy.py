import numpy as np
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.optimize import minimize
from scipy.integrate import quad
import sys
from collections import deque
from IStrategy import IStrategy
import pandas as pd
import pyvinecopulib as pv
from ouparams import ouparams

class CopulaStrategy(IStrategy):

    def __init__(self, name='CopulaStrategy', factor=2, formation=252, \
                 tradingperiod=126, indices = ['^GSPC', '^IXIC'], capital=100, \
                 leverage = 1, thresh = 0.6):
        
        super().__init__(name=name, capital = capital)
        self._factor = factor  # number of standard deviations as threshold
        self._formation = formation 
        self._tp = tradingperiod
        self._data = deque([],maxlen=self._formation)
        self._alter_data = deque([],maxlen=self._formation)
        self._normfactor = {indices[0]:0,indices[1]:0}
        self._counter = 0
        self._position = {indices[0]:0,indices[1]:0}
        self._indices = indices
        self._signx = 1
        self._signy = 1
        self._leverage = leverage
        self._thresh = thresh
        self._strategy_spread = pd.DataFrame(columns = ['flagx','flagy'])
        self._flagx = 0
        self._flagy = 0
        self._state = 'inactive'
        self._trade_dur = 0
        self._hlf = pd.DataFrame(columns = ['hlf_flagx','hlf_flagy'])
        self._half_life1 = 0
        self._half_life2 = 0

    def is_ready(self):
        return len(self._data) == self._formation
    
    def get_spread(self):
        return self._strategy_spread
    
    def refit(self):
        asset_ret = [[self._data[i+1][0] - self._data[i][0] for i in range(0,self._formation-1)], \
                     [self._data[i+1][1] - self._data[i][1] for i in range(0,self._formation-1)]]
        
        self._ecdfx = ECDF(asset_ret[0])
        self._ecdfy = ECDF(asset_ret[1])
        u_data = np.array([[self._ecdfx(self._data[i+1][0] - self._data[i][0]) , self._ecdfy(self._data[i+1][1] - self._data[i][1])] for i in range(0,self._formation-1)])
        self._model = pv.Vinecop(data = u_data)

    # Analytic risk management
    def half_life_check(self):
        return self._tp - self._counter > 2*self._half_life
            
    def trade_open_half_life_check(self):
        return self._trade_dur < 3*self._half_life

    def OUFit(self, spread1,spread2):
        mu, sigma, theta = ouparams.find(spread1)
        if abs(np.log(2)/theta) < 100: 
           self._half_life1 = np.log(2)/theta
        
        mu, sigma, theta = ouparams.find(spread2)
        if abs(np.log(2)/theta) < 100: 
           self._half_life2 = np.log(2)/theta

    def generate_signal(self,element,alter):
        signal = {self._indices[0]:0,self._indices[1]:0}
        self.add_price_pair(element,alter)

        if not self.is_ready():
           self._normfactor = {k:np.log(e) for k,e in element.items()}
           return signal
         
        self._counter += 1
        if self._counter == 1: self.refit()

        if self._counter == self._tp:
           self.refit()
           self._sign = 1
           self._counter = 0
           signal[self._indices[0]] = -self._position[self._indices[0]]
           signal[self._indices[1]] = -self._position[self._indices[1]]
           self._state = 'inactive'
           self._position = {self._indices[0]:0,self._indices[1]:0}
           self._strategy_spread.loc[len(self._strategy_spread)] = [0,0]
           self._normfactor = {k:np.log(e) for k,e in element.items()}
           self._flagx = 0
           self._flagy = 0
           self.OUFit(self._strategy_spread.iloc[-self._tp:]['flagx'].to_numpy(), self._strategy_spread.iloc[-self._tp:]['flagy'].to_numpy())
           self._hlf.loc[len(self._hlf)] = [self._half_life1, self._half_life2]
           return signal
        
        M_x, M_y = self._misprice_index(element)
        self._flagx += M_x - 0.5
        self._flagy += M_y - 0.5
        self._strategy_spread.loc[len(self._strategy_spread)] = [self._flagx, self._flagy]
        self._hlf.loc[len(self._hlf)] = [self._half_life1, self._half_life2]

        if self._flagx > self._thresh and self._flagy < -self._thresh and self._state == 'inactive':
           
           self._signx = np.sign(self._flagx)
           self._signy = np.sign(self._flagy)
           self._state = 'active'
           signal[self._indices[0]] = -self._leverage*self._capital*1.0/element[self._indices[0]]
           signal[self._indices[1]] = self._leverage*self._capital*1.0/element[self._indices[1]]
           self._position = signal 

        elif self._flagx < -self._thresh and self._flagy > self._thresh and self._state == 'inactive':
           self._signx = np.sign(self._flagx)
           self._signy = np.sign(self._flagy)
           self._state = 'active'
           signal[self._indices[0]] = self._leverage*self._capital*1.0/element[self._indices[0]]
           signal[self._indices[1]] = -self._leverage*self._capital*1.0/element[self._indices[1]]
           self._position = signal 

        elif self._state == 'active' and (self._signx != np.sign(self._flagx) or self._signy != np.sign(self._flagy)):
           self._state = 'inactive'
           signal[self._indices[0]] = -self._position[self._indices[0]]
           signal[self._indices[1]] = -self._position[self._indices[1]]
           self._state = 'inactive'
           self._position = {self._indices[0]:0,self._indices[1]:0}

        self._normfactor = {k:np.log(e) for k,e in element.items()}
        return signal
    
    def add_price_pair(self, element,alter):
        prices = [np.log(e) for k,e in element.items()]
        self._data.append(prices)
        self._alter_data.append(alter)

    def _misprice_index(self,element):
        '''Calculate mispricing index for every day in the trading period by using estimated copula
        Mispricing indices are the conditional probability P(U < u | V = v) and P(V < v | U = u)'''

        x = np.log(element[self._indices[0]]) - self._normfactor[self._indices[0]]
        y = np.log(element[self._indices[1]]) - self._normfactor[self._indices[1]]

        # Convert the two returns to uniform values u and v using the empirical distribution functions
        x = self._ecdfx(x)
        y = self._ecdfy(y)
        integrand1 = lambda x,y : self._model.pdf(np.array([[x,y]]))
        integrand2 = lambda y,x : self._model.pdf(np.array([[x,y]]))
        M_x = quad(integrand1,0,x,args=(y,))[0]/quad(integrand1,0,1,args=(y,))[0]
        M_y = quad(integrand2,0,y,args=(x,))[0]/quad(integrand2,0,1,args=(x,))[0]
        return M_x, M_y