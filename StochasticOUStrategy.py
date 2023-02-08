from BenchmarkStrategy import *
from IStrategy import *
from BacktestEngine import *
from MarketUtils import *
from MarketData import *
from Configuration import *
from AnalyticsEngine import *
import numpy as np
from sklearn.linear_model import LinearRegression

#number of days of history used to callibrate parameter
DAYS=60
#number of days before the need of re-callibration
SAFE=30

class StochasticOUStrategy(IStrategy):

    #parameters for dX=_miu(_theta-X)dt+_theta dW
    _beta=1
    _theta=0
    _miu=0
    _sigma=0
    #number of days remaining before re-callibration of parameters
    #including re-calculation of thresholds
    _remaining=0
    #thresholds: as, bs are short entry and exit; al, bl are long entry and exit
    _as=0
    _bs=0
    _al=0
    _bl=0
    _current_position=None

    #?????maybe weight can be outside of strategy
    def __init__(self, data = [], state="inactive", weight = 1.0, capital=100):
        #?????I suppose theh data we are given only contains the two columns of the two indices we are trading
        super().__init__(data=data, state=state, weight=weight, capital=capital, name = "Stochastic--OU")
        df = self._data
        _idx1 = df[df.columns[0]]
        _idx2 = df[df.columns[1]]

    def generate_signal(self,element,index):
        idx = [asset for asset in element]
        if(index<DAYS):
            self._current_position={idx[0]: 0, idx[1]: 0}
            self._state="inactive"
            return {idx[0]: 0, idx[1]: 0}
        if(self._remaining==0):
            #can use up to index-1 to callibrate
            self.callibration(idx0=np.array(self._idx1[index-60:index]),idx1=np.array(self._idx2[index-60:index]))
            _remaining=SAFE
        _remaining-=1
        current_spread=self._beta*element[idx[1]]
        if(self._current_position[idx[0]]>0 and element[idx[0]]-current_spread>=self._bl):
            self._current_position={idx[0]: 0, idx[1]: 0}
            self._state="inactive"
        if(self._current_position[idx[0]]<0 and current_spread<=self._bs):
            self._current_position={idx[0]: 0, idx[1]: 0}
            self._state="inactive"
        if(current_spread<=self._al):
            #?????this should depends on capital
            self._current_position[idx[0]]+=1
            self._current_position[idx[1]]-=self._beta*1
            self._state="active"
            return self._current_position.copy()
        if(current_spread>=self._as):
            self._current_position[idx[0]]-=1
            self._current_position[idx[1]]+=self._beta*1
            self._state="active"
            return self._current_position.copy()

            

    def callibration(self,idx0,idx1):
        idx0=np.log(idx0)
        idx1=np.log(idx1)
        def given_beta(b):
            xab=np.array([idx0[i]-idx1[i]*b for i in range(0,DAYS)])
            xx=sum(xab[:-1])
            xy=sum(xab[1:])
            xxx=sum(xab**2[:-1])
            xxy=xab.T[:-1]@ xab.T[1:]
            xyy=xab.T[1:]@ xab.T[1:]
            # theta=
            # miu=
            sigma=np.sqrt()
        #??????same as linear regression OLS?
        reg = LinearRegression(fit_intercept=False).fit(idx1.reshape(-1, 1), idx1)
        self._beta=reg.coef_


        pass

    def stop_loss():
        #TODO
        pass
        

