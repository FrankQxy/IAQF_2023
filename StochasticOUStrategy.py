from BenchmarkStrategy import *
from IStrategy import *
from BacktestEngine import *
from MarketUtils import *
from MarketData import *
from Configuration import *
from AnalyticsEngine import *
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize,minimize_scalar
from sympy import symbols, solve
import math
from mpmath import nsum, inf




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
    _spreads=[]

    #?????maybe weight can be outside of strategy
    def __init__(self, data = [], state="inactive", weight = 1.0, capital=100):
        #?????I suppose theh data we are given only contains the two columns of the two indices we are trading
        super().__init__(data=data, state=state, weight=weight, capital=capital, name = "Stochastic--OU")
        self._data={}

    def generate_signal(self,element):
        idx = [asset for asset in element]
        #storing data
        if(len(self._data)==0):
            self._data[idx[0]]=[]
            self._data[idx[1]]=[]
        else:
            self._data[idx[0]].append(element[idx[0]])
            self._data[idx[1]].append(element[idx[1]])
        if(len(self._data[idx[0]])<DAYS+1):
            self._current_position={idx[0]: 0, idx[1]: 0}
            self._state="inactive"
            return {idx[0]: 0, idx[1]: 0}
        if(self._remaining==0):
            #can use up to index-1 to callibrate
            self.callibration(idx0=np.array(self._data[idx[0]][-61:-1]),idx1=np.array(self._data[idx[1]][-61:-1]))
            print("self._beta   ",self._beta)
            self._remaining=SAFE
        self._remaining-=1
        current_spread=np.log(element[idx[0]])-self._beta*np.log(element[idx[1]])
        self._spreads.append(element[idx[0]]-self._beta*element[idx[1]])
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

        #for testing
        DAYS=6


        idx0=np.log(idx0)
        idx1=np.log(idx1)
        n=len(idx0)
        def given_beta(b):
            dt=1
            xab=np.array([idx0[i]-idx1[i]*b for i in range(0,DAYS)])
            # print(xab)
            xx=sum(xab[:-1])
            xy=sum(xab[1:])
            xxx=sum((xab**2)[:-1])
            xxy=np.dot(xab[:-1],xab[1:])
            xyy=np.dot(xab[1:],xab[1:])
            theta=(xy*xxx-xx*xxy)/(n*(xxx-xxy)-(xx**2-xx*xy))
            # print("@@@@@@@@@@@@@@@@@@@@@@ theta   ",theta)

            miu=-(1/dt)*np.log((xxy-theta*xx-theta*xy+n*(theta)**2)/(xxx-2*theta*xx+n*(theta)**2))
            # print("@@@@@@@@@@@@@@@@@@@@@@ miu   ",miu)

            sigma=np.sqrt(((2*miu)/(n*(1-np.exp(-2*miu*dt))))*(xyy-2*np.exp(-miu*(dt))*xxy+np.exp(-2*miu*dt)*xxx
            -2*theta*(1-np.exp(-miu*dt))*(xy-np.exp(-miu*dt)*xx)+n*(theta)**2*(1-np.exp(-miu*dt))**2))
            # print("@@@@@@@@@@@@@@@@@@@@@@ sigma   ",sigma)


            return theta, miu, sigma

        def l(b):
            #repeat
            dt=1
            xab=np.array([idx0[i]-idx1[i]*b for i in range(0,len(idx0))])

            theta, miu, sigma = given_beta(b)
            sigma_c = np.sqrt(sigma**2*((1-np.exp(-2*miu*dt))/(2*miu)))

            summation=0
            for i in range(1,len(xab)):
                summation = summation + (xab[i]-xab[i-1]*np.exp(-miu*dt)-theta*(1-np.exp(-miu*dt)))**2
            return -0.5*(np.log(2*np.pi))-np.log(sigma_c)-summation/(2*n*(sigma_c**2))
        
        def neg_l(b):
            # print("                      b  ",-l(b))
            return -l(b)

        self._beta = minimize_scalar(neg_l).x
        
        self._theta, self._miu, self._sigma = given_beta(self._beta)

        # #??????same as linear regression OLS?
        # reg = LinearRegression(fit_intercept=False).fit(idx1.reshape(-1, 1), idx0)
        # print("!!!!!!!!!!!!!!!!!!!OLS Beta ",reg.coef_)
        # print("                      OLS  ",-l(reg.coef_[0]))

        # #updating boundaries
        # a = symbols('a')
        # #define equation
        # def equation(a):
        #     def summand_1(n):
        #         product1=((np.sqrt(2)*a)**(2*int(n)+1))/math.factorial(2*int(n)+1)
        #         product2=math.gamma((2*n+1)/2)
        #         return product1*product2
        #     sum1=sum([summand_1(i) for i in range(0,10)])
        #     def summand_2(n):
        #         product1=((np.sqrt(2)*a)**(2*int(n)))/math.factorial(2*int(n))
        #         product2=math.gamma((2*n+1)/2)
        #         return product1*product2
        #     sum2=sum([summand_2(i) for i in range(0,10)])
        #     return 0.5*(sum1)-a*(np.sqrt(2)/2)*sum2
        # #solve the equation
        # expr = equation(a)
        # sol = solve(expr)
        # a_d = sol[0]
        # b_d = -a_d
        # def d_(k_d):
        #     return k_d*self._sigma/(np.sqrt(2*self._miu))+self._theta
        # self._as=d_(a_d)
        # self._bs=d_(b_d)
        # self._al=d_(-a_d)
        # self._bl=d_(-b_d)
        # print(self._as,self._bs,self._al,self._bl)
        # #calculating the boundaries

    def get_spreads(self) -> pd.DataFrame:
        return pd.DataFrame(self._spreads)
        

    def stop_loss():
        #TODO
        pass



# test = StochasticOUStrategy()
# test.callibration(np.array([220,190,220,190,220,190]),np.array([110,101,100,100,100,110]))
# print(test._beta)

price_data = get_data(type='index', col_list=['^GSPC', '^IXIC'], termDates=['2022-01-04','2022-12-30'])
strategy = StochasticOUStrategy()
backtest = backtest_walk_forward(price_data)
backtest.add_strategy(strategy)
trades = backtest.run_backtest()
print(trades)
spread = strategy.get_spreads()
for i in range(59,187):
    print(spread[i])
