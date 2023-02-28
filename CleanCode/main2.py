import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, accuracy_score, r2_score
from sklearn.preprocessing import PolynomialFeatures
import scipy as scp
import scipy.stats as ss
from scipy.optimize import minimize
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpl_toolkits import mplot3d
from matplotlib import cm
import scipy.special as scsp
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from MarketUtils import *
import pyfolio as pf
from datetime import datetime
from ouparams import ouparams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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
from copulas.multivariate import  VineCopula



def misprice_index(u,v,copula,theta):
        '''Calculate mispricing index for every day in the trading period by using estimated copula
        Mispricing indices are the conditional probability P(U < u | V = v) and P(V < v | U = u)'''

        if copula == 'clayton':
            MI_u_v = v ** (-theta - 1) * (u ** (-theta) + v ** (-theta) - 1) ** (
                        -1 / theta - 1)  # P(U<u|V=v)
            MI_v_u = u ** (-theta - 1) * (u ** (-theta) + v ** (-theta) - 1) ** (
                        -1 / theta - 1)  # P(V<v|U=u)

        elif copula == 'frank':
            A = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) + (np.exp(-theta * v) - 1)
            B = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) + (np.exp(-theta * u) - 1)
            C = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) + (np.exp(-theta) - 1)
            MI_u_v = B / C
            MI_v_u = A / C

        elif copula == 'gumbel':
            A = (-np.log(u)) ** theta + (-np.log(v)) ** theta
            C_uv = np.exp(-A ** (1 / theta))  # C_uv is gumbel copula function C(u,v)
            MI_u_v = C_uv * (A ** ((1 - theta) / theta)) * (-np.log(v)) ** (theta - 1) * (1.0 / v)
            MI_v_u = C_uv * (A ** ((1 - theta) / theta)) * (-np.log(u)) ** (theta - 1) * (1.0 / u)

        return MI_u_v , MI_v_u 

def _lpdf_copula(family, theta, u, v):
        '''Estimate the log probability density function of three kinds of Archimedean copulas
        '''

        if family == 'clayton':
            pdf = (theta + 1) * ((u ** (-theta) + v ** (-theta) - 1) ** (-2 - 1 / theta)) * (
                        u ** (-theta - 1) * v ** (-theta - 1))

        elif family == 'frank':
            num = -theta * (np.exp(-theta) - 1) * (np.exp(-theta * (u + v)))
            denom = ((np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) + (np.exp(-theta) - 1)) ** 2
            pdf = num / denom

        elif family == 'gumbel':
            A = (-np.log(u)) ** theta + (-np.log(v)) ** theta
            c = np.exp(-A ** (1 / theta))
            pdf = c * (u * v) ** (-1) * (A ** (-2 + 2 / theta)) * ((np.log(u) * np.log(v)) ** (theta - 1)) * (1 + (theta - 1) * A ** (-1 / theta))

        return np.log(pdf)

def _parameter(family, tau):
        ''' Estimate the parameters for three kinds of Archimedean copulas
        according to association between Archimedean copulas and the Kendall rank correlation measure
        '''

        if family == 'clayton':
            return 2 * tau / (1 - tau)

        elif family == 'frank':
            '''
            debye = quad(integrand, sys.float_info.epsilon, theta)[0]/theta  is first order Debye function
            frank_fun is the squared difference
            Minimize the frank_fun would give the parameter theta for the frank copula 
            '''
            integrand = lambda t: t / (np.exp(t) - 1)  # generate the integrand
            frank_fun = lambda theta: ((tau - 1) / 4.0 - (
                        quad(integrand, sys.float_info.epsilon, theta)[0] / theta - 1) / theta) ** 2

            return minimize(frank_fun, 4, method='BFGS', tol=1e-5).x[0]

        elif family == 'gumbel':
            return 1 / (1 - tau)

def get_strategy_copula2(x_train, y_train, x_test, y_test, isconver = True):

    #Get standard deviation for historical spread
    scaler = StandardScaler()
    x_train = np.diff(np.log(x_train))[1:]
    y_train = np.diff(np.log(y_train))[1:]
    x_train = scaler.fit_transform(x_train.reshape(-1, 1))
    ecdfx = ECDF(x_train.flatten())
    ecdfy = ECDF(y_train.flatten())
    u, v = [ecdfx(a) for a in x_train], [ecdfy(a) for a in y_train]
    tau = kendalltau(x_train, y_train)[0]  # estimate Kendall'rank correlation
    AIC = {}  # generate a dict with key being the copula family, value = [theta, AIC]
    s = ['clayton','frank', 'gumbel']
    for i in s:
        param = _parameter(i, tau)
        lpdf = [_lpdf_copula(i, param, x, y) for (x, y) in zip(u, v)]
        lpdf = np.nan_to_num(lpdf)
        loglikelihood = sum(lpdf)
        AIC[i] = [param, -2 * loglikelihood + 2]

    copula = min(AIC.items(), key=lambda x: x[1][1])[0]
    tau = kendalltau(x_train, y_train)[0]
    theta = _parameter(copula, tau)
    #Backtesting
    x_test = np.diff(np.log(x_test))[1:]
    x_test = scaler.transform(x_test.reshape(-1, 1))
    y_test = np.diff(np.log(y_test))[1:]

    # Convert the two returns to uniform values u and v using the empirical distribution functions
    x_test = ecdfx(x_test.flatten())
    y_test = ecdfy(y_test.flatten())
    
    M_x = np.array([misprice_index(x,y,copula,theta)[0] for x,y in zip(x_test,y_test)]) - 0.5
    M_y = np.array([misprice_index(x,y,copula,theta)[1] for x,y in zip(x_test,y_test)]) - 0.5
    flagx = M_x.cumsum()
    flagy = M_y.cumsum()


    #Get Strategy with Risk Management
    strategy = np.zeros(len(y_test))
    if isconver:
        for i in range(len(y_test) - len(flagx), len(flagx)):     
            if (flagx[i] > 0.6 and flagy[i] < -0.6) or (strategy[i-1] == -1 and (flagx[i] > 0 or flagy[i] < 0)):
                strategy[i] = -1
                
            elif (flagy[i] > 0.6 and flagx[i] < -0.6) or (strategy[i-1] == 1 and (flagy[i] > 0 or flagx[i] < 0)):
                strategy[i] = 1 

    else:
        for i in range(len(y_test) - len(flagx), len(flagx)):     
            if (flagx[i] > 0.6 or flagy[i] < -0.6) and strategy[i-1] == 0:
                strategy[i] = 1
                
            elif (flagy[i] > 0.6 or flagx[i] < -0.6) and strategy[i-1] == 0:
                strategy[i] = -1

            elif strategy[i-1] == 1 and (flagx[i] < 2 or flagy[i] > -2):
                strategy[i] = 1

            elif strategy[i-1] == -1 and (flagy[i] < 2 or flagx[i] > -2):
                strategy[i] = -1
                         
    return flagx, strategy


def calculate_return(strategy, x_test, y_test):
    returns = []
    short = False
    long = False

    for i in range(len(strategy)):
        #Case1: Open Short Position
        if strategy[i] < 0 and not short:
            returns.append(0)
            short = True
          
        #Case2: Open Long Position
        elif strategy[i] > 0 and not long:
            returns.append(0)
            long = True
    
        #Case3: Holding Short Position
        elif short:
            daily_return = 0.5*((x_test[i-1]-x_test[i])/x_test[i]) + 0.5*((y_test[i]-y_test[i-1])/y_test[i-1])
            returns.append(daily_return*abs(strategy[i-1]))
            #Exit
            if strategy[i] == 0:
                short = False
            else:
                continue
    
        #Case4: Holding Long Position
        elif long:
            daily_return = 0.5*((y_test[i-1]-y_test[i])/y_test[i]) + 0.5*((x_test[i]-x_test[i-1])/x_test[i-1])
            returns.append(daily_return*abs(strategy[i-1]))
            #Exit
            if strategy[i] == 0:
                long = False
            else:
                continue
            
        else:
            returns.append(0)
        
    return returns

#Get Data and set parameters
indices = ['^GSPC','^IXIC']
#Don't change this, we will keep the rest of the data as test data
termDates = ['01-01-2001','01-10-2005']
# termDates = ['01-01-2002','01-10-2022']
price_data = get_data(type ='index',col_list = indices, termDates = termDates)
alter_data = get_data(type = 'alter', termDates=termDates)
intersect = price_data.index.intersection(alter_data.index)
price_data = price_data.loc[intersect.values]
alter_data["Flat"] = alter_data['Yield_30Y'] - alter_data['fedfunds']
alter_data["vol_spread"] = alter_data[indices[0]] - alter_data[indices[1]]
alter_data["vix"] = alter_data[indices[0]]
alter_cols = ['fedfunds','vix','vol_spread','Flat','News Sentiment']
alter_data = alter_data[alter_cols]
alter_data = alter_data.loc[intersect.values]
# can change this as per your requirement
spy = price_data[indices[0]]
qqq = price_data[indices[1]]
spy_price = np.array(spy)
qqq_price = np.array(qqq)

#Train, Test split
x = np.array(spy_price)
y = np.array(qqq_price)

#trading params
formation = 252
trading_period = 126

alter_data = alter_data.values
all_returns = []
all_spread = []
all_startegy = []
sp2 = []
i = 0
while(i < len(x)- formation - trading_period):
    x_train, x_test = x[i:i+formation], x[i+formation:i+formation+trading_period]
    y_train, y_test = y[i:i+formation], y[i+formation:i+formation+trading_period]  
    # temp_spread,temp_s2,strategy = get_strategy_copula2(x_train, y_train, x_test, y_test, isconver=False)
    # temp_returns = calculate_return(strategy, x_test, y_test)
    # sp2.extend(temp_s2)
    temp_spread,strategy = get_strategy_copula2(x_train, y_train, x_test, y_test, isconver=True)
    temp_returns = calculate_return(strategy, x_test, y_test)
    all_startegy.extend(strategy)
    all_spread.extend(temp_spread)
    all_returns.extend(temp_returns)
    i += trading_period

dates = np.asarray(price_data.index.values[formation:], dtype='datetime64[s]')
dates = pd.to_datetime(price_data.index.values[formation:])
rets = pd.Series(data = all_returns, index = dates[-len(all_spread):])
rets = rets.tz_localize('UTC')
fig = pf.create_returns_tear_sheet(returns=rets, return_fig=False)
for ax in fig.axes:
        ax.tick_params(
        axis='x',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        bottom=True,
        top=False,
        labelbottom=True)    # labels along the bottom edge are on

fig.savefig("Plain_Copula.png")