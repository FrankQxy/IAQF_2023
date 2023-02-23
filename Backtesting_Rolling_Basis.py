from MachineLearningStrategy_v2 import NaiveMachineLearningStrategy
import numpy as np
import pandas as pd
import pyfolio as pf

def Backtesting_Rolling_Basis(backtesting_prices, test_size, train_test_ratio):
    #Enter the entire dataset as backtesting_prices
    #Price data should be in the format of dataframe with three columns:
    #column 1: date, column 2: index_x, column 3: index_y
    all_returns = []
    bt_Dates = np.asarray(backtesting_prices['Date'][train_test_ratio*test_size:], dtype='datetime64[D]')
    for i in range((len(backtesting_prices)-train_test_ratio*test_size)//test_size):
        data_train, data_test = backtesting_prices[test_size*i:test_size*(i+train_test_ratio)], backtesting_prices[test_size*(i+train_test_ratio):test_size*(i+train_test_ratio+1)]
        #Initialize the strategy, change to your own class when testing
        my_strategy = NaiveMachineLearningStrategy(data = data_train)
        strategy = my_strategy.generate_signal_with_risk_management(data_test)
        x_test = np.array(data_test['index_x'])
        y_test = np.array(data_test['index_y'])
        temp_returns = Get_return_equal_weighted_portfolio(strategy, x_test, y_test)
        all_returns.extend(temp_returns)

    #Deal with the last incomplete period
    data_train, data_remain = backtesting_prices[test_size*(i+1):test_size*(i+train_test_ratio+1)], backtesting_prices[test_size*(i+train_test_ratio+1):]  
    my_strategy = NaiveMachineLearningStrategy(data = data_train)
    strategy = my_strategy.generate_signal_with_risk_management(data_remain, close = True)
    x_remain = np.array(data_remain['index_x'])
    y_remain = np.array(data_remain['index_y'])
    temp_returns = Get_return_equal_weighted_portfolio(strategy, x_remain, y_remain)
    all_returns.extend(temp_returns)
    
    #Backtesting Visualization using Pyfolio
    rets = pd.Series(data = all_returns, index = bt_Dates)
    rets = rets.tz_localize('UTC')
    pf.create_full_tear_sheet(returns=rets)

def Get_return_equal_weighted_portfolio(strategy, x_test, y_test):
    returns = []
    short = False
    long = False

    for i in range(len(strategy)):
        #Case1: Open Short Position
        if strategy[i] == -1 and not short:
            returns.append(0)
            short = True
          
        #Case2: Open Long Position
        elif strategy[i] == 1 and not long:
            returns.append(0)
            long = True
    
        #Case3: Holding Short Position
        elif short:
            daily_return = 0.5*((x_test[i-1]-x_test[i])/x_test[i]) + 0.5*((y_test[i]-y_test[i-1])/y_test[i-1])
            returns.append(daily_return)
            #Exit
            if strategy[i] == 0:
                short = False
            else:
                continue
    
        #Case4: Holding Long Position
        elif long:
            daily_return = 0.5*((y_test[i-1]-y_test[i])/y_test[i]) + 0.5*((x_test[i]-x_test[i-1])/x_test[i-1])
            returns.append(daily_return)
            #Exit
            if strategy[i] == 0:
                long = False
            else:
                continue
            
        else:
            returns.append(0)
        
    return returns

if __name__ == "__main__":
    spy = pd.read_csv('SPY.csv')
    qqq = pd.read_csv('QQQ.csv')
    spy_price = np.array(spy['Close'])
    qqq_price = np.array(qqq['Close'])
    dates = pd.to_datetime(spy['Date'])
    backtesting_prices = pd.DataFrame()
    backtesting_prices['Date'] = dates
    backtesting_prices['index_x'] = spy_price
    backtesting_prices['index_y'] = qqq_price
    Backtesting_Rolling_Basis(backtesting_prices, 21, 3)
