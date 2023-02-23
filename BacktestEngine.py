#--- Backtesting and execution engine ------#
import pandas as pd
import numpy as np
from IStrategy import *
from MarketUtils import *
# from RiskManagement import *
from collections import Counter

def execution_engine(data,signal,positions) -> float:
    """Executes trade from the signal

    Args:
        data (dict): format asset:price
        signal (dict): format asset:position, negative for short and positive for long

    Returns:
        float: cash flow from trade execution
    """
    cash_flow = 0.0
    holdings = 0.0
    for asset in signal:
        price = data[asset]
        position = signal[asset]
        cash_flow += -position*price
        holdings += positions[asset]*price

    return cash_flow, holdings
        
class backtest_walk_forward():
    """Simple walk-forward backtesting engine could be used for Monte-Carlo paths
    """
    _QManager = [] # queue of strategy objects
    _price_data = [] 
    _alter_data = []
    _capital = 100 

    def __init__(self,price_data,alter_data,capital=0):
        """Initialization

        Args:
            price_data (DataFrame): date , name1, name2, ....
            alter_data (DataFrame): similar to price data
            capital (int, optional): portfolio value tracker. Defaults to 100.
            risk_manager (risk_manager object, optional): Will work on later. Defaults to lambdax:x.
        """
        if(len(price_data) != len(alter_data)) : raise("Data not updated correctly.")
        self._price_data = price_data
        self._capital = capital     # this will override the capital in strategy
        self._portfolio = 1
        self._position = {}
        self._alter_data = alter_data


    def add_strategy(self, strategy):

        if isinstance(strategy,IStrategy) and strategy not in self._QManager:
            self._QManager.append(strategy) 
        else:
            raise("Pass IStrategy type object or the startegy is already there.")
        return self

    def remove_startegy(self,strategy): 

        if isinstance(strategy,IStrategy) and strategy not in self._QManager:
            self._QManager.remove(strategy) 
        else:
            raise("Pass IStrategy type object or the startegy is not there")
        return self

    def run_backtest(self) -> pd.DataFrame:
        """primary backtesting function walk-forward method

        Returns:
            pd.DataFrame: positions on each asset and portfolio value
        """
        assets = list(self._price_data.copy())
        self._position = {assets[0]:0,assets[1]:0}
        # setting up trade dataframe
        trades = pd.DataFrame(index=self._price_data.index.copy())

        # iterating over each row in price data might update with a more optimal code
        for index,row in self._price_data.iterrows():
            curr_price = {}
            trades
            # constructing current price dict
            for asset in assets:
                curr_price[asset] = row[asset]

            # executing all strategies in queue
            for strategy in self._QManager:
                
                trade = strategy.generate_signal(curr_price, self._alter_data.loc[index,:])
                
            # position tracker
            for asset in assets:
                self._position[asset] += trade[asset]
           
            # executes trade and returns cash flow
            cash_flow, holdings = execution_engine(curr_price, trade, self._position)

            # updates trade object with final positions
            for asset in trade:
                trades.loc[index,asset+"_Signal"] = trade[asset]
            
            # calculates protfolio value
            self._capital += cash_flow
            trades.loc[index,"capital"] = self._capital
            trades.loc[index,"Portfolio"] = self._capital + holdings

        # trades["dailyReturn"] = (trades["Portfolio"] - self._capital)
        # trades["dailyReturn"] = trades["dailyReturn"].pct_change()
        # trades.drop(trades.index[0], inplace = True)
        # trades['cummulative_return'] = (1 + trades["dailyReturn"]).cumprod() - 1
        return trades

    def save_trades(self,trades,filename='backtest_'):
        """saves trade data to csv
        Args:
            trades (pd.DataFrame): output from run_backtest
        """
        path = DATA_PATH / (filename + f'{TIMESTAMP()}.csv')
        trades.to_csv(path)
    








    



