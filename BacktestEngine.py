#--- Backtesting and execution engine ------#
import pandas as pd
import numpy as np
from IStrategy import *
from MarketUtils import *
from RiskManagement import *
from collections import counter

def execution_engine(data,signal) -> float:
    """Executes trade from the signal

    Args:
        data (dict): format asset:price
        signal (dict): format asset:position, negative for short and positive for long

    Returns:
        float: cash flow from trade execution
    """
    cash_flow = 0.0
    for asset in data:
        price = data[asset]
        position = signal[asset]
        cash_flow += -position*price

    return cash_flow
        
class backtest_walk_forward():
    """Simple walk-forward backtesting engine could be used for Monte Carlo paths
    """
    _QManager = [] # queue of strategy objects
    _price_data = [] 
    _alter_data = []
    _capital = 100 
    _risk_manager = []

    def __init__(self,price_data,alter_data,capital=100,risk_manager = lambda x:x):
        """Initialization

        Args:
            price_data (DataFrame): date , name1, name2, ....
            alter_data (DataFrame): similar to price data
            capital (int, optional): _description_. Defaults to 100.
            risk_manager (risk_manager object, optional): _description_. Defaults to lambdax:x.
        """
        self._price_data = price_data
        self._alter_data = alter_data
        self._capital = capital
        self._risk_manager = risk_manager

    def add_strategy(self, strategy):

        if isinstance(strategy,IStrategy) and strategy not in self._QManager:
            self._QManager.append(IStrategy) 
        else:
            raise("Pass IStrategy type object or the startegy is already there.")
        return self

    def remove_startegy(self,strategy): 

        if isinstance(strategy,IStrategy) and strategy not in self._QManager:
            self._QManager.remove(IStrategy) 
        else:
            raise("Pass IStrategy type object or the startegy is not there")
        return self

    def run_backtest(self) -> pd.DataFrame:

        assets = list(self._price_data.copy())
        trades = pd.DataFrame(index=self._price_data.index.copy())
        trades[[asset + "signal" for asset in assets]] = 0
        trades["capital"] = self._capital

        for index,row in self._price_data:
            trade = {}
            curr_price = {}

            for asset in assets:
                curr_price[asset] = row[asset]

            for strategy in self._QManager:
                
                signal = strategy(curr_price)
                # maybe add risk manager here
                trade = dict(counter(trade) + counter(signal))

            # maybe add risk manager here
            cash_flow = execution_engine(curr_price, trade)
            for asset in trade:
                trades.loc[index,asset+"signal"] = trade[asset]
            
            self._capital += cash_flow
            trades[index,"capital"] = self._capital

        return trades

    def save_trades(self,trades):
        path = BASE_DIR + f'\\backtest_{TIMESTAMP()}.csv'
        trades.to_csv(path)
    








    



