#--- Backtesting and execution engine ------#
import pandas as pd
import numpy as np
from IStrategy import IStrategy

# pass data a dictionary of levels and signal as dictionary of positions
# positive is buy negative is sell
def execution_engine(self,data,signal,strategy,risk_manager = lambda x:x) -> float:

    signal = risk_manager.process_signal(signal,data)
    cash_flow = 0.0
    for asset in data:
        price = data[asset]
        position = signal[asset]
        cash_flow += position*price

    if strategy._capital + cash_flow < 0:
       raise(f"not enough capital to execute order {str(data)}")
     
    strategy._capital += cash_flow
    strategy._trade.append(signal)
    return cash_flow
        
class backtest_walk_forward():

    _QManager = []
    _data = []
    _capital = 100
    _assets = []
    _risk_manager = []

    def __init__(self,data,assets,capital=100, risk_manager = lambda x:x):
        self._data = data
        self._capital = capital
        self._assets = assets
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

    def run_backtest(self):
        


        
    
                


    



