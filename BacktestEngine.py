#--- Backtesting and execution engine ------#
import pandas as pd
import numpy as np
from IStrategy import IStrategy

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

    def run_backtest(self):
        

        


        
    
                


    



