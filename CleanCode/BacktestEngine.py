#--- Backtesting and execution engine ------#
import pandas as pd
import numpy as np
from IStrategy import *
from MarketUtils import *
import pyfolio as pf
        
class backtest_walk_forward():

    """Simple walk-forward backtesting engine could be used for Monte-Carlo paths
    """

    def __init__(self,price_data,alter_data,strategy):
        """Initialization
        Args:
            price_data (DataFrame): date , name1, name2, ....
            alter_data (DataFrame): similar to price data
            capital (int, optional): portfolio value tracker. Defaults to 100.
            risk_manager (risk_manager object, optional): Will work on later. Defaults to lambdax:x.
        """

        if(len(price_data) != len(alter_data)) : raise("Data not updated correctly.")
        self._price_data = (price_data - price_data.mean())/price_data.std()
        self._alter_data = alter_data
        self._ret = []
        self._strategy = strategy

    def run_backtest(self) -> pd.DataFrame:
        """primary backtesting function walk-forward method

        Returns:
            pd.DataFrame: positions on each asset and portfolio value
        """
        assets = list(self._price_data.copy())
        self._position = {assets[0]:0,assets[1]:0}
        # setting up trade dataframe
        trades = pd.DataFrame(index=self._price_data.index.copy())
        rets = self._price_data.pct_change().dropna()
        # iterating over each row in price data might update with a more optimal code
        for index,row in self._price_data.iterrows():
            curr_price = {}

            # constructing current price dict
            for asset in assets:
                curr_price[asset] = row[asset]
            
            trade = self._strategy.generate_signal(curr_price, self._alter_data.loc[index,:])
                
            # position tracker
            ret = 0.0
            for asset in assets:
                try:
                    ret += self._position[asset]*rets.loc[index, asset]
                except:
                    pass

                self._position[asset] += trade[asset]
            
            self._ret.append(ret)
            
        return self._ret

    def save_trades(self,trades,filename='backtest_'):
        """saves trade data to csv
        Args:
            trades (pd.DataFrame): output from run_backtest
        """
        path = DATA_PATH / (filename + f'{TIMESTAMP()}.csv')
        trades.to_csv(path)
    
    def doAnalytics(self):
        bt_dates = self._price_data.index.values
        rets = pd.Series(data = self._ret, index = bt_dates)
        rets = rets.tz_localize('UTC')
        pf.create_full_tear_sheet(returns=rets)    







    



