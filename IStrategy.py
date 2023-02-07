#------ strategy interface --------#
import numpy as np

class IStrategy():

    # local data object to store past data
    _data = [] 
    _state = "inactive" 
    _weight = 1.0 
    _capital = 100 
    _PnL = 0 # pnl, not needed right now, but again might need it
    _name = "" 
    _trades = [] # trades in the format in a list of dicts {"^DJI":100}

    def __init__(self, data = [], state="inactive", weight = 1.0, capital=100, name = "PCA"):
        """initialize your strategy

        Args:
            data (list, optional): stores past data for processing could be in any datatype . Defaults to [].
            state (str, optional): indicates when a strategy is active, when you enter a position you mark it active and inactive when you exit. Defaults to "inactive".
            weight (float, optional): weight assigned from asset allocation, we might not need it now. Defaults to 1.0.
            capital (int, optional): more flexibility just in case we need capital as input as well. Defaults to 100.
            name (str, optional): identifier of your strategy for backtester to output strategy specific results. Defaults to "PCA".
        """
        self._data = data
        self._state = state
        self._weight = weight
        self._capital = capital
        self._name = name

    def __str__(self) ->str:
        return self._name

    # to be implemented
    def add_data(self,element):
        self._data.append(element)
    
    # to be implemented    
    def generate_signal(self, element) -> dict:
        """generate signal from curretn price levels

        Args:
            element (dict): format asset:price

        Returns:
            dict: format asset:position, negative for short and positive for long
        """
        pass

    def get_state(self) -> str:
        return self._state

    def set_state(self,state):
        self._state = state

    # to be implmented
    def add_data_rule(element):
        """custom rule to add data like past 10 day price series

        Args:
            element (dict): current price levels or something else
        """
        pass
    
    # to be implemented
    def add_trade(self,trade):
        """add new trade to trades database

        Args:
            trade (dict): dict format asset:position, negative for short and positive for long
        """
        self._trades.append(trade)


# class test(IStrategy):
#     def __init__(self):
#         super().__init__()

#     def add_data(self,elem):
#         self._data.append(100)


# t = test()
# t.add_data(10)
# print(t._data)





