#------ strategy interface --------#
import numpy as np

class IStrategy():

    # local data object to store past data
    _data = []
    _state = "inactive"
    _weight = 1.0
    _capital = 100
    _PnL = 0

    def __init__(self, data = [], state="active", weight = 1.0, capital=100):
        self._data = data
        self._state = state
        self._weight = weight
        self._capital = capital

    # to be implemented
    def add_data(self,element):
        self._data.append(element)
    
    # to be implemented    
    def generate_signal(element) -> tuple:
        pass

    def get_state(self) -> str:
        return self._state

    def set_state(self,state) -> bool:
        self._state = state

    # to be implmented
    def add_data_rule(element):
        pass
    

# class test(IStrategy):
#     def __init__(self):
#         super().__init__()

#     def add_data(self,elem):
#         self._data.append(100)


# t = test()
# t.add_data(10)
# print(t._data)





