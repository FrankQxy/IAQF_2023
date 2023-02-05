#------ strategy interface --------#
import numpy as np

class IStrategy():

    # local data object to store past data
    _data = []
    _state = "inactive"
    _weight = 1.0
    _capital = 100
    _PnL = 0
    _name = ""


    def __init__(self, data = [], state="inactive", weight = 1.0, capital=100, name = "PCA"):
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
    def generate_signal(element) -> dict:
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





