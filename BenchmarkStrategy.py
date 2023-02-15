import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IStrategy import *


class BenchmarkStrategy(IStrategy):
    def __init__(self, name='BenchmarkStrategy', data=[]):
        super().__init__(name=name, data=data)
        self._factor = 2  # number of standard deviations as threshold
        self._trainlen = 260
        self._counter = 0
        self._idx1 = ''
        self._idx2 = ''
        self._idx1_price = []
        self._idx2_price = []

    def is_ready(self):
        return self._counter > self._trainlen

    def add_price_pair(self, element):
        self._idx1_price.append(element[self._idx1])
        self._idx2_price.append(element[self._idx2])

    def remove_price_pair(self):
        self._idx1_price.pop(0)
        self._idx2_price.pop(0)

    def unready_signal(self, element):
        if self._counter == 0:
            self._idx1 = list(element)[0]
            self._idx2 = list(element)[1]
        self._counter += 1
        self.add_price_pair(element)
        return {self._idx1: 0, self._idx2: 0}

    def get_thresh(self, element):
        p1 = np.array(self._idx1_price)
        p2 = np.array(self._idx2_price)

        norm1 = (p1 - max(p1)) / (max(p1) - min(p1))
        norm2 = (p2 - max(p2)) / (max(p2) - min(p2))
        norm_diff = norm1 - norm2

        thresh = self._factor * np.std(norm_diff)
        return thresh

    def save_spread(self, plot=True, name='BenchmarkSpread'):
        idx1 = self._data.columns[0]
        idx2 = self._data.columns[1]
        spread = self._data[idx1] - self._data[idx2]
        spread.name = 'Spread'
        path = './Benchmark/' + name + '_' + idx1[1:] + '_' + idx2[1:]
        spread.to_csv(path + '.csv')

        if plot:
            n = len(spread)
            plt.figure(figsize=(12, 8))
            plt.title('Distance approach SP500 - NASDAQ spread', size=20)
            plt.plot(spread.index, spread)
            plt.xticks(spread.index[::n // 6])
            plt.xlabel('time', size=15)
            plt.ylabel('spread', size=15)
            plt.savefig(path + '.png')

    def generate_signal(self, element):
        if not self.is_ready():
            return self.unready_signal(element)

        self.remove_price_pair()
        self.add_price_pair(element)
        thresh = self.get_thresh(element)
        norm_p1 = (element[self._idx1] - max(self._idx1_price)) / (max(self._idx1_price) - min(self._idx1_price))
        norm_p2 = (element[self._idx2] - max(self._idx2_price)) / (max(self._idx2_price) - min(self._idx2_price))
        spread = norm_p1 - norm_p2

        # Enter long position
        if spread < -thresh and self._state == 'inactive':
            self._state = 'long'
            return {self._idx1: 1, self._idx2: -1}

        # Close long position
        if spread > 0 and self._state == 'long':
            self._state = 'inactive'
            return {self._idx1: -1, self._idx2: 1}

        # Enter short position
        if spread > thresh and self._state == 'inactive':
            self._state = 'short'
            return {self._idx1: -1, self._idx2: 1}

        # Close short position
        if spread < 0 and self._state == 'short':
            self._state = 'inactive'
            return {self._idx1: 1, self._idx2: -1}

        return {self._idx1: 0, self._idx2: 0}
