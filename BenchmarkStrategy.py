import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IStrategy import *


class BenchmarkStrategy(IStrategy):
    def __init__(self, data=pd.DataFrame, name='Benchmark(Distance)',capital=0):
        super().__init__(data=data, name=name, capital=capital)
        self._factor = 0.5  # number of standard deviations as threshold
        self._length = data.shape[0]
        self._spread = []
        self._thresh = 0

    def get_length(self):
        return self._length

    def get_spread(self):
        df = self._data
        idx1 = df[df.columns[0]]
        idx2 = df[df.columns[1]]
        n = self._length
        spread = np.zeros(n)

        for i in range(n):
            norm1 = (idx1[i] - min(idx1)) / (max(idx1) - min(idx1))
            norm2 = (idx2[i] - min(idx2)) / (max(idx2) - min(idx2))
            spread[i] = norm1 - norm2

        return spread

    def get_thresh(self):
        thresh = self._factor * np.std(self._spread)
        return thresh

    def generate_signal(self, element, index):
        if index == 0:
            self._spread = self.get_spread()
            self._thresh = self.get_thresh()

        idx = [asset for asset in element]

        # Enter long position
        if self._spread[index] < -self._thresh and self._state == 'inactive':
            self._state = 'long'
            return {idx[0]: 1, idx[1]: -1}

        # Close long position
        if self._spread[index] > 0 and self._state == 'long':
            self._state = 'inactive'
            return {idx[0]: -1, idx[1]: 1}

        # Enter short position
        if self._spread[index] > self._thresh and self._state == 'inactive':
            self._state = 'short'
            return {idx[0]: -1, idx[1]: 1}

        # Close short position
        if self._spread[index] < 0 and self._state == 'short':
            self._state = 'inactive'
            return {idx[0]: 1, idx[1]: -1}

        return {idx[0]: 0, idx[1]: 0}

