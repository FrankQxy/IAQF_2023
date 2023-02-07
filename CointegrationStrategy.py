import numpy as np
import statsmodels.api as sm
from IStrategy import IStrategy


class CointegrationStrategy(IStrategy):
    def __init__(self):
        super().__init__()
        self._trading_ratio = self.compute_trading_ratio()
        self._spread = self.compute_spread()
        self._n = self._data.shape[0]
        self._threshold = self.compute_threshold()

    def compute_trading_ratio(self):
        # compute the trading ratio
        model = sm.OLS(self._data.Y.iloc[:90], self._data.X.iloc[:90])
        model = model.fit()
        trading_ratio = model.params[0]
        return trading_ratio

    def get_trading_ratio(self):
        return self._trading_ratio

    def compute_spread(self):
        data = self._data
        trading_ratio = self._trading_ratio
        spread = data.Y - trading_ratio * data.X
        return spread

    def get_spread(self):
        return self._spread

    def compute_threshold(self):
        thresh = np.std(self._spread)
        self._threshold = thresh
        return thresh

    def get_threshold(self):
        return self._threshold

    def generate_signal(self):
        # TODO: the strategy is -1 or 1, but the -1 and 1 here means a specific trading ratio
        strategy = np.zeros(self._n)

        for i in range(1, self._n):
            if self._spread[i] > self._threshold or (strategy[i - 1] == -1 and self._spread[i] > 0):
                strategy[i] = -1
            elif self._spread[i] < -self._threshold or (strategy[i - 1] == 1 and self._spread[i] < 0):
                strategy[i] = 1

        return strategy
