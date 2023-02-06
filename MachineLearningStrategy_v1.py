import numpy as np
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, accuracy_score, r2_score
from IStrategy import IStrategy

class NaiveMachineLearningStrategy(IStrategy):
    def __init__(self):
        super().__init__()
        self.model = None
        self.spread_std = self.model_training_get_spread_std()
        self._name = 'Machine Learning Strategy'
        self.state = 'inactive'

    def model_training_get_spread_std(self):
        X_train = np.array(self._data.X)
        y_train = np.array(self._data.y)
        #Use a Polynomial Regression with LASSO Regulations to determine spread
        poly = PolynomialFeatures(degree=4, include_bias=False)
        poly_features = poly.fit_transform(X_train.reshape(-1, 1))
        clf_Lasso = linear_model.Lasso(alpha=0.05) #Lasso
        clf_Lasso.fit(poly_features, y_train)
        #save trained model
        self.model = clf_Lasso
        #Get standard deviation for historical spread
        y_train_pred = clf_Lasso.predict(poly_features)
        spread = y_train-y_train_pred
        std = np.std(spread)
        
        return std

    def compute_z_score(self, backtesting_prices):
        X_test = np.array(backtesting_prices.X)
        y_test = np.array(backtesting_prices.y)
        poly = PolynomialFeatures(degree=4, include_bias=False)
        poly_features = poly.fit_transform(X_test.reshape(-1, 1))
        y_test_pred = self.model.predict(poly_features)
        #Calculate spread and z-score for testing data
        spread = y_test-y_test_pred
        z_score = spread/spread_std

        return z_score

    def generate_signal(self, backtesting_prices):
        # Currently the weight is 1 or -1 which indicates long and short
        # the two indices with equal weights
        n = len(backtesting_prices)
        strategy = {'asset x':np.zeros(n), 'asset y':np.zeros(n)}
        z_score = self.compute_z_score(backtesting_prices)
        for i in range(1,len(z_score)):
            if z_score[i] > 1.5 or (strategy['asset x'][i-1] == -1 and z_score[i] > 0):
                strategy['asset x'][i] == -1
                strategy['asset y'][i] == 1
                self.state = 'active'
                
            elif z_score[i] < -1.5 or (strategy['asset x'][i-1] == 1 and z_score[i] < 0):
                strategy['asset x'][i] == 1
                strategy['asset y'][i] == -1
                self.state = 'active'

            else:
                self.state = 'inactive'

        return strategy
    
Footer
© 2023 GitHub, Inc.
Footer navigation
Terms
