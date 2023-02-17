import numpy as np
from sklearn.tree import DecisionTreeRegressor 
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from IStrategy import IStrategy

class NaiveMachineLearningStrategy(IStrategy):
    def __init__(self,  data):
        super().__init__(data = data)
        self.model = None
        self.spread_std = self.model_training_get_spread_std()
        self._name = 'Machine Learning Strategy'
        self.state = 'inactive'
        self.spread = []

    def model_training_get_spread_std(self):
        X_train = np.array(self._data['index_x'])
        y_train = np.array(self._data['index_y'])
        #Use a Polynomial Regression with LASSO Regulations to determine spread
        poly = PolynomialFeatures(degree=4, include_bias=False)
        poly_features = poly.fit_transform(X_train.reshape(-1, 1))
        clf_Lasso = linear_model.Lasso(alpha=0.05) #Lasso
        clf_Lasso.fit(poly_features, y_train)
        #save trained model
        self.model = clf_Lasso
        #Get standard deviation for historical spread
        y_train_pred = clf_Lasso.predict(poly_features)
        spread = y_train_pred-y_train
        std = np.std(spread)
        
        return std
    
    #After initialize the strategy, call this function to get spread of test data
    def get_test_spread(self, backtesting_prices):
        X_test = np.array(backtesting_prices['index_x'])
        y_test = np.array(backtesting_prices['index_y'])
        poly = PolynomialFeatures(degree=4, include_bias=False)
        poly_features = poly.fit_transform(X_test.reshape(-1, 1))
        y_test_pred = self.model.predict(poly_features)
        #Calculate spread for testing data
        spread = y_test_pred-y_test
        self.spread = spread
        
        return spread
        

    def compute_z_score(self, backtesting_prices):
        #Calculate spread and z-score for testing data
        spread = self.get_test_spread(backtesting_prices)
        z_score = spread/self.spread_std

        return z_score

    def generate_signal(self, backtesting_prices):
        # Currently the weight is 1 or -1 which indicates long and short
        # the two indices with equal weights
        #strategy = {'asset x':[0], 'asset y':[0]}        
        z_score = self.compute_z_score(backtesting_prices)
        strategy = np.zeros(len(z_score))  
        for i in range(1,len(z_score)):
            if z_score[i] > 2 or (strategy[i-1] == -1 and z_score[i] > 0):
                strategy[i] = -1
                self.state = 'active'
                
            elif z_score[i] < -2 or (strategy[i-1] == 1 and z_score[i] < 0):
                strategy[i] = 1
                self.state = 'active'

            else:
                self.state = 'inactive'

        return strategy
    
