import numpy as np
import scipy as sp
from functools import partial
from sklearn import metrics
from collections import Counter
import json
from sklearn.metrics import cohen_kappa_score

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        
        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        
    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']
    
def find_threshold(valid_predictions, targets):
    optR = OptimizedRounder()
    #optR.fit(valid_predictions, targets)
    #coefficients = optR.coefficients()
    coefficients = [0.5, 1.5, 2.5, 3.5]
    valid_predictions = optR.predict(valid_predictions, coefficients)
    
    return coefficients, valid_predictions

def quadratic_kappa(y_hat, y):
    return cohen_kappa_score(np.argmax(y_hat,1), np.argmax(y,1), weights='quadratic')

def quadratic_kappa_v2(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')