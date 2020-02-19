import numpy as np
import scipy as sp
import sklearn.metrics

def metric(y_pred, y_true):
    return sklearn.metrics.recall_score(y_true, y_pred, average='macro')

