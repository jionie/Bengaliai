import numpy as np
import scipy as sp
import sklearn.metrics

def metric(y_pred, y_true):
    return sklearn.metrics.recall_score(y_true, y_pred, average='macro')

def compute_kaggle_metric_by_decode(prediction, class_map):
    
    num_test= len(prediction[0])
    p0,p1,p2,p3 = prediction
    for b in range(num_test):
        p0[b]= class_map[class_map['label'] == p3[b]]['grapheme_root']
        p1[b]= class_map[class_map['label'] == p3[b]]['vowel_diacritic']
        p2[b]= class_map[class_map['label'] == p3[b]]['consonant_diacritic']
    prediction = [p0,p1,p2,p3]
    
    return prediction

