import numpy as np
import scipy as sp
import sklearn.metrics

def metric(y_pred, y_true):
    return sklearn.metrics.recall_score(y_true, y_pred, average='macro')

def decode(prediction, class_map):
    
    p0, p1, p2, p3 = prediction[0], prediction[1], prediction[2], prediction[3]
    
    for b in range(p0.shape[0]):
        # print(p0[b], p1[b], p2[b], p3[b], class_map[class_map['label'] == p3[b]]['grapheme_root'].values[0], \
        #     class_map[class_map['label'] == p3[b]]['vowel_diacritic'].values[0], \
        #     class_map[class_map['label'] == p3[b]]['consonant_diacritic'].values[0])
        p0[b] = class_map[class_map['label'] == p3[b]]['grapheme_root'].values[0]
        p1[b] = class_map[class_map['label'] == p3[b]]['vowel_diacritic'].values[0]
        p2[b] = class_map[class_map['label'] == p3[b]]['consonant_diacritic'].values[0]
        # print(p0[b], p1[b], p2[b], p3[b], class_map[class_map['label'] == p3[b]]['grapheme_root'].values[0], \
        #     class_map[class_map['label'] == p3[b]]['vowel_diacritic'].values[0], \
        #     class_map[class_map['label'] == p3[b]]['consonant_diacritic'].values[0])
    
    return p0, p1 ,p2, p3

