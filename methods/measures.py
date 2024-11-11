import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csgraph
from sklearn.metrics import normalized_mutual_info_score, pair_confusion_matrix, silhouette_score


class Measures:
    @staticmethod
    def filter_labels(y, y_pred):
        """
        Filter out indicies with y = -1 or y_pred = -1
        """
        filtered_y = []
        filtered_y_pred = []
        for i in range(len(y)):
            # if y[i] != -1 and y_pred[i] != -1:
            if y[i] != -1:
                filtered_y.append(y[i])
                filtered_y_pred.append(y_pred[i])
        return np.array(filtered_y), np.array(filtered_y_pred)
    
    @staticmethod
    def safe_ari(y, y_pred, filtering=False):
        if filtering:
            y_, y_pred_ = Measures.filter_labels(y, y_pred)
        else:
            y_, y_pred_ = np.array(y), np.array(y_pred)
        (tn, fp), (fn, tp) = pair_confusion_matrix(y_, y_pred_)
        tn = int(tn)
        tp = int(tp)
        fp = int(fp)
        fn = int(fn)
        # Special cases: empty data or full agreement
        if fn == 0 and fp == 0:
            return 1.0
        return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    
    @staticmethod
    def nmi(y, y_pred, filtering=False):
        if filtering:
            y_, y_pred_ = Measures.filter_labels(y, y_pred)
        else:
            y_, y_pred_ = np.array(y), np.array(y_pred)
        return normalized_mutual_info_score(y_, y_pred_)
    
    @staticmethod
    def purity(y, y_pred, filtering=False):
        if filtering:
            y_, y_pred_ = Measures.filter_labels(y, y_pred)
        else:
            y_, y_pred_ = np.array(y), np.array(y_pred)
        if len(y) == 0:
            print(f'purity: empty filtered label\n')
            return 0
        (n,) = y_pred_.shape
        inv_list = dict()
        for i in range(n):
            if y_pred_[i] not in inv_list:
                inv_list[y_pred_[i]] = list()
            inv_list[y_pred_[i]].append(y_[i])
        sum_v = 0
        for _, l in inv_list.items():
            max_f = np.amax(np.bincount(l))
            sum_v = sum_v + max_f
        return sum_v / n
    
    