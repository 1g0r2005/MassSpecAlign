"""
набор данных - np.ndarray([100,150,200,500,680,700])   np.ndarray([106,144,203,495,705])
"""

import numpy as np
from munkres import Munkres

def munkres_align(x,y):
    x_len,y_len = len(x),len(y)
    x_n,y_n = __equal_size(x,y)
    matrix = __make_matrix(x_n,y_n)
    m = Munkres()
    indexes = m.compute(matrix)
    indexes = np.array(indexes)
    condition = (indexes[:, 0] < x_len) & (indexes[:, 1] < y_len)
    return x[indexes[condition][:,0]],y[indexes[condition][:,1]]

def __w(x, y, k = 20, threshold = 10, epsilon = 1):
    dist = abs(x - y)
    dist_norm =  np.tanh(dist/k)
    return dist_norm

def __make_matrix(x,y):
    matrix =  __w(x[:, np.newaxis], y[np.newaxis, :])
    return matrix

def __equal_size(x,y):
    x_len,y_len = len(x),len(y)
    if x_len == y_len:
        return x,y
    elif x_len > y_len:
        return x, np.concatenate([y,np.full(x_len-y_len,np.inf)])
    else:
        return np.concatenate([x, np.full(y_len - x_len, np.inf)]),y