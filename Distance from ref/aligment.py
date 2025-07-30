import numpy as np
from scipy.optimize import linear_sum_assignment

def munkres_align(x_arr,y_arr):

    convert = lambda arr: np.array([np.array(el).mean() for el in arr])
    x, y = convert(x_arr), convert(y_arr)

    x_len,y_len = len(x),len(y)
    x_n,y_n = __equal_size(x,y)
    matrix = __make_matrix(x_n,y_n)

    indexes = np.array(linear_sum_assignment(matrix))
    condition = (indexes[0,:] < x_len) & (indexes[1,:] < y_len)
    xind  = indexes[:,condition][0]
    yind = indexes[:,condition][1]
    aln_x = [x_arr[i] for i in xind]
    aln_y = [y_arr[i] for i in yind]

    return aln_x,aln_y


def __w(x:np.ndarray, y:np.ndarray, k = 20, threshold = 10, epsilon = 1):
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
