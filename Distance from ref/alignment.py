import numpy as np
from scipy.optimize import linear_sum_assignment

class LinkedList(np.ndarray):
    def __new__(cls, input_array, linked_array=None):
        obj = np.asarray(input_array).view(cls)
        obj.linked_array = linked_array
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.linked_array = getattr(obj, 'linked_array', None)

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        if self.linked_array is not None:
            self.linked_array[index] = value

    def sync_sort(self):
        sort_indices = np.argsort(self)
        sorted_self = self[sort_indices]
        sorted_linked = self.linked_array[sort_indices]

        self[:] = sorted_self
        self.linked_array[:] = sorted_linked

    def sync_delete(self, index):
        new_self = np.delete(self, index)
        if self.linked_array is not None:
            new_linked_array = np.delete(self.linked_array, index, axis=0)
            return LinkedList(new_self, new_linked_array)
        return new_self

def munkres_align(x_arr,y_arr,x_linked,y_linked,intens_x,intens_y):

    convert = lambda arr,arr_linked: LinkedList(np.array([np.array(el).mean() for el in arr]),arr_linked)
    x, y = convert(x_arr,intens_x), convert(y_arr,intens_y)

    x_len,y_len = len(x),len(y)
    x_n,y_n = __equal_size(x,y)

    matrix = __make_matrix(x_n,y_n)

    indexes = np.array(linear_sum_assignment(matrix))
    condition = (indexes[0,:] < x_len) & (indexes[1,:] < y_len)
    xind  = indexes[:,condition][0]
    yind = indexes[:,condition][1]
    aln_x = [x_arr[i] for i in xind]
    aln_y = [y_arr[i] for i in yind]

    aln_x_linked = np.asarray(x_linked)[xind]
    aln_y_linked = np.asarray(y_linked)[yind]
    return aln_x,aln_y,aln_x_linked,aln_y_linked


def __w(x, y,alpha_dist=0.1,alpha_int=3, k = 20):

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    x_linked = np.asarray(x.linked_array)
    y_linked = np.asarray(y.linked_array)

    scale = lambda var,alpha: np.exp(var*alpha)-1
    norm = lambda var,k: np.tanh(var/k)

    dist_scaled = scale(x_arr-y_arr,alpha_dist)
    dint_scaled = scale(x_linked-y_linked,alpha_int)

    dist_norm = norm(dist_scaled,k)
    dint_norm = norm(dint_scaled,k)

    weight = (dist_norm**2+dint_norm**2)**0.5

    return weight

def __make_matrix(x,y):
    matrix =  __w(x[:, np.newaxis], y[np.newaxis, :])
    return matrix

def __equal_size(x,y):

    change_linked = lambda linked_arr, delta: LinkedList(np.concatenate([linked_arr,np.full(delta,np.inf)]),linked_array=np.concatenate([linked_arr.linked_array,np.full(delta,np.inf)]))

    x_len,y_len = len(x),len(y)
    d_len = abs(x_len-y_len)
    if x_len == y_len:
        return x,y
    elif x_len > y_len:
        return x, change_linked(y,d_len)#np.concatenate([y,np.full(x_len-y_len,np.inf)])
    else:
        return change_linked(x,d_len),y#np.concatenate([x, np.full(y_len - x_len, np.inf)]),y
