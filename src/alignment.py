from scipy.optimize import linear_sum_assignment

from data_classes import *


def munkres_align(x_arr,y_arr,x_linked,y_linked,intens_x,intens_y,skip_fraction=0.2,skip_level=0.4,alpha_dist=0.1,alpha_int=0.05):

    convert = lambda arr, arr_linked: LinkedList(np.array([np.array(el).mean() for el in arr]), arr_linked)
    x, y = convert(x_arr, intens_x), convert(y_arr, intens_y)

    x_len,y_len = len(x),len(y)
    x_n,y_n = __equal_size(x,y)

    matrix = __make_matrix(x_n,y_n,skip_fraction,skip_level,alpha_dist,alpha_int)
    indexes = np.array(linear_sum_assignment(matrix))
    condition = (indexes[0,:] < x_len) & (indexes[1,:] < y_len)
    xind  = indexes[:,condition][0]
    yind = indexes[:,condition][1]

    aln_x = [x_arr[i] for i in xind]
    aln_y = [y_arr[i] for i in yind]


    aln_x_linked = np.asarray(x_linked)[xind]
    aln_y_linked = np.asarray(y_linked)[yind]
    return aln_x,aln_y,aln_x_linked,aln_y_linked

def __w(x, y,alpha_dist=1,alpha_int=0.01, k = 15):
    """
    Compute the pairwise cost between two LinkedLists combining distance and
    intensity differences with exponential scaling and tanh normalization.

    Parameters
    ----------
    x : LinkedList or array_like
        Values for the first axis. When `LinkedList`, its `linked_array` is used
        to compute intensity differences.
    y : LinkedList or array_like
        Values for the second axis. When `LinkedList`, its `linked_array` is
        used to compute intensity differences.
    alpha_dist : float, optional
        Exponential scaling factor for absolute distance differences. Default is
        0.01.
    alpha_int : float, optional
        Exponential scaling factor for absolute intensity differences. Default
        is 0.01.
    k : float, optional
        Scaling parameter for the tanh normalization. Default is 20.

    Returns
    -------
    ndarray
        A 2-D cost matrix broadcast from `x` and `y` shapes, with entries in
        the range (-1, 1), approaching 1 for large combined differences.
    """

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # Важно: linked_array тоже принудительно приводим к числовому типу,
    # чтобы избежать dtype=object и object-циклов ufunc'ов
    x_linked = np.asarray(x.linked_array, dtype=float)
    y_linked = np.asarray(y.linked_array, dtype=float)


    scale = lambda var, alpha: np.exp(var * alpha) - 1
    norm = lambda var,k: np.tanh(var/k)

    dist_scaled = scale(abs(x_arr-y_arr),alpha_dist)
    dint_scaled = scale(abs(x_linked-y_linked),alpha_int)

    
    weight = (dist_scaled**2+dint_scaled**2)**0.5

    return norm(weight,k)

def __make_matrix(x,y,skip_fraction=None,skip_level=1,alpha_dist=0.15,alpha_int=0.05):
    """
    Build a padded cost matrix for assignment, allowing optional skipping via
    dummy rows/columns filled with a constant penalty.

    Parameters
    ----------
    x : LinkedList
        Input values (possibly reshaped to 2-D) for the first axis.
    y : LinkedList
        Input values (possibly reshaped to 2-D) for the second axis.
    skip_fraction : float or None, optional
        Fraction of the base matrix size to determine how many dummy rows/cols
        to pad. If None, no padding is applied. Default is None.
    skip_level : float, optional
        Multiplier for the maximum base cost to set the pad penalty. Default is
        1.

    Returns
    -------
    ndarray
        The (possibly padded) 2-D cost matrix suitable for
        ``scipy.optimize.linear_sum_assignment``.
    """
    matrix =  __w(x.sync_reshape((-1,1)), y.sync_reshape((1,-1)),alpha_dist,alpha_int)
    if skip_fraction is not None:
        add_lines = round(matrix.shape[0]*skip_fraction)
        pad_value = matrix.max()*skip_level
        matrix = np.pad(matrix,((0,add_lines),(0,add_lines)),'constant',constant_values=pad_value)
    return matrix

def __equal_size(x,y):
    """
    Make two LinkedLists equal in length by padding the shorter one with
    ``np.inf`` both in values and in its linked array.

    Parameters
    ----------
    x : LinkedList
        First sequence.
    y : LinkedList
        Second sequence.

    Returns
    -------
    tuple of LinkedList
        A pair ``(x_equal, y_equal)`` with matching lengths.
    """

    change_linked = lambda linked_arr, delta: LinkedList(np.concatenate([linked_arr,np.full(delta,np.inf)]),linked_array=np.concatenate([linked_arr.linked_array,np.full(delta,np.inf)]))

    x_len,y_len = len(x),len(y)
    d_len = abs(x_len-y_len)

    if x_len == y_len:
        return x,y
    elif x_len > y_len:
        return x, change_linked(y,d_len)#np.concatenate([y,np.full(x_len-y_len,np.inf)])
    else:
        return change_linked(x,d_len),y#np.concatenate([x, np.full(y_len - x_len, np.inf)]),y
