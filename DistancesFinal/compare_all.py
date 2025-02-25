import os

import h5py
import numpy as np
import plotly.express as px
from numba import jit
from tqdm import tqdm

FileName = ("E031_032_033_5mg_60C_90sec_9AA_161023_rawdata.hdf5", 'E031_032_033_5mg_60C_90sec_9AA_161023.hdf5')
newFileName = ("rawdata.hdf5",
               "aligned.hdf5")

folderHDF = "HDF\\Data\\"

cash = 'Cash\\cash.hdf5'


@jit(nopython=True)
def get_long_and_short(raw, aln):
    size1, size2 = raw.shape[0], aln.shape[0]
    if size1 > size2:
        return raw, aln, True
    else:
        return aln, raw, False


@jit(nopython=False)
def opt_strip(arr_long, arr_short, flag):
    """
    return two arrays of equal size (1 - raw, 2 - aligned)
    """
    size = arr_short.shape[0]
    long_size = arr_long.shape[0]
    max_shift = long_size - size + 1
    shift_array = np.arange(max_shift)
    score_array = np.zeros(max_shift)
    for i in shift_array:
        fit_score = np.mean((arr_short - arr_long[i:i + size]) ** 2)
        score_array[i] = fit_score
        # if key_show:
        # print(f'{i}: arr1:{arr_short}, arr2:{arr_long[i:i+size]}, fit_score: {fit_score}')
    opt_shift = np.where(score_array == score_array.min())[0][0]
    opt_long = arr_long[opt_shift:opt_shift + size]

    if flag:
        return opt_long, arr_short
    else:
        return arr_short, opt_long


def indexes(dataset, ds_id: int):
    """
    return columns for spec with special ID
    dataset - hdf vis_dataset
    ID - ID of spec
    """
    index_data = dataset[0]
    return np.where(index_data == ds_id)


"""
def pad_arrays(arrays, fill_value=np.nan):

    max_len = max(len(arr) for arr in arrays)  # Находим максимальную длину

    padded_arrays = []
    for arr in arrays:
        pad_width = max_len - len(arr)
        padded_arr = np.pad(arr, pad_width=(0, pad_width), mode='constant', constant_values=fill_value)
        padded_arrays.append(padded_arr)

    return np.array(padded_arrays)  # Преобразуем в единый массив NumPy


def masked_arrays(arrays):
    array = np.array(arrays)
    return np.ma.masked_array(array,mask= np.isnan(array))

def distance(array):
    array = masked_arrays(array)

    shape = array.shape
    dist_matrix = np.zeros(shape,dtype=np.float64)

    with tqdm(total = array.size,desc = "DIST PROCESSING") as pbar:
        for (line, row), value in np.ndenumerate(array):
                #print(line,row)
                dot = array[line,row]
                dist_matrix[line,row] = (array-dot).mean()
                pbar.update(1)

    ans = dist_matrix.flatten()
    return ans[~np.isnan(ans)]


@jit(nopython=True,nogil=True,parallel=True)
def distance_2(arr,progress_proxy):
    size = np.sum(~np.isnan(arr))
    shape = arr.shape
    dist_array = np.full(shape,np.nan)

    for x in prange(shape[0]):
        for y in range(shape[1]):
            el = arr[x,y]
            if ~np.isnan(el): dist_array[x,y] = np.nansum(arr-el)/(size-1)
            progress_proxy.update(1)
    ans = dist_array.flatten()
    return ans[~np.isnan(ans)]

"""


def distance_1d(arr):
    n = arr.size
    return (n / (n - 1)) * (np.mean(arr) - arr)


"""
def numba_wrapper(arr):

    num_iteration = arr.size

    with ProgressBar(total=num_iteration) as progress:
        dist_matrix = distance_2(arr,progress)

    return dist_matrix
"""


def main():
    with h5py.File(os.path.join('..', folderHDF, FileName[0]), 'r') as old_raw:
        with h5py.File(os.path.join('..', folderHDF, FileName[1]), 'r') as old_aln:

            features_raw = old_raw['roi2_e033/00/features']
            features_aln = old_aln['roi2_e033/00/features']

            set_num = int(max(features_raw[0])) + 1
            set_num = 1000
            dot_array = []

            for ID in tqdm(range(set_num), desc="LOADING DATA"):
                index_raw = indexes(features_raw, ID)
                index_aln = indexes(features_aln, ID)
                data_raw = np.array(features_raw[1][index_raw])
                data_aln = np.array(features_aln[1][index_aln])

                if data_raw.size != data_aln.size:
                    data_raw, data_aln = opt_strip(*get_long_and_short(data_raw, data_aln))

                dot_array.append(data_raw)
                dot_array.append(data_aln)

            distance_array = distance_1d(np.hstack(dot_array))
            print(distance_array)
            fig = px.histogram(distance_array, nbins=300, title=f"Distribution of distances (n={set_num})",
                               histnorm="probability")
            fig.show()


if __name__ == '__main__':
    main()
