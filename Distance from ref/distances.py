import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy import stats
from tqdm import tqdm

FILENAMES = ('E031_032_033_5mg_60C_90sec_9AA_161023_rawdata.hdf5',
             'E031_032_033_5mg_60C_90sec_9AA_161023.hdf5')
FOLDERS = 'HDF\\Data\\'

CASH = 'Cash\\cash.hdf5'

DATASET = 'roi2_e033/00/features'

REF = 194.0804748353425

DEV = 0.2  # M/Z


class Dataset(np.ndarray):
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


class File:
    def __init__(self, fileName, folder):
        self.realpath = Path(os.path.join('..', folder, fileName))

    def exist(self):
        return self.realpath.exists()

    def read(self, dataset):
        try:
            with h5py.File(self.realpath, 'r') as f:
                if dataset in f:
                    data = f[dataset][:]
                    return data
        except FileNotFoundError:
            print(f"File {self.realpath} not found")
            return None
        except Exception as e:
            print(f'Reading Error {e}')
            return None


@jit(nopython=True)
def get_long_and_short(raw: np.ndarray, aln: np.ndarray):
    """
    input - 1 - raw, 2 - aligned, flag_raw - which (raw, aligned) is bigger (true if raw)
    return tuple of two arrays: 1 - long, 2 - short (1 and 2 have different sizes)
    """
    size1, size2 = raw.shape[0], aln.shape[0]
    if size1 > size2:
        return raw, aln, True
    else:
        return aln, raw, False


# @jit(nopython=True)
def opt_strip(arr_long: Dataset, arr_short: Dataset, flag: bool):
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

    # print(type(opt_long),type(arr_short))
    if flag:
        return opt_long, arr_short
    else:
        return arr_short, opt_long


def indexes(dataset: np.ndarray, dsID: int):
    """
    return columns for spec with special ID
    """
    indexData = dataset[0]
    return np.where(indexData == dsID)


def checkData(data1: Dataset, data2: Dataset, threshold=1):
    if data1.size != data2.size:
        data1_new, data2_new = opt_strip(*get_long_and_short(data1, data2))
    else:
        data1_new = data1
        data2_new = data2
    distArray = data1_new - data2_new
    scoreFit = np.max(np.abs(distArray))

    if scoreFit > 1:
        cut_index = np.array([np.where(np.abs(distArray) >= threshold)]).min()
        # print(f'cut_index = {cut_index}')
        if data1_new[cut_index] < data2_new[cut_index]:
            data1_new2 = np.delete(data1_new, cut_index)
            data2_new2 = data2_new
        else:
            data2_new2 = np.delete(data2_new, cut_index)
            data1_new2 = data1_new
        return opt_strip(*get_long_and_short(data1_new2, data2_new2))
    return data1_new, data2_new


def readDataset(datasetRaw: np.ndarray, datasetAln: np.ndarray, limit=None, ax=plt):
    """
    read datasets and check data
    """

    if limit is None:
        setNum = int(max(datasetRaw[0])) + 1
    else:
        setNum = int(limit)

    refList = []

    for index in tqdm(range(setNum)):
        indexRaw, indexAln = indexes(datasetRaw, index), indexes(datasetAln, index)

        dataRawUnsorted = datasetRaw[1:3, indexRaw[0][0]:indexRaw[0][-1] + 1]
        dataAlnUnsorted = datasetAln[1:3, indexAln[0][0]:indexAln[0][-1] + 1]

        dataRaw = dataRawUnsorted[:, np.argsort(dataRawUnsorted, axis=1)[0]]
        dataAln = dataAlnUnsorted[:, np.argsort(dataAlnUnsorted, axis=1)[0]]

        dataRawMZ, dataAlnMZ = dataRaw[0], dataAln[0]
        dataRawInt, dataAlnInt = dataRaw[1], dataAln[1]

        dataRawLinked = Dataset(dataRawMZ, dataRawInt)
        dataAlnLinked = Dataset(dataAlnMZ, dataAlnInt)

        checkedRaw, checkedAln = checkData(dataRawLinked, dataAlnLinked, 1)

        _, ref_mz_raw = findRef(checkedRaw, REF, DEV)
        _, ref_aln_raw = findRef(checkedAln, REF, DEV)

        refList.append(ref_mz_raw)
        refList.append(ref_aln_raw)

        '''
        if index <2:
            findRef(checkedRaw,REF,DEV)
            y = checkedRaw-checkedAln
            x = np.arange(len(y))
            ax.scatter(x,y)
        else:
            return
        '''
    kde = stats.gaussian_kde(refList)
    x_vals = np.linspace(min(refList), max(refList), 1000)
    y_vals = kde.evaluate(x_vals)

    ax.plot(x_vals, y_vals)


def findRef(dataset: Dataset, approx_mz: float, deviation=1.0):
    """find ref peak in ds and return index of peak"""
    condition_1 = approx_mz - deviation <= dataset
    condition_2 = approx_mz + deviation >= dataset

    where_construct = np.where(condition_1 & condition_2)
    if where_construct[0].size:
        ref_index = where_construct[0][np.argmax(dataset.linked_array[where_construct])]
    else:
        ref_index = np.argmin(np.abs(dataset - approx_mz))

    # print(f"ref_index = {ref_index},intens = {dataset.linked_array[ref_index]}, mz = {dataset[ref_index]}")
    return ref_index, dataset[ref_index]


def main():
    featuresRaw = File(FILENAMES[0], FOLDERS).read(DATASET)
    featuresAln = File(FILENAMES[1], FOLDERS).read(DATASET)

    fig, ax = plt.subplots()

    readDataset(featuresRaw, featuresAln, ax=ax)

    fig.show()


if __name__ == '__main__':
    main()
