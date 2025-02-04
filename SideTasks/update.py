import numpy as np
import h5py
from tqdm import tqdm
import os
from numba import jit
import distr_visual as vis
import matplotlib.pyplot as plt

FileName = ('E031_032_033_5mg_60C_90sec_9AA_161023_rawdata.hdf5',
            'E031_032_033_5mg_60C_90sec_9AA_161023.hdf5')
newFileName = ('rawdata.hdf5',
            'aligned.hdf5')

folderHDF = 'HDF\\Data\\'

cash = 'Cash\\cash.hdf5'

ref = [193.07712, 194.0804748353425, 195.08382967068502, 383.13021993, 384.1425151646575, 385.14587, 386.1492248353425,
       387.152579670685]

@jit(nopython=True)
def get_long_and_short(raw,aln):
    """
    input - 1 - raw, 2 - aligned, flag_raw - which (raw, aligned) is bigger (true if raw)
    return tuple of two arrays: 1 - long, 2 - short (1 and 2 have different sizes)
    """
    size1, size2 = raw.shape[0], aln.shape[0]
    if size1 > size2:
        return raw, aln, True
    else:
        return aln, raw, False

@jit(nopython=True)
def opt_strip(arr_long,arr_short,flag):
    """
    return two arrays of equal size (1 - raw, 2 - aligned)
    """
    size = arr_short.shape[0]
    long_size = arr_long.shape[0]
    maxshift = long_size - size + 1
    shift_array = np.arange(maxshift)
    score_array = np.zeros(maxshift)
    for i in shift_array:
        fit_score = np.mean(np.absolute(arr_short - arr_long[i:i + size]))
        score_array[i] = fit_score
        #if key_show:
            #print(f'{i}: arr1:{arr_short}, arr2:{arr_long[i:i+size]}, fit_score: {fit_score}')
    optshift = np.where(score_array == score_array.min())[0][0]
    optlong = arr_long[optshift:optshift+size]

    if flag:
        return optlong, arr_short
    else:
        return arr_short, optlong

def indexes(dataset, ds_id: int):
    """
    return colums for spec with special ID
    dataset - hdf vis_dataset
    ID - ID of spec
    """
    indexdata = dataset[0]
    return np.where(indexdata == ds_id)

def roi(dots, delta=2):
    '''
    return regions of interest around list of dots
    dost - array of dots in roi
    delta - the width of the region around a single point
    '''
    borders = []
    start = dots[0]
    end = dots[0]

    for i in range(1, len(dots)):
        if dots[i] - end <= 2 * delta:
            end = dots[i]
        else:
            borders.append([start - delta, end + delta])
            start = dots[i]
            end = dots[i]
            borders.append([start - delta, end + delta])
    return np.array(borders)

def dataFilter(rawdata, borders):
    minMZ, maxMZ = borders
    condition = (minMZ <= rawdata[0, :]) & (rawdata[0, :] <= maxMZ)
    dataFiltered = rawdata[:, np.where(condition)]
    return dataFiltered

def main():
    with h5py.File(os.path.join('..', folderHDF, FileName[0]), 'r') as old_raw:
        with h5py.File(os.path.join('..', folderHDF, FileName[1]), 'r') as old_aln:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

            features_raw = old_raw['roi2_e033/00/features']
            features_aln = old_aln['roi2_e033/00/features']

            setnum = int(max(features_raw[0])) + 1
            setnum = int(input("setnum for tests: "))

            maxpos = 10
            maxneg = -10

            for ID in tqdm(range(setnum)):
                index_raw = indexes(features_raw, ID)
                index_aln = indexes(features_aln, ID)
                data_raw = np.array(features_raw[1][index_raw])
                data_aln = np.array(features_aln[1][index_aln])
                if data_raw.size != data_aln.size:
                    #print(f'cannot be compared! ID:{ID}')
                    data_raw, data_aln = opt_strip(*get_long_and_short(data_raw, data_aln))
                #print(f'ID:{ID}, raw:{data_raw.shape}, aln: {data_aln.shape}')
                minsize = min(len(data_aln), len(data_raw))
                dist_array = np.sort(data_aln[:minsize]) - np.sort(data_raw[:minsize])
                vis.histogramm(dist_array, np.arange(maxneg, maxpos, 0.1), alpha=0.5, subplot=ax1)
                vis.histogramm(dist_array, np.arange(maxneg, maxpos, 1), alpha=0.5, subplot=ax2)
    vis.show()

if __name__ == '__main__':
    main()