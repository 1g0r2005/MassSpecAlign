import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from numba import jit
from scipy import stats
from tqdm import tqdm

FileName = ('E031_032_033_5mg_60C_90sec_9AA_161023_rawdata.hdf5',
            'E031_032_033_5mg_60C_90sec_9AA_161023.hdf5')
newFileName = ('rawdata.hdf5',
               'aligned.hdf5')

folderHDF = 'HDF\\Data\\'

cash = 'Cash\\cash.hdf5'

ref = np.array(
    [193.07712, 194.0804748353425, 195.08382967068502, 383.13021993, 384.1425151646575, 385.14587, 386.1492248353425,
     387.152579670685])


@jit(nopython=True)
def get_long_and_short(raw, aln):
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


def nearest(data, ref_dots):
    dist = np.abs(data[:, None] - ref_dots)
    nearest_id = np.argmin(dist, axis=1)
    nearest_dots = ref_dots[nearest_id.astype(int)]
    return np.vstack((data, data - nearest_dots, nearest_id))


def bin_stat_func(dist, dist_to_ref, resolution=5, max_bin=100):
    bins_calc = int(dist_to_ref.max() - dist_to_ref.min() / resolution)
    bins = bins_calc if bins_calc < max_bin else max_bin
    stat_res = stats.binned_statistic(dist_to_ref, dist, bins=bins, statistic='max')
    return stat_res


def bin_stat_show(stat_res):
    bins = stat_res.bin_edges
    bin_center = (bins[:-1] + bins[1:]) / 2

    bin_means = stat_res.statistic

    df = pd.DataFrame({'bin_center': bin_center, 'bin_mean': bin_means})
    fig = px.bar(df, x='bin_center', y='bin_mean')

    fig.show()

def main():
    with h5py.File(os.path.join('..', folderHDF, FileName[0]), 'r') as old_raw:
        with h5py.File(os.path.join('..', folderHDF, FileName[1]), 'r') as old_aln:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

            dot_array_pre = []

            features_raw = old_raw['roi2_e033/00/features']
            features_aln = old_aln['roi2_e033/00/features']

            set_num = int(max(features_raw[0])) + 1

            for ID in tqdm(range(set_num)):
                index_raw = indexes(features_raw, ID)
                index_aln = indexes(features_aln, ID)

                data_raw = features_raw[1:3, index_raw[0][0]:index_raw[0][-1] + 1]  # m/z | intensity
                data_aln = features_aln[1:3, index_aln[0][0]:index_aln[0][-1] + 1]  # m/z | intensity

                data_raw = data_raw[:, np.argsort(data_raw, axis=1)[0]]
                data_aln = data_aln[:, np.argsort(data_aln, axis=1)[0]]

                data_raw_mz = data_raw[0]
                data_aln_mz = data_aln[0]


                if data_raw_mz.size != data_aln_mz.size:
                    data_raw_mz, data_aln_mz = opt_strip(*get_long_and_short(data_raw_mz, data_aln_mz))
                    dist_array = data_aln_mz - data_raw_mz
                else:
                    dist_array = data_aln_mz - data_raw_mz

                _, dist_to_ref, _ = nearest(data_raw_mz, ref)  # data, dist to ref, id of ref

                score = np.max(np.abs(dist_array))
                if score > 1:
                    continue
                dot_array_pre.append(np.vstack((dist_array, dist_to_ref)))

    dot_array = np.hstack(dot_array_pre)
    print(dot_array)
    bin_stat_show(bin_stat_func(dot_array[0], dot_array[1], resolution=10, max_bin=100))

if __name__ == '__main__':
    main()
