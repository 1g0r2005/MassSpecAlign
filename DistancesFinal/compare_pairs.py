import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from tqdm import tqdm

FileName = ('E031_032_033_5mg_60C_90sec_9AA_161023_rawdata.hdf5',
            'E031_032_033_5mg_60C_90sec_9AA_161023.hdf5')
newFileName = ('rawdata.hdf5',
               'aligned.hdf5')

folderHDF = 'HDF\\Data\\'

cash = 'Cash\\cash.hdf5'

ref = [193.07712, 194.0804748353425, 195.08382967068502, 383.13021993, 384.1425151646575, 385.14587, 386.1492248353425,
       387.152579670685]


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


def main():
    with h5py.File(os.path.join('..', folderHDF, FileName[0]), 'r') as old_raw:
        with h5py.File(os.path.join('..', folderHDF, FileName[1]), 'r') as old_aln:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

            dot_array_pre = []

            features_raw = old_raw['roi2_e033/00/features']
            features_aln = old_aln['roi2_e033/00/features']

            set_num = int(max(features_raw[0])) + 1
            set_num = 100

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
                    data_raw_opt, data_aln_opt = opt_strip(*get_long_and_short(data_raw_mz, data_aln_mz))
                    dist_array = data_aln_opt - data_raw_opt
                else:
                    dist_array = data_aln_mz - data_raw_mz

                score = np.max(np.abs(dist_array))
                if score > 1:
                    continue
                dot_array_pre.append(dist_array)

    dot_array = np.hstack(dot_array_pre)

    max_pos = np.max(dot_array)
    min_pos = np.min(dot_array)

    width = 0.01

    cols = int(np.ceil((max_pos - min_pos) / width))
    borders = np.linspace(min_pos, max_pos, cols + 1)

    height, bord_hist = np.histogram(dot_array, borders)

    ax1.hist(dot_array, bins=borders, edgecolor='black')
    ax1.set_yscale('log')

    center = (borders[:-1] + borders[1:]) / 2

    for i, height in enumerate(height):
        if height < 10:
            ax1.plot(center[i], height, marker='o', color='r', markersize=3)

    plt.xlabel("Частоты сдвигов")
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # Добавляем сетку для лучшей читаемости
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black')

    plt.tight_layout()  # Улучшаем размещение элементов графика
    plt.show()


if __name__ == '__main__':
    main()
