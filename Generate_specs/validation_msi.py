import argparse
import math
import os.path
import warnings
from random import uniform
from typing import Iterable

import numpy as np
import yaml
from msalign import msalign
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from scipy import sparse
from tqdm import tqdm
from yaml import YAMLError

warnings.filterwarnings("ignore")

# для преобразования из centroid
STEP = .01
PADDING = 0.5

'''чтение данных из yaml'''
def yaml_read(path:str):
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print("File not found")
    except YAMLError as e:
        print(e)


'''отдельно добавляем шум + лишние пики''' # временно пустая
def add_noise(data,noise_int:float,noise_level:float,extra_peak:None|float ):
    return data

'''масштабирование по амплитуде'''
def resize_int(int_arr:Iterable[float], min_mult:float=0.95, max_mult:None|float=1.05):
    if max_mult is None: max_mult = min_mult
    multiplier = [uniform(min_mult,max_mult) for _ in range(len(int_arr))]
    return multiplier * int_arr


'''нелинейные искажения mz'''
def resize_mz(mz_arr:Iterable[float], min_mult:float, max_mult:None|float):
    if max_mult is None: max_mult = min_mult
    multiplier = [uniform(min_mult, max_mult) for _ in range(len(mz_arr))]

    distances = np.diff(mz_arr) * multiplier[1:]
    new_mz_arr = np.cumsum(np.concatenate(([0],distances))) + multiplier[0]*mz_arr[0]

    return new_mz_arr

'''поиск пиков'''
def find_peaks(X:Iterable[float], Y:Iterable[float], oversegmentation_filter=None, peak_location=1):
    n = X.size
    # Robust valley finding
    valley_dots = np.concatenate((np.where(np.diff(Y) != 0)[0], [n-1]))
    loc_min = np.diff(Y[valley_dots])
    loc_min = (np.array([True,*(loc_min < 0)])) & np.array(([*(loc_min > 0),True]))
    left_min = np.concatenate([[-1],valley_dots[:-1]])[loc_min][:-1] + 1
    right_min = valley_dots[loc_min][1:]
    # Compute max and min for every peak
    size = left_min.shape
    val_max = np.empty(size)
    pos_peak = np.empty(size)
    for idx, [lm, rm] in enumerate(zip(left_min, right_min)):
        pp = lm + np.argmax(Y[lm:rm])
        vm = np.max(Y[lm:rm])
        val_max[idx] = vm
        pos_peak[idx] = pp

    # Remove over-segmented peaks
    if oversegmentation_filter:
        while True:
            peak_threshold = val_max * peak_location - math.sqrt(np.finfo(float).eps)
            pk_x = np.empty(left_min.shape)

            for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_threshold)):
                mask = Y[lm:rm] >= th
                if np.sum(mask) == 0:
                    pk_x[idx]=np.nan
                else:
                    pk_x[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])
            dpk_x = np.concatenate(([np.inf], np.diff(pk_x), [np.inf]))

            j = np.where((dpk_x[1:-1] <= oversegmentation_filter) & (dpk_x[1:-1] <= dpk_x[:-2]) & (dpk_x[1:-1] < dpk_x[2:]))[0]
            if j.size == 0:
                break
            left_min = np.delete(left_min, j + 1)
            right_min = np.delete(right_min, j)
            val_max[j] = np.maximum(val_max[j], val_max[j + 1])
            val_max = np.delete(val_max, j + 1)
    else:
        peak_threshold = val_max * peak_location - math.sqrt(np.finfo(float).eps)
        pk_x = np.empty(left_min.shape)

        for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_threshold)):
            mask = Y[lm:rm] >= th
            if np.sum(mask) == 0:
                pk_x[idx]=np.nan
            else:
                pk_x[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])


    return pk_x, X[left_min], X[right_min]

def generator_step(path:str,params:dict,f_index:int,output:str):

    parser = ImzMLParser(path)
    number_of_spectra = len(parser.coordinates)

    mz_mul_max, mz_mul_min = params["mult_mz"]
    int_mul_max, int_mul_min = params["mult_int"]
    noise_level = params["n_lev"]
    noise_intensity = params["n_int"]
    reference_mz = params["ref_peaks"]

    names = [os.path.join(output,f'{f_index}_n.imzML'),os.path.join(output,f'{f_index}_na.imzML')]

    noise_writer, aln_writer = (ImzMLWriter(names[0], mode='processed'),
                                ImzMLWriter(names[1],mode='processed'))

    min_mz, max_mz = np.inf, -np.inf

    # первый проход для получения диапазона, с учетом максимальных отклонений
    for i in tqdm(range(number_of_spectra)):
        mz_i, _ = parser.getspectrum(i)
        if len(mz_i) == 0: continue
        min_mz_candidate = mz_i[0] * mz_mul_min
        max_mz_candidate = mz_i[-1] * mz_mul_max

        min_mz = min(min_mz_candidate, min_mz)
        max_mz = max(max_mz_candidate, max_mz)

    min_mz = max(0, min_mz - PADDING)
    max_mz += PADDING
    common_mz = np.arange(min_mz, max_mz + STEP, STEP)

    data_bin, row_bin, col_bin = [], [], []
    n_bins = len(common_mz)

    # второй проход для внесения искажений и выравнивания
    for i in tqdm(range(number_of_spectra)):
        mz_i, int_i = parser.getspectrum(i)
        coords = parser.coordinates[i]
        if f_index == 0:
            peaks_mz, _, _ = find_peaks(mz_i, int_i)
            peaks_int = np.interp(peaks_mz, mz_i, int_i)
        else:
            peaks_mz = mz_i
            peaks_int = int_i

        mz_i_new = resize_mz(peaks_mz,
                             min_mult=mz_mul_min,
                             max_mult=mz_mul_max)
        int_i_new = resize_int(peaks_int,
                               min_mult=int_mul_min,
                               max_mult=int_mul_max)

        noise_writer.addSpectrum(mz_i_new,
                                 int_i_new,
                                 coords)

        print(len(mz_i_new))
        print(len(int_i_new))
        print(len(mz_i))
        print(len(int_i))

        if len(mz_i_new) == 0: continue

        bin_idx = np.digitize(mz_i_new, common_mz) - 1
        valid = (bin_idx >= 0) & (bin_idx < n_bins)

        if np.any(valid):
            bin_indices = bin_idx[valid].astype(np.int64)
            bin_count = np.bincount(bin_indices, weights=int_i_new[valid], minlength=n_bins)
            nonzero = np.nonzero(bin_count)[0]
            row_bin.extend([i] * len(nonzero))
            col_bin.extend(nonzero)
            data_bin.extend(bin_count[nonzero])

    profile_matrix = sparse.csr_matrix((data_bin, (row_bin, col_bin)), shape=(number_of_spectra, n_bins)).toarray()
    aligned_ints = msalign(common_mz, profile_matrix, reference_mz, only_shift=True)
    total_int = np.sum(aligned_ints, axis=1)
    for i in tqdm(range(number_of_spectra)):
        coords = parser.coordinates[i]
        peaks_mz, _, _ = find_peaks(common_mz, aligned_ints[i])
        peaks_int = np.interp(peaks_mz, common_mz, aligned_ints[i])

        aln_writer.addSpectrum(peaks_mz,
                               peaks_int,
                               coords)
    aln_writer.close()
    noise_writer.close()

    return names[0]


def main():
    parser = argparse.ArgumentParser(description="Generate validation specs")
    parser.add_argument('data_name', help='MSI data')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('--ref_peaks' ,nargs='+', help='Reference peaks')
    parser.add_argument('--config', help='config file')

    parser.add_argument("--mult_mz", nargs='+', type=float, default=[0.999, 1.001], help="Multiplier for mz")
    parser.add_argument("--mult_int", nargs='+', type=float, default=[0.999, 1.001], help="Multiplier for intensity")
    parser.add_argument("--n_int", type=float, default=0., help="Noise intensity")
    parser.add_argument("--n_lev", type=float, default=0., help="Noise level")

    args = parser.parse_args()
    if args.config:
        params = yaml_read(args.config)
        print("Using config file: {}".format(args.config))
    else:

        params = {"data_name": args.data_name,
                  "n_int": args.n_int,
                  "n_lev": args.n_lev}
        if isinstance(args.mult_mz, float):
            params["mult_mz"] = [args.mult_mz, args.mult_mz]
        else:
            params["mult_mz"] = args.mult_mz
        if isinstance(args.mult_int, float):
            params["mult_int"] = [args.mult_int, args.mult_int]
        else:
            params["mult_int"] = args.mult_int

    # более удобное именование
    imzml_name = params["data_name"]

    for iter_index in range(3):
        print(f"\n=== Итерация {iter_index} ===")
        imzml_name = generator_step(imzml_name,
                                    params,
                                    output=params["out_dir"],
                                    f_index=iter_index)
        print(f"→ Следующая итерация будет использовать: {imzml_name}")


if __name__ == '__main__':
    main()