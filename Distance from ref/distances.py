import logging
import os
import sys
import tkinter as tk
from pathlib import Path
import multiprocessing as mp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numba import jit
from tqdm import tqdm

'''classes declaration'''


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


class Dataset(LinkedList):
    def __new__(cls, input_array, linked_array=None, reference=None):
        obj = super().__new__(cls, input_array, linked_array)
        obj.reference = reference
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None: return
        self.reference = getattr(obj, 'reference', None)

    def __setitem__(self, index, value):
        super().__setitem__(index, value)

class File:
    def __init__(self, file_name, folder):
        self.real_path = Path(os.path.join('..', folder, file_name))

    def exist(self):
        return self.real_path.exists()

    def read(self, dataset):
        try:
            with h5py.File(self.real_path, 'r') as f:
                if dataset in f:
                    data = f[dataset][:]
                    return data
        except FileNotFoundError:
            logging.error(f'File {self.real_path} not found')
            return None
        except Exception as err:
            logging.error(f'Reading Error: {err}')
            return None


class Root(tk.Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("MZ alignment estimation")
        self.geometry("800x600")
        self.resizable(width=False, height=False)


class TkPlot:
    def __init__(self, root, x, y, x_label="x", y_label='y'):
        self.root = root

        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel=x_label, ylabel=y_label)

        self.x, self.y = x, y
        self.line_plot = self.ax.plot(self.x, self.y)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update(self, x=None, y=None):
        self.x, self.y = x, y
        self.line_plot.relim()
        self.line_plot.autoscale_view()
        self.canvas.draw()


'''decorators declaration'''


def func_logger(full=False):
    def actual_decorator(func):
        def inner(*args, **kwargs):
            ret = func(*args, **kwargs)
            logger.info(f'Function {func.__name__}' + (f'with {args, kwargs} was called' if full else ''))
            return ret

        return inner

    return actual_decorator


def method_logger(method):
    def inner(self, *args, **kwargs):
        ret = method(*args, **kwargs)
        logger.info(f'Method {method.__name__} of {self} was called')
        return ret

    return inner


'''functions declaration'''


def setup_logger():
    real_logger = logging.getLogger()
    real_logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(f'{__name__}.log', "w")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.ERROR)

    real_logger.addHandler(file_handler)
    real_logger.addHandler(stream_handler)

    return real_logger

@jit(nopython=True)
def get_long_and_short(arr_1: np.ndarray, arr_2: np.ndarray):
    """
    input - arr_1, arr_2 - two np.ndarray, possibly not equal size
    return - tuple with two arrays (long, short)

    warning: using JIT compilation for acceleration
    """
    size1, size2 = arr_1.shape[0], arr_2.shape[0]
    if size1 > size2:
        return arr_1, arr_2, True
    else:
        return arr_2, arr_1, False


def get_opt_strip(arr_long: Dataset, arr_short: Dataset, flag: bool):
    """
    input - arr_long - bigger Dataset, arr_short - smaller Dataset with mz and int data
    return - tuple with two arrays of equal size (optional slice for bigger dataset)
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


def get_index(dataset: np.ndarray, ds_id: int):
    """
    input - dataset - np.ndarray with full recorded data from attached HDF file
          - ds_id - int with id of dataset column
    return - np.ndarray with indexes of columns with requested id
    """
    index_data = dataset[0]
    return np.where(index_data == ds_id)


@func_logger()
def verify_datasets(data_1: Dataset, data_2: Dataset, threshold=1.0):
    """
    input - data_1, data_2 - Dataset objects which need to be verified
          - threshold - int with maximum distance between values in data_1 and data_2 with one index
    return - two fitted Dataset objects which match threshold value
    """
    if data_1.size != data_2.size:
        data1_new, data2_new = get_opt_strip(*get_long_and_short(data_1, data_2))
    else:
        data1_new = data_1
        data2_new = data_2

    dist_array = data1_new - data2_new
    score_fit = np.max(np.abs(dist_array))

    if score_fit > threshold:
        cut_index = np.array([np.where(np.abs(dist_array) >= threshold)]).min()
        if data1_new[cut_index] < data2_new[cut_index]:
            data1_new2 = np.delete(data1_new, cut_index)
            data2_new2 = data2_new
        else:
            data2_new2 = np.delete(data2_new, cut_index)
            data1_new2 = data1_new
        return get_opt_strip(*get_long_and_short(data1_new2, data2_new2))
    return data1_new, data2_new


@func_logger()
def find_ref(dataset: Dataset, approx_mz: float, deviation=1.0):
    """
    input - dataset - Dataset object
          - approx_mz - float with initial guess for ref peak location
          - deviation - float with acceptable deviation from approx_mz value
    return - tuple (index of reference peak in current dataset, m/z value)
    """
    condition_1 = approx_mz - deviation <= dataset
    condition_2 = approx_mz + deviation >= dataset

    where_construct = np.where(condition_1 & condition_2)
    if where_construct[0].size:
        ref_index = where_construct[0][np.argmax(dataset.linked_array[where_construct])]
    else:
        ref_index = np.argmin(np.abs(dataset - approx_mz))

    return ref_index, dataset[ref_index]


@func_logger(full=False)
def read_dataset(dataset_raw: np.ndarray, dataset_aln: np.ndarray, limit=None):
    """
    initial data verifying and recording into Dataset objects
    input - dataset_raw, dataset_aln - np.ndarray arrays with full recorded data,
          - limit - maximum number of mass spectra to be processed (for debugging use only, otherwise should be zero)
    return - (void function)
    """
    if limit is None:
        set_num = int(max(dataset_raw[0])) + 1
    else:
        set_num = int(limit)

    ref_list = []

    for index in tqdm(range(set_num)):
        index_raw, index_aln = get_index(dataset_raw, index), get_index(dataset_aln, index)

        data_raw_unsorted = dataset_raw[1:3, index_raw[0][0]:index_raw[0][-1] + 1]
        data_aln_unsorted = dataset_aln[1:3, index_aln[0][0]:index_aln[0][-1] + 1]

        data_raw = data_raw_unsorted[:, np.argsort(data_raw_unsorted, axis=1)[0]]
        data_aln = data_aln_unsorted[:, np.argsort(data_aln_unsorted, axis=1)[0]]

        data_raw_mz, data_aln_mz = data_raw[0], data_aln[0]
        data_raw_int, data_aln_int = data_raw[1], data_aln[1]

        data_raw_linked = Dataset(data_raw_mz, data_raw_int)
        data_aln_linked = Dataset(data_aln_mz, data_aln_int)

        checked_raw, checked_aln = verify_datasets(data_raw_linked, data_aln_linked, 1)

        _, ref_aln = find_ref(checked_aln, REF, DEV)
        _, ref_raw = find_ref(checked_raw, REF, DEV)

        checked_raw.reference = ref_aln
        checked_aln.reference = ref_raw

        ref_list.append(ref_aln)

    """
    kde = stats.gaussian_kde(ref_list)

    x_vals = np.linspace(min(ref_list), max(ref_list), 1000)
    y_vals = kde.evaluate(x_vals)

    ax.plot(x_vals, y_vals)
    ax.scatter(ref_list, np.zeros(len(ref_list)),marker='x', color='black')
    """


'''processes functions'''


def front_main(q):
    root = Root()
    root.mainloop()

'''program main function'''
def main():
    q = mp.Queue()
    front_process = mp.Process(target=front_main,args=(q,))
    front_process.start()
    front_process.join()

    features_raw = File(FILE_NAMES[0], FOLDERS).read(DATASET)
    features_aln = File(FILE_NAMES[1], FOLDERS).read(DATASET)
    read_dataset(features_raw, features_aln, limit=100)



if __name__ == '__main__':
    logger = setup_logger()
    try:
        with open("config.yaml", 'r') as yml_file:
            yaml_config = yaml.load(yml_file, Loader=yaml.FullLoader)
            FILE_NAMES, FOLDERS, CASH, DATASET, REF, DEV = (yaml_config["FILE_NAMES"],
                                                            yaml_config["FOLDERS"],
                                                            yaml_config["CASH"],
                                                            yaml_config["DATASET"],
                                                            yaml_config["REF"],
                                                            yaml_config["DEV"])
            logger.info("Configuration loaded")

    except FileNotFoundError:
        logger.error('File "config.yaml" not found')
    try:
        main()
    except Exception as fatal:
        logger.fatal(f'Fatal error: {fatal}. Program terminated.', exc_info=True)
        quit()
