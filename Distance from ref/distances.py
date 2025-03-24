import logging
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('MZ alignment quality evaluation')
        self.geometry('800x600')

        self.base = ttk.Notebook(self)
        self.base.pack(fill='both', expand=True)

    def add_page(self, page: tk.Frame, page_title):
        self.base.add(page, text=page_title)


class TkPlot:
    """Pyplot dynamic plot wrapped into tkinter canvas object"""

    def __init__(self, root, x=None, y=None, x_label="x", y_label='y'):
        self.root = root
        self.fig, self.ax = plt.subplots()

        self.ax.set(xlabel=x_label, ylabel=y_label)

        if x is None or y is None:
            self.x, self.y = np.array([]), np.array([])
        else:
            self.x, self.y = x, y

        self.line_plot, = self.ax.plot(self.x, self.y, markersize=8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.warning_label = tk.Label(self.root, text='No data found')
        self.warning_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        if self.x.size > 0 or self.y.size > 0:
            self.warning_label.place_forget()

    def update(self, x=None, y=None):
        self.x, self.y = x, y
        if x is None or y is None:
            self.x, self.y = np.array([]), np.array([])
        else:
            self.x, self.y = x, y

        self.line_plot.set_ydata(self.y)
        self.line_plot.set_xdata(self.x)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        if self.x.size > 0 or self.y.size > 0:
            self.warning_label.place_forget()
        else:
            self.warning_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)


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


def kde_process(dataset, num_dots=1000):
    kde = stats.gaussian_kde(dataset)

    x_vals = np.linspace(min(dataset), max(dataset), num_dots)
    y_vals = kde.evaluate(x_vals)

    return x_vals, y_vals


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

    dataset_list = np.empty((set_num, 2), dtype=Dataset)

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

        checked_raw.reference = ref_raw
        checked_aln.reference = ref_aln

        dataset_list[index, 0] = checked_raw
        dataset_list[index, 1] = checked_aln

    return dataset_list


'''program main function'''
def main():
    root = App()
    features_raw = File(FILE_NAMES[0], FOLDERS).read(DATASET)
    features_aln = File(FILE_NAMES[1], FOLDERS).read(DATASET)

    dataset_list = read_dataset(features_raw, features_aln)
    raw_ref_list = np.array([ds.reference for ds in dataset_list[:, 0]])
    aln_ref_list = np.array([ds.reference for ds in dataset_list[:, 1]])
    '''
    x1,y1 = raw_ref_list,np.full(raw_ref_list.shape,1)
    x2,y2 = aln_ref_list,np.full(aln_ref_list.shape,2)
    '''
    x1, y1 = kde_process(raw_ref_list)
    x2, y2 = kde_process(aln_ref_list)

    plot_data = {'Raw refs': (x1, y1), 'Aln refs': (x2, y2)}

    def on_plot_select(event=None):
        selection = plot_combobox.get()
        x, y = plot_data[selection]
        print(selection)
        plot.update(x, y)

    graph_page = ttk.Frame(root)
    root.add_page(graph_page, 'graph')
    plot = TkPlot(graph_page)

    plot_combobox = ttk.Combobox(graph_page, values=list(plot_data.keys()))
    plot_combobox.pack(side=tk.TOP)
    plot_combobox.set('Raw refs')
    plot_combobox.bind('<<ComboboxSelected>>', on_plot_select)

    on_plot_select()
    root.mainloop()

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
