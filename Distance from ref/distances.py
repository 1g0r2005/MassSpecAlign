import logging
import multiprocessing as mul
import os
import sys
from multiprocessing import Process
from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg
import scipy.stats as stats
import yaml
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout
from numba import jit
from tqdm import tqdm

'''classes declaration'''

class Const:
    FILE_NAMES = None
    FOLDERS = None
    CASH = None
    DATASET = None
    REF = None
    DEV = None

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


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('MZ alignment quality evaluation')
        self.resize(800, 600)
        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabs.setMovable(True)
        self.setCentralWidget(self.tabs)
        self.tabs.currentChanged.connect(self.adjust_tab_sizes)
        self.const = Const

        self.graph = GraphPage('REF')
        self.tabs.addTab(self.graph, self.graph.title)
    def add_page(self, page: QWidget, page_title: str):
        self.tabs.addTab(page, page_title)


    def adjust_tab_sizes(self):
        """resize core widget"""
        tab_size = self.tabs.size()
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            tab.resize(tab_size)

    def resizeEvent(self, event):
        """resize core widget - event catch"""
        super().resizeEvent(event)
        self.adjust_tab_sizes()

    def setup_calc(self, graph):
        """setup main calculation function separate from gui"""

        self.parent_conn, self.child_conn = mul.Pipe()
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_pipe)
        self.process = None

    def start_calc(self):
        if self.process and self.process.is_alive():
            pass  # already started
        self.process = Process(target=calc_process, args=(self.child_conn,
                                                          self.const.FILE_NAMES,
                                                          self.const.FOLDERS,
                                                          self.const.DATASET,
                                                          self.const.REF,
                                                          self.const.DEV))
        self.process.start()
        self.timer.start(100)

    def stop_calc(self):
        if self.process and self.process.is_alive():
            self.parent_conn.send(('command', 'stop'))
            self.process.join()
        self.timer.stop()
        self.parent_conn.close()

    def check_pipe(self):
        while self.parent_conn.poll():
            try:
                msg_type, msg = self.parent_conn.recv()
                if msg_type == 'data':
                    self.graph.update(msg[0], msg[1])
                    QApplication.processEvents()
                elif msg_type == 'update':
                    print(f'status : {msg}')
                # self.signals.data.emit(msg)
            except Exception as err:
                print(err)
                self.timer.stop()


class GraphPage(QWidget):

    def __init__(self, title, x=None, y=None, x_label='x', y_label='y', color=(255, 255, 255), bg_color=(0, 0, 0)):
        super().__init__()
        self.layout = QVBoxLayout()
        self.title = title
        self.Plot = pg.PlotWidget()
        self.layout.addWidget(self.Plot)
        self.setLayout(self.layout)

        self.Plot.setLabel('bottom', x_label)
        self.Plot.setLabel('left', y_label)

        self.Plot.setBackground(bg_color)
        if x is not None and y is not None:
            self.x = x
            self.y = y
        else:
            self.x = []
            self.y = []

        pen = pg.mkPen(color=color)
        self.data_line = self.Plot.plot(self.x, self.y, pen=pen)

    def update(self, x=None, y=None):
        self.x = x
        self.y = y
        self.data_line.setData(self.x, self.y)

        print('update!', self.x, self.y)
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


def read_dataset(dataset_raw: np.ndarray, dataset_aln: np.ndarray, REF, DEV, limit=None):
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


'''process main function'''


def calc_process(conn, FILE_NAMES, FOLDERS, DATASET, REF, DEV):
    try:
        features_raw = File(FILE_NAMES[0], FOLDERS).read(DATASET)
        features_aln = File(FILE_NAMES[1], FOLDERS).read(DATASET)
        dataset_list = read_dataset(features_raw, features_aln, REF, DEV, limit=200)
        raw_ref_list = np.array([ds.reference for ds in dataset_list[:, 0]])
        aln_ref_list = np.array([ds.reference for ds in dataset_list[:, 1]])
        x1, y1 = kde_process(aln_ref_list)
        conn.send(('data', (x1, y1)))
    finally:
        conn.send(('status', 'end'))
        conn.close()

'''program main function'''
def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()

    main_window.setup_calc(main_window.graph)
    main_window.start_calc()
    main_window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    logger = setup_logger()
    try:
        with open("config.yaml", 'r') as yml_file:
            yaml_config = yaml.load(yml_file, Loader=yaml.FullLoader)
            c = Const
            c.FILE_NAMES, c.FOLDERS, c.CASH, c.DATASET, c.REF, c.DEV = (yaml_config["FILE_NAMES"],
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
