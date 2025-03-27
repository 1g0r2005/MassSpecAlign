import logging
import multiprocessing as mul
import sys
from multiprocessing import Process
from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg
import scipy.stats as stats
import yaml
from IPython.external.qt_for_kernel import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QLineEdit, QLabel, QPushButton, \
    QFormLayout, QFileDialog
from numba import jit
from tqdm import tqdm

'''decorators declaration'''


def func_logger(add_info='', full=False):
    def actual_decorator(func):
        def inner(*args, **kwargs):
            logger = logging.getLogger('main')
            ret = func(*args, **kwargs)
            logger.info(f'Function {func.__name__}' + (f'with {args, kwargs} was called' if full else '') + (
                f'({add_info})' if add_info else ''))
            return ret

        return inner

    return actual_decorator


"""classes declaration"""

class Const:
    """Class for handling constants """
    RAW = None
    ALN = None
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
    def __init__(self, file_name):
        self.real_path = Path(file_name)

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


class SpecialHandler(logging.StreamHandler, QObject):
    message = pyqtSignal(str)

    def __init__(self):
        logging.StreamHandler.__init__(self)
        QtCore.QObject.__init__(self)

    def emit(self, record: str):
        log_message = self.format(record)
        if 'DEBUG' in log_message:
            text = f"""<span style="color:#000000">{log_message}</span>"""
        elif 'INFO' in log_message:
            text = f"""<span style="color:#000000">{log_message}</span>"""
        elif 'WARNING' in log_message:
            text = f"""<span style="color:#cecc21">{log_message}</span>"""
        elif 'ERROR' in log_message:
            text = f"""<span style="color:#ff7e00">{log_message}</span>"""
        elif 'CRITICAL' in log_message:
            text = f"""<span style="color:#ff0000">{log_message}</span>"""
        self.message.emit(text)
        self.flush()


class LogWidget(QtWidgets.QTextEdit):
    def __init__(self, handler: logging.StreamHandler, parent=None):
        QtWidgets.QTabWidget.__init__(self, parent)
        super().__init__(parent)
        self.handler = handler
        self.handler.message.connect(self.__updateText)

    def __scrollDown(self):
        scroll = self.verticalScrollBar()
        end_text = scroll.maximum()
        scroll.setValue(end_text)

    def __updateText(self, msg: str):
        self.append(msg)
        self.__scrollDown()

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
        self.logger, self.special_handler = setup_logger()
        self.main = MainPage(self, 'Main')
        self.tabs.addTab(self.main, self.main.title)
        self.graph = GraphPage(self, 'Graph')
        self.tabs.addTab(self.graph, self.graph.title)


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

    def setup_calc(self):
        """setup main calculation function separate from gui"""

        self.parent_conn, self.child_conn = mul.Pipe()
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_pipe)
        self.process = None

    def start_calc(self):
        if self.process and self.process.is_alive():
            pass  # already started
        self.process = Process(target=calc_process, args=(self.child_conn,
                                                          self.const.RAW,
                                                          self.const.ALN,
                                                          self.const.DATASET,
                                                          self.const.REF,
                                                          self.const.DEV
                                                          ))
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
                elif msg_type == 'Info':
                    self.logger.info(msg)
                elif msg_type == 'Error':
                    self.logger.error(msg)

            except Exception as err:
                print(err)
                self.timer.stop()


class MainPage(QWidget):
    def __init__(self, parent, title):
        super().__init__()
        self.title = title
        self.parent = parent
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(self)

        form_panel = QtWidgets.QWidget()
        form_layout = QFormLayout()
        form_panel.setLayout(form_layout)

        config_panel = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout()
        config_panel.setLayout(config_layout)
        self.logger, self.special_handler = parent.logger, parent.special_handler
        self.setLayout(self.main_layout)

        # Raw
        self.raw_layout = QtWidgets.QHBoxLayout()
        self.raw_filename = QLineEdit()
        self.raw_open_button = QPushButton("Browse")
        self.raw_open_button.clicked.connect(lambda: self.open_file(self.raw_filename))
        self.raw_layout.addWidget(self.raw_filename)
        self.raw_layout.addWidget(self.raw_open_button)
        # aln
        self.aln_layout = QtWidgets.QHBoxLayout()
        self.aln_filename = QLineEdit()
        self.aln_open_button = QPushButton("Browse")
        self.aln_open_button.clicked.connect(lambda: self.open_file(self.aln_filename))
        self.aln_layout.addWidget(self.aln_filename)
        self.aln_layout.addWidget(self.aln_open_button)
        # ref and dev
        self.dataset = QLineEdit()
        self.ref_set = QLineEdit()
        self.dev_set = QLineEdit()
        form_layout.addRow(QLabel("Raw data:"), self.raw_layout)
        form_layout.addRow(QLabel("Alignment data:"), self.aln_layout)
        form_layout.addRow(QLabel("Dataset:"), self.dataset)
        form_layout.addRow(QLabel("Reference point:"), self.ref_set)
        form_layout.addRow(QLabel("Acceptable deviation:"), self.dev_set)

        self.config_button = QPushButton("Browse config")
        self.config_button.clicked.connect(lambda: self.open_config())
        self.load_config_button = QPushButton("Load")
        self.load_config_button.clicked.connect(lambda: self.save_config())
        self.calc_button = QPushButton("Calculate")
        config_layout.addWidget(self.config_button)
        config_layout.addWidget(self.load_config_button)
        config_layout.addWidget(self.calc_button)
        self.calc_button.clicked.connect(lambda: self.signal())
        self.calc_button.setEnabled(False)
        self.splitter.addWidget(form_panel)
        self.splitter.addWidget(config_panel)
        self.main_layout.addWidget(self.splitter)
        self.main_layout.addWidget(LogWidget(self.special_handler))

    @func_logger(full=True)
    def open_file(self, raw_filename):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "HDF (*.hdf,*.h5,,'*.hdf5');;All Files (*)")
        if not filename: return
        raw_filename.setText(filename)

    @func_logger(full=True)
    def open_config(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "yaml (*.yaml);;All Files (*)")
        if not filename: return
        with open(filename, 'r', encoding='utf8') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            self.raw_filename.setText(yaml_config['FILE_NAMES'][0])
            self.aln_filename.setText(yaml_config['FILE_NAMES'][1])
            self.ref_set.setText(str(yaml_config['REF']))
            self.dev_set.setText(str(yaml_config['DEV']))
            self.dataset.setText(str(yaml_config['DATASET']))

    @func_logger(full=True)
    def save_config(self):
        try:
            data = (self.raw_filename.text(),
                    self.aln_filename.text(),
                    self.ref_set.text(),
                    self.dev_set.text(),
                    self.dataset.text())
            if '' in data:
                raise Exception('Empty string')
            Const.RAW, Const.ALN, Const.REF, Const.DEV, Const.DATASET = data[0], data[1], float(data[2]), float(
                data[3]), data[4]
            self.calc_button.setEnabled(True)
        except Exception as e:
            print(e)

    @func_logger()
    def signal(self):
        self.parent.setup_calc()
        self.parent.start_calc()

class GraphPage(QWidget):
    def __init__(self, parent, title='graph', x=None, y=None, x_label='x', y_label='y', color=(255, 255, 255),
                 bg_color=(0, 0, 0)):
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

    @func_logger(add_info='GRAPH PAGE UPDATE')
    def update(self, x=None, y=None):
        self.x = x
        self.y = y
        self.data_line.setData(self.x, self.y)

'''functions declaration'''
def kde_process(dataset, num_dots=1000):
    kde = stats.gaussian_kde(dataset)
    x_vals = np.linspace(min(dataset), max(dataset), num_dots)
    y_vals = kde.evaluate(x_vals)
    return x_vals, y_vals

def setup_logger():
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(f'{__name__}.log', "w")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    special_handler = SpecialHandler()
    special_handler.setFormatter(log_formatter)
    logger.addHandler(special_handler)
    logger.addHandler(file_handler)

    return logger, special_handler

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
    opt_shift = np.where(score_array == score_array.min())[0][0]
    opt_long = arr_long[opt_shift:opt_shift + size]
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


def calc_process(conn, RAW, ALN, DATASET, REF, DEV):
    conn.send(('Info', f'Process {calc_process.__name__} started'))
    try:
        features_raw = File(RAW).read(DATASET)
        features_aln = File(ALN).read(DATASET)
        dataset_list = read_dataset(features_raw, features_aln, REF, DEV)
        raw_ref_list = np.array([ds.reference for ds in dataset_list[:, 0]])
        aln_ref_list = np.array([ds.reference for ds in dataset_list[:, 1]])
        x1, y1 = kde_process(aln_ref_list)
        conn.send(('data', (x1, y1)))
    except Exception as e:
        conn.send('Error', e)
    finally:
        conn.send(('Info', f'Process {calc_process.__name__} ended successfully'))
        conn.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
