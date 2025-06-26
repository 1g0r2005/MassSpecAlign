import copy
import math
import sys
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty

import h5py
import numpy as np
import pyqtgraph as pg
import scipy.stats as stats
import yaml
from IPython.external.qt_for_kernel import QtCore
from KDEpy import FFTKDE
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLineEdit, QLabel, QPushButton, \
    QFormLayout, QFileDialog
from diptest import diptest
from numba import jit
from tqdm import tqdm

"""classes declaration"""


class Const:
    """Class for handling constants """
    RAW = None
    ALN = None
    CASH = None
    DATASET = None
    REF = None
    DEV = None
    N_DOTS = None
    BW = None


class WorkerSignals(QObject):
    output = pyqtSignal(str)
    error = pyqtSignal(str)
    result = pyqtSignal(object)


class StreamRedirect:
    def __init__(self, q):
        self.q = q

    def write(self, msg: str):
        if msg.strip():
            self.q.put(msg)

    def flush(self):
        pass


class ProcessManager:
    def __init__(self, signals):
        self.signals = signals
        self.output_q = Queue()
        self.error_q = Queue()
        self.return_q = Queue()
        self.process_set = set()

    def run_process(self, target, target_name, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        p = Process(target=self._std_wrapper, args=(target, self.output_q, self.error_q, self.return_q, args, kwargs))
        self.process_set.add(target_name)
        p.start()
        return p

    def end_process(self, process, target_name):
        if target_name in self.process_set:
            try:
                process.join()
                if len(self.process_set) == 0:
                    self.shared_memory.unlink()
            except Exception as e:
                self.error_q.put(e)
        else:
            self.error_q.put(str(Exception('no process with name {} running'.format(target_name))))

    @staticmethod
    def _std_wrapper(target, out_q, error_q, ret_q, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        try:
            sys.stdout = StreamRedirect(out_q)
            sys.stderr = StreamRedirect(error_q)
            result = target(*args, **kwargs)
            ret_q.put((target.__name__, result))

        except Exception as e:
            error_q.put(e)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

    def __check_return(self):
        while not self.return_q.empty():
            try:
                func_name, content = self.return_q.get_nowait()
                print('return from {}'.format(func_name))
                self.signals.result.emit(content)
            except Empty:
                break
            except Exception as e:
                self.error_q.put(e)

    def __check_error(self):
        while not self.error_q.empty():
            try:
                msg = self.error_q.get_nowait()
                # print(msg, end='')
                self.signals.error.emit(str(msg))
            except Empty:
                break

    def __check_out(self):
        while not self.output_q.empty():
            try:
                msg = self.output_q.get_nowait()
                # print(msg)
                self.signals.output.emit(msg)
            except Empty:
                break

    def check_queues(self):
        self.__check_return()
        self.__check_error()
        self.__check_out()


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

    def sync_delete(self, index):
        new_self = np.delete(self, index)
        if self.linked_array is not None:
            new_linked_array = np.delete(self.linked_array, index, axis=0)
            return LinkedList(new_self, new_linked_array)
        return new_self


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
            print(f'File {self.real_path} not found')
            return None
        except Exception as err:
            print(f'Reading Error: {err}')
            return None


class LogWidget(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        QtWidgets.QTabWidget.__init__(self, parent)
        super().__init__(parent)
        self.setReadOnly(True)

    def __scrollDown(self):
        scroll = self.verticalScrollBar()
        end_text = scroll.maximum()
        scroll.setValue(end_text)

    def updateText(self, msg: str):
        try:
            self.append(str(msg))
            self.__scrollDown()
        except Exception as e:
            print(e)


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
        self.console_log = LogWidget()

        self.main = MainPage(self, 'Main')
        self.tabs.addTab(self.main, self.main.title)

        self.graph = GraphPage(self, x_labels=['m/z'], y_labels=['dens'], title='KDE', title_plots=('kde',))
        self.tabs.addTab(self.graph, self.graph.title)

        self.stats = StatGraphPage(self, title='Statistics')
        self.tabs.addTab(self.stats, self.stats.title)

        self.signals = WorkerSignals()
        self.manager = ProcessManager(self.signals)

        self.timer = QTimer()
        self.timer.timeout.connect(self.manager.check_queues)
        self.timer.start(100)
        self.signals.output.connect(self.console_log.updateText)
        self.signals.error.connect(self.console_log.updateText)
        self.signals.result.connect(self.console_log.updateText)
        # self.signals.result.connect(lambda ds: self.graph.add_plot_mul(ds))
        self.signals.result.connect(self.redirect_outputs)

    def redirect_outputs(self, ret):
        self.aval_func = {'show': self.graph.add_plot_mul, 'stats': self.stats.add_plot_mul}
        for output in ret:
            self.aval_func[output[0]](output[1])


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


    def start_calc(self, target, process_name=None, args=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if args is None:
            args = []
        if process_name is None:
            process_name = target.__name__

        if process_name in self.manager.process_set:
            pass  #already started

        self.manager.run_process(target, process_name, args, kwargs)


class MainPage(QWidget):
    def __init__(self, parent, title):
        super().__init__()
        self.title = title
        self.parent = parent
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(self)
        self.const = Const

        form_panel = QtWidgets.QWidget()
        form_layout = QFormLayout()
        form_panel.setLayout(form_layout)

        config_panel = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout()
        config_panel.setLayout(config_layout)
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
        self.bw_set = QLineEdit()
        self.n_dots_set = QLineEdit()
        form_layout.addRow(QLabel("Raw data:"), self.raw_layout)
        form_layout.addRow(QLabel("Alignment data:"), self.aln_layout)
        form_layout.addRow(QLabel("Dataset:"), self.dataset)
        form_layout.addRow(QLabel("Reference point:"), self.ref_set)
        form_layout.addRow(QLabel("Acceptable deviation for msalign:"), self.dev_set)
        form_layout.addRow(QLabel("Bandwidth:"), self.bw_set)
        form_layout.addRow(QLabel("Number of dots:"), self.n_dots_set)

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
        self.main_layout.addWidget(self.parent.console_log)

    def open_file(self, raw_filename):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "HDF (*.hdf,*.h5,,'*.hdf5');;All Files (*)")
        if not filename: return
        raw_filename.setText(filename)

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
            self.bw_set.setText(str(yaml_config['BW']))
            self.n_dots_set.setText(str(yaml_config['NDOTS']))


    def save_config(self):
        try:
            data = (self.raw_filename.text(),
                    self.aln_filename.text(),
                    self.ref_set.text(),
                    self.dev_set.text(),
                    self.dataset.text(),
                    self.bw_set.text(),
                    self.n_dots_set.text())
            if '' in data:
                raise Exception('Empty string')
            Const.RAW, Const.ALN, Const.REF, Const.DEV, Const.DATASET, Const.BW, Const.N_DOTS = data[0], data[1], float(
                data[2]), float(
                data[3]), data[4], float(data[5]), int(data[6])
            self.calc_button.setEnabled(True)
        except Exception as e:
            print(e)

    def signal(self):
        self.parent.start_calc(target=find_dots_process, args=(self.const.RAW,
                                                               self.const.ALN,
                                                               self.const.DATASET,
                                                               self.const.REF,
                                                               self.const.DEV,
                                                               self.const.BW,
                                                               self.const.N_DOTS
                                                               ))

class GraphPage(QWidget):
    def __init__(self, parent, canvas_count=1, title='PlotPage', title_plots=None, x_labels=None, y_labels=None,
                 color=(255, 255, 255), bg_color=(0, 0, 0)):
        super().__init__()

        if x_labels is None: x_labels = ['x'] * canvas_count
        if y_labels is None: y_labels = ['y'] * canvas_count
        if title_plots is None: title_plots = [f'plot{i}' for i in range(canvas_count)]

        self.canvas_adj = {title_plots[i]: i for i in range(canvas_count)}

        self.parent = parent
        self.title = title
        self.plot_spaces = [pg.PlotWidget() for i in range(canvas_count)]
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        for i in range(canvas_count):
            self.plot_spaces[i].showGrid(x=True, y=True)
            self.plot_spaces[i].setTitle(title_plots[i])
            self.layout.addWidget(self.plot_spaces[i])
            self.plot_spaces[i].setLabel('bottom', x_labels[i])
            self.plot_spaces[i].setLabel('left', y_labels[i])

    def add_plot(self, data, plot_name, color, canvas_name=None):
        if canvas_name is None:
            plot_id = 0
        else:
            plot_id = self.canvas_adj[canvas_name]
        pen = pg.mkPen(color=color)
        self.plot_spaces[plot_id].plot(data[0], data[1], name=plot_name, pen=pen)
        self.plot_spaces[plot_id].getAxis('bottom').setVisible(True)

    def add_line(self, data, y_max, color, canvas_name=None):
        try:
            if canvas_name is None:
                plot_id = 0
            else:
                plot_id = self.canvas_adj[canvas_name]

            pen = pg.mkPen(color=color, style=QtCore.Qt.DashLine)
            y_min = 0
            x = np.column_stack([data,
                                 data,
                                 np.full_like(data, np.nan)]).ravel()
            y = np.column_stack([np.full_like(data, y_min),
                                 np.full_like(data, y_max),
                                 np.full_like(data, np.nan)]).ravel()
            self.plot_spaces[plot_id].plot(x, y, pen=pen)
            self.plot_spaces[plot_id].getAxis('bottom').setVisible(True)
        except Exception as e:
            print(e)
            # self.plot_space.addItem(pg.InfiniteLine(pos=x,angle=90,pen=pen,movable=False))

    def add_plot_mul(self, ds):
        # print(ds)
        for data in ds:
            if data[-2] == 'p':
                self.add_plot(data[0], data[1], data[2], data[-1])
            elif data[-2] == 'vln':
                self.add_line(data[0], data[1], data[3], data[-1])


class StatGraphPage(GraphPage):
    def __init__(self, parent, title='StatPage', x_labels=None, y_labels=None,
                 color=(255, 255, 255), bg_color=(0, 0, 0), p_val=0.05):
        super().__init__(parent, canvas_count=4, title=title, title_plots=('DEV', 'MOD', 'SKEW', 'KURT'),
                         x_labels=x_labels, y_labels=y_labels, color=color, bg_color=bg_color)

        self.table = pg.TableWidget()  # сколько всего точек, медианное отклонение, число точек не мономодальных
        self.p = p_val
        self.table_data = np.zeros((3, 2))

        self.layout.setStretch(0, 1)  # Виджет 1
        self.layout.setStretch(1, 1)  # Виджет 2
        self.layout.setStretch(2, 1)  # Виджет 3
        self.layout.setStretch(3, 1)
        self.layout.addWidget(self.table)

    def add_plot_mul(self, ds):
        self.fixed_colors = [
            pg.mkColor('blue'),  # Синий
            pg.mkColor('red'),  # Красный
            pg.mkColor('green'),  # Зеленый
            pg.mkColor('yellow'),  # Желтый
            pg.mkColor('purple'),  # Фиолетовый
            pg.mkColor('cyan'),  # Голубой
        ]
        for n in range(len(ds)):
            data = ds[n]
            ds_color = self.fixed_colors[n]

            self.table_data[0, n] = len(data[0])

            self.add_plot(data[0], f'st_dev_{n}', ds_color, 'DEV')
            self.add_plot(data[1], f'dip_{n}', ds_color, 'MOD')
            self.add_plot(data[3], f'skew_{n}', ds_color, 'SKEW')
            self.add_plot(data[4], f'kurt_{n}', ds_color, 'KURT')
            self.table_data[1, n] = np.where(data[2] < self.p)[0].size
            self.table_data[2, n] = np.median(data[0])
        self.table.setData(self.table_data)
        self.table.setHorizontalHeaderLabels([str(i) for i in range(len(ds))])
        self.table.setVerticalHeaderLabels(['Total', 'Multimodal?', 'Median std dev'])

    def add_plot(self, data, plot_name, color, canvas_name=None):
        if canvas_name is None:
            plot_id = 0
        else:
            plot_id = self.canvas_adj[canvas_name]
        pen = pg.mkPen(color=color)
        no_nan = lambda arr: arr[~np.isnan(arr)]
        y, x = np.histogram(no_nan(data), bins=60)

        self.plot_spaces[plot_id].plot(x, y, stepMode=True, name=plot_name, pen=pen)
        self.plot_spaces[plot_id].getAxis('bottom').setVisible(True)




'''functions declaration'''


def peak_picking(X, Y, oversegmentation_filter=0, peak_location=1):
    n = X.size
    # Robust valley finding
    h = np.concatenate(([-1], np.where(np.diff(Y) != 0)[0], [n - 1]))
    g = (np.diff(Y[[h[1], *h[1:]]]) <= 0) & (np.diff(Y[[*h[1:], h[-1]]]) >= 0)

    left_min = h[np.concatenate([g, [False]])] + 1
    right_min = h[np.concatenate([[False], g])]
    left_min = left_min[:-1]
    right_min = right_min[1:]
    # Compute max and min for every peak
    size = left_min.shape
    val_max = np.empty(size)
    pos_peak = np.empty(size)
    for idx, [lm, rm] in enumerate(zip(left_min, right_min)):
        pp = lm + np.argmax(Y[lm:rm])
        vm = np.max(Y[lm:rm])
        val_max[idx] = vm
        pos_peak[idx] = pp

    # Remove oversegmented peaks
    while True:
        peak_thld = val_max * peak_location - math.sqrt(np.finfo(float).eps)
        pkX = np.empty(left_min.shape)

        for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
            mask = Y[lm:rm] >= th
            if np.sum(mask) == 0:
                pkX[idx] = np.nan
            else:
                pkX[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])
        dpkX = np.concatenate(([np.inf], np.diff(pkX), [np.inf]))

        j = np.where((dpkX[1:-1] <= oversegmentation_filter) & (dpkX[1:-1] <= dpkX[:-2]) & (dpkX[1:-1] < dpkX[2:]))[0]
        if j.size == 0:
            break
        left_min = np.delete(left_min, j + 1)
        right_min = np.delete(right_min, j)

        val_max[j] = np.maximum(val_max[j], val_max[j + 1])
        val_max = np.delete(val_max, j + 1)
    # print(pkX,X[left_min],X[right_min])
    return pkX, X[left_min], X[right_min]


@jit(nopython=True)
def sort_dots(ds: np.ndarray, left: np.ndarray, right: np.ndarray) -> list:
    mask = (ds[:, None] >= left) & (ds[:, None] <= right)
    return [ds[m] for m in mask.T]


def get_long_and_short(arr_1: np.ndarray, arr_2: np.ndarray) -> (np.ndarray, np.ndarray, bool):
    """
    input - arr_1, arr_2 - two np.ndarray, possibly not equal size
    return - tuple with two arrays (long, short)
    """
    size1, size2 = arr_1.shape[0], arr_2.shape[0]
    if size1 > size2:
        return arr_1, arr_2, True
    else:
        return arr_2, arr_1, False


def get_opt_strip(arr_long: Dataset, arr_short: Dataset, flag: bool) -> (Dataset, Dataset):
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


def get_index(dataset: np.ndarray, ds_id: int) -> np.ndarray:
    """
    input - dataset - np.ndarray with full recorded data from attached HDF file
          - ds_id - int with id of dataset column
    return - np.ndarray with indexes of columns with requested id
    """
    index_data = dataset[0]
    return np.where(index_data == ds_id)


def verify_datasets(data_1: LinkedList, data_2: LinkedList, threshold=1.0) -> (LinkedList, LinkedList):
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
    if threshold == 'dist_based':
        threshold = np.mean(dist_array)
    score_fit = np.max(np.abs(dist_array))

    if score_fit > threshold:
        cut_index = np.array([np.where(np.abs(dist_array) >= threshold)]).min()
        if data1_new[cut_index] < data2_new[cut_index]:
            data1_new2 = data1_new.sync_delete(cut_index)
            data2_new2 = data2_new
        else:
            data2_new2 = data2_new.sync_delete(cut_index)
            data1_new2 = data1_new
        return get_opt_strip(*get_long_and_short(data1_new2, data2_new2))
    return data1_new, data2_new


def find_ref(dataset: Dataset, approx_mz: float, deviation=1.0) -> [float, float]:
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

    dataset_list = np.empty((2, set_num), dtype=Dataset)

    for index in tqdm(range(set_num), desc='Прогресс'):
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

        dataset_list[0, index] = np.array(checked_raw)  # - checked_raw.reference
        dataset_list[1, index] = np.array(checked_aln)  # - checked_aln.reference

    return dataset_list


def prepare_array(distances):
    """prepare distances dataset"""
    concatenated = np.array([np.concatenate(sub) for sub in distances])
    indexes = np.repeat(np.arange(len(distances[0])), [len(sub_arr) for sub_arr in distances[0]])
    pre_sorted = np.vstack((concatenated, indexes))
    sorted = pre_sorted[:, pre_sorted[0].argsort()]
    return sorted

# вычислить среднее и дисперсии, проверить нормальность, проверить гипотезы о значимости различия средних и дисперсий, возможно посчитать форму распределения
def stat_params_paired(ds_raw, ds_aln, p_value=0.05):
    mean_r, mean_a = np.mean(ds_raw), np.mean(ds_aln)
    var_r, var_a = np.var(ds_raw), np.var(ds_aln)

    neq_mean, neq_var = np.nan, np.nan
    check_normal = (stats.shapiro(ds_raw)[1] > p_value) & (stats.shapiro(ds_aln)[1] > p_value)
    if check_normal:
        neq_var = stats.levene(ds_raw, ds_aln)[1] < p_value
        neq_mean = stats.ttest_rel(ds_raw, ds_aln)[1] < p_value

    return mean_r - mean_a, var_r - var_a, neq_mean, neq_var


def stat_params_unpaired(ds):
    res = np.array([[np.var(dot), *diptest(dot), stats.skew(dot), stats.kurtosis(dot)] for dot in ds])
    return res


def moving_average(a, n=2):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def out_criteria(arr, inten):
    min_int = np.max(inten) * 0.01
    max_diff = 0.3
    width_eps = 0.3
    first_or = inten < min_int
    int_criteria = abs(inten[:-1] / inten[1:] - 1) < max_diff

    width_criteria = np.diff(arr) / moving_average(np.diff(arr.linked_array).flatten()) <= width_eps
    print(int_criteria.shape, width_criteria.shape)
    second_or = np.full(arr.shape, False)
    second_or[1:] = np.logical_and(int_criteria, width_criteria)

    return np.where(np.logical_or(first_or, second_or))[0]


def criteria_apply(arr,inten):
    arr_out = copy.deepcopy(arr)
    indexes = out_criteria(arr,inten)
    for index in indexes:
        arr_out.linked_array[index-1] = sorted([arr.linked_array[index-1,0],arr.linked_array[index,1]])

    return arr_out.sync_delete(indexes)


'''process main function'''


def find_dots_process(RAW, ALN, DATASET, REF, DEV, BW, N_DOTS):
        #epsilon = 5*10**-5
        features_raw = File(RAW).read(DATASET)
        features_aln = File(ALN).read(DATASET)
        distance_list = read_dataset(features_raw, features_aln, REF, DEV)

        distance_list_prepared = prepare_array(distance_list)
        raw_concat, aln_concat, id_concat = distance_list_prepared

        kde_x_raw, kde_y_raw = FFTKDE(bw=BW, kernel='gaussian').fit(raw_concat).evaluate(N_DOTS)
        kde_x_aln, kde_y_aln = FFTKDE(bw=BW, kernel='gaussian').fit(aln_concat).evaluate(N_DOTS)

        epsilon = np.max(kde_y_raw) * 0.01

        center_r, left_r, right_r = peak_picking(kde_x_raw, kde_y_raw)
        center_a, left_a, right_a = peak_picking(kde_x_aln, kde_y_aln)
        # восстановим высоту пиков
        max_center_r, max_center_a = np.interp(center_r, kde_x_raw, kde_y_raw), np.interp(center_a, kde_x_aln,
                                                                                          kde_y_aln)


        # print(max_center_r,max_center_a)

        borders_r = np.stack((left_r, right_r), axis=1)
        borders_a = np.stack((left_a, right_a), axis=1)
        # print('borders_r')
        # print(borders_r)
        ds_raw = LinkedList(center_r, borders_r)#.sync_delete(np.where(max_center_r <= epsilon)[0])
        ds_aln = LinkedList(center_a, borders_a)#.sync_delete(np.where(max_center_a <= epsilon)[0])

        c_ds_raw,c_ds_aln = criteria_apply(ds_raw, max_center_r),criteria_apply(ds_aln, max_center_a)


        peak_lists_raw = sort_dots(raw_concat, c_ds_raw.linked_array[:, 0], c_ds_raw.linked_array[:, 1])
        peak_lists_aln = sort_dots(aln_concat, c_ds_aln.linked_array[:, 0], c_ds_aln.linked_array[:, 1])
        print(len(peak_lists_raw))
        print(len(peak_lists_aln))

        # peak_lists_raw = sort_dots(raw_concat,left_r,right_r)
        # peak_lists_aln = sort_dots(aln_concat,left_a,right_a)
        # statistics = [stat_params(peak_lists_raw[i],peak_lists_aln[i]) for i in range(len(peak_lists_raw))]

        ret = (
            ('show', (((kde_x_raw, kde_y_raw), 'raw', 'red', 'p', 'kde'),
                      ((kde_x_aln, kde_y_aln), 'aln', 'blue', 'p', 'kde'),
                      (c_ds_raw, np.max(kde_y_raw), 'raw_peaks', 'red', 'vln', 'kde'),
                      (c_ds_aln, np.max(kde_y_aln), 'aln_peaks', 'blue', 'vln', 'kde'))),
            ('stats', (stat_params_unpaired(peak_lists_raw).T, stat_params_unpaired(peak_lists_aln).T)),
        )

        return ret


'''
except Exception as e:
    print(e)
'''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
