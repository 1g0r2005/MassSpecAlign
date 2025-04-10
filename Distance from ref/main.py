import logging
import sys
from functools import partial
from multiprocessing import Process, Queue, Pool, freeze_support
from pathlib import Path
from queue import Empty

import h5py
import numpy as np
import yaml
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QLineEdit, QLabel, QPushButton, \
    QFormLayout, QFileDialog
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from numba import jit
from scipy.stats import levene
from sklearn.neighbors import KernelDensity
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

        self.graph = MPLGraph(self, x_label='m/z', y_label='dens')
        self.tabs.addTab(self.graph, self.graph.title)

        self.heat = MPLHeat(self)
        self.tabs.addTab(self.heat, self.heat.title)

        self.signals = WorkerSignals()
        self.manager = ProcessManager(self.signals)

        self.timer = QTimer()
        self.timer.timeout.connect(self.manager.check_queues)
        self.timer.start(100)
        self.signals.output.connect(self.console_log.updateText)
        self.signals.error.connect(self.console_log.updateText)
        self.signals.result.connect(self.console_log.updateText)
        # self.signals.result.connect(self.graph.set_plots)

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

    def signal(self):
        self.parent.start_calc(target=find_dots_process, args=(self.const.RAW,
                                                               self.const.ALN,
                                                               self.const.DATASET,
                                                               self.const.REF,
                                                               self.const.DEV
                                                               ))


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class MPLGraph(QWidget):

    def __init__(self, parent, title='PlotPage', title_plot='plot', x_label='x', y_label='y', color=(255, 255, 255),
                 bg_color=(0, 0, 0)):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout()
        self.title = title
        self.figure = Figure()
        self.canvas = MplCanvas(parent)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

        self.canvas.axes.set_title(title_plot)
        self.canvas.axes.set_xlabel(x_label)
        self.canvas.axes.set_ylabel(y_label)

    def set_plots(self, datapack, names=None):
        self.canvas.axes.clear()
        plot_count = len(datapack)
        if names is None: names = [i for i in range(plot_count)]
        for i in range(plot_count):
            data = datapack[i]
            self.canvas.axes.scatter(data[0], data[1], label=names[i])
        self.canvas.axes.legend()
        self.canvas.axes.grid(True)
        self.canvas.draw()


class MPLHeat(QWidget):
    def __init__(self, parent, title='HeatmapPage', title_plot='heatmap'):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout()
        self.title = title
        self.canvas = MplCanvas(parent)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

        self.canvas.axes.set_title(title_plot)

    def set_heatmaps(self, datapack, names):
        self.canvas.axes.clear()
        plot_count = len(names)
        vmin = min([np.min(data) for data in datapack])
        vmax = max([np.max(data) for data in datapack])
        for i in range(plot_count):
            data = datapack[i]
            self.canvas.axes.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax, label=names[i])
        self.canvas.axes.legend()
        self.canvas.axes.grid(True)
        self.canvas.draw()

'''functions declaration'''


def kde_points(data, num_bins, left=None, right=None, center=False):
    """gaussian KDE with fixed borders"""
    if left is None: left = np.min(data)
    if right is None: right = np.max(data)
    if center: data = data - data.mean()
    data = data.reshape(-1, 1)
    x_grid = np.linspace(left, right, num_bins).reshape(-1, 1)
    # Настройка KDE (можно менять ядро и bandwidth)
    kde_obj = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data)
    # Логарифм плотности → преобразуем в обычную шкалу
    log_density = kde_obj.score_samples(x_grid)
    dens = np.exp(log_density)
    return dens


def heatmap_prepare_data(data, DEV, y):
    """create matrix for heatmap"""
    n = len(data)
    worker = partial(kde_points, **{"num_bins": y, "left": (-DEV), "right": (+DEV), 'center': True})
    with Pool(4) as pool:
        matrix = list(tqdm(pool.imap(worker, data, chunksize=1), total=n))
    return np.array(matrix)

@jit(nopython=True)
def get_long_and_short(arr_1: np.ndarray, arr_2: np.ndarray):
    """
    input - arr_1, arr_2 - two np.ndarray, possibly not equal size
    return - tuple with two arrays (long, short)
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

        dataset_list[0, index] = np.array(checked_raw) - checked_raw.reference
        dataset_list[1, index] = np.array(checked_aln) - checked_aln.reference

    return dataset_list


def prepare_array(distances):
    """prepare distances dataset"""
    concatenated = np.array([np.concatenate(sub) for sub in distances])
    indexes = np.repeat(np.arange(len(distances[0])), [len(sub_arr) for sub_arr in distances[0]])
    pre_sorted = np.vstack((concatenated, indexes))
    sorted = pre_sorted[:, pre_sorted[0].argsort()]
    return sorted


def find_dots(distances, DEV):
    dots = []
    counter = 0
    is_in_list = set()
    for i in tqdm(range(distances.shape[1])):
        if not dots:
            dots.append([])
            dots[counter].append(distances[:2, i])
            is_in_list.add(distances[2, i])
        else:
            if abs(distances[0, i] - dots[counter][0][0]) <= (2 * DEV):
                if distances[2, i] in is_in_list:
                    print(f'ERROR with {distances[2, i]}')
                    pass
                dots[counter].append(distances[:2, i])
                is_in_list.add(distances[2, i])
            else:
                is_in_list.clear()
                dots.append([distances[:2, i]])
                is_in_list.add(distances[2, i])
                counter += 1
    print(dots)
    dots_concat = [np.array(group).T for group in dots]
    print(dots_concat)
    raw_dots_concat = [group[0] for group in dots_concat]  # delete arrays from one element (outliers)
    aln_dots_concat = [group[1] for group in dots_concat]

    return raw_dots_concat, aln_dots_concat


def variance(arr1, arr2, REF, p=0.05):
    """calculate standard deviations for two arrays and check """
    coord = np.mean(np.hstack((arr1, arr2))) + REF  # restore m/z for dots
    var_1, var_2 = np.std(arr1), np.std(arr2)
    # print(f'var1 = {var_1}, var2 = {var_2}')
    delta_dev = var_2 - var_1
    stat, p_value = levene(arr1, arr2)
    if np.isnan(p_value):
        # print(f'ERROR with {arr1} and {arr2}')
        p_value = 0.0
    return coord, delta_dev, p_value


'''process main function'''


def process_wrapper(args):
    """Обертка для обработки одной итерации"""
    i, raw, aln, REF = args
    coord, delta_dev, p_value = variance(raw, aln, REF)
    return i, coord, delta_dev, p_value


def find_dots_process(RAW, ALN, DATASET, REF, DEV):
    P = 0.05
    try:
        features_raw = File(RAW).read(DATASET)
        features_aln = File(ALN).read(DATASET)
        distance_list = read_dataset(features_raw, features_aln, REF, DEV, limit=300)

        raw_dots, aln_dots = find_dots(prepare_array(distance_list), 1)
        n_dots = len(raw_dots)
        print(n_dots)
        delta_dev_list = np.empty((2, n_dots), dtype=float)
        p_value_list = np.empty((2, n_dots), dtype=float)

        tasks = [(i, raw_dots[i], aln_dots[i], REF) for i in range(n_dots)]

        with Pool(processes=4) as pool:
            # Используем imap_unordered для прогресс-бара и экономии памяти
            results = list(tqdm(pool.imap_unordered(process_wrapper, tasks),
                                total=n_dots))
        for i, coord, delta_dev, p_value in results:
            delta_dev_list[:, i] = [coord, delta_dev]
            p_value_list[:, i] = [coord, p_value]

        return delta_dev_list, p_value_list
    except Exception as e:
        print(e)


if __name__ == '__main__':
    freeze_support()
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
