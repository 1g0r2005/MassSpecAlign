import os

os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import copy
import math
import sys
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
import traceback

import h5py
import numpy as np
import pyqtgraph as pg
import scipy.stats as stats
import yaml
from IPython.external.qt_for_kernel import QtCore
from KDEpy import FFTKDE
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, QObject, pyqtSignal, Qt, QThread
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLineEdit, QLabel, QPushButton, \
    QFormLayout, QFileDialog, QTableWidgetItem, QTableWidget, QVBoxLayout, QHeaderView, QAbstractItemView, QSplitter, \
    QHBoxLayout, QTreeWidget, QTreeWidgetItem, QProgressBar
from diptest import diptest
from numba import jit

import alignment

"""classes declaration"""


class Const:
    """Class for handling constants """
    RAW = None
    ALN = None
    CASH = None
    DATASET_RAW = None
    DATASET_ALN = None
    REF = None
    DEV = None
    N_DOTS = None
    BW = None


class WorkerSignals(QObject):
    output = pyqtSignal(str)
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    create_pbar = pyqtSignal(tuple)

    def find_dots_process(self):
        try:

            features_raw, attrs_raw = File(Const.RAW).read(Const.DATASET_RAW)
            features_aln, attrs_aln = File(Const.ALN).read(Const.DATASET_ALN)

            distance_list = read_dataset(self,features_raw, attrs_raw, features_aln, attrs_aln, Const.REF, Const.DEV,limit=1000)

            distance_list_prepared = prepare_array(distance_list)
            raw_concat, aln_concat, id_concat = distance_list_prepared

            kde_x_raw, kde_y_raw = FFTKDE(bw=Const.BW, kernel='gaussian').fit(raw_concat).evaluate(Const.N_DOTS)
            kde_x_aln, kde_y_aln = FFTKDE(bw=Const.BW, kernel='gaussian').fit(aln_concat).evaluate(Const.N_DOTS)


            center_r, left_r, right_r = peak_picking(kde_x_raw, kde_y_raw)
            center_a, left_a, right_a = peak_picking(kde_x_aln, kde_y_aln)
            # восстановим высоту пиков
            max_center_r, max_center_a = np.interp(center_r, kde_x_raw, kde_y_raw), np.interp(center_a, kde_x_aln,
                                                                                            kde_y_aln)

            borders_r = np.stack((left_r, right_r), axis=1)
            borders_a = np.stack((left_a, right_a), axis=1)
            ds_raw = LinkedList(center_r, borders_r)#.sync_delete(np.where(max_center_r <= epsilon)[0])
            ds_aln = LinkedList(center_a, borders_a)#.sync_delete(np.where(max_center_a <= epsilon)[0])

            c_ds_raw,c_ds_aln = criteria_apply(ds_raw, max_center_r),criteria_apply(ds_aln, max_center_a)
            c_ds_raw_intensity, c_ds_aln_intensity = np.interp(c_ds_raw, kde_x_raw, kde_y_raw), np.interp(c_ds_aln, kde_x_aln,
                                                                                                    kde_y_aln)

            peak_lists_raw = sort_dots(raw_concat, c_ds_raw.linked_array[:, 0], c_ds_raw.linked_array[:, 1])
            peak_lists_aln = sort_dots(aln_concat, c_ds_aln.linked_array[:, 0], c_ds_aln.linked_array[:, 1])

            aln_peak_lists_raw, aln_peak_lists_aln, aln_kde_raw, aln_kde_aln = alignment.munkres_align(peak_lists_raw,
                                                                                                      peak_lists_aln,
                                                                                                      c_ds_raw,
                                                                                                      c_ds_aln,
                                                                                                      c_ds_raw_intensity,
                                                                                                      c_ds_aln_intensity)

            print(len(aln_peak_lists_aln))
            s_p = np.array(
                [stat_params_paired_single(x_el, y_el) for x_el, y_el in zip(aln_peak_lists_raw, aln_peak_lists_aln)],
                dtype='object')

            ret = (
                ('show', (((kde_x_raw, kde_y_raw), 'raw', 'red', 'p', 'kde'),
                        ((kde_x_aln, kde_y_aln), 'aln', 'blue', 'p', 'kde'),
                        (aln_kde_aln,np.max(kde_y_aln),'raw_peaks','mult','vln','kde'),
                        (aln_kde_raw, np.max(kde_y_aln), 'raw_peaks', 'mult', 'vln', 'kde'))),
                #(c_ds_raw, np.max(kde_y_raw), 'raw_peaks', 'red', 'vln', 'kde'),
                #(c_ds_aln, np.max(kde_y_aln), 'aln_peaks', 'blue', 'vln', 'kde'))),
                #('stats', (stat_params_unpaired(peak_lists_raw).T, stat_params_unpaired(peak_lists_aln).T)),
                #('stats_p', (stat_params_unpaired(aln_peak_lists_raw).T, stat_params_unpaired(aln_peak_lists_aln).T)),

                ('stats',
                 ((stat_params_unpaired(peak_lists_raw).T, 'raw'), (stat_params_unpaired(peak_lists_aln).T, 'aln'))),
                ('stats_p', ((stat_params_unpaired(aln_peak_lists_raw).T, 'raw'),
                             (stat_params_unpaired(aln_peak_lists_aln).T, 'aln'))),
                ('stats_table',s_p)
            )
            print('_____________________')
            print(stat_params_unpaired(peak_lists_raw).T.shape)
            print(stat_params_unpaired(peak_lists_raw).T[0])
            print('_____________________')
            self.result.emit(ret)
            self.finished.emit()

        except Exception as error:
#            self.error.emit(str(error))
            self.error.emit(traceback.format_exc()) #temporary
            self.finished.emit()

# class Worker_processing(QObject):

        # return ret
class DatasetHeaders:
    def __init__(self,attrs):
        self.index = {}
        self.name = [0]*len(attrs)
        for index, name in enumerate(attrs):
            self.name.append(name)
            self.index[name]=index

    def __call__(self,index_value):

        if isinstance(index_value,list):
            list_ind = [0]*len(self.name)
            if isinstance(index_value[0],int):
                for i,ind in enumerate(index_value):
                    list_ind[i] = self.name[ind]
            elif isinstance(index_value[0],str):
                for i,ind in enumerate(index_value):
                    list_ind[i]=self.index[ind]
            return list_ind

        else:
            if isinstance(index_value,int):
                return self.name[index_value]
            elif isinstance(index_value,str):
                return self.index[index_value]

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
                self.signals.result.emit(content)
            except Empty:
                break
            except Exception as e:
                self.error_q.put(e)

    def __check_error(self):
        while not self.error_q.empty():
            try:
                msg = self.error_q.get_nowait()
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

    def sync_reshape(self,size):
        new_self = np.reshape(self, size)
        if self.linked_array is not None:
            new_linked_array = np.reshape(self.linked_array, size)
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
        try:# TODO: add regex for more flexible input
            with h5py.File(self.real_path, 'r') as f:
                if dataset in f:
                    data = f[dataset][:]
                    attr = f[dataset].attrs["Column headers"]

                    if len(attr) != f[dataset].shape[0] and len(attr) == f[dataset].shape[1]:
                        data = data.T
                    elif len(attr) != f[dataset].shape[0] and len(attr) != f[dataset].shape[1]:
                        raise Exception("The number of columns does not match the number of headers")
                    return data, attr
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

class TreeWidget(QWidget):
    path_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.tree = QTreeWidget()
        self.initUI()

    def initUI(self):
        # Создаем layout и tree widget
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.itemDoubleClicked.connect(self.get_path)
        self.layout.addWidget(self.tree)

        self.tree.expandAll()
        self.tree.setHeaderLabels(['Name','Type','Shape','DType'])

        self.setLayout(self.layout)

    def populate_tree(self,path):
        def get_node(name, obj, indent='',parent=None):
            """Helper function to recursively print group and dataset info."""
            child = None
            if isinstance(obj, h5py.Group):
                if parent is not None:
                    child = QTreeWidgetItem(parent)
                    child.setText(0, name)
                    child.setText(1, 'Group')
                    child.setIcon(0, QIcon("folder_ico.png"))

                for key, item in obj.items():
                    get_node(key, item, indent + '  ',parent=child)

            elif isinstance(obj, h5py.Dataset):
                if parent is not None:
                    child = QTreeWidgetItem(parent)
                    child.setText(0, name)
                    child.setText(1,'Dataset')
                    child.setText(2,str(obj.shape))
                    child.setText(3,str(obj.dtype))
                    child.setIcon(0, QIcon("ds_ico.png"))

        with h5py.File(path,'r') as f:
            root = QTreeWidgetItem(self.tree)
            root.setText(0,'root')
            get_node('/',f,parent=root)

    def get_path(self):
        selection = self.tree.currentItem()
        path = []
        current = selection
        while current is not None:
            path.insert(0,current.text(0))
            current = current.parent()

        path_join = "/".join(path).replace('root//','')
        self.path_signal.emit(path_join)
        return path_join

    def update_tree(self,path):
        self.tree.clear()
        self.populate_tree(path)

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

        self.stats = StatGraphPage(self, title='Statistics(Unpaired)')
        self.tabs.addTab(self.stats, self.stats.title)

        self.stats_p = StatGraphPage(self, title='Statistics(Paired)')
        self.tabs.addTab(self.stats_p, self.stats_p.title)

        self.table = TablePage(self,title ='Stat per peak',columns=6)
        self.tabs.addTab(self.table, self.table.title)


        self.table.set_title(['distance','var(raw)','var(aln)','is normal distributed?','neq_mean?','neq_var?'])
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
        self.aval_func = {'show': self.graph.add_plot_mul, 'stats': self.stats.add_plot_mul, 'stats_p': self.stats_p.add_plot_mul,'stats_table':self.table.add_data}
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

        self.thread = QThread()
        self.processing = WorkerSignals()

        self.title = title
        self.parent = parent

        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        self.main_splitter = QSplitter()

        self.left_main_widget = QWidget()
        self.right_main_widget = QWidget()

        self.left_layout = QVBoxLayout()

        self.right_layout = QVBoxLayout()

        self.left_main_widget.setLayout(self.left_layout)
        self.right_main_widget.setLayout(self.right_layout)

        self.main_splitter.addWidget(self.left_main_widget)
        self.main_splitter.addWidget(self.right_main_widget)

        self.main_layout.addWidget(self.main_splitter)

        self.left_splitter = QSplitter(Qt.Vertical)

        self.raw_tree = TreeWidget()
        self.aln_tree = TreeWidget()

        self.left_splitter.addWidget(self.raw_tree)
        self.left_splitter.addWidget(self.aln_tree)
        self.left_layout.addWidget(self.left_splitter)

        self.splitter = QSplitter()
        self.const = Const

        form_panel = QtWidgets.QWidget()
        form_layout = QFormLayout()
        form_panel.setLayout(form_layout)

        config_panel = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout()
        config_panel.setLayout(config_layout)
        #self.setLayout(self.right_layout)

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
        self.dataset_raw = QLineEdit()
        self.dataset_aln = QLineEdit()

        self.ref_set = QLineEdit()

        self.raw_filename.setEnabled(False)
        self.aln_filename.setEnabled(False)

        self.dev_set = QLineEdit()
        self.bw_set = QLineEdit()
        self.n_dots_set = QLineEdit()

        form_layout.addRow(QLabel("Raw data:"), self.raw_layout)
        form_layout.addRow(QLabel("Alignment data:"), self.aln_layout)
        form_layout.addRow(QLabel("Dataset (raw):"), self.dataset_raw)
        form_layout.addRow(QLabel("Dataset (aln):"), self.dataset_aln)
        form_layout.addRow(QLabel("Reference point:"), self.ref_set)
        form_layout.addRow(QLabel("Acceptable deviation for msalign:"), self.dev_set)
        form_layout.addRow(QLabel("Bandwidth:"), self.bw_set)
        form_layout.addRow(QLabel("Number of dots:"), self.n_dots_set)

        self.config_button = QPushButton("Open config file")
        self.config_button.clicked.connect(lambda: self.open_config())
        # self.load_config_button = QPushButton("Save configs")
        # self.load_config_button.clicked.connect(lambda: self.save_config())
        self.calc_button = QPushButton("Calculate")
        config_layout.addWidget(self.config_button)
        # config_layout.addWidget(self.load_config_button)
        config_layout.addWidget(self.calc_button)
        self.calc_button.clicked.connect(lambda: self.signal())
        # self.calc_button.setEnabled(False)
        self.pbar_widget = QWidget()
        self.pbar_layout = QFormLayout(self.pbar_widget)
        self.pbar = QProgressBar()
        self.pbar_label = QLabel("Spectra processing:")
        self.pbar_layout.addRow(self.pbar_label,self.pbar)
        self.splitter.addWidget(form_panel)
        self.splitter.addWidget(config_panel)
        self.right_layout.addWidget(self.splitter)
        self.right_layout.addWidget(self.parent.console_log)

        self.raw_filename.textChanged.connect(lambda text: self.raw_tree.update_tree(text))
        self.aln_filename.textChanged.connect(lambda text: self.aln_tree.update_tree(text))

        self.raw_tree.path_signal.connect(lambda path: self.dataset_raw.setText(path))
        self.aln_tree.path_signal.connect(lambda path: self.dataset_aln.setText(path))

        self.right_layout.addWidget(self.pbar_widget)
        self.pbar_widget.hide()

        try:
            with open("last_config.yaml", 'r', encoding='utf8') as f:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
                self.raw_filename.setText(yaml_config['FILE_NAMES'][0])
                self.aln_filename.setText(yaml_config['FILE_NAMES'][1])
                self.ref_set.setText(str(yaml_config['REF']))
                self.dev_set.setText(str(yaml_config['DEV']))
                self.dataset_raw.setText(str(yaml_config['DATASET_R']))
                self.dataset_aln.setText(str(yaml_config['DATASET_A']))
                self.bw_set.setText(str(yaml_config['BW']))
                self.n_dots_set.setText(str(yaml_config['NDOTS']))
        except Exception as error:
            print(error)
    def open_file(self, raw_filename):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "HDF (*.hdf *.hdf5 *.h5);;All Files (*)")
        if not filename: return
        raw_filename.setText(filename)

    def open_config(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "yaml (*.yaml);;All Files (*)")
        if not filename: return
        with open(filename, 'r', encoding='utf8') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
                self.raw_filename.setText(yaml_config['FILE_NAMES'][0])
                self.aln_filename.setText(yaml_config['FILE_NAMES'][1])
                self.ref_set.setText(str(yaml_config['REF']))
                self.dev_set.setText(str(yaml_config['DEV']))
                self.dataset_raw.setText(str(yaml_config['DATASET_R']))
                self.dataset_aln.setText(str(yaml_config['DATASET_A']))
                self.bw_set.setText(str(yaml_config['BW']))
                self.n_dots_set.setText(str(yaml_config['NDOTS']))
            except Exception as e:
                print(e)

    def Pbar_set_ranges(self, ranges):
        self.pbar.setRange(*ranges)
        self.pbar.setValue(ranges[0])
    def Pbar_forwarder(self, n):
        self.pbar.setValue(n)
    def signal(self):
        self.pbar_widget.show()
        self.pbar.show()
        self.pbar_label.setText("Spectra processing:")
        try:
            data = (self.raw_filename.text(),
                    self.aln_filename.text(),
                    self.ref_set.text(),
                    self.dev_set.text(),
                    self.dataset_raw.text(),
                    self.dataset_aln.text(),
                    self.bw_set.text(),
                    self.n_dots_set.text())
            if '' in data:
                raise Exception('Empty string')
            else:

                with open('last_config.yaml', 'w') as outfile:
                    yaml.dump({
                        'FILE_NAMES':(data[0],data[1]),
                        'REF': float(data[2]),
                        'DEV': float(data[3]),
                        'DATASET_R':data[4],
                        'DATASET_A':data[5],
                        'BW':float(data[6]),
                        'NDOTS':int(data[7])
                    }, outfile, default_flow_style=False)
            Const.RAW, Const.ALN, Const.REF, Const.DEV, Const.DATASET_RAW,Const.DATASET_ALN, Const.BW, Const.N_DOTS = data[0], data[1], float(
                data[2]), float(
                data[3]), data[4],data[5], float(data[6]), int(data[7])
            # self.calc_button.setEnabled(True)
        except Exception as e:
            print(e)


        self.processing.moveToThread(self.thread)
        self.thread.started.connect(self.processing.find_dots_process)
        self.processing.finished.connect(self.thread.quit)
        self.processing.finished.connect(self.processing.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.processing.create_pbar.connect(self.Pbar_set_ranges)
        self.processing.progress.connect(self.Pbar_forwarder)
        self.thread.start()
        self.config_button.setEnabled(False)
        self.calc_button.setEnabled(False)
        self.processing.result.connect(main_window.redirect_outputs)
        self.processing.error.connect(main_window.console_log.updateText)
        self.thread.finished.connect(
            lambda: self.config_button.setEnabled(True)
        )
        self.thread.finished.connect(
            lambda: self.calc_button.setEnabled(True)
        )
        self.processing.finished.connect(
            lambda: self.pbar_label.setText("Process done")
        )
        self.thread.finished.connect(
            lambda: self.pbar.hide()
        )
        self.processing.error.connect(
            lambda: self.pbar_label.setText("Error occurred during processing")
        )
        self.processing.error.connect(
            lambda: self.config_button.setEnabled(True)
        )
        self.processing.error.connect(
            lambda: self.calc_button.setEnabled(True)
        )
        self.processing.error.connect(
            lambda: self.pbar.hide()
        )

class TablePage(QWidget):
    def __init__(self,parent,title='TablePage',columns=1):
        super().__init__()
        self.parent = parent
        self.title = title
        self.layout = QVBoxLayout()

        self.splitter = QSplitter(Qt.Vertical)
        self.table = QTableWidget()
        self.aver_table = QTableWidget()

        self.setLayout(self.layout)


        self.splitter.addWidget(self.table)
        self.splitter.addWidget(self.aver_table)

        self.splitter.setSizes([int(self.height()*0.95),int(self.height()*0.05)])
        self.layout.addWidget(self.splitter)
        self.table.setColumnCount(columns)
        self.aver_table.setColumnCount(columns)
        self.aver_table.setRowCount(1)

        self.aver_table.verticalHeader().setDefaultSectionSize(self.aver_table.height())

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.aver_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self.average_selected)

    def set_title(self,title):
        self.table.setHorizontalHeaderLabels(title)
        self.aver_table.setHorizontalHeaderLabels(title)
    def add_row(self,data):
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)
        for col,value in enumerate(data):
            self.table.setItem(row_index, col, QTableWidgetItem(str(value)))

    def add_data(self,data):
        for line in data:
            self.add_row(line)

    def average_selected(self):
        selected = self.table.selectedItems()
        temp = np.array([el.text() for el in selected]).reshape(-1, 6).T
        data = temp.astype(float).mean(axis=1)

        for col,value in enumerate(data):
            self.aver_table.setItem(0, col, QTableWidgetItem(str(value)))

class GraphPage(QWidget):
    def __init__(self, parent, canvas_count=1, title='PlotPage', title_plots=None, x_labels=None, y_labels=None,
                 color=(255, 255, 255), bg_color=(0, 0, 0),n_colors = 8,autoSize = True):
        super().__init__()
        self.autoSize = autoSize
        self.bg_color = bg_color
        self.color = color

        self.fixed_colors = [
            pg.mkColor('blue'),  # Синий
            pg.mkColor('red')
        ]

        if x_labels is None: x_labels = ['x'] * canvas_count
        if y_labels is None: y_labels = ['y'] * canvas_count
        if title_plots is None: title_plots = [f'plot{i}' for i in range(canvas_count)]

        self.canvas_adj = {title_plots[i]: i for i in range(canvas_count)}

        self.parent = parent
        self.title = title
        self.plot_spaces = [pg.PlotWidget() for _ in range(canvas_count)]
        [self.pyqt_settings(pw_id) for pw_id in self.plot_spaces]

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        for i in range(canvas_count):
            self.plot_spaces[i].showGrid(x=True, y=True)
            self.plot_spaces[i].setTitle(title_plots[i])
            self.layout.addWidget(self.plot_spaces[i])
            self.plot_spaces[i].setLabel('bottom', x_labels[i])
            self.plot_spaces[i].setLabel('left', y_labels[i])

        self.palette_colors =  [pg.intColor(i,hues=n_colors) for i in range(n_colors)]


    def pyqt_settings(self,plot_widget):
        plot_widget.addLegend(brush='black')
        plot_widget.setMouseEnabled(y=False, x=True)

        vb = plot_widget.getViewBox()

        # Включить автомасштабирование по Y
        if self.autoSize:
            vb.enableAutoRange(axis='y')
        # Установить видимое автомасштабирование
            vb.setAutoVisible(y=True)

    def add_plot(self, data, plot_name, color='w', canvas_name=None):
        if canvas_name is None:
            plot_id = 0
        else:
            plot_id = self.canvas_adj[canvas_name]
        pen = pg.mkPen(color=color)
        self.plot_spaces[plot_id].plot(data[0], data[1], name=plot_name, pen=pen)
        self.plot_spaces[plot_id].getAxis('bottom').setVisible(True)

    def add_line(self, data, y_max, color='w',canvas_name=None):
        try:
            if canvas_name is None:
                plot_id = 0
            else:
                plot_id = self.canvas_adj[canvas_name]

            y_min = 0
            x = np.column_stack([data,
                                 data,
                                 np.full_like(data, np.nan)])
            y = np.column_stack([np.full_like(data, y_min),
                                 np.full_like(data, y_max),
                                 np.full_like(data, np.nan)])

            if color == 'mult':

                length = x.shape[0]
                indices = [np.arange(i,length,len(self.palette_colors)) for i in range(len(self.palette_colors))]

                x_s = [np.take(x,idx,axis=0) for idx in indices]
                y_s = [np.take(y,idy,axis=0) for idy in indices]
                pens = [pg.mkPen(color=self.palette_colors[i%len(self.palette_colors)]) for i in range(y.shape[0])]

                for figure_index in range(len(self.palette_colors)):
                    self.plot_spaces[plot_id].plot(x_s[figure_index].ravel(),y_s[figure_index].ravel(),pen = pens[figure_index])
                #self.plot_spaces[plot_id].plot(x,y,pen=pens)
            else:
                x = x.ravel()
                y = y.ravel()
                pen = pg.mkPen(color=color, style=QtCore.Qt.DashLine)
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
        super().__init__(parent, canvas_count=4, title=title, title_plots=('std_dev', 'modality (dip test)', 'skewness', 'kurtosis'),
                         x_labels=x_labels, y_labels=y_labels, color=color, bg_color=bg_color,autoSize=False)


        #self.table_un = pg.TableWidget()  # сколько всего точек, медианное отклонение, число точек не мономодальных
        #self.table_list = {'unpaired': self.table_un}

        self.p = p_val
        #self.table_data = np.zeros((3, 2))

        self.layout.setStretch(0, 1)  # Виджет 1
        self.layout.setStretch(1, 1)  # Виджет 2
        self.layout.setStretch(2, 1)  # Виджет 3
        self.layout.setStretch(3, 1)

        self.table_layout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.table_layout)
        #self.table_layout.addWidget(self.table_un)

    def add_row(self,table_name,data):
        row_index = self.table_data[table_name].rowCount()
        self.table_data[table_name].insertRow(row_index)
        for col,value in enumerate(data):
            self.table_data[table_name].setItem(row_index, col, QTableWidgetItem(str(value)))
    def add_data(self,table_name,data):
        for line in data:
            self.add_row(table_name,line)

    def add_plot_mul(self, ds):



        for n in range(len(ds)):
            data = ds[n][0]
            data_name = ds[n][1]
            ds_color = self.fixed_colors[n]

            #self.table_data[0, n] = len(data[0])

            self.add_plot(data[0], f'st dev {data_name}', ds_color, 'std_dev')
            self.add_plot(data[1], f'dip {data_name}', ds_color, 'modality (dip test)')
            self.add_plot(data[3], f'skew {data_name}', ds_color, 'skewness')
            self.add_plot(data[4], f'kurt {data_name}', ds_color, 'kurtosis')
            #self.table_data[1, n] = np.where(data[2] < self.p)[0].size
            #self.table_data[2, n] = np.median(data[0])
        #self.table_un.setData(self.table_data)
        #self.table_un.setHorizontalHeaderLabels([str(i) for i in range(len(ds))])
        #self.table_un.setVerticalHeaderLabels(['total', 'is multimodal', 'median std dev'])

    def add_plot(self, data, plot_name, color, canvas_name=None):
        if canvas_name is None:
            plot_id = 0
        else:
            plot_id = self.canvas_adj[canvas_name]
        pen = pg.mkPen(color=color)
        no_nan = lambda arr: arr[~np.isnan(arr)]
        y, x = np.histogram(no_nan(data), bins=1000)

        self.plot_spaces[plot_id].plot(x, y, stepMode=True, name=plot_name, pen=pen)
        self.plot_spaces[plot_id].getAxis('bottom').setVisible(True)


'''functions declaration'''


def peak_picking(X, Y, oversegmentation_filter=None, peak_location=1):
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


def read_dataset(self, dataset_raw: np.ndarray, attrs_raw: list, dataset_aln: np.ndarray, attrs_aln: list, REF, DEV, limit=None):
    """
    initial data verifying and recording into Dataset objects
    input - dataset_raw, dataset_aln - np.ndarray arrays with full recorded data,
          - limit - maximum number of mass spectra to be processed (for debugging use only, otherwise should be zero)
    return - (void function)
    """

    row_raw = DatasetHeaders(attrs_raw)
    row_aln = DatasetHeaders(attrs_aln)

    int_type = None

    if "mz" in attrs_raw:
        mz_type = "mz"
    else:
        mz_type = "peak"
    if "Intensity" not in attrs_raw:
        for column in ["Area","SNR"]:
            if column in attrs_raw:
                int_type = column
                break
    else:
        int_type = "Intensity"

    if int_type is None:
        raise Exception('Intensity type not stated in attrs_raw, check file input')

    index_row_raw = dataset_raw[row_raw("spectra_ind")]
    index_row_aln = dataset_aln[row_aln("spectra_ind")]

    start_index, end_index = int(min(index_row_raw)), int(max(index_row_raw))
    if limit is not None:
        if start_index + limit <= end_index:
            end_index = start_index + limit

    set_num = end_index - start_index + 1

    dataset_list = np.empty((2, set_num), dtype=Dataset)

    self.create_pbar.emit((0,end_index-start_index))

    for spec_n, index in enumerate(range(start_index,end_index+1)):
        index_raw, index_aln = np.where(index_row_raw == index)[0], np.where(index_row_aln == index)[0]
        # get_index(dataset_raw, index), get_index(dataset_aln, index)
        data_raw_unsorted = dataset_raw[row_raw([mz_type,int_type]),index_raw[0]:index_raw[-1] + 1]
        data_aln_unsorted = dataset_aln[row_aln([mz_type,int_type]),index_aln[0]:index_aln[-1] + 1]

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
        self.progress.emit(spec_n)

    return dataset_list


def prepare_array(distances):
    """prepare distances dataset"""
    concatenated = np.array([np.concatenate(sub) for sub in distances])
    indexes = np.repeat(np.arange(len(distances[0])), [len(sub_arr) for sub_arr in distances[0]])
    pre_sorted = np.vstack((concatenated, indexes))
    result = pre_sorted[:, pre_sorted[0].argsort()]
    return result

def concover(arr1:np.ndarray,arr2:np.ndarray):
    dev = lambda data: np.abs(data - np.median(data))

    dev1 = dev(arr1)
    dev2 = dev(arr2)

    all_devs = np.hstack((dev1, dev2))
    ranks = stats.rankdata(all_devs)

    rank1 = ranks[:len(dev1)]
    rank2 = ranks[len(dev1):]

    n = len(dev1)+len(dev2)
    mean_rank = (n+1)/2

    ss_between = (
        len(dev1)*(np.mean(rank1)-mean_rank)**2 +
        len(dev2)*(np.mean(rank2)-mean_rank)**2)

    ss_total = np.sum((ranks-mean_rank)**2)

    t = (n-1)*ss_between/ss_total
    p_value = 1 - stats.chi2.cdf(t, 2)
    return p_value


# вычислить среднее и дисперсии, проверить нормальность, проверить гипотезы о значимости различия средних и дисперсий, возможно посчитать форму распределения
def stat_params_paired_single(peak_raw, peak_aln, p_value=0.05):
    """paired peak comparison"""

    norm_var = lambda data: np.var(data-np.mean(data),ddof=1)
    mean_r, mean_a = np.mean(peak_raw), np.mean(peak_aln)
    var_r, var_a = norm_var(peak_raw), norm_var(peak_aln)

    check_normal_func = lambda data,p: stats.kstest(data,'norm',args=(np.mean(data),np.std(data)))[1]>p
    check_normal = check_normal_func(peak_raw,p_value) & check_normal_func(peak_aln,p_value)
    if check_normal:
        neq_var = stats.levene(peak_raw, peak_aln)[1] < p_value
        neq_mean = stats.ttest_ind(peak_raw, peak_aln,nan_policy='omit')[1] < p_value
    else:
        neq_mean = stats.mannwhitneyu(peak_raw, peak_aln)[1] < p_value
        neq_var = stats.ansari(peak_raw, peak_aln)[1] < p_value

    return mean_r - mean_a, var_r, var_a,float(check_normal), float(neq_mean), float(neq_var)


def stat_params_unpaired(ds):
    res = np.array([[np.var(dot), *diptest(dot), stats.skew(dot), stats.kurtosis(dot)] for dot in ds])
    return res


def moving_average(a, n=2):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def out_criteria(mz, intensity, int_threshold = 0.01, max_diff = 0.3, width_eps = 0.1):# TODO: add documentation (see TG for descr.)
    min_int = np.max(intensity) * int_threshold
    first_or = intensity < min_int
    int_criteria = abs(intensity[:-1] / intensity[1:] - 1) < max_diff

    width_criteria = np.diff(mz) / moving_average(np.diff(mz.linked_array).flatten()) <= width_eps
    second_or = np.full(mz.shape, False)
    second_or[1:] = np.logical_and(int_criteria, width_criteria)

    return np.where(np.logical_or(first_or, second_or))[0]


def criteria_apply(arr, intensity):
    arr_out = copy.deepcopy(arr)
    indexes = out_criteria(arr, intensity)
    for index in indexes:
        arr_out.linked_array[index-1] = sorted([arr.linked_array[index-1,0],arr.linked_array[index,1]])
    return arr_out.sync_delete(indexes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app_icon = QIcon('main_ico.png')
    app.setWindowIcon(app_icon)
    main_window = MainWindow()
    main_window.showMaximized()

    sys.exit(app.exec_())