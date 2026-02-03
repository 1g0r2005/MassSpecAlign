import multiprocessing
import os

import numpy as np
import pyimzml
from pyimzml.ImzMLParser import ImzMLParser

os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import copy
import math
import sys
from multiprocessing import Process, Queue, Pool
from pathlib import Path
from queue import Empty
import traceback

import h5py
import pyqtgraph as pg
import scipy.stats as stats
import yaml
from PyQt5 import QtCore
from KDEpy import FFTKDE
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, QObject, pyqtSignal, Qt, QThread
from PyQt5.QtGui import QIcon, QFont, QResizeEvent
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLineEdit, QLabel, QPushButton, \
    QFormLayout, QFileDialog, QTableWidgetItem, QTableWidget, QVBoxLayout, QHeaderView, QAbstractItemView, QSplitter, \
    QHBoxLayout, QTreeWidget, QTreeWidgetItem, QProgressBar
from diptest import diptest
import alignment
from data_classes import *
from scipy.special import rel_entr
import pandas as pd
"""classes declaration"""


class Const:
    """
    Container for application-wide constants.

    Attributes
    ----------
    DATASET_RAW : str or None
        HDF5 path to the raw spectra dataset.
    DATASET_ALN : str or None
        HDF5 path to the aligned spectra dataset.
    DEV : float or None
        Acceptable deviation (±) around `REF` when searching for the reference peak.
    N_DOTS : int or None
        Number of points for KDE evaluation.
    BW : float or None
        Bandwidth parameter for KDE.
    """


    #class attr
    DATASET_RAW: str | None= None
    DATASET_ALN: str | None = None

    DEV: float | None = None
    N_DOTS: int | None = None
    BW: float | None = None

    #file extensions
    HDF = ['.hdf5', '.hdf', '.he5', '.h5']
    IMZML = ['.imzML']


class WorkerSignals(QObject):
    """
    Signals and processing pipeline for background computations.

    Signals
    -------
    output : pyqtSignal(str)
        Emitted for redirected standard output messages.
    error : pyqtSignal(str)
        Emitted for redirected standard error messages or exceptions.
    result : pyqtSignal(object)
        Emitted with computation results to be consumed by the main thread.
    finished : pyqtSignal()
        Emitted when the processing pipeline finishes.
    progress : pyqtSignal(int)
        Emitted to update a progress bar.
    create_pbar : pyqtSignal(tuple)
        Emitted to initialize a progress bar. Expected tuple is (min, max).
    """
    output = pyqtSignal(str)
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    create_pbar = pyqtSignal(tuple)

    def find_dots_process(self):
        """
        Run the main data processing pipeline.

        The pipeline reads raw and aligned spectra from HDF5, computes KDEs,
        performs peak picking, aligns peak lists, and computes
        descriptive and inferential statistics. Results are emitted via the
        `result` signal as a tuple of render instructions and statistics.

        Notes
        -----
        Emits
            - ``create_pbar``: tuple of (min, max) for a progress bar.
            - ``progress``: updates during dataset iteration.
            - ``result``: composite payload for UI updates.
            - ``finished``: upon completion or on handled exception.
            - ``error``: formatted traceback on exception.
        """

        try:
            processes = max(1, min((os.cpu_count() or 2) - 1, 3))

            file_raw, file_aln = File(Const.RAW), File(Const.ALN)
            ext_raw, ext_aln = file_raw.extension, file_aln.extension

            if ext_raw in Const.IMZML and ext_aln in Const.IMZML:
                p_raw = file_raw.read()
                p_aln = file_aln.read()
                distance_list = read_imz_dataset(self,p_raw,p_aln,processes=processes)
            elif ext_raw in Const.HDF and ext_aln in Const.HDF:
                features_raw, attrs_raw = file_raw.read(Const.DATASET_RAW)
                features_aln, attrs_aln = file_aln.read(Const.DATASET_ALN)
                distance_list = read_dataset(self, features_raw, attrs_raw, features_aln, attrs_aln, Const.DEV,
                                             processes=processes)
            else:
                raise Exception('Different file formats are not supported')

            print('Data loaded')
            # включаем параллельную обработку по умолчанию (все ядра минус одно)





            distance_list_prepared = prepare_array(distance_list)
            raw_concat, aln_concat, id_concat = distance_list_prepared

            kde_x_raw, kde_y_raw = FFTKDE(bw=Const.BW, kernel='gaussian').fit(raw_concat).evaluate(Const.N_DOTS)
            kde_x_aln, kde_y_aln = FFTKDE(bw=Const.BW, kernel='gaussian').fit(aln_concat).evaluate(Const.N_DOTS)


            center_r, left_r, right_r = peak_picking(kde_x_raw, kde_y_raw)
            center_a, left_a, right_a = peak_picking(kde_x_aln, kde_y_aln)
            # восстановим высоту пиков
            max_center_r, max_center_a = np.interp(center_r, kde_x_raw, kde_y_raw), np.interp(center_a, kde_x_aln,
                                                                                            kde_y_aln)
            print('Peak evaluation finished')
            borders_r = np.stack((left_r, right_r), axis=1)
            borders_a = np.stack((left_a, right_a), axis=1)
            c_ds_raw = LinkedList(center_r, borders_r)
            c_ds_aln = LinkedList(center_a, borders_a)

            print('LinkedList created')

            c_ds_raw_intensity, c_ds_aln_intensity = np.interp(c_ds_raw, kde_x_raw, kde_y_raw), np.interp(c_ds_aln, kde_x_aln,
                                                                                                    kde_y_aln)

            print('Just before sortind')
            ##### заменили sort_dots на sorting_dots
            peak_lists_raw = sorting_dots(raw_concat, c_ds_raw.linked_array[:, 0], c_ds_raw.linked_array[:, 1])
            peak_lists_aln = sorting_dots(aln_concat, c_ds_aln.linked_array[:, 0], c_ds_aln.linked_array[:, 1])

            print(raw_concat)
            print(peak_lists_raw)
            aln_peak_lists_raw, aln_peak_lists_aln, aln_kde_raw, aln_kde_aln = alignment.munkres(peak_lists_raw,
                                                                                                 peak_lists_aln,
                                                                                                 c_ds_raw,
                                                                                                 c_ds_aln,
                                                                                                 c_ds_raw_intensity,
                                                                                                 c_ds_aln_intensity,
                                                                                                 segmentation_threshold=400)

            s_p = np.array(pd.DataFrame(np.array(
                [stat_params_paired_single(x_el, y_el) for x_el, y_el in zip(aln_peak_lists_raw, aln_peak_lists_aln)],
                dtype='float')).dropna())

            result_text,result_type = construct_output(p_value=s_p[:,-1],
                                                       var_raw=s_p[:,1],
                                                       var_aln=s_p[:,2])


            ret = (
                ('show', (((kde_x_raw, kde_y_raw), 'raw', 'red', 'p', 'kde'),
                        ((kde_x_aln, kde_y_aln), 'aln', 'blue', 'p', 'kde'),
                        (aln_kde_aln,np.max(kde_y_aln),'raw_peaks','mult','vln','kde'),
                        (aln_kde_raw, np.max(kde_y_aln), 'raw_peaks', 'mult', 'vln', 'kde'))),
                ('stats',
                 ((stat_params_unpaired(peak_lists_raw).T, 'raw'), (stat_params_unpaired(peak_lists_aln).T, 'aln'))),
                ('stats_p', ((stat_params_unpaired(aln_peak_lists_raw).T, 'raw'),
                             (stat_params_unpaired(aln_peak_lists_aln).T, 'aln'))),
                ('stats_table',s_p),
                ('final',(result_text,result_type))
            )
            print(s_p.shape)
            self.result.emit(ret)
            self.finished.emit()

        except Exception as error:
#            self.error.emit(str(error))
            self.error.emit(traceback.format_exc()) #temporary
            self.finished.emit()


class DatasetHeaders:
    """
    Helper to access HDF5 dataset attributes by name or index.

    Parameters
    ----------
    attrs : Sequence[str]
        List of attribute names as provided by the HDF5 dataset.

    Attributes
    ----------
    index : dict
        Mapping from attribute name to its integer index.
    name : list
        List of attribute names in positional order.
    """
    def __init__(self,attrs):
        """
        Build the name-to-index and index-to-name mappings.

        Parameters
        ----------
        attrs : Sequence[str]
            Attribute names from the dataset.
        """
        self.index = {}
        self.name = [0]*len(attrs)
        for index, name in enumerate(attrs):
            self.name.append(name)
            self.index[name]=index

    def __call__(self,index_value):
        """
        Convert between names and indices for single values or lists.

        Parameters
        ----------
        index_value : int, str, list[int], or list[str]
            Input specification to convert.

        Returns
        -------
        int, str, or list
            The converted value(s): index for name input, name for index input.
        """

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
    """
    Redirect-like object writing messages into a multiprocessing queue.

    Parameters
    ----------
    q : multiprocessing.Queue
        Target queue where messages will be put.
    """
    def __init__(self, q):
        """
        Initialize the redirector.

        Parameters
        ----------
        q : multiprocessing.Queue
            Target queue for messages.
        """
        self.q = q

    def write(self, msg: str):
        """
        Write a message to the queue if it's not empty or whitespace only.

        Parameters
        ----------
        msg : str
            Message to forward.
        """
        if msg.strip():
            self.q.put(msg)

    def flush(self):
        """
        Placeholder
        """
        pass


class ProcessManager:
    """
    Manage background processes and multiplex their stdout, stderr and results.

    Parameters
    ----------
    signals : WorkerSignals
        Signals object to emit collected outputs to the main thread.

    Attributes
    ----------
    output_q, error_q, return_q : multiprocessing.Queue
        Internal queues used to collect outputs from child processes.
    process_set : set[str]
        Names of currently running processes.
    """
    def __init__(self, signals):
        """
        Create the manager and internal queues.

        Parameters
        ----------
        signals : WorkerSignals
            Signals sink for emitting messages.
        """
        self.signals = signals
        self.output_q = Queue()
        self.error_q = Queue()
        self.return_q = Queue()
        self.process_set = set()

    def run_process(self, target, target_name, args=None, kwargs=None):
        """
        Start a target function in a separate process.

        Parameters
        ----------
        target : callable
            Function to execute in a child process.
        target_name : str
            Name used to track the process.
        args : list, optional
            Positional arguments for `target`.
        kwargs : dict, optional
            Keyword arguments for `target`.

        Returns
        -------
        multiprocessing.Process
            The started process instance.
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        p = Process(target=self._std_wrapper, args=(target, self.output_q, self.error_q, self.return_q, args, kwargs))
        self.process_set.add(target_name)
        p.start()
        return p

    def end_process(self, process, target_name):
        """
        Join the process if tracked by name and report join errors to `error_q`.

        Parameters
        ----------
        process : multiprocessing.Process
            Process to join.
        target_name : str
            Name that identifies the process in `process_set`.
        """
        if target_name in self.process_set:
            try:
                process.join()
            except Exception as e:
                self.error_q.put(e)
        else:
            self.error_q.put(str(Exception('no process with name {} running'.format(target_name))))

    @staticmethod
    def _std_wrapper(target, out_q, error_q, ret_q, args=(), kwargs=None):
        """
        Wrap a target callable to redirect stdio and return its result via queues.

        Parameters
        ----------
        target : callable
            Function to execute.
        out_q, error_q, ret_q : multiprocessing.Queue
            Queues for stdout, stderr, and return payloads.
        args : tuple
            Positional arguments for `target`.
        kwargs : dict, optional
            Keyword arguments for `target`.
        """
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
        """
        Drain the return queue and emit results via `signals.result`.
        """
        while not self.return_q.empty():
            try:
                func_name, content = self.return_q.get_nowait()
                self.signals.result.emit(content)
            except Empty:
                break
            except Exception as e:
                self.error_q.put(e)

    def __check_error(self):
        """
        Drain the error queue and emit messages via `signals.error`.
        """
        while not self.error_q.empty():
            try:
                msg = self.error_q.get_nowait()
                self.signals.error.emit(str(msg))
            except Empty:
                break

    def __check_out(self):
        """
        Drain the stdout queue and emit messages via `signals.output`.
        """
        while not self.output_q.empty():
            try:
                msg = self.output_q.get_nowait()
                # print(msg)
                self.signals.output.emit(msg)
            except Empty:
                break

    def check_queues(self):
        """
        Poll all internal queues and forward their content via signals.
        """
        self.__check_return()
        self.__check_error()
        self.__check_out()


class Dataset(LinkedList):
    """
    LinkedList with an optional reference value attached.

    Parameters
    ----------
    input_array : array_like
        Primary data.
    linked_array : array_like or None, optional
        Secondary linked data.
    reference : float or None, optional
        Reference m/z value associated with the dataset.

    Attributes
    ----------
    reference : float or None
        The attached reference value.
    """
    def __new__(cls, input_array, linked_array=None, reference=None):
        """
        Create a Dataset and attach an optional reference value.

        Parameters
        ----------
        input_array : array_like
            Primary data.
        linked_array : array_like or None, optional
            Secondary linked data.
        reference : float or None, optional
            Reference value to attach.
        """
        obj = super().__new__(cls, input_array, linked_array)
        obj.reference = reference
        return obj

    def __array_finalize__(self, obj):
        """
        Propagate `reference` and linked array when creating views.

        Parameters
        ----------
        obj : ndarray or None
            Source object for the view.
        """
        super().__array_finalize__(obj)
        if obj is None: return
        self.reference = getattr(obj, 'reference', None)

    def __setitem__(self, index, value):
        """
        Assign items in the primary array and mirror to the linked array.
        """
        super().__setitem__(index, value)


class File:
    """
    Thin wrapper around an data (hdf5 of imzml) file to read datasets and their headers.

    Parameters
    ----------
    file_name : str or Path
        Path to the file.

    Attributes
    ----------
    real_path : Path
        Resolved path to the file.
    """
    def __init__(self, file_name):
        """
        Initialize the file wrapper.

        Parameters
        ----------
        file_name : str or Path
            Path to the HDF5 file.
        """
        self.real_path = Path(file_name)
        self.extension = self.real_path.suffix

    def exist(self):
        """
        Check whether the file exists.

        Returns
        -------
        bool
            True if the file exists.
        """
        return self.real_path.exists()

    def read(self, dataset = None):
        """
        Read a dataset and its column headers from the HDF5 file.

        Parameters
        ----------
        dataset : str
            HDF5 path to the dataset to read.

        Returns
        -------
        tuple
            A tuple ``(data, attr)`` where ``data`` is a NumPy array and
            ``attr`` is a list/array of column headers. Returns None on error.

        Raises
        ------
        Exception
            If the number of headers does not match the number of columns.
        FileNotFoundError
            If the file does not exist.
        """
        try:
            if not self.exist():
                raise FileNotFoundError

            if self.extension in Const.IMZML:

                return ImzMLParser(self.real_path, include_spectra_metadata="full")

            elif self.extension in Const.HDF:
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
    """
    Read-only widget to display log and info messages.
    """
    def __init__(self, parent=None):
        """
        Initialize the text widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        """
        QtWidgets.QTabWidget.__init__(self, parent)
        super().__init__(parent)
        self.setReadOnly(True)

    def __scrollDown(self):
        """
        Scroll to the bottom of the text area.
        """
        scroll = self.verticalScrollBar()
        end_text = scroll.maximum()
        scroll.setValue(end_text)

    def updateText(self, msg: str):
        """
        Append a message and scroll to the end.

        Parameters
        ----------
        msg : str
            Message to append.
        """
        try:
            self.append(str(msg))
            self.__scrollDown()
        except Exception as e:
            print(e)


class TreeWidget(QWidget):
    """
    Widget for browsing HDF5 groups and datasets as a tree.

    Signals
    -------
    path_signal : pyqtSignal(str)
        Emitted with the path of the double-clicked node.
    """
    path_signal = pyqtSignal(str)

    def __init__(self):
        """
        Initialize the tree widget and layout.
        """
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.tree = QTreeWidget()
        self.__initUI()

        self.overlay_info = QLabel('Unable to load structure for this data format',self)
        self.overlay_info.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.overlay_info.setStyleSheet("""
                    color: #FFFFFF;
                    background: rgba(0, 0, 0, 180);
                    border-radius: 6px;
                    padding: 8px 16px;
                """)
        self.overlay_info.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.overlay_info.setAlignment(Qt.AlignCenter)
        self.overlay_info.hide()


    def setEnabled(self, enabled: bool):
        """
        Redefining setEnabled to add QLabel for info text
        """
        super().setEnabled(enabled)
        if not enabled:
            self.overlay_info.show()
            self.overlay_info.raise_()
        else:
            self.overlay_info.hide()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.overlay_info.resize(self.size())
    def __initUI(self):
        """
        Configure the internal QTreeWidget and layout.
        """
        # Создаем layout и tree widget
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.itemDoubleClicked.connect(self.get_path)
        self.layout.addWidget(self.tree)

        self.tree.expandAll()
        self.tree.setHeaderLabels(['Name','Type','Shape','DType'])

        self.setLayout(self.layout)

    def populate_tree(self,path):
        """
        Populate the tree with the hierarchy of an HDF5 file.

        Parameters
        ----------
        path : str
            Path to an HDF5 file on disk.
        """
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

        if Path(path).suffix in Const.HDF:
            self.setEnabled(True)
        else:
            #для imzML не получаем
            print(f"File doesn't have hierarchical structure")
            self.setEnabled(False)
            return


        with h5py.File(path,'r') as f:
            root = QTreeWidgetItem(self.tree)
            root.setText(0,'root')
            get_node('/',f,parent=root)

    def get_path(self):
        """
        Return the HDF5-like path of the selected node and emit it.

        Returns
        -------
        str
            The constructed path of the selected item.
        """
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
        """
        Clear and rebuild the tree from an HDF5 file.

        Parameters
        ----------
        path : str
            Path to an HDF5 file.
        """
        self.tree.clear()
        self.populate_tree(path)


class MainWindow(QMainWindow):
    """
    Main application window that hosts pages and coordinates background work.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the main window, tabs, and signal wiring.
        """
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


        self.table.set_title(['Distance','Var(raw)','Var(aln)','JSD','neq_mean?','neq_var?'])
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
        """
        Dispatch a composite results payload to the respective UI pages.

        Parameters
        ----------
        ret : Sequence[tuple]
            Iterable of (key, payload) pairs where key selects a handler.
        """
        self.aval_func = {'show': self.graph.add_plot_mul, 'stats': self.stats.add_plot_mul, 'stats_p': self.stats_p.add_plot_mul,'stats_table':self.table.add_data,'final':self.main.result}
        for output in ret:
            self.aval_func[output[0]](output[1])


    def adjust_tab_sizes(self):
        """
        Resize tab widgets to fit the current tab area.
        """
        tab_size = self.tabs.size()
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            tab.resize(tab_size)

    def resizeEvent(self, event):
        """
        Handle window resize events and adjust child sizes.

        Parameters
        ----------
        event : QResizeEvent
            The resize event.
        """
        super().resizeEvent(event)
        self.adjust_tab_sizes()


    def start_calc(self, target, process_name=None, args=None, kwargs=None):
        """
        Start a background calculation using the process manager.

        Parameters
        ----------
        target : callable
            Function to run in background.
        process_name : str, optional
            Name for the process; defaults to ``target.__name__``.
        args : list, optional
            Positional arguments for the target.
        kwargs : dict, optional
            Keyword arguments for the target.
        """
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
    """
    Main configuration page for selecting files, datasets and parameters.

    Parameters
    ----------
    parent : QWidget
        Parent main window.
    title : str
        Page title.
    """
    def __init__(self, parent, title):
        """
        Build the configuration UI and wire controls.
        """
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



        self.raw_filename.setEnabled(False)
        self.aln_filename.setEnabled(False)

        self.dev_set = QLineEdit()
        self.bw_set = QLineEdit()
        self.n_dots_set = QLineEdit()

        self.global_result = QLabel()

        form_layout.addRow(QLabel("Raw data:"), self.raw_layout)
        form_layout.addRow(QLabel("Alignment data:"), self.aln_layout)
        form_layout.addRow(QLabel("Dataset (raw):"), self.dataset_raw)
        form_layout.addRow(QLabel("Dataset (aln):"), self.dataset_aln)

        form_layout.addRow(QLabel("Acceptable deviation for msalign:"), self.dev_set)
        form_layout.addRow(QLabel("Bandwidth:"), self.bw_set)
        form_layout.addRow(QLabel("Number of dots:"), self.n_dots_set)

        self.config_button = QPushButton("Open config file")
        self.config_button.clicked.connect(lambda: self.open_config())

        self.calc_button = QPushButton("Calculate")
        config_layout.addWidget(self.config_button)

        config_layout.addWidget(self.calc_button)
        self.calc_button.clicked.connect(lambda: self.signal())

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
        #temp: result out
        self.right_layout.addWidget(self.global_result)
        self.global_result.hide()
        self.pbar_widget.hide()

        try:
            with open("last_config.yaml", 'r', encoding='utf8') as f:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
                self.raw_filename.setText(yaml_config['FILE_NAMES'][0])
                self.aln_filename.setText(yaml_config['FILE_NAMES'][1])

                self.dev_set.setText(str(yaml_config['DEV']))
                self.dataset_raw.setText(str(yaml_config['DATASET_R']))
                self.dataset_aln.setText(str(yaml_config['DATASET_A']))
                self.bw_set.setText(str(yaml_config['BW']))
                self.n_dots_set.setText(str(yaml_config['NDOTS']))
        except Exception as error:
            print(error)
    def open_file(self, raw_filename):
        """
        Open a file dialog and set the selected path to the provided line edit.

        Parameters
        ----------
        raw_filename : QLineEdit
            Line edit to receive the selected file path.
        """
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "HDF (*.hdf *.hdf5 *.h5);;All Files (*)")
        if not filename: return
        raw_filename.setText(filename)

    def open_config(self):
        """
        Load configuration from a YAML file and populate the UI fields.
        """
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "yaml (*.yaml);;All Files (*)")
        if not filename: return
        with open(filename, 'r', encoding='utf8') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
                self.raw_filename.setText(yaml_config['FILE_NAMES'][0])
                self.aln_filename.setText(yaml_config['FILE_NAMES'][1])

                self.dev_set.setText(str(yaml_config['DEV']))
                self.dataset_raw.setText(str(yaml_config['DATASET_R']))
                self.dataset_aln.setText(str(yaml_config['DATASET_A']))
                self.bw_set.setText(str(yaml_config['BW']))
                self.n_dots_set.setText(str(yaml_config['NDOTS']))
            except Exception as e:
                print(e)

    def Pbar_set_ranges(self, ranges):
        """
        Initialize the progress bar range and reset its value.

        Parameters
        ----------
        ranges : tuple[int, int]
            Minimum and maximum for the progress bar.
        """
        self.pbar.setRange(*ranges)
        self.pbar.setValue(ranges[0])
    def Pbar_forwarder(self, n):
        """
        Update progress bar value.

        Parameters
        ----------
        n : int
            New progress value.
        """
        self.pbar.setValue(n)
    def signal(self):
        """
        Validate inputs, persist the last configuration, and start processing.
        """
        self.pbar_widget.show()
        self.pbar.show()
        self.pbar_label.setText("Spectra processing:")
        try:
            data = (self.raw_filename.text(),
                    self.aln_filename.text(),
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
                        'DEV': float(data[2]),
                        'DATASET_R':data[3],
                        'DATASET_A':data[4],
                        'BW':float(data[5]),
                        'NDOTS':int(data[6])
                    }, outfile, default_flow_style=False)
            Const.RAW, Const.ALN, Const.DEV, Const.DATASET_RAW,Const.DATASET_ALN, Const.BW, Const.N_DOTS = (data[0],
                                                                                                            data[1],
                                                                                                            float(data[2]),
                                                                                                            data[3],
                                                                                                            data[4],
                                                                                                            float(data[5]),
                                                                                                            int(data[6]))
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

    def result(self, result):
        """
        Show final result

        Parameters
        ----------
        result : tuple
            (string which will be displayed in final_result QLabel, type of result (for text color))
        """
        result_text,result_type = result

        match result_type:
            case 0:
                color_name = 'black'
            case -1:
                color_name = 'red'
            case 1:
                color_name = 'green'

        self.global_result.setStyleSheet("""
        QLabel {
        color: %s};
        font-size: 16px;
        padding: 10px;
        }
        """%color_name)
        self.global_result.setText(result_text)
        self.global_result.show()


class TablePage(QWidget):
    """
    Page containing a detailed statistics table and its row-wise average.

    Parameters
    ----------
    parent : QWidget
        Parent widget.
    title : str, optional
        Page title.
    columns : int, optional
        Number of columns in the tables.
    """
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
        """
        Set column headers for both the main and average tables.

        Parameters
        ----------
        title : list[str]
            Column titles.
        """
        self.table.setHorizontalHeaderLabels(title)
        self.aver_table.setHorizontalHeaderLabels(title)
    def add_row(self,data):
        """
        Append a single row to the main table.

        Parameters
        ----------
        data : Sequence
            Row values.
        """
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)
        for col,value in enumerate(data):
            self.table.setItem(row_index, col, QTableWidgetItem(str(value)))

    def add_data(self,data):
        """
        Append multiple rows to the main table.

        Parameters
        ----------
        data : Iterable[Sequence]
            Rows to append.
        """
        for line in data:
            self.add_row(line)

    def average_selected(self):
        """
        Compute the column-wise average for selected rows and show it below.
        """
        selected = self.table.selectedItems()
        temp = np.array([el.text() for el in selected]).reshape(-1, 6).T
        data = temp.astype(float).mean(axis=1)

        for col,value in enumerate(data):
            self.aver_table.setItem(0, col, QTableWidgetItem(str(value)))


class GraphPage(QWidget):
    """
    Page with one or more plotting canvases built on pyqtgraph.

    Parameters
    ----------
    parent : QWidget
        Parent widget.
    canvas_count : int, optional
        Number of plot canvases.
    title : str, optional
        Page title.
    title_plots : Sequence[str] or None, optional
        Titles for each canvas.
    x_labels, y_labels : Sequence[str] or None, optional
        Axis labels for each canvas.
    color : tuple, optional
        Default foreground color.
    bg_color : tuple, optional
        Background color.
    n_colors : int, optional
        Size of the categorical color palette.
    autoSize : bool, optional
        Whether to enable auto-ranging on the Y axis.
    """
    def __init__(self, parent, canvas_count=1, title='PlotPage', title_plots=None, x_labels=None, y_labels=None,
                 color=(255, 255, 255), bg_color=(240, 240, 230),n_colors = 8,autoSize = True):
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
        """
        Apply common pyqtgraph settings to a plot widget.

        Parameters
        ----------
        plot_widget : pg.PlotWidget
            Target plot widget.
        """
        plot_widget.setBackground(self.bg_color)
        plot_widget.addLegend(brush=self.bg_color)
        plot_widget.setMouseEnabled(y=False, x=True)

        vb = plot_widget.getViewBox()

        # Включить автомасштабирование по Y
        if self.autoSize:
            vb.enableAutoRange(axis='y')
        # Установить видимое автомасштабирование
            vb.setAutoVisible(y=True)

    def add_plot(self, data, plot_name, color='w', canvas_name=None):
        """
        Plot a 2D curve on the specified canvas.

        Parameters
        ----------
        data : tuple(ndarray, ndarray)
            X and Y arrays.
        plot_name : str
            Name for the legend.
        color : str or tuple, optional
            Pen color.
        canvas_name : str or None, optional
            Canvas identifier; when None, use the first canvas.
        """
        if canvas_name is None:
            plot_id = 0
        else:
            plot_id = self.canvas_adj[canvas_name]
        pen = pg.mkPen(color=color)
        self.plot_spaces[plot_id].plot(data[0], data[1], name=plot_name, pen=pen)
        self.plot_spaces[plot_id].getAxis('bottom').setVisible(True)

    def add_line(self, data, y_max, color='w',canvas_name=None):
        """
        Draw vertical reference lines at X positions up to `y_max`.

        Parameters
        ----------
        data : array_like
            X positions of lines.
        y_max : float
            Maximum Y extent for the lines.
        color : str or tuple, optional
            Pen color or 'mult' to use a color palette.
        canvas_name : str or None, optional
            Canvas identifier; when None, use the first canvas.
        """
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

    def add_dot(self,data,y_level,color = 'w',canvas_name = None,symbol = 'o'):
        """
        Scatter plot of points at a fixed Y level.

        Parameters
        ----------
        data : array_like
            X positions for the markers.
        y_level : float
            Y coordinate for all markers.
        color : str or tuple, optional
            Color or 'mult' to use a palette.
        canvas_name : str or None, optional
            Canvas identifier; when None, use the first canvas.
        symbol : str, optional
            Marker symbol.
        """

        if canvas_name is None:
            plot_id = 0
        else:
            plot_id = self.canvas_adj[canvas_name]

        x = data.ravel()
        y = np.full_like(x, y_level)

        if color == 'mult':
            length = x.size

            indices = [np.arange(i, length, len(self.palette_colors)) for i in range(len(self.palette_colors))]

            x_s = [np.take(x, idx) for idx in indices]
            y_s = [np.take(y, idy) for idy in indices]

            colors = [self.palette_colors[i % len(self.palette_colors)] for i in range(length)]
            
            print(len(self.palette_colors))
            for figure_index in range(len(self.palette_colors)):
                print('!!!!!!!!!!!',figure_index)
                print(x_s[figure_index],y_s[figure_index])
                self.plot_spaces[plot_id].plot(x=x_s[figure_index], y=y_s[figure_index], symbol=symbol,symbol_size=10,symbolBrush=self.palette_colors[figure_index])
                # self.plot_spaces[plot_id].plot(x,y,pen=pens)

        else:

            self.plot_spaces[plot_id].plot(x=x, y=y,symbol=symbol,symbol_size=10,symbolBrush=color)

        self.plot_spaces[plot_id].getAxis('bottom').setVisible(True)

    def add_plot_mul(self, ds):
        """
        Render multiple plot primitives given a compact descriptor list.

        Parameters
        ----------
        ds : Iterable[tuple]
            Each entry encodes a plot instruction; see producer for details.
        """
        # print(ds)
        for data in ds:
            if data[-2] == 'p':
                self.add_plot(data[0], data[1], data[2], data[-1])
            elif data[-2] == 'vln':
                self.add_dot(data[0], 0, data[3], data[-1])



class StatGraphPage(GraphPage):
    """
    Page for visualizing summary statistics distributions across datasets.

    Plots include standard deviation, dip test statistic/p-value, skewness,
    and kurtosis histograms for raw and aligned data.
    """
    def __init__(self, parent, title='StatPage', x_labels=None, y_labels=None,
                 color=(255, 255, 255), bg_color=(240, 240, 230), p_val=0.05):
        super().__init__(parent, canvas_count=4, title=title, title_plots=('std_dev', 'modality (dip test) p-value', 'skewness', 'kurtosis'),
                         x_labels=x_labels, y_labels=y_labels, color=color, bg_color=bg_color,autoSize=False)

        self.p = p_val
        self.layout.setStretch(0, 1)  # Виджет 1
        self.layout.setStretch(1, 1)  # Виджет 2
        self.layout.setStretch(2, 1)  # Виджет 3
        self.layout.setStretch(3, 1)

        self.table_layout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.table_layout)
        #self.table_layout.addWidget(self.table_un)

    def add_row(self,table_name,data):
        """
        Append a row into an auxiliary table by name.
        """
        row_index = self.table_data[table_name].rowCount()
        self.table_data[table_name].insertRow(row_index)
        for col,value in enumerate(data):
            self.table_data[table_name].setItem(row_index, col, QTableWidgetItem(str(value)))
    def add_data(self,table_name,data):
        """
        Append multiple rows into an auxiliary table by name.
        """
        for line in data:
            self.add_row(table_name,line)

    def add_plot_mul(self, ds):
        """
        Plot multiple histogram-based statistics for provided datasets.

        Parameters
        ----------
        ds : Sequence
            Sequence of ``((data_arrays), label)`` pairs.
        """

        for n in range(len(ds)):
            data = ds[n][0]
            data_name = ds[n][1]
            ds_color = self.fixed_colors[n]



            self.add_plot(data[0], f'st dev {data_name}', ds_color, 'std_dev')
            self.add_plot(data[2], f'dip {data_name}', ds_color, 'modality (dip test) p-value')
            self.add_line(data = 0.05,y_max=len(data[2]),canvas_name='modality (dip test) p-value',color='black')
            self.add_plot(data[3], f'skew {data_name}', ds_color, 'skewness')
            self.add_plot(data[4], f'kurt {data_name}', ds_color, 'kurtosis')


    def add_plot(self, data, plot_name, color, canvas_name=None):
        """
        Plot a histogram-like step curve of the provided data.

        Parameters
        ----------
        data : array_like
            Data to histogram.
        plot_name : str
            Name for the legend.
        color : str or tuple
            Pen color.
        canvas_name : str or None, optional
            Canvas identifier.
        """
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
def construct_output(p_value, var_raw, var_aln,alpha = 0.05):
    """
        Detect peaks in a KDE curve and return their centers and boundaries.

        Parameters
        ----------
        p_value : np.ndarray
            Array with all p-values
        var_raw : np.ndarray
            Array with dispersion for all peaks in raw data
        var_aln : np.ndarray
            Array with dispersion for all peaks in aln data
        alpha: float
            Confidence level. Default is 0.05.
        Returns
        -------
        result_type: float
            type of result: -1 is negative, +1 is positive, 0 is not statistically significant.
        result_text: str
            exact text of result message which will be displayed.
        """
    s_val, simes_significance = simes(p_value, alpha)
    delta_var_all= np.mean(var_raw - var_aln)
    delta_var_significant = np.mean((var_raw - var_aln)[np.where(p_value<=alpha)])

    if simes_significance: #if significant
        if delta_var_significant>0:
            result_type = 1
            result_text = f"""
                            <div style='font-family: Times New Roman, font-size=12px;'>
                                <b>Alignment is better than raw data</b><br>
                                Simes = <b>{s_val:.2e}</b><br>
                                &Delta;&sigma;<sup>2</sup>(total) = <b>{delta_var_all:.2e}</b><br>
                                &Delta;&sigma;<sup>2</sup>(sign.) = <b>{delta_var_significant:.2e}</b>
                            </div>
                        """
        else:
            result_type = -1
            result_text = f"""
                <div style='font-family: Times New Roman, font-size=12px;'>
                    <b>Alignment is worse than raw data</b><br>
                    Simes = <b>{s_val:.2e}</b><br>
                    &Delta;&sigma;<sup>2</sup>(total) = <b>{delta_var_all:.2e}</b><br>
                    &Delta;&sigma;<sup>2</sup>(sign.) = <b>{delta_var_significant:.2e}</b>
                </div>
            """
    else:
        result_type = 0
        result_text = f"""
            <div style='font-family: Times New Roman, font-size=12px;'>
                <b>The differences are not significant</b><br>
                Simes = <b>{s_val:.2e}</b><br>
                &Delta;&sigma;<sup>2</sup>(total) = <b>{delta_var_all:.2e}</b><br>
                &Delta;&sigma;<sup>2</sup>(sign.) = <b>{delta_var_significant:.2e}</b>
            </div>
        """
    return result_text, result_type


def peak_picking(X, Y, oversegmentation_filter=None, peak_location=1):
    """
    Detect peaks in a KDE curve and return their centers and boundaries.

    Parameters
    ----------
    X : ndarray
        Monotonic array of X coordinates (e.g., m/z grid).
    Y : ndarray
        Corresponding density/height values.
    oversegmentation_filter : float or None, optional
        Minimal allowed separation between adjacent peaks; when provided, peaks
        closer than this threshold are merged.
    peak_location : float, optional
        Fraction of the peak height to compute a barycentric center; used in
        boundary calculations as a threshold. Default is 1.

    Returns
    -------
    pk_x : ndarray
        Estimated peak centers (X positions). May contain NaNs if a region has
        no samples above the threshold.
    left : ndarray
        Left boundary (valley position) for each peak.
    right : ndarray
        Right boundary (valley position) for each peak.
    """
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

def sorting_dots(ds: np.ndarray, left: np.ndarray, right: np.ndarray) -> list:
    if len(left) == 0:
        return []

    ds_sorted = np.sort(ds)

    # Находим границы каждого бина в отсортированном массиве
    starts = np.searchsorted(ds_sorted, left, side='left')
    ends = np.searchsorted(ds_sorted, right, side='right')

    # Формируем список срезов (views, без лишнего копирования)
    ret = [ds_sorted[start:end] for start, end in zip(starts, ends)]
    return ret

def get_long_and_short(arr_1: np.ndarray, arr_2: np.ndarray) -> (np.ndarray, np.ndarray, bool):
    """
    Return the longer and shorter of two arrays and a flag indicating order.

    Parameters
    ----------
    arr_1, arr_2 : ndarray
        Arrays to compare by first-dimension length.

    Returns
    -------
    long : ndarray
        The longer array.
    short : ndarray
        The shorter array.
    flag : bool
        True if ``arr_1`` is the longer array, else False.
    """
    size1, size2 = arr_1.shape[0], arr_2.shape[0]
    if size1 > size2:
        return arr_1, arr_2, True
    else:
        return arr_2, arr_1, False


def get_opt_strip(arr_long: Dataset, arr_short: Dataset, flag: bool) -> (Dataset, Dataset):
    """
    Align two sequences by shifting the longer to minimize mean squared error.

    Parameters
    ----------
    arr_long : Dataset
        Longer dataset.
    arr_short : Dataset
        Shorter dataset.
    flag : bool
        True if `arr_long` corresponds to the original first argument from
        ``get_long_and_short``.

    Returns
    -------
    Dataset, Dataset
        Sliced/shifted versions with equal length, ordered to match the flag.
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
    Verify and co-trim two sorted datasets so that element-wise differences are bounded.

    The function optionally removes one outlier (by index) and re-aligns to
    satisfy the threshold, returning two arrays of equal length.

    Parameters
    ----------
    data_1, data_2 : LinkedList
        Input datasets to verify.
    threshold : float or str, optional
        Maximum allowed absolute difference between paired values. If
        ``'dist_based'``, the mean difference is used as the threshold.

    Returns
    -------
    LinkedList, LinkedList
        Verified (possibly trimmed) datasets of equal size.
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


_DATA_RAW = None
_DATA_ALN = None
_IDX = None          # (mz_idx_raw, intensity_idx_raw, spectra_idx_raw, mz_idx_aln, intensity_idx_aln, spectra_idx_aln)
_REF_DEV = None      # (REF, DEV)


def pool_initializer(data_raw, data_aln, idx_tuple, dev):
    """
    Pool initializer: store global references to datasets, indices, and params.

    Parameters
    ----------
    data_raw : ndarray
        Raw dataset array loaded from HDF5.
    data_aln : ndarray
        Aligned dataset array loaded from HDF5.
    idx_tuple : tuple[int, int, int, int, int, int]
        ``(mz_idx_raw, intensity_idx_raw, spectra_idx_raw, mz_idx_aln,
        intensity_idx_aln, spectra_idx_aln)`` indices into the datasets.
    dev : float
        Allowed deviation (±) around ``ref`` for reference search.

    Notes
    -----
    Stores the arguments into module-level globals (``_DATA_RAW``, ``_DATA_ALN``,
    ``_IDX``, ``_DEV``) to avoid repeated pickling and argument passing to
    worker processes.
    """
    global _DATA_RAW, _DATA_ALN, _IDX, _DEV
    _DATA_RAW = data_raw
    _DATA_ALN = data_aln
    _IDX = idx_tuple
    _DEV = dev

def pool_initializer_imz(fn_raw,fn_aln):
    global parser_raw_g, parser_aln_g
    parser_raw_g = ImzMLParser(fn_raw)
    parser_aln_g = ImzMLParser(fn_aln)

def process_spectrum(task):
    """
        Process a single spectrum task and return datasets for raw and aligned.

        Parameters
        ----------
        task : tuple[int, int, int, int, int]
            ``(spec_id, r0, r1, a0, a1)`` where ``[r0:r1]`` and ``[a0:a1]`` are
            inclusive slices for raw and aligned blocks belonging to ``spec_id``.

        Returns
        -------
        tuple
            ``(spec_id, arr_raw, arr_aln)`` where ``arr_raw`` and ``arr_aln`` are
            NumPy arrays representing ``Dataset`` instances for the spectrum.
        """
    spec_id, r0, r1, a0, a1 = task
    mz_idx_raw, intensity_idx_raw, _s_idx_r, mz_idx_aln, intensity_idx_aln, _s_idx_a = _IDX
    DEV = _DEV

    # извлечь и отсортировать по m/z
    data_raw_unsorted = _DATA_RAW[[mz_idx_raw, intensity_idx_raw], r0:r1 + 1]
    data_aln_unsorted = _DATA_ALN[[mz_idx_aln, intensity_idx_aln], a0:a1 + 1]

    order_raw = np.argsort(data_raw_unsorted[0])
    order_aln = np.argsort(data_aln_unsorted[0])
    data_raw = data_raw_unsorted[:, order_raw]
    data_aln = data_aln_unsorted[:, order_aln]

    data_raw_mz, data_aln_mz = data_raw[0], data_aln[0]
    data_raw_int, data_aln_int = data_raw[1], data_aln[1]

    data_raw_linked = Dataset(data_raw_mz, data_raw_int)
    data_aln_linked = Dataset(data_aln_mz, data_aln_int)

    checked_raw, checked_aln = verify_datasets(data_raw_linked, data_aln_linked, 1)

    return spec_id, np.array(checked_raw), np.array(checked_aln)

def process_spectrum_imz(num):
    mzs_raw, intensities_raw = parser_raw_g.getspectrum(num)
    mzs_aln, intensities_aln = parser_aln_g.getspectrum(num)
    data_raw_linked = Dataset(mzs_raw, intensities_raw)
    data_aln_linked = Dataset(mzs_aln, intensities_aln)

    checked_raw, checked_aln = verify_datasets(data_raw_linked, data_aln_linked, 1)

    return (num,np.array(checked_raw),np.array(checked_aln))


def read_imz_dataset(self,parser_raw:pyimzml.ImzMLParser,parser_aln:pyimzml.ImzMLParser,limit=None,processes: int = 0):
    len_raw, len_aln = len(parser_raw.coordinates), len(parser_aln.coordinates)
    if len_raw != len_aln:
        raise Exception('Length of raw and aln is not equal!')

    set_num = len_aln
    if limit is not None:
        set_num = min(set_num, limit)

    dataset_list = np.empty((2, set_num), dtype=Dataset)
    tasks = list(range(set_num))

    self.create_pbar.emit((0, len(tasks)))

    if processes <= 0:
        for num in tasks:
            mzs_raw, intensities_raw = parser_raw.getspectrum(num)
            mzs_aln, intensities_aln = parser_aln.getspectrum(num)
            data_raw_linked = Dataset(mzs_raw, intensities_raw)
            data_aln_linked = Dataset(mzs_aln, intensities_aln)

            checked_raw, checked_aln = verify_datasets(data_raw_linked, data_aln_linked, 1)

            dataset_list[0, num] = np.array(checked_raw)
            dataset_list[1, num] = np.array(checked_aln)
            self.progress.emit(num+1)
        return dataset_list

    init_args = (parser_raw.filename, parser_aln.filename)

    with Pool(processes=processes,
              initializer=pool_initializer_imz,
              initargs=init_args) as pool:
        completed_tasks = 0
        for num, raw_ds, aln_ds in pool.imap_unordered(process_spectrum_imz,tasks):
            dataset_list[0, num] = raw_ds
            dataset_list[1, num] = aln_ds
            completed_tasks += 1
            self.progress.emit(completed_tasks)
    return dataset_list



def read_dataset(self, dataset_raw: np.ndarray, attrs_raw: list, dataset_aln: np.ndarray,
                 attrs_aln: list, DEV, limit=None, processes: int = 0):
    """
    Prepare per-spectrum datasets and emit progress for the UI, with optional
    sequential or parallel execution (multiprocessing.Pool).

    Overview
    --------
    - Resolve indices of required columns by headers (m/z and intensity).
    - Build contiguous segments for each spectrum id based on the spectra index.
    - Create tasks only for spectrum ids present in both raw and aligned inputs.
    - For each task: slice the subarrays, sort by m/z, verify alignment
      (``verify_datasets``), find a reference peak around ``REF`` within ``DEV``
      (``find_ref``), and store the result as a ``Dataset`` with a ``reference``.
    - Emit progress after each spectrum is processed.

    Modes
    -----
    - Sequential (``processes <= 0``): runs in the main thread, preserving
      existing variable names and logic.
    - Parallel (``processes > 0``): uses ``multiprocessing.Pool`` with an
      initializer (``pool_initializer``) and worker (``process_spectrum``).
      Tasks are processed in parallel; results may arrive unordered and are
      placed by ``spec_id``.

    Parameters
    ----------
    self : WorkerSignals
        Object used to emit progress bar initialization and updates.
    dataset_raw, dataset_aln : ndarray
        Raw and aligned datasets read from HDF5.
    attrs_raw, attrs_aln : list of str
        Column headers for the respective datasets.
    DEV : float
        Acceptable deviation (±) around ``REF`` for reference search.
    limit : int or None, optional
        Optional limit on the number of spectra to process (debugging).
    processes : int, optional
        Number of processes for ``multiprocessing.Pool``. ``<= 0`` means
        sequential mode. Default is 0.

    Returns
    -------
    ndarray
        Array of shape ``(2, N)`` with ``dtype=Dataset``, where ``N`` is the
        number of processed spectra. ``dataset_list[0, spec_id]`` corresponds to
        the raw dataset; ``dataset_list[1, spec_id]`` to the aligned dataset.

    Notes
    -----
    - Only spectrum ids present in both raw and aligned datasets are processed.
    - The progress bar is initialized based on the number of tasks (common ids).
    - In parallel mode, result arrival order is not guaranteed.
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

    # предрасчёт сегментов по каждому индексу спектра
    segments_raw = build_segments(index_row_raw)
    segments_aln = build_segments(index_row_aln)

    # список задач только для тех индексов, которые есть и в raw, и в aln
    tasks = []
    for spec_id in range(start_index, end_index + 1):
        if spec_id in segments_raw and spec_id in segments_aln:
            r0, r1 = segments_raw[spec_id]
            a0, a1 = segments_aln[spec_id]
            tasks.append((spec_id, r0, r1, a0, a1))

    # инициализация прогресса по числу задач
    self.create_pbar.emit((0, len(tasks)))

    # последовательная ветка — сохранить существующую логику имен/переменных
    if processes <= 0:
        for spec_n, (spec_id, r0, r1, a0, a1) in enumerate(tasks):
            index_raw, index_aln = np.where(index_row_raw == spec_id)[0], np.where(index_row_aln == spec_id)[0]
            data_raw_unsorted = dataset_raw[row_raw([mz_type,int_type]), index_raw[0]:index_raw[-1] + 1]
            data_aln_unsorted = dataset_aln[row_aln([mz_type,int_type]), index_aln[0]:index_aln[-1] + 1]

            data_raw = data_raw_unsorted[:, np.argsort(data_raw_unsorted, axis=1)[0]]
            data_aln = data_aln_unsorted[:, np.argsort(data_aln_unsorted, axis=1)[0]]

            data_raw_mz, data_aln_mz = data_raw[0], data_aln[0]
            data_raw_int, data_aln_int = data_raw[1], data_aln[1]

            data_raw_linked = Dataset(data_raw_mz, data_raw_int)
            data_aln_linked = Dataset(data_aln_mz, data_aln_int)

            checked_raw, checked_aln = verify_datasets(data_raw_linked, data_aln_linked, 1)

            dataset_list[0, spec_id] = np.array(checked_raw)
            dataset_list[1, spec_id] = np.array(checked_aln)
            self.progress.emit(spec_n)
        return dataset_list

    # параллельная ветка — Pool с минимальными изменениями
    mz_idx_raw = row_raw(mz_type)
    intensity_idx_raw = row_raw(int_type)
    spectra_idx_raw = row_raw("spectra_ind")

    mz_idx_aln = row_aln(mz_type)
    intensity_idx_aln = row_aln(int_type)
    spectra_idx_aln = row_aln("spectra_ind")

    init_args = (
        dataset_raw,
        dataset_aln,
        (mz_idx_raw, intensity_idx_raw, spectra_idx_raw,
         mz_idx_aln, intensity_idx_aln, spectra_idx_aln),
        DEV,
    )

    multiprocessing.util.FINALIZE_MAX_DELAY = 10
    with Pool(processes=processes, initializer=pool_initializer, initargs=init_args) as pool:
        for spec_n, (spec_id, arr_raw, arr_aln) in enumerate(pool.imap_unordered(process_spectrum, tasks)):
            dataset_list[0, spec_id] = arr_raw
            dataset_list[1, spec_id] = arr_aln
            self.progress.emit(spec_n)

    return dataset_list


def build_segments(spectra_index_row: np.ndarray) -> dict[int, tuple[int, int]]:
    """
    Build contiguous [start, end] slices for each value of ``spectra_ind``.

    Parameters
    ----------
    spectra_index_row : ndarray
        1-D array of spectrum identifiers, typically the ``spectra_ind`` row
        from an HDF5 dataset.

    Returns
    -------
    dict[int, tuple[int, int]]
        Mapping from spectrum id to an inclusive ``(start, end)`` slice within
        ``spectra_index_row`` covering its contiguous block.
    """
    segments: dict[int, tuple[int, int]] = {}
    if spectra_index_row.size == 0:
        return segments
    change_pos = np.where(np.diff(spectra_index_row) != 0)[0]
    starts = np.concatenate(([0], change_pos + 1))
    ends = np.concatenate((change_pos, [spectra_index_row.size - 1]))
    ids = spectra_index_row[ends]
    for s, e, spec_id in zip(starts, ends, ids):
        segments[int(spec_id)] = (int(s), int(e))
    return segments


def prepare_array(distances):
    """
    Concatenate per-peak distances and build a 2-row sorted view with indices.

    Parameters
    ----------
    distances : ndarray or Sequence
        Pair or sequence of sequences to concatenate and index.

    Returns
    -------
    ndarray
        A 2 x K array with sorted values in row 0 and original indices in row 1.
    """
    concatenated = np.array([np.concatenate(sub) for sub in distances])
    indexes = np.repeat(np.arange(len(distances[0])), [len(sub_arr) for sub_arr in distances[0]])
    pre_sorted = np.vstack((concatenated, indexes))
    result = pre_sorted[:, pre_sorted[0].argsort()]
    return result

def simes(p_value, alpha = 0.05):
    """
    Calculate Simes method p-value for whole spectrum

    Parameters
    ----------
    p_value : ndarray
        p-value array
    alpha : float
        Confidence level. Default is 0.05

    Returns
    -------
    float
        simes value
    bool
        is test statistically significant
    """

    p_vals = np.sort(p_value)
    count = len(p_vals)

    simes_value = np.min(count*p_vals/np.arange(1, count+1))

    simes_significance = simes_value < alpha

    return simes_value, simes_significance

def concover(arr1:np.ndarray,arr2:np.ndarray):
    """
    Compare two distributions using a rank-based variance (Conover-like) test.

    Parameters
    ----------
    arr1, arr2 : ndarray
        Samples from two distributions.

    Returns
    -------
    float
        p-value for the test of equal scale/dispersion.
    """
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


def stat_params_paired_single(peak_raw, peak_aln, alpha=0.05,return_p = True):
    """
    Compute paired statistics between raw and aligned peak positions.

    For each matched peak, compute mean difference, variances, normality check (to choose acceptable hypothesis tests) and JS-divergence

    Parameters
    ----------
    peak_raw, peak_aln : array_like
        Samples of raw and aligned values for a single peak.
    alpha : float, optional
        Significance level used in tests. Default is 0.05.
    return_p : bool, optional
        If True, function will return exact p-value, otherwise result of comparison with significance level. Default is True.

    Returns
    -------
    tuple
        ``(mean_diff, var_raw, var_aln, js_div, neq_mean, neq_var)``
        where boolean flags are returned as floats (0.0/1.0).
    """
    # вычислить среднее и дисперсии, проверить нормальность, проверить гипотезы о значимости различия средних и дисперсий, возможно посчитать форму распределения
    jsd = lambda p,q: 0.5*(sum(rel_entr(p,q))+sum(rel_entr(q,p)))
    kde_single_peak = lambda dots,n_eval: FFTKDE(bw='silverman',kernel='gaussian').fit(dots).evaluate(n_eval)[1]

    norm_var = lambda data: np.var(data-np.mean(data),ddof=1)
    mean_r, mean_a = np.mean(peak_raw), np.mean(peak_aln)
    var_r, var_a = norm_var(peak_raw), norm_var(peak_aln)

    check_normal_func = lambda data,p: stats.kstest(data,'norm',args=(np.mean(data),np.std(data)))[1]>p
    check_normal = check_normal_func(peak_raw, alpha) & check_normal_func(peak_aln, alpha)
    neq_var_p_val = stats.levene(peak_raw, peak_aln)[1]
    if check_normal:

        neq_mean_p_val = stats.ttest_ind(peak_raw, peak_aln,nan_policy='omit')[1]
    else:
        neq_mean_p_val = stats.mannwhitneyu(peak_raw, peak_aln)[1]


    if return_p:
        neq_var = neq_var_p_val
        neq_mean = neq_mean_p_val
    else:
        neq_var = neq_var_p_val < alpha
        neq_mean = neq_mean_p_val < alpha
    return np.array([mean_r - mean_a, var_r, var_a,jsd(kde_single_peak(peak_raw,20),kde_single_peak(peak_aln,20)), float(neq_mean), float(neq_var)])


def stat_params_unpaired(ds):
    """
    Compute unpaired per-group statistics for a list of arrays.

    Parameters
    ----------
    ds : Sequence[array_like]
        Sequence of samples (e.g., peak positions per bin).

    Returns
    -------
    ndarray
        Array with columns: variance, dip statistic, dip p-value, skewness,
        kurtosis for each group.
    """
    res = np.array([[np.var(dot), *diptest(dot), stats.skew(dot), stats.kurtosis(dot)] for dot in ds])
    return res


def moving_average(a, n=2):
    """
    Compute the simple moving average over a 1D array.

    Parameters
    ----------
    a : ndarray
        Input array.
    n : int, optional
        Window size. Default is 2.

    Returns
    -------
    ndarray
        Averaged array of length ``len(a) - n + 1``.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def out_criteria(mz, intensity, int_threshold = 0.01, max_diff = 0.4, width_eps = 0.1):
    """
    .. warning::
        Current version of pipeline doesn't use this function

    Identify outlier peak intervals based on intensity and width heuristics.

    Parameters
    ----------
    mz : Dataset or LinkedList
        Peak centers with linked boundaries in `linked_array`.
    intensity : ndarray
        Intensities corresponding to `mz` centers.
    int_threshold : float, optional
        Fraction of the maximum intensity below which points are flagged.
    max_diff : float, optional
        Maximum relative change between consecutive intensities (as |a/b - 1|).
    width_eps : float, optional
        Threshold on normalized width ratio used for flagging.

    Returns
    -------
    ndarray
        Indices of points considered outliers.
    """
    min_int = np.max(intensity) * int_threshold
    first_or = intensity < min_int
    int_criteria = abs(intensity[:-1] / intensity[1:] - 1) < max_diff

    width_criteria = np.diff(mz) / moving_average(np.diff(mz.linked_array).flatten()) <= width_eps
    second_or = np.full(mz.shape, False)
    second_or[1:] = np.logical_and(int_criteria, width_criteria)

    return np.where(np.logical_or(first_or, second_or))[0]


def criteria_apply(arr, intensity):
    """
    .. warning::
        Current version of pipeline doesn't use this function

    Merge narrow neighboring intervals and drop flagged indices.

    Parameters
    ----------
    arr : LinkedList
        Peak centers with linked left/right boundaries.
    intensity : ndarray
        Intensities used to evaluate the criteria.

    Returns
    -------
    LinkedList
        Filtered peaks with adjusted boundaries.
    """
    arr_out = copy.deepcopy(arr)
    indexes = out_criteria(arr, intensity)
    for index in indexes:
        arr_out.linked_array[index-1] = sorted([arr.linked_array[index-1,0],arr.linked_array[index,1]])
    return arr_out.sync_delete(indexes)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')
    app = QApplication(sys.argv)
    app_icon = QIcon('main_ico.png')
    app.setWindowIcon(app_icon)
    main_window = MainWindow()
    main_window.showMaximized()

    sys.exit(app.exec_())
