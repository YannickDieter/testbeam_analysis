import os
import inspect
import logging
import math
from multiprocessing import Pool
from collections import OrderedDict
from numpydoc.docscrape import FunctionDoc
from PyQt5 import QtWidgets, QtCore, QtGui

from testbeam_analysis.gui import option_widget


def get_default_args(func):
    """
    Returns a dictionary of arg_name:default_values for the input function
    """
    args, _, _, defaults = inspect.getargspec(func)
    return dict(zip(args[-len(defaults):], defaults))


def get_parameter_doc(func, dtype=False):
    """ 
    Returns a dictionary of paramerter:pardoc for the input function
    Pardoc is either the parameter description (dtype=False) or the data type (dtype=False)
    """
    doc = FunctionDoc(func)
    pars = {}
    for par, datatype, descr in doc['Parameters']:
        if not dtype:
            pars[par] = '\n'.join(descr)
        else:
            pars[par] = datatype
    return pars


class AnalysisWidget(QtWidgets.QWidget):
    """
    Implements a generic analysis gui.

    There are two separated widget areas. One the left one for plotting
    and on the right for function parameter options.
    There are 3 kind of options:
      - needed ones on top
      - optional options that can be deactivated below
      - fixed option that cannot be changed
    Below this is a button to call the underlying function with given
    keyword arguments from the options.

    Introprospection is used to determine function argument types and
    documentation from the function implementation automatically.
    """

    # Signal emitted after all funcs are called
    analysisDone = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list=None):
        super(AnalysisWidget, self).__init__(parent)
        self.setup = setup
        self.options = options
        self.option_widgets = {}
        self._setup()
        # Holds functions with kwargs
        self.calls = OrderedDict()
        # List of tabs which will be enabled after analysis
        if isinstance(tab_list, list):
            self.tab_list = tab_list
        else:
            self.tab_list = [tab_list]

    def _setup(self):
        # Plot area
        self.left_widget = QtWidgets.QWidget()
        self.plt = QtWidgets.QHBoxLayout()
        self.left_widget.setLayout(self.plt)
        # Options
        self.opt_needed = QtWidgets.QVBoxLayout()
        self.opt_optional = QtWidgets.QVBoxLayout()
        self.opt_fixed = QtWidgets.QVBoxLayout()
        # Option area

        self.layout_options = QtWidgets.QVBoxLayout()
        self.label_option = QtWidgets.QLabel('Options')
        self.layout_options.addWidget(self.label_option)
        self.layout_options.addLayout(self.opt_needed)
        self.layout_options.addLayout(self.opt_optional)
        self.layout_options.addLayout(self.opt_fixed)
        self.layout_options.addStretch(0)

        # Proceed button
        self.button_ok = QtWidgets.QPushButton('OK')
        self.button_ok.clicked.connect(self._call_funcs)
        self.layout_options.addWidget(self.button_ok)

        # Right widget
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setLayout(self.layout_options)

        # Make right widget scroll able
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setBackgroundRole(QtGui.QPalette.Light)
        scroll.setWidget(self.right_widget)

        # Split plot and option area
        self.widget_splitter = QtWidgets.QSplitter(parent=self)
        self.widget_splitter.addWidget(self.left_widget)
        self.widget_splitter.addWidget(scroll)
        self.widget_splitter.setStretchFactor(0, 10)
        self.widget_splitter.setStretchFactor(1, 2.5)
        self.widget_splitter.setChildrenCollapsible(False)
        # Add complete layout to this widget
        layout_widget = QtWidgets.QVBoxLayout()
        layout_widget.addWidget(self.widget_splitter)
        self.setLayout(layout_widget)

    def _option_exists(self, option):
        """
        Check if option is already defined
        """
        for call in self.calls.values():
            for kwarg in call:
                if option == kwarg:
                    return True
        return False

    def add_options_auto(self, func):
        """
        Inspect a function to create options for kwargs
        """

        for name in get_default_args(func):
            # Only add as function parameter if the info is not
            # given in setup/option data structures
            if name in self.setup:
                if not self._option_exists(option=name):
                    self.add_option(option=name, default_value=self.setup[name],
                                    func=func, fixed=True)
                else:
                    self.calls[func][name] = self.setup[name]
            elif name in self.options:
                if not self._option_exists(option=name):
                    self.add_option(option=name, default_value=self.options[name],
                                    func=func, fixed=True)
                else:
                    self.calls[func][name] = self.options[name]
            else:
                self.add_option(func=func, option=name)

    def add_option(self, option, func, dtype=None, name=None, optional=None, default_value=None, fixed=False, tooltip=None):
        """
        Add an option to the gui to set function arguments

        option: str
            Function argument name
        func: function
            Function to be used for the option
        dtype: str
            Type string to select proper input method, if None determined from default parameter type
        name: str
            Name shown in gui
        optional: bool
            Show as optional option, If optional is not defined all parameters with default value
            None are set as optional. The common behavior is that None deactivates a parameter
        default_value : object
            Default value for option
        fixed : boolean
            Fix option value  default value
        """

        # Check if option exists already
        if option in self.calls[func]:
            self._delete_option(option=option, func=func)

        # Get name from argument name
        if not name:
            name = option.replace("_", " ").capitalize()

        # Get default argument value
        if default_value is None:
            default_value = get_default_args(func)[option]

        # Get parameter description from numpy style docstring
        if not tooltip:
            try:
                tooltip = get_parameter_doc(func)[option]
            except KeyError:  # No parameter docu available
                logging.warning(
                    'Parameter %s in function %s not documented', option, func.__name__)
                tooltip = None

        # Get parameter dtype from numpy style docstring
        if not dtype:
            try:
                dtype = get_parameter_doc(func, dtype=True)[option]
            except KeyError:  # No parameter docu available
                pass

        # Get dtype from default arg
        if not dtype:
            if default_value is not None:
                dtype = str(type(default_value).__name__)
            else:
                raise RuntimeError(
                    'Cannot deduce data type for %s in function %s, because no default parameter exists', option, func.__name__)

        # Get optional argument from default function argument
        if optional is None and default_value is None:
            optional = True

        if not fixed:  # Option value can be changed
            try:
                widget = self._select_widget(dtype, name, default_value,
                                             optional, tooltip)
            except NotImplementedError:
                logging.warning('Cannot create option %s for dtype "%s" for function %s',
                                option, dtype, func.__name__)
                return

            self._set_argument(
                func, option, default_value if not optional else None)
            self.option_widgets[option] = widget
            self.option_widgets[option].valueChanged.connect(
                lambda value: self._set_argument(func, option, value))

            if optional:
                self.opt_optional.addWidget(self.option_widgets[option])
            else:
                self.opt_needed.addWidget(self.option_widgets[option])
        else:  # Fixed value
            if default_value is None:
                raise RuntimeError(
                    'Cannot create fixed option without default value')
            text = QtWidgets.QLabel()
            text.setWordWrap(True)
            # Fixed options cannot be changed --> grey color
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.darkGray)
            text.setPalette(palette)
            text.setToolTip(tooltip)
#            metrics = QtGui.QFontMetrics(self.font())
#            elided_text = metrics.elidedText(str(default_value), QtCore.Qt.ElideMiddle, 500)
            text.setText(name + ': ' + str(default_value))
            self.opt_fixed.addWidget(text)
            self.calls[func][option] = default_value

    def _select_widget(self, dtype, name, default_value, optional, tooltip):
        # Create widget according to data type
        if ('scalar' in dtype and ('tuple' in dtype or 'iterable' in dtype) or
                'int' in dtype and ('tuple' in dtype or 'iterable' in dtype) or
                ('iterable' in dtype and 'iterable of iterable' not in dtype)):
            widget = option_widget.OptionMultiSlider(
                name=name, labels=self.setup['dut_names'],
                default_value=default_value,
                optional=optional, tooltip=tooltip, parent=self)
        elif 'iterable of iterable' in dtype:
            widget = option_widget.OptionMultiBox(
                name=name, labels_x=self.setup['dut_names'],
                default_value=default_value,
                optional=optional, tooltip=tooltip, labels_y=self.setup['dut_names'], parent=self)
        elif 'str' in dtype:
            widget = option_widget.OptionText(
                name, default_value, optional, tooltip, parent=self)
        elif 'int' in dtype:
            widget = option_widget.OptionSlider(
                name, default_value, optional, tooltip, parent=self)
        elif 'float' in dtype:
            widget = option_widget.OptionSlider(
                name, default_value, optional, tooltip, parent=self)
        elif 'bool' in dtype:
            widget = option_widget.OptionBool(
                name, default_value, optional, tooltip, parent=self)
        else:
            raise NotImplementedError('Cannot use type %s', dtype)

        return widget

    def _delete_option(self, option, func):
        """
        Delete existing option. Needed if option is set manually.
        """

        # Delete option widget
        self.option_widgets[option].close()
        del self.option_widgets[option]
        # Update widgets
        self.opt_optional.update()
        self.opt_needed.update()
        # Delete kwarg
        del self.calls[func][option]

    def add_function(self, func):
        """
        Add an analysis function
        """

        self.calls[func] = {}
        # Add tooltip from function docstring
        doc = FunctionDoc(func)
        label_option = self.label_option.toolTip()
        self.label_option.setToolTip(label_option +
                                     '\n'.join(doc['Summary']))
        # Add function options to gui
        self.add_options_auto(func)

    def _set_argument(self, func, name, value):
        # Workaround for https://www.riverbankcomputing.com/pipermail/pyqt/2016-June/037662.html
        # Cannot transmit None for signals with string (likely also float)
        if type(value) == str and 'None' in value:
            value = None
        if type(value) == float and math.isnan(value):
            value = None
        if type(value) == list and None in value:
            value = None
        self.calls[func][name] = value

    def _call_func(self, func, kwargs):
        """
        Call an analysis function with given kwargs
        Setup info and generic options are added if needed.
        """

        # Set missing kwargs from setting data structures
        args = inspect.getargspec(func)[0]
        for arg in args:
            if arg not in self.calls[func]:
                if arg in self.setup:
                    kwargs[arg] = self.setup[arg]
                elif arg in self.options or 'file' in arg:
                    try:
                        if 'input' in arg or 'output' in arg:
                            kwargs[arg] = os.path.join(self.options['working_directory'],
                                                       self.options[arg])
                        else:
                            kwargs[arg] = self.options[arg]
                    except KeyError:
                        logging.error(
                            'File I/O %s not defined in settings', arg)
                else:
                    raise RuntimeError('Function argument %s not defined', arg)
        # print(func.__name__, kwargs)
        func(**kwargs)

    def _call_funcs(self):
        """ 
        Call all functions in a row
        """

        pool = Pool()
        for func, kwargs in self.calls.iteritems():
            # print(func.__name__, kwargs)
            pool.apply_async(self._call_func(func, kwargs))
        pool.close()
        pool.join()

        # Emit signal to indicate end of analysis
        if self.tab_list is not None:
            self.analysisDone.emit(self.tab_list)


class ParallelAnalysisWidget(QtWidgets.QWidget):
    """
    AnalysisWidget for functions that need to run for every input data file.
    Creates UI with one tab widget per respective input file
    """

    parallelAnalysisDone = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list=None):

        super(ParallelAnalysisWidget, self).__init__(parent)

        # Make main layout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

        # Add sub-layout and ok button and progressbar
        self.sub_layout = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton('Ok')
        self.btn_ok.clicked.connect(lambda: self._call_parallel_funcs())
        self.p_bar = QtWidgets.QProgressBar()
        self.p_bar.setVisible(False)

        # Set alignment in sub-layout
        self.sub_layout.addWidget(self.p_bar)
        self.sub_layout.addWidget(self.btn_ok)
        self.sub_layout.setAlignment(self.p_bar, QtCore.Qt.AlignLeading)
        self.sub_layout.setAlignment(self.btn_ok, QtCore.Qt.AlignTrailing)

        # Tab related widgets
        self.tabs = QtWidgets.QTabWidget()
        self.tw = {}

        # Add to main layout
        self.main_layout.addWidget(self.tabs)
        self.main_layout.addLayout(self.sub_layout)

        # Initialize options and setup
        self.setup = setup
        self.options = options

        # List of tabs which will be enabled after analysis
        if isinstance(tab_list, list):
            self.tab_list = tab_list
        else:
            self.tab_list = [tab_list]

        self._init_tabs()
        self.connect_tabs()

    def _init_tabs(self):

        # Clear widgets
        self.tabs.clear()
        self.tw = {}

        for i in range(self.setup['n_duts']):

            tmp_setup = {}
            tmp_options = {}

            for s_key in self.setup.keys():

                if isinstance(self.setup[s_key], list) or isinstance(self.setup[s_key], tuple):
                    if isinstance(self.setup[s_key][i], str):
                        tmp_setup[s_key] = [self.setup[s_key][i]]  # FIXME: Does not work properly without list
                    else:
                        tmp_setup[s_key] = self.setup[s_key][i]
                elif isinstance(self.setup[s_key], int) or isinstance(self.setup[s_key], str):
                    tmp_setup[s_key] = self.setup[s_key]

            for o_key in self.options.keys():

                if isinstance(self.options[o_key], list) or isinstance(self.options[o_key], tuple):
                    if isinstance(self.options[o_key][i], str):
                        tmp_options[o_key] = [self.options[o_key][i]]  # FIXME: Does not work properly without list
                    else:
                        tmp_options[o_key] = self.options[o_key][i]
                elif isinstance(self.options[o_key], int) or isinstance(self.options[o_key], str):
                    tmp_options[o_key] = self.options[o_key]

            widget = AnalysisWidget(parent=self.tabs, setup=tmp_setup, options=tmp_options, tab_list=self.tab_list)
            widget.button_ok.deleteLater()

            self.tw[self.setup['dut_names'][i]] = widget
            self.tabs.addTab(self.tw[self.setup['dut_names'][i]], self.setup['dut_names'][i])

    def connect_tabs(self):

        self.tabs.currentChanged.connect(lambda tab: self.handle_sub_layout(tab=tab))

        for tab_name in self.tw.keys():
            self.tw[tab_name].widget_splitter.splitterMoved.connect(
                lambda: self.handle_sub_layout(tab=self.tabs.currentIndex()))

    def resizeEvent(self, QResizeEvent):
        self.handle_sub_layout(tab=self.tabs.currentIndex())

    def showEvent(self, QShowEvent):
        self.handle_sub_layout(tab=self.tabs.currentIndex())

    def handle_sub_layout(self, tab):

        offset = 10
        sub_widths = self.tw[self.tabs.tabText(tab)].widget_splitter.sizes()

        self.p_bar.setFixedWidth(sub_widths[0] + offset)
        self.btn_ok.setFixedWidth(sub_widths[1] + offset)

    def add_parallel_function(self, func):
        for i in range(self.setup['n_duts']):
            self.tw[self.setup['dut_names'][i]].add_function(func=func)

    def add_parallel_option(self, option, default_value, func, name=None, dtype=None, optional=None, fixed=False, tooltip=None):

        for i in range(self.setup['n_duts']):
            self.tw[self.setup['dut_names'][i]].add_option(option=option, func=func, dtype=dtype, name=name,
                                                           optional=optional, default_value=default_value[i],
                                                           fixed=fixed, tooltip=tooltip)

    def _call_parallel_funcs(self):

        self.btn_ok.setDisabled(True)

        self.p_bar.setRange(0, len(self.tw.keys()))
        self.p_bar.setVisible(True)

        for i, tab in enumerate(self.tw.keys()):
            self.p_bar.setValue(i+1)
            self.tw[tab]._call_funcs()
            # QtCore.QCoreApplication.processEvents()  # FIXME: Multi-threading probably needed here

        if self.tab_list is not None:
            self.parallelAnalysisDone.emit(self.tab_list)

        self.btn_ok.setDisabled(False)


class AnalysisLogger(logging.Handler):
    """
    Implements a logging handler which allows redirecting log
    into QPlainTextEdit to display in AnalysisWindow
    """

    def __init__(self, parent):

        super(AnalysisLogger, self).__init__()

        # Widget to display log in, we only want to read log
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

        # Dock in which text widget is placed to make it closable without losing log content
        self.dock = QtWidgets.QDockWidget(parent)
        self.dock.setWidget(self.widget)
        self.dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable)
        self.dock.setWindowTitle('Logger')

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

