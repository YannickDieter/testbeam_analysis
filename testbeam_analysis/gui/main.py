import sys
import logging
import platform

from email import message_from_string
from pkg_resources import get_distribution, DistributionNotFound

from PyQt5 import QtCore, QtWidgets, QtGui

from data import DataTab
from setup import SetupTab
from sub_windows import SettingsWindow, ExceptionWindow
from analysis_logger import AnalysisLogger, AnalysisStream
import testbeam_analysis
from testbeam_analysis.gui import tab_widget

PROJECT_NAME = 'Testbeam Analysis'
GUI_AUTHORS = 'Pascal Wolf, David-Leon Pohl'
MINIMUM_RESOLUTION = (1366, 768)

# Create all tabs at start up for debugging purpose
_DEBUG = False

try:
    pkgInfo = get_distribution('testbeam_analysis').get_metadata('PKG-INFO')
    AUTHORS = message_from_string(pkgInfo)['Author']
except (DistributionNotFound, KeyError):
    AUTHORS = 'Not defined'


class AnalysisWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        """
        Initializes the analysis window
        """
        super(AnalysisWindow, self).__init__(parent)

        # Get default settings
        self.setup = SettingsWindow().default_setup
        self.options = SettingsWindow().default_options

        # Make variable for SettingsWindow
        self.settings_window = None

        # Make variable for ExceptionWindow
        self.exception_window = None

        # Variable to store tab name from which consecutive analysis starts
        self.starting_tab_rca = None

        # Flag to interrupt consecutive analysis
        self.flag_interrupt = False

        # Make dict to access tab widgets
        self.tw = {}

        # Icon do indicate tab completed
        self.icon_complete = QtWidgets.qApp.style().standardIcon(QtWidgets.qApp.style().SP_DialogApplyButton)

        self._init_UI()

    def _init_UI(self):
        """
        Initializes the user interface and displays "Hello"-message
        """

        # Main window settings
        self.setWindowTitle(PROJECT_NAME)
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.setMinimumSize(MINIMUM_RESOLUTION[0], MINIMUM_RESOLUTION[1])
        self.resize(0.8 * self.screen.width(), 0.8 * self.screen.height())
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Create main layout
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.setCentralWidget(self.main_widget)

        # Main splitter
        self.main_splitter = QtWidgets.QSplitter()
        self.main_splitter.setOrientation(QtCore.Qt.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([int(0.8*self.height()), int(0.2*self.height())])

        self.main_layout.addWidget(self.main_splitter)

        # Init widgets and add to main window
        self._init_menu()
        self._init_tabs()
        self._init_logger()
        self.connect_tabs()

        # Show welcome message
        self.handle_messages("Hello and welcome to a simple and easy to use testbeam analysis!", 4000)

    def _init_tabs(self):
        """
        Initializes the tabs for the analysis window
        """

        # Add tab_widget and widgets for the different analysis steps
        self.tab_order = ('Files', 'Setup', 'Noisy Pixel', 'Clustering', 'Pre-alignment', 'Track finding',
                          'Alignment', 'Track fitting', 'Residuals', 'Efficiency')

        # Add QTabWidget for tab_widget
        self.tabs = QtWidgets.QTabWidget()

        # Initialize each tab
        for name in self.tab_order:
            if name == 'Files':
                widget = DataTab(parent=self.tabs)
            else:
                # Add dummy widget
                widget = QtWidgets.QWidget(parent=self.tabs)

            self.tw[name] = widget
            self.tabs.addTab(self.tw[name], name)

        # Disable all tabs but DataTab. Enable tabs later via self.enable_tabs()
        if not _DEBUG:
            self.handle_tabs(enable=False)
        else:
            self.handle_tabs(enable=True)

        # Add to main layout
        self.main_splitter.addWidget(self.tabs)

    def _init_logger(self, init=True):
        """
        Initializes a custom logging handler for analysis and set its
        visibility to False. The logger can be shown/hidden via the
        appearance menu in the GUI or closed button
        """

        if init:

            # Set logging level
            logging.getLogger().setLevel(logging.INFO)

            # Create logger instance
            self.logger = AnalysisLogger(self.main_widget)
            self.logger.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            # Add custom logger
            logging.getLogger().addHandler(self.logger)

            # Connect logger signal to logger console
            AnalysisStream.stdout().messageWritten.connect(lambda msg: self.logger_console.appendPlainText(msg))
            AnalysisStream.stderr().messageWritten.connect(lambda msg: self.logger_console.appendPlainText(msg))

        # Add widget to display log and add it to dock
        # Widget to display log in, we only want to read log
        self.logger_console = QtWidgets.QPlainTextEdit()
        self.logger_console.setReadOnly(True)

        # Dock in which text widget is placed to make it closable without losing log content
        self.console_dock = QtWidgets.QDockWidget()
        self.console_dock.setWidget(self.logger_console)
        self.console_dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.console_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable)
        self.console_dock.setWindowTitle('Logger')

        # Set visibility to false at init
        self.console_dock.setVisible(False)

        # Add to main layout
        self.main_splitter.addWidget(self.console_dock)

        logging.info('Started "testbeam_analysis" on %s' % platform.system())

    def _init_menu(self):
        """
        Initialize the menubar of the AnalysisWindow
        """

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.file_quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.file_menu.addAction('&New', self.new_analysis,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_N)
        self.menuBar().addMenu(self.file_menu)

        self.settings_menu = QtWidgets.QMenu('&Settings', self)
        self.settings_menu.addAction('&Global', self.global_settings)
        self.menuBar().addMenu(self.settings_menu)

        self.run_menu = QtWidgets.QMenu('&Run', self)
        self.run_menu.setToolTipsVisible(True)
        self.run_menu.addAction('&Run consecutive analysis', self.run_consecutive_analysis, QtCore.Qt.CTRL + QtCore.Qt.Key_R)
        # Disable consecutive analysis until setup is done
        self.run_menu.actions()[0].setEnabled(False)
        self.run_menu.actions()[0].setToolTip('Finish data selection and testbeam setup to enable')
        self.menuBar().addMenu(self.run_menu)

        self.appearance_menu = QtWidgets.QMenu('&Appearance', self)
        self.appearance_menu.setToolTipsVisible(True)
        self.appearance_menu.addAction('&Show/hide logger', self.handle_logger, QtCore.Qt.CTRL + QtCore.Qt.Key_L)
        self.appearance_menu.addAction('&Show current analysis tab', self.view_current_tab, QtCore.Qt.CTRL + QtCore.Qt.Key_C)
        # Disable until setup is done
        self.appearance_menu.actions()[1].setEnabled(False)
        self.appearance_menu.actions()[1].setToolTip('Finish data selection and testbeam setup to enable')
        self.menuBar().addMenu(self.appearance_menu)

        self.session_menu = QtWidgets.QMenu('&Session', self)
        self.session_menu.addAction('&Save', self.save_session, QtCore.Qt.CTRL + QtCore.Qt.Key_S)
        self.session_menu.addAction('&Load', self.load_session, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.menuBar().addMenu(self.session_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.help_menu.addAction('&About', self.about)
        self.help_menu.addAction('&Documentation', self.open_docu)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    "Version\n%s.\n\n"
                                    "Authors\n%s\n\n"
                                    "GUI authors\n%s" % (testbeam_analysis.VERSION,
                                                         AUTHORS.replace(', ', '\n'),
                                                         GUI_AUTHORS.replace(', ', '\n')))

    def open_docu(self):
        link = r'https://silab-bonn.github.io/testbeam_analysis/'
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(link))

    def handle_messages(self, message, ms):
        """
        Handles messages from the tabs shown in QMainWindows statusBar
        """

        self.statusBar().showMessage(message, ms)

    def handle_logger(self):
        """
        Handle whether logger is visible or not
        """

        if self.console_dock.isVisible():
            self.console_dock.setVisible(False)
        else:
            self.console_dock.setVisible(True)

    def handle_tabs(self, tabs=None, enable=True):
        """
        Enables/Disables a specific tab with name 'names' or loops over list of tab names to en/disable them
        """

        if _DEBUG:
            return

        # Dis/enable all tabs but Files
        if tabs is None:
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) != 'Files':
                    self.tabs.setTabEnabled(i, enable)

        # Dis/enable specific tab
        elif type(tabs) is unicode:
            if tabs in self.tab_order:
                self.tabs.setTabEnabled(self.tab_order.index(tabs), enable)

        # Dis/enable several tabs
        else:
            for tab in tabs:
                if tab in self.tab_order:
                    self.tabs.setTabEnabled(self.tab_order.index(tab), enable)

    def connect_tabs(self, tabs=None):
        """
        Connect statusMessage and proceedAnalysis signal of all tabs
        """

        if tabs is None:
            tab_list = self.tab_order
        else:
            if isinstance(tabs, list):
                tab_list = tabs
            else:
                tab_list = [tabs]

        for name in tab_list:
            try:
                if name == 'Files':
                    for x in [lambda: self.update_tabs(tabs='Setup'),
                              lambda: self.tw['Setup'].input_data(self.tw['Files'].data),
                              lambda: self.tabs.setCurrentIndex(self.tabs.currentIndex() + 1)]:
                        self.tw[name].proceedAnalysis.connect(x)
                    self.tw[name].statusMessage.connect(lambda message: self.handle_messages(message, 4000))

                if name == 'Setup':
                    msg_0 = 'Run consecutive analysis with default options without user interaction'
                    msg_1 = 'Go to currently running or next to be running analysis tab'
                    for xx in [lambda: self.update_tabs(data=self.tw['Setup'].data, skip='Setup'),
                               lambda: self.appearance_menu.actions()[1].setEnabled(True),
                               lambda: self.appearance_menu.actions()[1].setToolTip(msg_1),
                               lambda: self.tabs.setCurrentIndex(self.tabs.currentIndex() + 1),
                               lambda: self.run_menu.actions()[0].setEnabled(True),  # Enable consecutive analysis
                               lambda: self.run_menu.actions()[0].setToolTip(msg_0)]:
                        self.tw[name].proceedAnalysis.connect(xx)
                    self.tw[name].statusMessage.connect(lambda message: self.handle_messages(message, 4000))

                if name == 'Alignment':
                    for xxx in [lambda: self.update_tabs(data={'skip_alignment': True},
                                                         tabs=['Track fitting', 'Residuals', 'Efficiency'])]:
                        self.tw[name].skipAlignment.connect(xxx)

                self.tw[name].proceedAnalysis.connect(lambda tab_names: self.handle_tabs(tabs=tab_names))
                self.tw[name].proceedAnalysis.connect(lambda tab_names: self.tab_completed(tab_names))
                self.tw[name].exceptionSignal.connect(lambda e, trc_bck, tab, cause: self.handle_exceptions(exception=e,
                                                                                                            trace_back=trc_bck,
                                                                                                            tab=tab,
                                                                                                            cause=cause))

            except (AttributeError, KeyError) as e:
                if _DEBUG:
                    logging.warning(e.message)
                else:
                    pass

    def update_tabs(self, data=None, tabs=None, skip=None):
        """
        Updates the setup and options with data from the SetupTab and then updates the tabs

        :param tabs: list of strings with tab names that should be updated, if None update all
        :param data: dict with all information necessary to perform analysis, if None only update tabs
        :param skip: str or list of tab names which should be skipped when updating tabs
        """

        # Save users current tab position
        current_tab = self.tabs.currentIndex()

        if data is not None:
            for key in data:

                # Store setup data in self.setup and everything else in self.options
                if key in self.setup.keys():
                    self.setup[key] = data[key]
                else:
                    self.options[key] = data[key]

        if tabs is None:
            update_tabs = list(self.tab_order)
        else:
            if isinstance(tabs, list):
                update_tabs = tabs
            else:
                update_tabs = [tabs]

        if skip is not None:
            if isinstance(skip, list):
                for t_name in skip:
                    if t_name in update_tabs:
                        update_tabs.remove(t_name)
            else:
                if skip in update_tabs:
                    update_tabs.remove(skip)

        # Remove tabs from being updated if they are already finished
        for t in self.tab_order:
            try:
                if self.tw[t].isFinished:
                    if t in update_tabs:
                        update_tabs.remove(t)
            except AttributeError:
                pass

        # Make temporary dict for updated tabs
        tmp_tw = {}
        for name in update_tabs:

            if name == 'Setup':
                widget = SetupTab(parent=self.tabs)

            elif name == 'Noisy Pixel':
                widget = tab_widget.NoisyPixelsTab(parent=self.tabs,
                                                   setup=self.setup,
                                                   options=self.options,
                                                   name=name,
                                                   tab_list='Clustering')
            elif name == 'Clustering':
                widget = tab_widget.ClusterPixelsTab(parent=self.tabs,
                                                     setup=self.setup,
                                                     options=self.options,
                                                     name=name,
                                                     tab_list='Pre-alignment')

            elif name == 'Pre-alignment':
                widget = tab_widget.PrealignmentTab(parent=self.tabs,
                                                    setup=self.setup,
                                                    options=self.options,
                                                    name=name,
                                                    tab_list='Track finding')

            elif name == 'Track finding':
                widget = tab_widget.TrackFindingTab(parent=self.tabs,
                                                    setup=self.setup,
                                                    options=self.options,
                                                    name=name,
                                                    tab_list='Alignment')
            elif name == 'Alignment':
                widget = tab_widget.AlignmentTab(parent=self.tabs,
                                                 setup=self.setup,
                                                 options=self.options,
                                                 name=name,
                                                 tab_list='Track fitting')
            elif name == 'Track fitting':
                widget = tab_widget.TrackFittingTab(parent=self.tabs,
                                                    setup=self.setup,
                                                    options=self.options,
                                                    name=name,
                                                    tab_list=['Residuals', 'Efficiency'])
            elif name == 'Residuals':
                widget = tab_widget.ResidualTab(parent=self.tabs,
                                                setup=self.setup,
                                                options=self.options,
                                                name=name,
                                                tab_list='Efficiency')
            elif name == 'Efficiency':
                widget = tab_widget.EfficiencyTab(parent=self.tabs,
                                                  setup=self.setup,
                                                  options=self.options,
                                                  name=name,
                                                  tab_list='Last')  # Random string for last tab, NOT in self.tab_order
            else:
                continue

            tmp_tw[name] = widget

        for tab in self.tab_order:
            if tab in tmp_tw.keys():

                # Replace tabs in self.tw with updated tabs
                self.tw[tab] = tmp_tw[tab]

                # Get tab status of tab which is updated to set status of updated tab
                enable = self.tabs.isTabEnabled(self.tab_order.index(tab))

                # Remove old tab, insert updated tab at same index and set status
                self.tabs.removeTab(self.tab_order.index(tab))
                self.tabs.insertTab(self.tab_order.index(tab), self.tw[tab], tab)
                self.tabs.setTabEnabled(self.tab_order.index(tab), enable)

        # Set the tab index to stay at the same tab after replacing old tabs
        self.tabs.setCurrentIndex(current_tab)

        # Connect updated tabs
        self.connect_tabs(update_tabs)  # tabs

    def tab_completed(self, tabs):

        tab = None

        # Sender is the one completed so only first dut in tabs matters
        if isinstance(tabs, list):

            sorted_tabs = {}
            for dut in tabs:
                if dut in self.tab_order:
                    sorted_tabs[self.tab_order.index(dut)] = dut

            if sorted_tabs:
                tab = sorted_tabs[min(sorted_tabs.iterkeys())]

        elif isinstance(tabs, unicode):
            tab = tabs

        if tab in self.tab_order:
            # Set icon for sender which is the one before tab
            self.tabs.setTabIcon(self.tab_order.index(tab) - 1, self.icon_complete)

        else:
            # Set icon for last tab
            self.tabs.setTabIcon(len(self.tab_order) - 1, self.icon_complete)

    def global_settings(self):
        """
        Creates a child SettingsWindow of the analysis window to change global settings
        """
        self.settings_window = SettingsWindow(self.setup, self.options, parent=self)
        self.settings_window.show()
        self.settings_window.settingsUpdated.connect(lambda: self.update_globals())

    def update_globals(self):
        """
        Updates the global settings which are applied in the SettingsWindow
        """

        self.setup = self.settings_window.setup
        self.options = self.settings_window.options

        try:
            if self.tw['Setup'].isFinished:
                self.update_tabs()  # skip='Setup'
        except AttributeError:
            pass

    def save_session(self):
        """
        Creates a child SessionWindow of the analysis window to save current session
        """
        caption = 'Save session'
        session = QtWidgets.QFileDialog.getSaveFileName(parent=self,
                                                        caption=caption,
                                                        directory='./sessions',
                                                        filter='*.yaml')[0]

        message = 'Congratulations! Your session would have been saved in %s' \
                  ' ...if we had implemented saving sessions, which we have not.' % session
        logging.info(message)
        pass

    def load_session(self):
        """
        Opens dialog to select previously saved session. Must be yaml-file and lie in ./sessions
        """
        caption = 'Load session'
        session = QtWidgets.QFileDialog.getOpenFileName(parent=self,
                                                        caption=caption,
                                                        directory='./sessions',
                                                        filter='*.yaml')[0]

        message = 'Congratulations! Your session would have been loaded from %s' \
                  ' ...if we had implemented loading sessions, which we have not.' % session
        logging.info(message)
        pass

    def new_analysis(self):
        """
        Restores the initial state of the AnalysisWindow to re-start analysis
        """

        # Get default settings
        self.setup = SettingsWindow().default_setup
        self.options = SettingsWindow().default_options

        # Make variable for SettingsWindow
        self.settings_window = None

        # Make dict to access tab widgets
        self.tw = {}

        # Disable consecutive analysis until setup is done
        self.run_menu.actions()[0].setEnabled(False)
        self.run_menu.actions()[0].setToolTip('Finish data selection and testbeam setup to enable')

        for i in reversed(range(self.main_splitter.count())):
            w = self.main_splitter.widget(i)
            w.hide()
            w.deleteLater()

        # Remove progressbar of consecutive analysis if there is one
        try:
            self.remove_widget(widget=self.widget_rca, layout=self.main_layout)

        # RuntimeError if progressbar has been removed previously
        except (AttributeError, RuntimeError):
            pass

        self._init_tabs()
        self.connect_tabs()
        self._init_logger(init=False)
        self.tabs.setCurrentIndex(0)

    def run_consecutive_analysis(self):
        """
        Method to start a consecutive call of all analysis functions with their default values
        as defined in tab_widget.py. Acronym rca==run constructive analysis
        """
        if self.tw[self.current_analysis_tab()].analysis_thread.isRunning():
            msg = 'Can not start consecutive analysis while %s analysis is running.' % self.current_analysis_tab()
            logging.warning(msg=msg)
            return

        # Whenever starting rca restore flag state
        self.flag_interrupt = False

        # Make sub-layout for consecutive analysis progressbar with label
        self.widget_rca = QtWidgets.QWidget()
        self.layout_rca = QtWidgets.QHBoxLayout()
        self.widget_rca.setLayout(self.layout_rca)
        self.main_layout.addWidget(self.widget_rca)

        # Make widgets to fill rca layout
        self.label_rca = QtWidgets.QLabel('Running consecutive analysis...')
        self.p_bar_rca = QtWidgets.QProgressBar()
        self.p_bar_rca.setRange(0, len(self.tab_order))
        self.btn_interrupt_rca = QtWidgets.QPushButton('Interrupt')
        self.btn_interrupt_rca.setToolTip('Interrupt consecutive analysis after finishing current analysis tab')
        self.btn_interrupt_rca.clicked.connect(lambda: handle_rca(interrupt=True))
        self.layout_rca.addWidget(self.label_rca)
        self.layout_rca.addWidget(self.p_bar_rca)
        self.layout_rca.addWidget(self.btn_interrupt_rca)

        # Get starting tab
        self.starting_tab_rca = self.current_analysis_tab()

        for tab in self.tab_order:

            # Connect starting tab and all following
            if self.tab_order.index(tab) >= self.tab_order.index(self.starting_tab_rca):

                # Handle consecutive analysis
                self.tw[tab].proceedAnalysis.connect(lambda tab_list: handle_rca(tab_list))

                # Check whether or not alignment is skipped
                if tab == 'Track finding' and self.options['skip_alignment']:
                    for x in [lambda tab_list: self.tw[tab_list[0]].skipAlignment.emit(),
                              lambda tab_list: self.tw[tab_list[0]].proceedAnalysis.emit(self.tw[tab_list[0]].tl)]:
                        self.tw[tab].proceedAnalysis.connect(x)

                else:
                    if tab != self.tab_order[-1]:
                        self.tw[tab].proceedAnalysis.connect(lambda tab_list: self.tw[tab_list[0]].btn_ok.clicked.emit())

        # Start analysis by clicking ok button on starting tab
        self.tw[self.starting_tab_rca].btn_ok.clicked.emit()
        self.p_bar_rca.setValue(self.tab_order.index(self.starting_tab_rca))
        self.p_bar_rca.setFormat(self.starting_tab_rca)

        def handle_rca(tab_list=None, interrupt=False):
            """
            Helper function to run consecutive analysis
            :param tab_list: list or str of tab name whichs analysis step is to be started
            :param interrupt: bool whether interrupt btn was clicked
            """

            # Redundant call
            if tab_list is None and not interrupt:
                return

            # If interrupt btn was clicked set flag
            if interrupt:
                self.flag_interrupt = True
                self.btn_interrupt_rca.setDisabled(True)

                # Get analysis tab that is currently doing analysis step
                current_analysis = self.current_analysis_tab()

                self.label_rca.setText('Finishing %s...' % current_analysis)
                self.p_bar_rca.setDisabled(True)

            else:

                if self.flag_interrupt:

                    # Disconnect and re-connect tabs
                    for tab_name in self.tw.keys():
                        self.tw[tab_name].proceedAnalysis.disconnect()
                    self.connect_tabs()

                    # Remove consecutive analysis progressbar
                    self.remove_widget(widget=self.widget_rca, layout=self.main_layout)

                else:
                    if isinstance(tab_list, list):
                        tab_name = tab_list[0]
                    else:
                        tab_name = tab_list

                    if tab_name in self.tab_order:

                        # Update progressbar
                        self.p_bar_rca.setFormat(tab_name)
                        self.p_bar_rca.setValue(self.tab_order.index(tab_name))

                    else:
                        # Last tab finished
                        self.p_bar_rca.setValue(len(self.tab_order))
                        self.label_rca.setText('Done!')
                        # Remove consecutive analysis progressbar
                        self.remove_widget(widget=self.widget_rca, layout=self.main_layout)

    def handle_exceptions(self, exception, trace_back, tab, cause):
        """
        Handles exceptions which are raised on sub-thread where "ViTables" or analysis is done.
        Re-raises unexpected exceptions and and handles expected ones.

        :param exception: Any Exception
        :param trace_back: traceback of exception
        :param tab: analysis tab
        :param cause: "vitables" or "analysis"
        """

        # Make list of expected exceptions. Under Windows missing ViTables will produce WindowsError.
        # WindowsError will raise NameError under Linux
        try:
            expected_exceptions = [OSError, ImportError, WindowsError]
        except NameError:
            expected_exceptions = [OSError, ImportError]

        # If vitables raises exception, only disable button and log
        if type(exception) in expected_exceptions and cause == 'vitables':

            msg = 'ViTables not found. Try installing ViTables'
            self.tw[tab].btn_ok.setToolTip('Try installing or re-installing ViTables')
            self.tw[tab].btn_ok.setText('ViTables not found')
            self.tw[tab].btn_ok.setDisabled(True)
            logging.error(msg)

        else:

            # Set index to tab where exception occurred
            self.tabs.setCurrentIndex(self.tab_order.index(tab))

            # Make instance of exception window
            self.exception_window = ExceptionWindow(exception=exception, trace_back=trace_back,
                                                    tab=tab, cause=cause, parent=self)
            self.exception_window.show()
            self.exception_window.exceptionRead.connect(lambda: self.update_tabs(tabs=tab))

            # Remove progressbar of consecutive analysis if there is one
            try:
                self.remove_widget(widget=self.widget_rca, layout=self.main_layout)

            # RuntimeError if progressbar has been removed previously
            except (AttributeError, RuntimeError):
                pass

    def check_resolution(self):
        """
        Checks for resolution and gives pop-up warning if too low
        """

        # Show message box with warning if screen resolution is lower than required
        if self.screen.width() < MINIMUM_RESOLUTION[0] or self.screen.height() < MINIMUM_RESOLUTION[1]:
            msg = "Your screen resolution (%d x %d) is below the required minimum resolution of %d x %d." \
                  " This may affect the appearance!" % (self.screen.width(), self.screen.height(),
                                                        MINIMUM_RESOLUTION[0], MINIMUM_RESOLUTION[1])
            title = "Screen resolution low"
            msg_box = QtWidgets.QMessageBox.information(self, title, msg, QtWidgets.QMessageBox.Ok)

        else:
            pass

    def remove_widget(self, widget, layout):
        """
        Removes a widget with all its child widgets from layout
        :param widget: QtWidget.QWidget
        :param layout: QtWidget.QLayout
        """
        for i in reversed(range(widget.layout().count())):
            item = widget.layout().itemAt(i)
            item.widget().deleteLater()

        layout.removeWidget(widget)
        widget.deleteLater()

    def current_analysis_tab(self):
        """
        Returns the currently running or next to be running analysis tab
        """

        current_tab = None

        for tab in self.tab_order:

            tab_index = self.tab_order.index(tab)

            if tab not in ['Files', 'Setup']:

                # Get current tab of analysis
                if self.tabs.isTabEnabled(tab_index) and not self.tabs.isTabEnabled(tab_index + 1):
                    current_tab = tab
                else:
                    pass

        # All tabs are finished
        if not current_tab:
            current_tab = self.tab_order[-1]

        return current_tab

    def view_current_tab(self):
        """
        Goes to currently running or next to be running analysis tab
        """
        self.tabs.setCurrentIndex(self.tab_order.index(self.current_analysis_tab()))

    def file_quit(self):
        self.close()

    def closeEvent(self, _):
        self.file_quit()

def main():
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont()
    font.setPointSize(11)
    app.setFont(font)
    aw = AnalysisWindow()
    aw.show()
    aw.check_resolution()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
