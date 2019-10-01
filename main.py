#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Application

License: GPL version 3
Ulrich Lukas 2019-07-29
"""
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# FIXME: Set levels for debug only
logging.getLogger("matplotlib.axes._base").setLevel(logging.INFO)
logging.getLogger("matplotlib.pyplot").setLevel(logging.INFO)
logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)
logger = logging.getLogger("MainWindow")

import os
import pickle

import numpy as np
from numpy import NaN

from PyQt5.QtCore import Qt, QDir, QSize, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
        QMainWindow, QApplication, QAction, QStyle, QFileDialog,
        )

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from digitizer import Digitizer
from default_configuration import APPLICATION, DATA_MODEL, TRACE, X_AXIS, Y_AXIS

from embedded_ipython import EmbeddedIPythonKernel

class RuntimeConfig():
    """Modified configuration settings and application state can be written to
    file for persistent storage
    """
    def __init__(self):
        self.app_conf = APPLICATION()
        self.model_conf = DATA_MODEL()
        self.trace_conf = TRACE()
        self.x_ax_conf = X_AXIS()
        self.y_ax_conf = Y_AXIS()

        self.x_ax_state = None
        self.y_ax_state = None

    def load_from_configfile(self):
        try:
            with open(self.app_conf.config_file_name, "rb") as f:
                vars(self).update(pickle.load(f))
        except FileNotFoundError:
            logger.info("Config file not found, using defaults...")
        except IOError as e:
            logger.critical("Error loading config file: ", e)
 
    def store_to_configfile(self):
        logger.info("Storing configuration...")
        try:
            with open(self.app_conf.config_file_name, "wb") as f:
                pickle.dump(vars(self), f)
        except IOError as e:
            logger.error("Error saving config file: ", e)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load configuration either from defaults or from config file
        self.conf = conf = RuntimeConfig()
        conf.load_from_configfile()

        # Working directory stored in config file
        self.set_wdir(conf.app_conf.wdir)
        self.set_last_image_file(conf.app_conf.last_image_file)
        self.setMinimumSize(QSize(*conf.app_conf.window_size))
        self.setWindowTitle("Plot Workbench -- Main Window -- "
                            "Digitize, Interpolate, Optimize, Approximate")

        ########## Central Widget
        self.cw = Digitizer(self, conf)
        self.setCentralWidget(self.cw)
        ########## Main window toolbar
        self.main_tb = MainToolbar(self.cw.mplw.canvas_qt, self)
        self.addToolBar(self.main_tb)
        ########## Custom Dialogs
        self.dlg_open_image = QFileDialog(
                self,
                "Open Source Image",
                self.wdir,
                "Images (*.png *.jpg *.jpeg)")
        self.dlg_export_csv = QFileDialog(
                self, "Export CSV", self.wdir, "Text/CSV (*.csv *.txt)")
        self.dlg_export_xlsx = QFileDialog(
                self, "Export XLS/XLSX", self.wdir, "Excel (*.xlsx)")
        
        ########## Embedded IPython Kernel and Jupyter Console Launcher
        self.ipyconsole = EmbeddedIPythonKernel(self)

        ########## Connect main window signals
        # Main toolbar signals
        self.main_tb.act_load_clipboard.triggered.connect(
                self.cw.load_clipboard_image)
        self.main_tb.act_open.triggered.connect(self.dlg_open_image.open)
        self.main_tb.act_export_csv.triggered.connect(self.dlg_export_csv.open)
        self.main_tb.act_export_xlsx.triggered.connect(
                self.dlg_export_xlsx.open)
        self.main_tb.act_put_clipboard.triggered.connect(self.cw.put_clipboard)
        
        # Embedded Jupyter Console Button signal
        self.cw.btn_console.clicked.connect(
                self.ipyconsole.launch_jupyter_console_process)

        # Dialog box signals
        self.dlg_open_image.fileSelected.connect(self.cw.mplw.load_image)
        self.dlg_open_image.fileSelected.connect(self.set_last_image_file)
        self.dlg_open_image.directoryEntered.connect(self.set_wdir)
        self.dlg_export_csv.fileSelected.connect(self.cw.export_csv)
        
        # Reopen last file if requested and if it exists
        if self.last_image_file != "":
            self.cw.mplw.load_image(self.last_image_file)
        

    def closeEvent(self, event):
        """closeEvent() inherited from QMainWindow, reimplemented here.
        This saves current configuration and axes properties if flag is set.
        """
        # Store axis configuration if requested
        if self.cw.model.wants_persistent_storage:
            logger.info("Storing axis configuration to disk..")
            self.conf.model_conf.wants_persistent_storage = True
            # Getting plot configuration from Digitizer widget:
            self.conf.x_ax_state = self.cw.model.x_ax.restorable_state()
            self.conf.y_ax_state = self.cw.model.y_ax.restorable_state()
        else:
            self.conf.x_ax_state = None
            self.conf.y_ax_state = None
        # Save working directory
        self.conf.app_conf.wdir = self.wdir
        self.conf.app_conf.last_image_file = self.last_image_file
        # Store all configuration
        self.conf.store_to_configfile()
        self.ipyconsole.shutdown_kernel()
        event.accept()

    @pyqtSlot(str)
    def set_wdir(self, abs_path):
        # Set working directory to last opened file directory
        self.wdir = abs_path if os.path.isdir(abs_path) else QDir.homePath()

    @pyqtSlot(str)
    def set_last_image_file(self, abs_path):
        # This property will be reloaded if persistent_storage property of
        # digitizer instance is set. Then, the input image file will be
        # re-loaded on next start
        self.last_image_file = abs_path if os.path.isfile(abs_path) else ""


class MainToolbar(NavigationToolbar2QT):
    # When coordinates display is enabled, this has influence on the toolbar
    # sizeHint, which causes trouble in case of vertical orientation.
    def __init__(self, canvas, parent, coordinates=False):
        super().__init__(canvas, parent, coordinates)
        ########## Patch original API action buttons with new text etc:
        for act in self.actions():
            text = act.text()
            # We want to insert our buttons before the external matplotlib
            # API buttons where the "Home" is the leftmost
            if text == "Home":
                act.setText("Reset Zoom")
                api_home = act
            # The matplotlib save button only saves a screenshot thus it should
            # be appropriately renamed
            elif text == "Save":
                act.setText("Save as Image")
                api_save = act
            elif text == "Customize":
                act.setText("Figure Options")
                api_customize = act            
            elif text in ("Back", "Forward", "Subplots"):
                self.removeAction(act)
        api_actions = {api_home: "Reset{}Zoom",
                       api_save: "Save{}Image",
                       api_customize: "Figure{}Options",
                       }
        ########## Define new actions
        icon_open = QIcon.fromTheme(
                "document-open",
                self.style().standardIcon(QStyle.SP_DialogOpenButton))
        icon_export = QIcon.fromTheme(
                "document-save",
                self.style().standardIcon(QStyle.SP_DialogSaveButton))
        icon_send = QIcon.fromTheme(
                "document-send",
                self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.act_put_clipboard = QAction(
                icon_send,
                "Put to Clipboard",
                self)
        self.act_export_xlsx = QAction(
                icon_export,
                "Export data as XLSX",
                self)
        self.act_export_csv = QAction(
                icon_export,
                "Export data as CSV",
                self)
        self.act_open = QAction(
                icon_open,
                "Open an image file",
                self)
        self.act_load_clipboard = QAction(
                icon_open,
                "Load Image from Clipboard",
                self)
        # Dict of our new actions plus text
        self._custom_actions = {self.act_load_clipboard: "From{}Clipboard",
                                self.act_open: "Open File",
                                self.act_export_csv: "Export{}CSV",
                                self.act_export_xlsx: "Export{}XLSX", 
                                self.act_put_clipboard: "Put Into{}Clipboard",
                                }
        # Separator before first external API buttons
        sep = self.insertSeparator(api_home)
        ########## Add new actions to the toolbar
        self.insertActions(sep, self._custom_actions.keys())
        ########## Add original buttons to the dict of all custom actions
        self._custom_actions.update(api_actions)
        ########## Connect own signals
        self.orientationChanged.connect(self._on_orientationChanged)
        ########## Initial view setup
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._on_orientationChanged(Qt.Horizontal)
        #self.setIconSize(self.iconSize() * 0.8)
        self.setStyleSheet("spacing:2px")
        

    def sizeHint(self):
        # Matplotlib returns a minimum of 48 by default, we don't want this
        # size.setHeight(max(48, size.height()))
        return super(NavigationToolbar2QT, self).sizeHint()
    
    @pyqtSlot(Qt.Orientation)
    def _on_orientationChanged(self, new_orientation):
        logger.debug("Main Toolbar orientation change handler called")
        if new_orientation == Qt.Horizontal:
            self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            # Set text without line break
            for act, text in self._custom_actions.items():
                act.setIconText(text.format(" "))
        else:
            self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            # Set text with line break
            for act, text in self._custom_actions.items():
                act.setIconText(text.format("\n"))


if __name__ == "__main__":
    # When run interactively in IPython shell, app instance might already exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    mainw = MainWindow()
    mainw.show()
    mainw.activateWindow()

    # Set up some shortcuts for command line interactive use.
    mplw = mainw.cw.mplw
    ax = mainw.cw.mplw.mpl_ax
    fig = mainw.cw.mplw.fig
    cv = mainw.cw.mplw.canvas_qt
    redraw = mainw.cw.mplw.canvas_qt.draw_idle
    print_model_view_items = mainw.cw.mplw.print_model_view_items

    model = mainw.cw.model
    traces = model.traces
    tr1, tr2, tr3 = model.traces[0:3]

    # When running from ipython shell, use its Qt GUI event loop
    # integration. Otherwise, start embedded IPython kernel.
    # If import not available, start pyqt event loop via app.exec_()
    try:
        from IPython import get_ipython
        ipy_inst = get_ipython()
        if ipy_inst is not None:
            # Running in IPython interactive shell. Configure Qt integration
            ipy_inst.run_line_magic("gui", "qt5")
            ipy_inst.run_line_magic("matplotlib", "qt5")
        else:
            # Launch embedded IPython Kernel and start GUI event loop
            mainw.ipyconsole.start_ipython_kernel(locals(), gui="qt5")
    except ImportError:
        # Run normal Qt app without any IPython integration
        sys.exit(app.exec_())

