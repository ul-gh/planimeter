#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Application

License: GPL version 3
Ulrich Lukas 2019-07-29
"""
import os
import sys
import pickle

import numpy as np

from PyQt5.QtCore import Qt, QDir, QSize, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
        QMainWindow, QApplication, QAction, QStyle, QFileDialog,
        )

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from upylib.pyqt import ipython_pyqt_boilerplate
from upylib.pyqt_debug import patch_pyqt_event_exception_hook

from digitizer import Digitizer
from default_configuration import APPLICATION, DATA_MODEL, TRACE, X_AXIS, Y_AXIS


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
            print("Config file not found, using defaults...")
        except IOError as e:
            print("Error loading config file: ", e)
 
    def store_to_configfile(self):
        print("Storing configuration...")
        try:
            with open(self.app_conf.config_file_name, "wb") as f:
                pickle.dump(vars(self), f)
        except IOError as e:
            print("Error saving config file: ", e)


class MainToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        for act in self.actions():
            # We want to insert our buttons before the external matplotlib
            # API buttons where the "Home" is the leftmost
            if act.text() == "Home":
                api_first_action = act
            # The matplotlib save button only saves a screenshot thus it should
            # be appropriately renamed
            if act.text() == "Save":
                act.setText("Screenshot")
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
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
            icon_send, "Put to Clipboard", self, iconText="Put Into\nClipboard")
        self.act_export_xlsx = QAction(
            icon_export, "Export data as XLSX", self, iconText="Export XLSX")
        self.act_export_csv = QAction(
            icon_export, "Export data as CSV", self, iconText="Export CSV")
        self.act_open = QAction(
            icon_open, "Open an image file", self, iconText="Open File")
        # Separator before first external API buttons
        sep = self.insertSeparator(api_first_action)
        # Inserting new buttons
        self.insertActions(sep, [self.act_open, self.act_export_csv,
                                 self.act_export_xlsx, self.act_put_clipboard])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load configuration either from defaults or from config file
        self.conf = conf = RuntimeConfig()
        conf.load_from_configfile()

        # Working directory stored in config file might not exist
        self.wdir = (conf.app_conf.wdir
                     if os.path.exists(conf.app_conf.wdir)
                     else QDir.homePath())
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
            self, "Open Source Image", self.wdir, "Images (*.png *.jpg *.jpeg)")
        self.dlg_export_csv = QFileDialog(
            self, "Export CSV", self.wdir, "Text/CSV (*.csv *.txt)")
        self.dlg_export_xlsx = QFileDialog(
            self, "Export XLS/XLSX", self.wdir, "Excel (*.xlsx)")

        ########## Connect main window signals
        # Main toolbar signals
        self.main_tb.act_open.triggered.connect(self.dlg_open_image.open)
        self.main_tb.act_export_csv.triggered.connect(self.dlg_export_csv.open)
        self.main_tb.act_export_xlsx.triggered.connect(
            self.dlg_export_xlsx.open)
        self.main_tb.act_put_clipboard.triggered.connect(self.cw.put_clipboard)

        # Dialog box signals
        self.dlg_open_image.fileSelected.connect(self.cw.load_image)
        self.dlg_open_image.directoryEntered.connect(self.set_wdir)
        self.dlg_export_csv.fileSelected.connect(self.cw.on_dlg_export_csv)

    def closeEvent(self, event):
        """closeEvent() inherited from QMainWindow, reimplemented here.
        This saves current configuration and axes properties if flag is set.
        """
        # Store axis configuration if requested
        if self.cw.model.store_ax_conf:
            print("Storing axis configuration to disk..")
            self.conf.model_conf.store_ax_conf = True
            # Getting plot configuration from Digitizer widget:
            self.conf.x_ax_state = self.cw.model.x_ax._get_state()
            self.conf.y_ax_state = self.cw.model.y_ax._get_state()
        else:
            self.conf.x_ax_state = None
            self.conf.y_ax_state = None
        # Save working directory
        self.conf.app_conf.wdir = self.wdir
        # Store all configuration
        self.conf.store_to_configfile()
        event.accept()

    @pyqtSlot(str)
    def set_wdir(self, abs_path):
        """Set working directory to last opened file directory"""
        self.wdir = abs_path


if __name__ == "__main__":
    # When run interactively in IPython shell, app instance might already exist
    if "app" not in globals() or not isinstance(app, QApplication):
        app = QApplication(sys.argv)

    # Debug: Enable display of exceptions occurring in pyqtSlots:
    patch_pyqt_event_exception_hook(app)

    mainw = MainWindow()
    mainw.show()
    mainw.activateWindow()
    
    # IPython integration via IPython Qt5 event loop
    #ipython_pyqt_boilerplate(app)

    # Set up some shortcuts for command line interactive use.
    ax = mainw.cw.mplw.mpl_ax
    fig = mainw.cw.mplw.fig
    cv = mainw.cw.mplw.canvas_qt
    redraw = mainw.cw.mplw.canvas_qt.draw_idle

    model = mainw.cw.model
    traces = model.traces
    tr1, tr2, tr3 = model.traces[0:3]

    mainw.cw.console.push_vars(locals())
    app.exec_()

