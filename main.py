#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pickle

from PyQt5.QtCore import Qt, QDir, QSize, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
        QMainWindow, QApplication, QAction, QStyle, QFileDialog,
        )

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from upylib.pyqt import ipython_pyqt_boilerplate, get_qobj_base_state
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

        # Matplotlib toolbar, we are now adding more buttons to it
        self.main_tb = NavigationToolbar2QT(self.cw.mplw.canvas_qt, self)
        self.main_tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        icon_open = QIcon.fromTheme(
            "document-open",
            self.style().standardIcon(QStyle.SP_DialogOpenButton))
        act_open = QAction(
            icon_open, "Open an image file", self, iconText="Open File")
        self.main_tb.addAction(act_open)
        self.addToolBar(self.main_tb)
        # Open image file dialog
        self.open_image_dialog = QFileDialog(
            self, "Open Source Image", self.wdir, "Images (*.png *.jpg *.jpeg)")

        ########## Connect main window signals
        # Central Widget has its own signals; these are main window specific
        act_open.triggered.connect(self.open_image_dialog.open)
        self.open_image_dialog.fileSelected.connect(self.cw.load_image)
        self.open_image_dialog.directoryEntered.connect(self.set_wdir)

    def closeEvent(self, event):
        """closeEvent() inherited from QMainWindow, reimplemented here.
        This saves current configuration and axes properties if flag is set.
        """
        # Store axis configuration if requested
        if self.cw.model.store_ax_conf:
            print("Storing axis configuration to disk..")
            # Getting plot configuration from Digitizer widget:
            self.conf.x_ax_state = get_qobj_base_state(self.cw.model.x_ax)
            self.conf.y_ax_state = get_qobj_base_state(self.cw.model.y_ax)
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
    ipython_pyqt_boilerplate(app)

    # Set up some shortcuts for command line interactive use.
    mpl_ax = mainw.cw.mplw.mpl_ax
    redraw = mainw.cw.mplw.canvas_qt.draw_idle
    model = mainw.cw.model
    traces = model.traces
    tr1, tr2, tr3 = model.traces[0:3]


