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

from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication

from plot_model_assistant import PlotModelAssistant
from default_configuration import APPLICATION, PLOT_MODEL, TRACE, X_AXIS, Y_AXIS

from embedded_ipython import EmbeddedIPythonKernel

class RuntimeConfig():
    """Modified configuration settings and application state can be written to
    file for persistent storage
    """
    def __init__(self):
        self.app_conf = APPLICATION()
        self.plot_conf = PLOT_MODEL()
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

        #self.setMinimumSize(QSize(*conf.app_conf.window_size))
        self.setWindowTitle("-- Plot Model Assistant -- Main Window --")

        ########## Embedded IPython Kernel and Jupyter Console Launcher
        self.ipyconsole = EmbeddedIPythonKernel(self)
        self._add_toolbar()
        ########## Central Widget
        self.cw = PlotModelAssistant(self, conf)
        self.setCentralWidget(self.cw)

        self.set_last_image_file(conf.app_conf.last_image_file)
        # Reopen last file if requested and if it exists
        if self.last_image_file != "":
            self.cw.curr_digitizer.mpl_widget.load_image_file(self.last_image_file)
        # Add to the central widget size for mainw autoscaling
        self.autoscale_window()


    @pyqtSlot()
    def autoscale_window(self):
        sh = self.cw.sizeHint()
        logger.debug(f"Autoscale Window Called. SizeHint is: {sh}")
        width, height = sh.width(), sh.height()
        # Don't know where these pixels come from
        height += 26
        aspect = width / height
        # +20 px for mainw contentsMargins
        width += 20
        #height += 20
        width_max, height_max = self.conf.app_conf.autoscale_max_window_size
        if width > width_max or height > height_max:
            limit_aspect = width_max / height_max
            if aspect > limit_aspect:
                width = width_max
                height = int(width_max / aspect)
            else:
                width = int(height_max * aspect)
                height = height_max
        curr_width, curr_height = self.width(), self.height()
        if width > curr_width or height > curr_height:
            self.resize(width, height)


    def closeEvent(self, event):
        """closeEvent() inherited from QMainWindow, reimplemented here.
        This saves current configuration and axes properties if flag is set.
        """
        # Store axis configuration if requested
        if self.cw.curr_plot.wants_persistent_storage:
            logger.info("Storing axis configuration to disk..")
            self.conf.plot_conf.wants_persistent_storage = True
            # Getting plot configuration from Digitizer widget:
            self.conf.x_ax_state = self.cw.curr_plot.x_ax.restorable_state()
            self.conf.y_ax_state = self.cw.curr_plot.y_ax.restorable_state()
        else:
            self.conf.x_ax_state = None
            self.conf.y_ax_state = None
        # Save working directory
        self.conf.app_conf.wdir = self.cw.wdir
        self.conf.app_conf.last_image_file = self.last_image_file
        # Store all configuration
        self.conf.store_to_configfile()
        self.ipyconsole.shutdown_kernel()
        event.accept()

    @pyqtSlot(str)
    def set_last_image_file(self, abs_path):
        logger.warning("This should go into each plot data model")
        # This property will be reloaded if persistent_storage property of
        # digitizer instance is set. Then, the input image file will be
        # re-loaded on next start
        self.last_image_file = abs_path if os.path.isfile(abs_path) else ""

    def _add_toolbar(self):
        self.toolbar = toolbar = self.addToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        style = self.style()
        icon = QIcon.fromTheme("arrow-up", style.standardIcon(style.SP_ArrowUp))
        act = toolbar.addAction(icon, "Live Console")
        act.triggered.connect(self.ipyconsole.launch_jupyter_console_process)


if __name__ == "__main__":
    # When run interactively in IPython shell, app instance might already exist
    app = QApplication.instance()
    if app is None:
        logger.debug("Creating new QApplication")
        app = QApplication(sys.argv)

    mainw = MainWindow()
    mainw.show()
    mainw.activateWindow()

    # Set up some shortcuts for command line interactive use.
    digitizer = mainw.cw.digitizers[0]
    mplw = digitizer.mpl_widget
    ax = mplw.mpl_ax
    fig = mplw.fig
    cv = mplw.canvas_qt
    redraw = mplw.canvas_qt.draw_idle
    print_model_view_items = mplw.print_model_view_items

    plots = mainw.cw.phys_model.plots
    tr1, tr2, tr3 = plots[0].traces[0:3]

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

