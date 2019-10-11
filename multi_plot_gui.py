#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Digitizer Widget

License: GPL version 3
"""
import logging
logger = logging.getLogger(__name__)

import os
import inspect

import numpy as np

from PyQt5.QtCore import Qt, QDir, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QSplitter, QSizePolicy, QPlainTextEdit, QMessageBox,
        QPushButton, QCheckBox, QTabWidget, QFileDialog, QLabel, QComboBox,
        )
from custom_standard_widgets import CustomisedMessageBox

from digitizer import Digitizer
from plot_model import PlotModel
import physical_models

from upylib.pyqt_debug import logExceptionSlot


class MultiPlotWidget(QWidget):
    """PyQt5 widget for GUI interactive digitizing of data models.
    
    Each data model consists of one or more plots and specific
    export and processing functions.
    
    Each plot in turn has an individual coordinate system setup
    plus one or more data traces.

    This class assembles the main components:
        * Data model of plot data, physical data representation and
          associated data manipulation and export functions
        * A matplotlib based view component providing
          plot and graphic display with mouse interaction
        * Various text and button input widgets for setting model properties
        * Clipboard access and file import/export functions
    
    2019-10-10 Ulrich Lukas
    """
    def __init__(self, mainw, conf):
        super().__init__(mainw)
        self.mainw = mainw
        self.conf = conf
        self.set_wdir(conf.app_conf.wdir)

        # General text or warning message. This is accessed by some
        # sub-widgets so the instance must be created early
        self.messagebox = CustomisedMessageBox(self)
        ########## Tab display on the left and right column
        self.tabs = QTabWidget(self)

        self.plots = []
        self.digitizers = []
        self.curr_plot = None
        self.curr_digitizer = None

        # List of physical model specialised class objects,
        # all imported from physical_models.py
        self.phys_models = [
                member[1] for member
                in inspect.getmembers(physical_models, inspect.isclass)
                if member[1].__module__ == physical_models.__name__
                ]
        self.phys_model_names = [model.name for model in self.phys_models]

        self.phys_model = None
        new_model = physical_models.MosfetDynamic(self, conf)
        self.set_model(new_model)

        # Select, configure and export physical data models
        self.tab_physical_model = PhysicalModelTab(self, self.phys_models)
        # Launch Jupyter Console button
        self.btn_console = QPushButton(
                "Launch Jupyter Console\nIn Application Namespace", self)
        # The first tab is supposed to be always present for setting the
        # multi-plot-model, while the other right-side tabs 
        self.tabs.addTab(self.tab_physical_model, "Physical Model")
        self.tabs.addTab(self.btn_console, "IPython Console")
        # Setup layout
        self._set_layout()

        ########## Connect own signals
        self.tabs.currentChanged.connect(self.switch_plot_index)
        ########## Connect foreign signals


    def set_model(self, phys_model):
        if phys_model is self.phys_model:
            return
        #if phys_model.hasData() and not self.confirm_delete():
        #    return
        # Remove all plots from current (old) model
        if self.phys_model is not None:
            for index in range(len(self.phys_model.plots)):
                self.remove_plot(index)
        # Set new model
        self.phys_model = phys_model
        for plot in phys_model.plots:
            self.add_plot(plot)

    @pyqtSlot(int)
    def switch_plot_index(self, new_index):
        if new_index == self.current_plot_index:
            return
        # Disable current mplw toolbar
        self.digitizers[self.current_plot_index].mplw.toolbar.setVisible(False)
        self.curr_digitizer = self.digitizers[new_index]
        self.curr_plot = self.plots[new_index]
        # Set new index, set shortcut properties and activate everything
        self.digitizers[new_index].mplw.toolbar.setVisible(True)
        self.tab_coordinate_system.switch_plot_index(new_index)
        self.current_plot_index = new_index

    @pyqtSlot(int)
    def remove_plot(self, index):
        logger.debug(
                f"Removing plot: {index} FIXME: Not yet complete? Must check.")
        if index == self.current_plot_index:
            self.switch_plot_index[0]
        digitizer = self.digitizers[index]
        digitizer.mplw.canvas_rescaled.disconnect()
        self.mainw.removeToolBar(digitizer.mplw.toolbar)
        self.tabs.removeTab(index)
        self.plot_model.remove_plot(index)
        digitizer.deleteLater()
        del self.digitizers[index]
        self.plots[index].value_error.disconnect()
        del self.plots[index]

    @pyqtSlot(PlotModel)
    def add_plot(self, plot_model):
        logger.debug(f"Adding plot: {plot_model.name}")
        plot_model.value_error.connect(self.messagebox.show_warning)
        self.plots.append(plot_model)
        digitizer = Digitizer(self, plot_model, self.conf)
        digitizer.mplw.canvas_rescaled.connect(self.mainw.autoscale_window)
        self.mainw.addToolBar(digitizer.mplw.toolbar)
        self.tabs.addTab(digitizer, plot_model.name)
        self.digitizers.append(digitizer)
        self.switch_plot_index(len(self.digitizers) - 1)

    @pyqtSlot(str)
    def set_wdir(self, abs_path):
        # Set working directory to last opened file directory
        self.wdir = abs_path if os.path.isdir(abs_path) else QDir.homePath()

    def _set_layout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)


class PhysicalModelTab(QWidget):
    def __init__(self, digitizer, phys_models):
        super().__init__(digitizer)
        self.digitizer = digitizer

        ########## Widgets setup
        ##### Physical model preset selector
        self.lbl_physical_model = QLabel("Physical Model:")
        presets = {"mos_sw": "MOSFET Dynamic", 
                   "igbt_sw": "IGBT Dynamic",
                   "magnetic-hb": "Magnetic Hysteresis", 
                   "pn-diode": "PN-Diode",
                   "custom": "Custom Model",
                   }
        self.combo_presets = QComboBox()
        for key, value in presets.items():
            self.combo_presets.addItem(value, key)
        ##### Custom plot format and button
        custom_plots = ["Dynamic Properties", "Tr1 Integral"]
        self.lbl_custom_plot = QLabel("Custom Plot:")
        self.combo_custom_plot = QComboBox()
        self.combo_custom_plot.addItems(custom_plots)
        self.btn_custom_plot = QPushButton("Plot Now!")
        ##### Export format and button
        # formats = self.model.physical.export_formats
        formats = ["LTSpice", "Spice Netlist", "QUCS"]
        self.lbl_export_format = QLabel("Export Format:")
        self.combo_export_format = QComboBox()
        self.combo_export_format.addItems(formats)
        self.btn_do_export = QPushButton("Export Now!")
        self._set_layout()
        
        ########## Initialise view from model
        self.update_model_view()

        ########## Connect own and sub-widget signals
        self.btn_custom_plot.clicked.connect(self.wip_do_plot)

        ########## Connect foreign signals

    @logExceptionSlot()
    def update_model_view(self):
        pass

    @logExceptionSlot(bool)
    def wip_do_plot(self, state):
        trace = self.model.traces[self.mplw.curr_trace_no]
        self.model.wip_export(trace)
        self.model.wip_plot_cap_charge_e_stored(trace)

    def _set_layout(self):
        layout = QGridLayout(self)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 1)
        # Make last row expand to the end
        layout.setRowStretch(3, 1)
        # Row 0
        layout.addWidget(self.lbl_physical_model, 0, 0)
        layout.addWidget(self.combo_presets, 0, 1, 1, 2)
        # Row 1
        layout.addWidget(self.lbl_custom_plot, 1, 0)
        layout.addWidget(self.combo_custom_plot, 1, 1)
        layout.addWidget(self.btn_custom_plot, 1, 2)
        # Row 2
        layout.addWidget(self.lbl_export_format, 2, 0)
        layout.addWidget(self.combo_export_format, 2, 1)
        layout.addWidget(self.btn_do_export, 2, 2)
        # Row 3: Empty space

