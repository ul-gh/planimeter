#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Digitizer Widget

License: GPL version 3
"""
import logging
logger = logging.getLogger(__name__)

import io
import os
import tempfile
import inspect

import numpy as np

from PyQt5.QtCore import Qt, QDir, QMimeData, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
        QWIDGETSIZE_MAX, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QSizePolicy, QPlainTextEdit, QMessageBox,
        QPushButton, QCheckBox, QTabWidget, QFileDialog
        )

from mpl_widget import MplWidget
from digitizer_widgets import (
        CoordinateSystemTab, TraceDataModelTab, ExportSettingsTab,
        DigitizerToolBar,
        )
from plot_model import PlotModel
import physical_models

from upylib.pyqt_debug import logExceptionSlot


class Digitizer(QWidget):
    """PyQt5 widget for GUI interactive plot digitizing.

    This class assembles the main components:
        * Data model of plot data, physical data representation and
          associated data manipulation and export functions
        * A matplotlib based view component providing
          plot and graphic display with mouse interaction
        * Various text and button input widgets for setting model properties
        * Clipboard access and file import/export functions
    
    2019-09-17 Ulrich Lukas
    """
    def __init__(self, mainw, conf):
        super().__init__(mainw)
        self.mainw = mainw
        self.conf = conf
        self.set_wdir(conf.app_conf.wdir)
        
        # Current plot index, used for mplw tabs and model indexing
        self.current_plot_index = 0
        # List of physical model specialised class objects,
        # all imported from physical_models.py
        self.phys_models = [
                member[1] for member
                in inspect.getmembers(physical_models, inspect.isclass)
                if member[1].__module__ == physical_models.__name__
                ]
        self.phys_model_names = [model.name for model in self.phys_models]

        # self.model = physical_models.Custom(self, conf)
        self.model = physical_models.MosfetDynamic(self, conf)
        # System clipboard access
        self.clipboard = QApplication.instance().clipboard()
        # General text or warning message. This is accessed by some
        # sub-widgets so the instance must be created early
        self.messagebox = QMessageBox(self)
        ########## Tab display on the left and right column
        self.tabs_left = QTabWidget(self)
        self.tabs_right = QTabWidget(self)
        ########## Digitizer Toolbar
        self.toolbar = DigitizerToolBar(self)
        mainw.addToolBar(self.toolbar)
        ########## Custom Dialogs
        self.dlg_export_csv = QFileDialog(
                self, "Export CSV", self.wdir, "Text/CSV (*.csv *.txt)")
        self.dlg_export_xlsx = QFileDialog(
                self, "Export XLS/XLSX", self.wdir, "Excel (*.xlsx)")

        # Matplotlib widgets, one for each plot
        self.mplws = []
        self.plot_names = [plot.name for plot in self.model.plots]
        for plot in self.model.plots:
            self.add_plot(plot)
        plot = self.model.plots[0]
        mplw = self.mplws[0]
        # Push buttons and axis value input fields widget.
        self.tab_coordinate_system = CoordinateSystemTab(self, plot, mplw)
        # Trace Data Model tab
        self.tab_trace_data_model = TraceDataModelTab(self, plot, mplw)
        # Export options box
        self.tab_export_settings = ExportSettingsTab(self, plot, mplw)
        # Launch Jupyter Console button
        self.btn_console = QPushButton(
                "Launch Jupyter Console\nIn Application Namespace", self)

        for mplw, name in zip(self.mplws, self.plot_names):
            self.tabs_left.addTab(mplw, name)
        self.tabs_right.addTab(self.tab_coordinate_system, "Coordinate System")
        self.tabs_right.addTab(self.tab_trace_data_model, "Traces Data Model")
        self.tabs_right.addTab(self.tab_export_settings, "Export Settings")
        self.tabs_right.addTab(self.btn_console, "IPython Console")
        # Setup layout
        self._set_layout()

        ########## Connect own signals
        self.tabs_left.currentChanged.connect(self.switch_plot_index)
        # ToolBar signals
        self.toolbar.act_export_csv.triggered.connect(self.dlg_export_csv.open)
        self.toolbar.act_export_xlsx.triggered.connect(self.dlg_export_xlsx.open)
        self.toolbar.act_put_clipboard.triggered.connect(self.put_clipboard)
        # Dialog box signals
        self.dlg_export_csv.fileSelected.connect(self.export_csv)
        self.dlg_export_xlsx.fileSelected.connect(
                lambda _: self.show_text("Not yet implemented!"))
        ########## Connect foreign signals
        plot.value_error.connect(self.show_text)


    # This is connected to from the main window toolbar!
    @logExceptionSlot(str)
    def export_csv(self, filename):
        """Export CSV textstring to file
        """
        trace = self.model.traces[self.mplw.curr_trace_no]
        pts_i = trace.pts_i
        if self.conf.app_conf.decimal_chr.lower() == "system":
            decimal_chr = self.locale().decimalPoint()
        else:
            decimal_chr = self.conf.app_conf.decimal_chr
        num_fmt = self.conf.app_conf.num_fmt_export
        logger.info(f"Storing CSV output to file: {filename}\n"
                    f"Number format string used is: {num_fmt}"
                    f'==> Decimal point character used is: "{decimal_chr}" <==')
        pts_i_csv = self._array2csv(pts_i, decimal_chr, num_fmt)
        try:
            with open(filename, "x") as f:
                f.write(pts_i_csv)
        except IOError as e:
            self.show_error(e)

    @logExceptionSlot(bool)
    def put_clipboard(self, state=True, pts_data=None):
        trace = self.model.traces[self.mplw.curr_trace_no]
        if pts_data is None:
            pts_data = trace.pts
        if self.conf.app_conf.decimal_chr.lower() == "system":
            decimal_chr = self.locale().decimalPoint()
        else:
            decimal_chr = self.conf.app_conf.decimal_chr
        num_fmt = self.conf.app_conf.num_fmt_export
        logger.info(f"Putting CSV and HTML table data into clipboard!"
                    f"Number format string used is: {num_fmt}"
                    f'==> Decimal point character used is: "{decimal_chr}" <==')
        pts_csv = self._array2csv(pts_data, decimal_chr, num_fmt)
        pts_html = self._array2html(pts_data, decimal_chr, num_fmt)
        qmd = QMimeData()
        qmd.setData("text/csv", bytes(pts_csv, encoding="utf-8"))
        qmd.setData("text/plain", bytes(pts_csv, encoding="utf-8"))
        qmd.setHtml(pts_html)
        self.clipboard.setMimeData(qmd)

    @logExceptionSlot(Exception)
    def show_error(self, e: Exception):
        self.messagebox.setIcon(QMessageBox.Critical)
        self.messagebox.setText("<b>Error!</b>")
        text = f"Uncaught exception of {type(e)} occurred:\n{str(e)}"
        if hasattr(e, "filename"):
                text += f"\nFilename: {e.filename}"
        logger.critical(text)
        self.messagebox.setInformativeText(text)
        self.messagebox.setWindowTitle("Plot Workbench Error")
        self.messagebox.raise_()
        self.messagebox.exec_()

    @logExceptionSlot(str)
    def show_text(self, text):
        self.messagebox.setIcon(QMessageBox.Warning)
        self.messagebox.setText("<b>Please note:</b>")
        logger.info(text)
        self.messagebox.setInformativeText(text)
        self.messagebox.setWindowTitle("Plot Workbench Notification")
        self.messagebox.exec_()

    @pyqtSlot(str)
    def set_wdir(self, abs_path):
        # Set working directory to last opened file directory
        self.wdir = abs_path if os.path.isdir(abs_path) else QDir.homePath()

    @pyqtSlot(int)
    def switch_plot_index(self, new_index):
        # Disable current mplw toolbar
        self.mplws[self.current_plot_index].mpl_toolbar.setVisible(False)
        # Set new index, set shortcut properties and activate everything
        self.mplws[new_index].mpl_toolbar.setVisible(True)
        self.current_plot_index = new_index

    @pyqtSlot(int)
    def remove_plot(self, index):
        logger.debug(
                f"Removing plot: {index} FIXME: Not yet complete? Must check.")
        mplw = self.mplws[index]
        mplw.canvas_rescaled.disconnect()
        self.mainw.removeToolBar(mplw.mpl_toolbar)
        self.tabs_left.removeTab(index)
        self.model.remove_plot(index)
        mplw.deleteLater()
        del self.mplws[index]

    @pyqtSlot(PlotModel)
    def add_plot(self, plot):
        logger.debug(f"Adding plot: {plot.name}")
        mplw = MplWidget(self, plot)
        mplw.canvas_rescaled.connect(self.mainw.autoscale_window)
        self.mainw.addToolBar(mplw.mpl_toolbar)
        self.tabs_left.addTab(mplw, plot.name)
        self.mplws.append(mplw)
        self.switch_plot_index(len(self.mplws) - 1)
        
    # Layout is two columns of widgets, arranged by movable splitter widgets
    def _set_layout(self):
        #self._set_v_stretch(self.tab_trace_data_model, 1)
        # Horizontal splitter layout is left and right side combined
        self.hsplitter = hsplitter = QSplitter(Qt.Horizontal, self)
        hsplitter.setChildrenCollapsible(False)
        hsplitter.addWidget(self.tabs_left)
        hsplitter.addWidget(self.tabs_right)
        # All combined
        digitizer_layout = QHBoxLayout(self)
        digitizer_layout.addWidget(hsplitter)

    @staticmethod
    def _set_v_stretch(widget, value: int):
        # Set widget size policy stretch factor to value
        sp = widget.sizePolicy()
        sp.setVerticalStretch(value)
        widget.setSizePolicy(sp)

    @staticmethod
    def _array2html(array, decimal_chr, num_fmt):
        """Make a HTML table with two columns from 2D numpy array
        """
        def row2html(columns):
            r = '<tr>'
            for i in columns:
                r += f'<td>{i:{num_fmt}}</td>'
            return r + '</tr>'
        header = (
          '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">'
          '<html><head>'
          '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
          '<title></title>'
          '<meta name="generator" content="Plot Workbench Export"/>'
          '</head><body><table>'
          )
        
        footer = '</table></body></html>'
        s = ""
        for row in array.tolist():
            s += row2html(row)
        if decimal_chr != ".":
            s = s.replace(".", decimal_chr)
        return header + s + footer

    @staticmethod
    def _array2csv(array, decimal_chr, num_fmt):
        """Output np.array as CSV text.
        """
        #datastring = np.array2string(pts_i, separator="\t")
        strio = io.StringIO()
        np.savetxt(strio, array, delimiter=" ", fmt=f"%{num_fmt}")
        s = strio.getvalue()
        strio.close()
        if decimal_chr != ".":
            s = s.replace(".", decimal_chr)
        return s
