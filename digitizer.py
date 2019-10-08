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

import numpy as np

from PyQt5.QtCore import Qt, QMimeData, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
        QWIDGETSIZE_MAX, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QSizePolicy, QPlainTextEdit, QMessageBox,
        QPushButton, QCheckBox, QTabWidget,
        )

from mpl_widget import MplWidget
from digitizer_widgets import (
        CoordinateSystemTab, TraceDataModelTab, ExportSettingsTab,
        )
from plot_model import DataModel

from upylib.pyqt_debug import logExceptionSlot


class Digitizer(QWidget):
    """PyQt5 widget for GUI interactive plot digitizing.

    This class assembles the main components:
        * Data model of plot data, physical data representation and
          associated data manipulation and export functions
        * A matplotlib based view component providing
          plot and graphic displa/y with mouse interaction
        * Various text and button input widgets for setting model properties
        * Clipboard access and file import/export functions
    
    2019-09-17 Ulrich Lukas
    """
    def __init__(self, mainw, conf):
        super().__init__(mainw)
        self.conf = conf
        # Filename for temporary storage of clipboard images
        self.temp_filename = os.path.join(
                tempfile.gettempdir(),
                f"plot_workbench_clipboard_paste_image.png"
                )

        # List of physical model specialised class objects,
        # all imported from physical_models.py
        self.phys_models = [
                member[1] for member
                in inspect.getmembers(physical_models, inspect.isclass)
                if member[1].__module__ == physical_models.__name__
                ]
        self.phys_model_names = [model.name for model in self.phys_models]
        # Plot interactive data model
        self.model = model = DataModel(self, conf)
        # System clipboard access
        self.clipboard = QApplication.instance().clipboard()
        # General text or warning message. This is accessed by some
        # sub-widgets so the instance must be created early
        self.messagebox = QMessageBox(self)
        # Matplotlib widget
        self.mplw = mplw = MplWidget(self, model)
        # Tab display on the right column
        self.tabs = QTabWidget(self)
        # Push buttons and axis value input fields widget.
        self.tab_coordinate_system = CoordinateSystemTab(self, model, mplw)
        # Trace Data Model tab
        self.tab_trace_data_model = TraceDataModelTab(self, model, mplw)
        # Export options box
        self.tab_export_settings = ExportSettingsTab(self, model, mplw)
        # Launch Jupyter Console button
        self.btn_console = QPushButton(
                "Launch Jupyter Console\nIn Application Namespace", self)
        # Setup layout
        self._set_layout()

        ########## Connect foreign signals
        model.value_error.connect(self.show_text)


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

    @logExceptionSlot()
    def load_clipboard_image(self, state):
        image = self.clipboard.image()
        if image.isNull():
            self.show_text("There is no image data in the system clipboard!")
            return
        image.save(self.temp_filename, format="png")
        self.mplw.load_image(self.temp_filename)

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

    # Layout is two columns of widgets, arranged by movable splitter widgets
    def _set_layout(self):
        # Resizing policy arranges the widgets with sane preset positions
        self._set_v_stretch(self.mplw, 1)
        self._set_v_stretch(self.tab_coordinate_system, 0)
        self._set_v_stretch(self.tab_trace_data_model, 1)
        self._set_v_stretch(self.tab_export_settings, 0)
        self._set_v_stretch(self.btn_console, 0)
        
        self.tabs.addTab(self.tab_coordinate_system, "Coordinate System")
        self.tabs.addTab(self.tab_trace_data_model, "Traces Data Model")
        self.tabs.addTab(self.tab_export_settings, "Export Settings")
        self.tabs.addTab(self.btn_console, "IPython Console")
        # Horizontal splitter layout is left and right side combined
        hsplitter = QSplitter(Qt.Horizontal, self)
        hsplitter.setChildrenCollapsible(False)
        hsplitter.addWidget(self.mplw)
        hsplitter.addWidget(self.tabs)
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
