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
        QPushButton, QCheckBox
        )

from mpl_widget import MplWidget
from digitizer_widgets import (
        AxConfWidget, TraceConfTable, ExportSettingsBox, DataCoordProps,
        )
from plot_model import DataModel

from upylib.pyqt_debug import logExceptionSlot


class Digitizer(QWidget):
    """PyQt5 widget for GUI interactive plot digitizing.

    This is the controller instance with direct association to the
    interactive data model, a matplotlib based view component providing
    plot and graphic displa/y with mouse interaction,
    a text and button input widget for setup and control and some
    auxiliary methods e.g. system clipboard access and helper functions.

    The matplotlib widget has internal state (MODE) and direct access
    to the data model which leaves only part of the controller functions
    to this module, mainly connecting the public Qt signals and slots.
    
    2019-07-29 Ulrich Lukas
    """
    def __init__(self, mainw, conf):
        super().__init__(mainw)
        self.conf = conf
        # System clipboard access
        self.clipboard = QApplication.instance().clipboard()
        # Filename for temporary storage of clipboard images
        self.temp_filename = os.path.join(
                tempfile.gettempdir(),
                f"plot_workbench_clipboard_paste_image.png"
                )
        
        # Plot interactive data model
        self.model = model = DataModel(self, conf)

        # General text or warning message
        self.messagebox = QMessageBox(self)

        # Matplotlib widget
        self.mplw = mplw = MplWidget(self, model)

        # Launch Jupyter Console button
        self.btn_console = QPushButton(
                "Launch Jupyter Console\nIn Application Namespace", self)

        # Data Coordinate Display and Edit Box
        self.data_coord_props = DataCoordProps(self, model, mplw)
        
        # Export options box
        self.export_settings = ExportSettingsBox(self, model, mplw)

        # Traces properties are displayed in a QTableWidget
        self.tr_conf_table = TraceConfTable(self, model, mplw)
        
        # Push buttons and axis value input fields widget.
        self.axconfw = AxConfWidget(self, model, mplw)

        # Layout is two columns of widgets, right is data output and console,
        # left is inputwidget and mpl_widget
        self.hsplitter = hsplitter = QSplitter(Qt.Horizontal, self)
        hsplitter.setChildrenCollapsible(False)
        layout = QHBoxLayout(self)
        layout.addWidget(hsplitter)

        # Left side layout is vertical widgets, divided by a splitter
        self.mplw_splitter = mplw_splitter = QSplitter(Qt.Vertical, self)
        mplw_splitter.setChildrenCollapsible(False)
        self._set_v_stretch(self.axconfw, 0)
        self._set_v_stretch(self.mplw, 1)
        mplw_splitter.addWidget(self.axconfw)
        mplw_splitter.addWidget(self.mplw)

        # Right side layout just the same
        self.io_splitter = io_splitter = QSplitter(Qt.Vertical, self)
        io_splitter.setChildrenCollapsible(False)
        self._set_v_stretch(self.data_coord_props, 0)
        self._set_v_stretch(self.export_settings, 0)
        self._set_v_stretch(self.tr_conf_table, 1)
        self._set_v_stretch(self.btn_console, 0)
        io_splitter.addWidget(self.data_coord_props)
        io_splitter.addWidget(self.export_settings)
        io_splitter.addWidget(self.tr_conf_table)
        io_splitter.addWidget(self.btn_console)

        # All combined
        hsplitter.addWidget(self.mplw_splitter)
        io_splitter.setChildrenCollapsible(False)
        hsplitter.addWidget(io_splitter)

        # Error message
        model.value_error.connect(self.show_text)

    def array2html(self, array, decimal_chr, num_fmt):
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

    def array2csv(self, array, decimal_chr, num_fmt):
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

    @logExceptionSlot(str)
    def on_dlg_export_csv(self, filename):
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
        pts_i_csv = self.array2csv(pts_i, decimal_chr, num_fmt)
        try:
            with open(filename, "x") as f:
                f.write(pts_i_csv)
        except IOError as e:
            self.show_error(e)

    @logExceptionSlot()
    def load_clipboard_image(self):
        image = self.clipboard.image()
        if image.isNull():
            self.show_text("There is no image data in the system clipboard!")
            return
        image.save(self.temp_filename, format="png")
        self.mplw.load_image(self.temp_filename)

    @logExceptionSlot()
    def put_clipboard(self):
        trace = self.model.traces[self.mplw.curr_trace_no]
        pts_i = trace.pts_i
        if self.conf.app_conf.decimal_chr.lower() == "system":
            decimal_chr = self.locale().decimalPoint()
        else:
            decimal_chr = self.conf.app_conf.decimal_chr
        num_fmt = self.conf.app_conf.num_fmt_export
        logger.info(f"Putting CSV and HTML table data into clipboard!"
                    f"Number format string used is: {num_fmt}"
                    f'==> Decimal point character used is: "{decimal_chr}" <==')
        pts_i_csv = self.array2csv(pts_i, decimal_chr, num_fmt)
        pts_i_html = self.array2html(pts_i, decimal_chr, num_fmt)
        qmd = QMimeData()
        qmd.setData("text/csv", bytes(pts_i_csv, encoding="utf-8"))
        qmd.setData("text/plain", bytes(pts_i_csv, encoding="utf-8"))
        qmd.setHtml(pts_i_html)
        self.clipboard.setMimeData(qmd)

    def show_error(self, e: Exception):
        self.messagebox.setIcon(QMessageBox.Critical)
        self.messagebox.setText("<b>Error!</b>")
        text = f"Uncaught exception of {type(e)} occurred:\n{str(e)}"
        if hasattr(e, "filename"):
                text += f"\nFilename: {e.filename}"
        logger.critical(text)
        self.messagebox.setInformativeText(text)
        self.messagebox.setWindowTitle("Plot Workbench Error")
        self.messagebox.exec_()

    @logExceptionSlot(str)
    def show_text(self, text):
        self.messagebox.setIcon(QMessageBox.Warning)
        self.messagebox.setText("<b>Please note:</b>")
        logger.info(text)
        self.messagebox.setInformativeText(text)
        self.messagebox.setWindowTitle("Plot Workbench Notification")
        self.messagebox.exec_()

    @staticmethod
    def _set_v_stretch(widget, value: int):
        # Set widget size policy stretch factor to value
        sp = widget.sizePolicy()
        sp.setVerticalStretch(value)
        widget.setSizePolicy(sp)