#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Digitizer Widget

License: GPL version 3
"""
import io
from functools import partial

import numpy as np

from PyQt5.QtCore import Qt, QMimeData, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (
        QWIDGETSIZE_MAX, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QPlainTextEdit, QMessageBox,
        QPushButton, QCheckBox
        )

from mpl_widget import MplWidget
from input_widgets import AxConfWidget, TraceConfTable, ExportSettingsBox
from plot_model import DataModel


class Digitizer(QWidget):
    """PyQt5 widget for GUI interactive plot digitizing.

    This is the controller instance with direct association to the
    interactive data model, a matplotlib based view component providing
    plot and graphic display with mouse interaction,
    a text and button input widget for setup and control and some
    auxiliary methods e.g. system clipboard access and helper functions.

    The matplotlib widget has internal state (MODE) and direct access
    to the data model which leaves only part of the controller functions
    to this module, mainly connecting the public Qt signals and slots.
    
    2019-07-29 Ulrich Lukas
    """
    ########## Qt signals
    # This triggers a load or reload of the digitizer source input image
    load_image = pyqtSignal(str)
   
    def __init__(self, parent, conf):
        super().__init__(parent)
        self.conf = conf
        # System clipboard access
        self.clipboard = QApplication.instance().clipboard()

        # Plot interactive data model
        self.model = model = DataModel(self, conf)

        # General text or warning message
        self.messagebox = QMessageBox(self)
        
        # Text widget
        self.txtw = QPlainTextEdit()
        self.txtw.insertPlainText("Clipboard Copy and Paste.")
        #self.txtw.setFixedWidth(80)

        # Jupyter Console button
        self.btn_console = QPushButton(
            "Launch Jupyter Console\nIn Application Namespace", self)
        
        # Export options box
        self.export_settings_box = ExportSettingsBox(self, model)

        # Traces properties are displayed in a QTableWidget
        self.tr_conf_table = TraceConfTable(self, model)
        
        # Push buttons and axis value input fields widget.
        self.inputw = inputw = AxConfWidget(self, model)
        
        # Matplotlib widget
        self.mplw = mplw = MplWidget(self, model, conf)
       

        # Layout is two columns of widgets, right is data output and console,
        # left is inputwidget and mpl_widget
        hsplitter = QSplitter(Qt.Horizontal, self)
        hsplitter.setChildrenCollapsible(False)
        layout = QHBoxLayout(self)
        layout.addWidget(hsplitter)

        # Left side layout is vertical widgets, divided by a splitter
        self.mplw_splitter = QSplitter(Qt.Vertical, self)
        self.mplw_splitter.setChildrenCollapsible(False)
        self.mplw_splitter.addWidget(inputw)
        self.mplw_splitter.addWidget(mplw)

        # Right side layout just the same
        io_splitter = QSplitter(Qt.Vertical, self)
        io_splitter.setChildrenCollapsible(False)
        io_splitter.addWidget(self.export_settings_box)
        io_splitter.addWidget(self.tr_conf_table)
        io_splitter.addWidget(self.btn_console)
        #io_splitter.addWidget(self.txtw)

        # All combined
        hsplitter.addWidget(self.mplw_splitter)
        io_splitter.setChildrenCollapsible(False)
        hsplitter.addWidget(io_splitter)

        ########## All model and GUI signals connection, this is the
        ########## application controller logic
        ##### Connect model state changes to update the GUI widgets
        # Update input widget immediately when axis config changes
        model.x_ax.config_changed.connect(inputw.update_axes_view)
        model.y_ax.config_changed.connect(inputw.update_axes_view)
        # Update traces view when model has updated data.
        # Since the input data is normally set by the view itself, a redraw
        # of raw input data is not performed when this signal is received.
        model.output_data_changed.connect(mplw.update_output_view)
        model.output_data_changed[int].connect(mplw.update_output_view)
        # Update plot view displaying axes points as well
        model.redraw_ax_pts_px.connect(mplw.using_model_redraw_ax_pts_px)
        # Re-draw display of the raw (pixel-space) input points if requested.
        model.redraw_tr_pts_px.connect(mplw.using_model_redraw_tr_pts_px)
        model.redraw_tr_pts_px[int].connect(mplw.using_model_redraw_tr_pts_px)

        # Error message
        model.value_error.connect(self.show_text)

        ##### Matplotlib widget state is displayed on the input widget
        # Matplotlib widget signals a complete axis setup
        mplw.valid_x_axis_setup.connect(inputw.btn_pick_x.set_green)
        mplw.valid_y_axis_setup.connect(inputw.btn_pick_y.set_green)
        # This checks or unchecks the input widget buttons to reflect the
        # corresponding matplotlib widget current operating mode
        mplw.mode_sw_default.connect(inputw.uncheck_all_buttons)
        mplw.mode_sw_setup_x_axis.connect(
            partial(inputw.btn_pick_x.setChecked, True))
        mplw.mode_sw_setup_y_axis.connect(
            partial(inputw.btn_pick_y.setChecked, True))
        mplw.mode_sw_add_trace_pts.connect(
            lambda i: inputw.btns_pick_trace[i].setChecked(True))

        ##### Input widget button signals connect in turn to matplotlib widget
        # to set the corresponding operation mode. Signals emit bool values.
        inputw.btn_pick_x.clicked.connect(mplw.toggle_setup_x_axis)
        inputw.btn_pick_y.clicked.connect(mplw.toggle_setup_y_axis)
        for btn in inputw.export_settings_box.btns_pick_trace:
            btn.i_clicked.connect(mplw.toggle_add_trace_pts_mode)

        ##### Input widget signals also trigger model updates of the axes config
        inputw.btn_log_x.toggled.connect(model.x_ax.set_log_scale)
        inputw.btn_log_y.toggled.connect(model.y_ax.set_log_scale)
        inputw.btn_store_config.toggled.connect(model.set_store_config)
        # Number input boxes. These signals emit float values.
        inputw.xstartw.valid_number_entered.connect(model.x_ax.set_ax_start)
        inputw.ystartw.valid_number_entered.connect(model.y_ax.set_ax_start)
        inputw.xendw.valid_number_entered.connect(model.x_ax.set_ax_end)
        inputw.yendw.valid_number_entered.connect(model.y_ax.set_ax_end)

        # When the splitter is clicked, release the fixed height constraint
        self.mplw_splitter.splitterMoved.connect(
            partial(self.txtw.setMaximumHeight, 10000))
        # Update source input image for digitizing. Argument is file path.
        self.load_image.connect(mplw.load_image)
        # Clipboard handling
        #self.clipboard.dataChanged.connect(self.clipboardChanged)


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


    @pyqtSlot(str)
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
        print(f"Storing CSV output to file: {filename}")
        print(f"Number format string used is: {num_fmt}")
        print(f'==> Decimal point character used is: "{decimal_chr}" <==')
        pts_i_csv = self.array2csv(pts_i, decimal_chr, num_fmt)
        try:
            with open(filename, "x") as f:
                f.write(pts_i_csv)
        except IOError as e:
            self.show_error(e)


    @pyqtSlot()
    def clipboardChanged(self):
        # Get the system clipboard contents
        text = self.clipboard.text()
        print(text)
        self.txtw.insertPlainText(text + '\n')
    
    def do_clip(self):
        buf = io.BytesIO()
        self.fig.savefig(buf)
        self.clipboard.setImage(QImage.fromData(buf.getvalue()))
        buf.close()

    @pyqtSlot()
    def put_clipboard(self):
        trace = self.model.traces[self.mplw.curr_trace_no]
        pts_i = trace.pts_i
        if self.conf.app_conf.decimal_chr.lower() == "system":
            decimal_chr = self.locale().decimalPoint()
        else:
            decimal_chr = self.conf.app_conf.decimal_chr
        num_fmt = self.conf.app_conf.num_fmt_export
        print("Putting CSV and HTML table data into clipboard!")
        print(f"Number format string used is: {num_fmt}")
        print(f'==> Decimal point character used is: "{decimal_chr}" <==')
        pts_i_csv = self.array2csv(pts_i, decimal_chr, num_fmt)
        pts_i_html = self.array2html(pts_i, decimal_chr, num_fmt)
        qmd = QMimeData()
        qmd.setData("text/csv", bytes(pts_i_csv, encoding="utf-8"))
        qmd.setData("text/plain", bytes(pts_i_csv, encoding="utf-8"))
        qmd.setHtml(pts_i_html)
        self.clipboard.setMimeData(qmd)


    def show_error(self, exception):
        self.messagebox.setIcon(QMessageBox.Critical)
        self.messagebox.setText("<b>Error!</b>")
        text = " ".join([str(i) for i in exception.args])
        if hasattr(exception, "filename"):
            text += f"\nFilename: {exception.filename}"
        print(text)
        self.messagebox.setInformativeText(text)
        self.messagebox.setWindowTitle("Plot Workbench Error")
        self.messagebox.exec_()

    @pyqtSlot(str)
    def show_text(self, text):
        self.messagebox.setIcon(QMessageBox.Warning)
        self.messagebox.setText("<b>Please note:</b>")
        self.messagebox.setInformativeText(text)
        self.messagebox.setWindowTitle("Plot Workbench Notification")
        self.messagebox.exec_()


