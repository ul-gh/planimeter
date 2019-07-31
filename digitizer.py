#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Digitizer Widget

License: GPL version 3
"""
import io
from functools import partial

from PyQt5.QtCore import Qt, QMimeData, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout,  QSplitter, QPlainTextEdit,
        QMessageBox,
        )

from mpl_widget import MplWidget
from input_widget import InputWidget
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
        self.model = DataModel(self, conf)
        model = self.model

        # General text or warning message
        self.messagebox = QMessageBox(self)
        
        # Text widget
        self.txtw = QPlainTextEdit()
        self.txtw.insertPlainText("Clipboard Copy and Paste.")
        self.txtw.setFixedHeight(25)
        
        # Push buttons and axis value input fields widget.
        self.inputw = InputWidget(self, model)
        inputw = self.inputw
        
        # Matplotlib widget
        self.mplw = MplWidget(self, model, conf)
        mplw = self.mplw
       
        # Layout is vertical widgets, divided by a splitter
        self.splitter = QSplitter(Qt.Vertical, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.txtw)
        self.splitter.addWidget(inputw)
        self.splitter.addWidget(mplw)
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.splitter)
        self.setLayout(vbox)

        ########## Set up all widgets public signals for interactive GUI
        ##### Connect model state changes to the GUI widgets
        # Update input widget immediately when axis config changes
        model.x_ax.config_changed.connect(inputw.update_axes_view)
        model.y_ax.config_changed.connect(inputw.update_axes_view)
        # Update plot view displaying axes points as well
        model.x_ax.redraw_pts_px.connect(mplw.using_model_redraw_ax_pts_px)
        model.y_ax.redraw_pts_px.connect(mplw.using_model_redraw_ax_pts_px)
        # Update plot view again when origin has been calculated
        model.affine_transformation_defined.connect(
            mplw.using_model_redraw_ax_pts_px)
        # Update traces view when model has updated data.
        # Since the input data is normally set by the view itself, a redraw
        # of raw input data is not performed when this signal is received.
        model.output_data_changed.connect(mplw.update_output_view)
        model.output_data_changed[int].connect(mplw.update_output_view)
        # Re-draw display of the raw (pixel-space) input points if requested.
        model.redraw_tr_pts_px.connect(mplw.using_model_redraw_tr_pts_px)
        model.redraw_tr_pts_px[int].connect(mplw.using_model_redraw_tr_pts_px)

        # Error message
        model.value_error.connect(self.show_text)

        ##### Matplotlib widget state is displayed on the input widget
        # Matplotlib widget signals a complete axis setup
        mplw.valid_x_axis_setup.connect(inputw.set_green_btn_pick_x)
        mplw.valid_y_axis_setup.connect(inputw.set_green_btn_pick_y)
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
        for btn in inputw.btns_pick_trace:
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
        self.splitter.splitterMoved.connect(
            partial(self.txtw.setMaximumHeight, 10000))
        # Update source input image for digitizing. Argument is file path.
        self.load_image.connect(mplw.load_image)
        # Clipboard handling
        #self.clipboard.dataChanged.connect(self.clipboardChanged)


    def array2html(self, array, decimal_chr):
        """Make a HTML table with two columns from 2D numpy array"""
        def row2html(columns):
            r = '<tr>'
            for i in columns:
                r += f'<td>{i}</td>'
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
        print("Putting CSV and HTML table data into clipboard!")
        print(f'==> Decimal point character used is: "{decimal_chr}" <==')
        #datastring = np.array2string(pts_i, separator="\t")
        #strio = io.StringIO()
        #np.savetxt(strio, pts_i, delimiter="\t", fmt="%.6e")
        #datastring = strio.getvalue()
        #strio.close()
        #databytes = bytes(datastring, encoding="utf-8")
        qmd = QMimeData()
        #qmd.setData("Csv", databytes)
        qmd.setHtml(self.array2html(pts_i, decimal_chr))
        self.clipboard.setMimeData(qmd)
        #self.clipboard.setText(datastring)


    def show_error(self, exception):
        self.messagebox.setIcon(QMessageBox.Warning)
        self.messagebox.setText("<b>Error!</b>")
        self.messagebox.setInformativeText(exception.args[0])
        self.messagebox.setWindowTitle("Plot Workbench Error")
        import traceback
        traceback.print_tb(exception.__traceback__)
        self.messagebox.exec_()

    @pyqtSlot(str)
    def show_text(self, text):
        self.messagebox.setIcon(QMessageBox.Warning)
        self.messagebox.setText("<b>Please note:</b>")
        self.messagebox.setInformativeText(text)
        self.messagebox.setWindowTitle("Plot Workbench Notification")
        self.messagebox.exec_()


