#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Input Widget

License: GPL version 3
2019-07-29 Ulrich Lukas
"""
from functools import partial

from numpy import isclose, isnan

from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QMessageBox,
        QGroupBox, QLabel, QPushButton, QRadioButton, QCheckBox, QComboBox,
        QTableWidget, QTableWidgetItem
        )

class TraceConfTable(QTableWidget):
    show_xrange = pyqtSignal(bool)

    def __init__(self, parent, model):
        headers = ["Name", "Pick Points", "Export", "X Start", "X End",
                   "Interpolation", "N Points"]
        self.col_xstart = headers.index("X Start")
        self.col_xend = headers.index("X End")
        n_traces = len(model.traces)
        n_headers = len(headers)
        super().__init__(n_traces, n_headers, parent)
        self.setHorizontalHeaderLabels(headers)
        # Data Export options
        ###
        n_interp_presets_text = ["10", "25", "50", "100", "250", "500", "1000"]
        n_interp_presets_values = [10, 25, 50, 100, 250, 500, 1000]
        ###
        i_types_text = ["Linear", "Cubic", "Sin(x)/x"]
        i_types_values = ["linear", "cubic", "sinc"]
        for row, tr in enumerate(model.traces):
            name = QTableWidgetItem(tr.name)
            btn_pick_trace = NumberedButton(row, f"Pick Trace {row+1}", self)
            checkbox_export = CenteredCheckbox(self)
            x_start = QTableWidgetItem(f"{tr.x_start_export}")
            x_end = QTableWidgetItem(f"{tr.x_end_export}")
            combo_i_type = QComboBox(self)
            combo_i_type.addItems(i_types_text)
            combo_n_interp = QComboBox(self)
            combo_n_interp.addItems(n_interp_presets_text)
            self.setItem(row, 0, name)
            self.setCellWidget(row, 1, btn_pick_trace)
            self.setCellWidget(row, 2, checkbox_export)
            self.setItem(row, 3, x_start)
            self.setItem(row, 4, x_end)
            self.setCellWidget(row, 5, combo_i_type)
            self.setCellWidget(row, 6, combo_n_interp)

        ##### Signals
        self.itemSelectionChanged.connect(self._handle_selection)

    def _handle_selection(self):
        self.sel_traces = sel_traces = {
                s.row() for s in self.selectedIndexes()
                if s.column() in (self.col_xstart, self.col_xend)
                and s.row() < self.n_traces
                }
        if sel_traces:
            self._show_xrange = True
            inf = float("inf")
            x_start = -inf
            x_end = inf
            for i in sel_traces:
                item_start = self.item(i, self.col_xstart)
                item_end = self.item(i, self.col_xend)
                xs_new = float(item_start.text())
                xe_new = float(item_end.text())
                if xs_new < x_start:
                    x_start = xs_new
                if xe_new < x_end:
                    x_end = xe_new
            for i in sel_traces:
                self.model.traces[i].x_start_export = x_start
                self.model.traces[i].x_end_export = x_end
                self.x_start_export = x_start
                self.x_end_export = x_end
            self.show_xrange.emit(True)
        elif self._show_xrange:
            self._show_xrange = False
            self.show_xrange.emit(False)

    @pyqtSlot()
    def update_from_model(self):
        for i in range(self.n_traces):
            self.item(i, col_xstart).setText(f"{self.x_start_export}")
            self.item(i, col_xend).setText(f"{self.x_end_export}")


class AxConfWidget(QWidget):
    def __init__(self, parent, model):
        super().__init__(parent)
        ######### Shortcuts to the data model
        self.model = model
        self.x_ax = model.x_ax
        self.y_ax = model.y_ax

        ######### Qt widget setup
        # Error message box used in set_prop()
        self.messagebox = QMessageBox(self)
        
        ######### Group box for X Coordinate picker and input boxes
        self.group_x = QGroupBox("Enter X Axis Start and End Values")
        group_x_layout = QHBoxLayout(self.group_x)
        # Group X contents
        self.btn_pick_x = StyledButton("Pick Points", self)
        self.btn_pick_x.setAutoExclusive(True)
        self.xstartw = SciLineEdit(
                self.x_ax.pts_data[0], "X Axis Start Value", model.num_fmt)
        self.xendw = SciLineEdit(
                self.x_ax.pts_data[1], "X Axis End Value", model.num_fmt)
        self.btn_lin_x = QRadioButton("Lin")
        self.btn_log_x = QRadioButton("Log")
        self.btn_lin_x.setChecked(not self.x_ax.log_scale)
        self.btn_log_x.setChecked(self.x_ax.log_scale)
        group_x_layout.addWidget(self.xstartw)
        group_x_layout.addWidget(self.xendw)
        group_x_layout.addWidget(self.btn_lin_x)
        group_x_layout.addWidget(self.btn_log_x)
        group_x_layout.addWidget(self.btn_pick_x)
        
        ######### Group box for Y Coordinate picker and input boxes
        self.group_y = QGroupBox("Enter Y Axis Start and End Values")
        group_y_layout = QHBoxLayout(self.group_y)
        # Group X contents
        self.btn_pick_y = StyledButton("Pick Points", self)
        self.btn_pick_y.setAutoExclusive(True)
        self.ystartw = SciLineEdit(
                self.y_ax.pts_data[0], "Y Axis Start Value", model.num_fmt)
        self.yendw = SciLineEdit(
                self.y_ax.pts_data[1], "Y Axis End Value", model.num_fmt)
        self.btn_lin_y = QRadioButton("Lin")
        self.btn_log_y = QRadioButton("Log")
        self.btn_log_y.setChecked(self.y_ax.log_scale)
        self.btn_lin_y.setChecked(not self.y_ax.log_scale)
        group_y_layout.addWidget(self.ystartw)
        group_y_layout.addWidget(self.yendw)
        group_y_layout.addWidget(self.btn_lin_y)
        group_y_layout.addWidget(self.btn_log_y)
        group_y_layout.addWidget(self.btn_pick_y)

        # Store plot config button
        self.btn_store_config = QCheckBox("Store Config")
        self.btn_store_config.setChecked(self.model.store_ax_conf)

        # Pick traces buttons
        self.btns_pick_trace = (
                NumberedButton(0, "Pick\nTrace 1", self),
                NumberedButton(1, "Pick\nTrace 2", self),
                NumberedButton(2, "Pick\nTrace 3", self),
                )
        
        # Export Data button
        self.btn_export = StyledButton("Export\nData", self)

        
        # This is all input boxes plus label
        inputw_layout = QHBoxLayout(self)
        inputw_layout.addWidget(self.group_x)
        inputw_layout.addWidget(self.group_y)
        inputw_layout.addWidget(self.btn_store_config)
        for i in self.btns_pick_trace:
            i.setAutoExclusive(True)
            inputw_layout.addWidget(i)
        inputw_layout.addWidget(self.btn_export)

        ########## Initialise view from model
        self.update_axes_view()


    ########## Slots
    @pyqtSlot()
    def update_axes_view(self):
        """Updates buttons and input boxes to represent the data model state"""
        x_ax = self.x_ax
        y_ax = self.y_ax
        num_fmt = self.model.num_fmt
        # Update axis section value input boxes
        x0, x1, y0, y1 = *x_ax.pts_data, *y_ax.pts_data
        self.xstartw.setText("" if x0 is None else f"{x0:{num_fmt}}")
        self.xendw.setText("" if x1 is None else f"{x1:{num_fmt}}")
        self.ystartw.setText("" if y0 is None else f"{y0:{num_fmt}}")
        self.yendw.setText("" if y1 is None else f"{y1:{num_fmt}}")
        invalid_x = x_ax.log_scale and isclose(
            x_ax.pts_data, 0.0, atol=x_ax.atol).any()
        invalid_y = y_ax.log_scale and isclose(
            y_ax.pts_data, 0.0, atol=y_ax.atol).any()
        self.set_green_x_ax(not invalid_x and None not in x_ax.pts_data)
        self.set_green_y_ax(not invalid_y and None not in y_ax.pts_data)
        # Update log/lin radio buttons.
        self.btn_lin_x.setChecked(not x_ax.log_scale)
        self.btn_log_x.setChecked(x_ax.log_scale)
        self.btn_lin_y.setChecked(not y_ax.log_scale)
        self.btn_log_y.setChecked(y_ax.log_scale)
        # Store config button
        self.btn_store_config.setChecked(self.model.store_ax_conf)


    @pyqtSlot(bool)
    def set_green_x_ax(self, state):
        # Background set to green when model has valid data
        style = "QLineEdit { background-color: Palegreen; }" if state else ""
        self.group_x.setStyleSheet(style)

    @pyqtSlot(bool)
    def set_green_y_ax(self, state):
        # Background set to green when model has valid data
        style = "QLineEdit { background-color: Palegreen; }" if state else ""
        self.group_y.setStyleSheet(style)

    
    @pyqtSlot()
    def uncheck_all_buttons(self):
        self.btn_pick_x.setChecked(False)
        self.btn_pick_y.setChecked(False)
        for i in self.btns_pick_trace:
            i.setChecked(False)
        # Set focus to the default in order to unfocus all other buttons
        self.btn_export.setFocus()



########## Custom Widgets Used Above ##########
class StyledButton(QPushButton):
    """This checkable button has a minimum size set to the initial
    text contents requirements and can be used as a state indicator by
    connecting the set_green slot.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCheckable(True)
        size_ref_text = "A" + self.text() + "A"
        min_size = self.fontMetrics().size(Qt.TextShowMnemonic, size_ref_text)
        self.setMinimumSize(min_size)

    @pyqtSlot(bool)
    def set_green(self, state):
        style = "background-color: Palegreen" if state else ""
        self.setStyleSheet(style)


class NumberedButton(StyledButton):
    """This subclass of QPushButton adds a number index property and
    emits a corresponding integer signal
    """
    i_clicked = pyqtSignal(int)
    def __init__(self, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.clicked.connect(
                partial(self.i_clicked.emit, index)
                )


class SciLineEdit(QLineEdit):
    """QLineEdit with added validator for scientific notation input.
    Also, this takes a number as preset value instead of a string.
    """
    valid_number_entered = pyqtSignal(float)
    def __init__(
            self,
            preset_value,
            placeholderText,
            num_fmt=".6G",
            *args,
            **kwargs
            ):
        text = None if isnan(preset_value) else f"{preset_value:{num_fmt}}"
        super().__init__(
                text,
                *args,
                placeholderText=placeholderText,
                **kwargs
                )
        v = QDoubleValidator(self, notation=QDoubleValidator.ScientificNotation)
        self.setValidator(v)
        self.editingFinished.connect(self._emit_number)

    @pyqtSlot()
    def _emit_number(self):
        number, is_valid = self.locale().toDouble(self.text())
        if is_valid:
            self.valid_number_entered.emit(number)


class CenteredCheckbox(QWidget):
    # This is a checkbox inside a QWidget to get centered alignment - see:
    # https://bugreports.qt.io/browse/QTBUG-5368
    def __init__(self, parent):
        super().__init__(parent)
        self.cbox = QCheckBox(self, checked=True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cbox)

    def isChecked(self):
        return self.cbox.isChecked()
