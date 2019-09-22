#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Digitizer Widgets

License: GPL version 3
2019-07-29 Ulrich Lukas
"""
import logging
logger = logging.getLogger(__name__)

from functools import partial
from numpy import NaN, isclose, isnan

from PyQt5.QtCore import Qt, QLocale, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QMessageBox,
        QGroupBox, QLabel, QPushButton, QRadioButton, QCheckBox, QComboBox,
        QTableWidget, QTableWidgetItem, QSizePolicy
        )

from upylib.pyqt_debug import logExceptionSlot


class DataCoordProps(QGroupBox):
    def __init__(self, digitizer, model, mplw):
        super().__init__("Data Coordinate System", digitizer)
        self.digitizer = digitizer
        self.model = model
        self.mplw = mplw
        
        ########## Widgets setup
        self.cursor_xy_label = QLabel("Cursor X and Y data:")
        self.cursor_x_display = SciLineEdit()
        self.cursor_y_display = SciLineEdit()
        self.cursor_x_display.setReadOnly(True)
        self.cursor_y_display.setReadOnly(True)
        self.cursor_x_display.setStyleSheet("background-color: LightGrey")
        self.cursor_y_display.setStyleSheet("background-color: LightGrey")
        self.x_range_label = QLabel("Canvas X Data Extent:")
        self.x_min_edit = SciLineEdit()
        self.x_max_edit = SciLineEdit()
        self.y_range_label = QLabel("Canvas Y Data Extent:")
        self.y_min_edit = SciLineEdit()
        self.y_max_edit = SciLineEdit()
        # Layout setup
        self._set_layout()
        
        ########## Initialise view from model
        self.update_model_view()
        self.update_mplw_view(mplw.MODE_DEFAULT)

        ########## Connect own and sub-widget signals
        self.x_min_edit.valid_number_entered.connect(self._set_model_px_bounds)
        self.x_max_edit.valid_number_entered.connect(self._set_model_px_bounds)
        self.y_min_edit.valid_number_entered.connect(self._set_model_px_bounds)
        self.y_max_edit.valid_number_entered.connect(self._set_model_px_bounds)

        ########## Connect foreign signals
        model.coordinate_system_changed.connect(self.update_model_view)
        # Update when matplotlib widget changes operating mode
        mplw.canvas_rescaled.connect(self.update_mplw_view)

    @logExceptionSlot()
    def update_model_view(self):
        x_min, x_max = self.model.x_ax.pts_data
        y_min, y_max = self.model.y_ax.pts_data
        self.x_min_edit.setValue(x_min)
        self.x_max_edit.setValue(x_max)
        self.y_min_edit.setValue(y_min)
        self.y_max_edit.setValue(y_max)

    @logExceptionSlot(int)
    def update_mplw_view(self, op_mode):
        #x_min, x_max = self.mplw.mpl_ax.get_xbound()
        #y_min, y_max = self.mplw.mpl_ax.get_ybound()
        self.mplw.mouse_coordinates_updated.connect(self.update_xy_display)

    @logExceptionSlot(float, float)
    def update_xy_display(self, px_x: float, px_y: float):
        self.cursor_x_display.setValue(px_x)
        self.cursor_y_display.setValue(px_y)
        
    @logExceptionSlot()
    def _set_model_px_bounds(self, _): # Signal value not needed
        x_min_max = self.x_min_edit.value(), self.x_max_edit.value()
        y_min_max = self.y_min_edit.value(), self.y_max_edit.value()
        # Displays error message box for invalid data
        bbox = self.model.px_from_data_bounds(x_min_max, y_min_max)
        if bbox is not None:
            x_min_max_px, y_min_max_px = bbox
            self.mplw.mpl_ax.set_xbound(x_min_max)
            self.mplw.mpl_ax.set_ybound(y_min_max)
            self.mplw.canvas_qt.draw_idle()

    def _set_layout(self):
        layout = QGridLayout(self)
        # self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.cursor_xy_label, 0, 0)
        layout.addWidget(self.cursor_x_display, 0, 1)
        layout.addWidget(self.cursor_y_display, 0, 2)
        layout.addWidget(self.x_range_label, 1, 0)
        layout.addWidget(self.x_min_edit, 1, 1)
        layout.addWidget(self.x_max_edit, 1, 2)
        layout.addWidget(self.y_range_label, 2, 0)
        layout.addWidget(self.y_min_edit, 2, 1)
        layout.addWidget(self.y_max_edit, 2, 2)


class ExportSettingsBox(QGroupBox):
    def __init__(self, digitizer, model, mplw):
        super().__init__("Trace Export Settings", digitizer)
        ######### Shortcuts to the data model
        self.model = model
        self.mplw = mplw

        ######### Setup widgets
        # Export Data button
        self.btn_export = StyledButton("Export\nData", self)
        self.x_start_export_edit = SciLineEdit(
                self.model.x_start_export,
                "X Axis Start Value",
                self.model.num_fmt
                )
        self.x_end_export_edit = SciLineEdit(
                self.model.x_end_export,
                "X Axis End Value",
                self.model.num_fmt
                )
        self.btn_lin_export = QRadioButton("Lin")
        self.btn_log_export = QRadioButton("Log")
        # Setup layout
        self._set_layout()
        
        ########## Initialise view from model
        self.update_model_view()
        self.update_mplw_view(mplw.MODE_DEFAULT)

        ########## Connect own and sub-widget signals

        ########## Connect foreign signals
        model.export_settings_changed.connect(self.update_model_view)
        # Update when matplotlib widget changes operating mode
        mplw.mode_sw.connect(self.update_mplw_view)

    @logExceptionSlot()
    def update_model_view(self):
        self.btn_lin_export.setChecked(not self.model.x_log_scale_export)
        self.btn_log_export.setChecked(self.model.x_log_scale_export)

    @logExceptionSlot(int)
    def update_mplw_view(self, op_mode):
        pass
    
    def _set_layout(self):
        layout = QHBoxLayout(self)
        # self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.x_start_export_edit)
        layout.addWidget(self.x_end_export_edit)
        layout.addWidget(self.btn_lin_export)
        layout.addWidget(self.btn_log_export)
        layout.addWidget(self.btn_export)


class TraceConfTable(QTableWidget):
    def __init__(self, digitizer, model, mplw):
        self.model = model
        self.mplw = mplw
        headers = ["Name", "Pick Points", "Export", "X Start", "X End",
                   "Interpolation", "N Points"]
        self.col_xstart = headers.index("X Start")
        self.col_xend = headers.index("X End")
        n_traces = len(digitizer.model.traces)
        n_headers = len(headers)
        self.btns_pick_trace = []
        super().__init__(n_traces, n_headers, digitizer)
#        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        self.setHorizontalHeaderLabels(headers)
        # Data Export options
        ###
        n_interp_presets_text = ["10", "25", "50", "100", "250", "500", "1000"]
        n_interp_presets_values = [10, 25, 50, 100, 250, 500, 1000]
        ###
        i_types_text = ["Linear", "Cubic", "Sin(x)/x"]
        i_types_values = ["linear", "cubic", "sinc"]
        for row, tr in enumerate(self.model.traces):
            name = QTableWidgetItem(tr.name)
            btn_pick_trace = NumberedButton(row, f"Pick Trace {row+1}", self)
            self.btns_pick_trace.append(btn_pick_trace)
            checkbox_export = CenteredCheckbox(self)
            #x_start = QTableWidgetItem(f"{tr.x_start_export}")
            #x_end = QTableWidgetItem(f"{tr.x_end_export}")
            combo_i_type = QComboBox(self)
            combo_i_type.addItems(i_types_text)
            combo_n_interp = QComboBox(self)
            combo_n_interp.addItems(n_interp_presets_text)
            self.setItem(row, 0, name)
            self.setCellWidget(row, 1, btn_pick_trace)
            self.setCellWidget(row, 2, checkbox_export)
            #self.setItem(row, 3, x_start)
            #self.setItem(row, 4, x_end)
            self.setCellWidget(row, 5, combo_i_type)
            self.setCellWidget(row, 6, combo_n_interp)

        ########## Initialise view from model
        self.update_model_view()
        self.update_mplw_view(mplw.MODE_DEFAULT)

        ########## Connect own and sub-widget signals
        #self.itemSelectionChanged.connect(self._handle_selection)
        for btn in self.btns_pick_trace:
            btn.i_toggled.connect(mplw.set_mode_add_trace_pts)

        ########## Connect foreign signals
        # Update when trace config changes, e.g. if traces are added or renamed
        model.tr_input_data_changed.connect(self.update_model_view)
        # Update when matplotlib widget changes operating mode
        mplw.mode_sw.connect(self.update_mplw_view)

    @logExceptionSlot()
    def update_model_view(self):
        pass

    @logExceptionSlot(int)
    def update_mplw_view(self, op_mode):
        if op_mode == self.mplw.MODE_ADD_TRACE_PTS:
            for i, btn in enumerate(self.btns_pick_trace):
                btn.setChecked(i == self.mplw.curr_trace_no)
        else:
            for btn in self.btns_pick_trace:
                btn.setChecked(False)

#    def _handle_selection(self):
#        self.sel_traces = sel_traces = {
#                s.row() for s in self.selectedIndexes()
#                if s.column() in (self.col_xstart, self.col_xend)
#                and s.row() < self.n_traces
#                }
#        if sel_traces:
#            self._show_xrange = True
#            inf = float("inf")
#            x_start = -inf
#            x_end = inf
#            for i in sel_traces:
#                item_start = self.item(i, self.col_xstart)
#                item_end = self.item(i, self.col_xend)
#                xs_new = float(item_start.text())
#                xe_new = float(item_end.text())
#                if xs_new < x_start:
#                    x_start = xs_new
#                if xe_new < x_end:
#                    x_end = xe_new
#            for i in sel_traces:
#                self.model.traces[i].x_start_export = x_start
#                self.model.traces[i].x_end_export = x_end
#                self.x_start_export = x_start
#                self.x_end_export = x_end
#            #self.show_xrange.emit(True)
#        elif self._show_xrange:
#            self._show_xrange = False
#            #self.show_xrange.emit(False)


class AxConfWidget(QWidget):
    def __init__(self, digitizer, model, mplw):
        super().__init__(digitizer)
        ######### Access to the data model
        self.model = model
        # Matplotlib widget state
        self.mplw = mplw

        ######### Qt widget setup
        #### Group box for X Coordinate picker and input boxes
        self.group_x = QGroupBox("Enter X Axis Start and End Values")
        self.btn_pick_x = StyledButton("Pick Points", self)
        self.xstart_edit = SciLineEdit(
                model.x_ax.pts_data[0],
                "X Axis Start Value",
                model.num_fmt)
        self.xend_edit = SciLineEdit(
                model.x_ax.pts_data[1],
                "X Axis End Value",
                model.num_fmt)
        self.btn_lin_x = QRadioButton("Lin")
        self.btn_log_x = QRadioButton("Log")
        #### Group box for Y Coordinate picker and input boxes
        self.group_y = QGroupBox("Enter Y Axis Start and End Values")
        self.btn_pick_y = StyledButton("Pick Points", self)
        self.ystart_edit = SciLineEdit(
                model.y_ax.pts_data[0],
                "Y Axis Start Value",
                model.num_fmt)
        self.yend_edit = SciLineEdit(
                model.y_ax.pts_data[1],
                "Y Axis End Value",
                model.num_fmt)
        self.btn_lin_y = QRadioButton("Lin")
        self.btn_log_y = QRadioButton("Log")
        # Store plot config button
        self.btn_store_config = QCheckBox("Store Config")
        # Setup Layout
        self._set_layout()

        ########## Initialise view from model
        self.update_model_view()
        self.update_mplw_view(mplw.MODE_DEFAULT)

        ########## Connect own and sub-widget signals
        self.btn_pick_x.toggled.connect(mplw.set_mode_setup_x_axis)
        self.btn_pick_y.toggled.connect(mplw.set_mode_setup_y_axis)
        self.btn_log_x.toggled.connect(model.x_ax.set_log_scale)
        self.btn_log_y.toggled.connect(model.y_ax.set_log_scale)
        self.btn_store_config.toggled.connect(model.set_persistent_storage)
        # Number input boxes emit float signals.
        self.xstart_edit.valid_number_entered.connect(model.x_ax.set_ax_start)
        self.ystart_edit.valid_number_entered.connect(model.y_ax.set_ax_start)
        self.xend_edit.valid_number_entered.connect(model.x_ax.set_ax_end)
        self.yend_edit.valid_number_entered.connect(model.y_ax.set_ax_end)

        ########## Connect foreign signals
        # Update when axes config changes
        model.ax_input_data_changed.connect(self.update_model_view)
        # Update when matplotlib widget changes operating mode
        mplw.mode_sw.connect(self.update_mplw_view)

    ########## Slots
    # Updates state of the Matplotlib widget display by setting down the
    # the buttons when each mode is active
    @logExceptionSlot(int)
    def update_mplw_view(self, op_mode):
        self.btn_pick_x.setChecked(op_mode == self.mplw.MODE_SETUP_X_AXIS)
        self.btn_pick_y.setChecked(op_mode == self.mplw.MODE_SETUP_Y_AXIS)

    # Updates buttons and input boxes to represent the data model state
    # and also the new and current matplotlib widget operation mode.
    @logExceptionSlot()
    def update_model_view(self):
        x_ax = self.model.x_ax
        y_ax = self.model.y_ax
        self.set_green_x_ax(x_ax.is_complete())
        self.set_green_y_ax(y_ax.is_complete())
        # Update axis section value input boxes
        self.xstart_edit.setValue(x_ax.pts_data[0])
        self.xend_edit.setValue(x_ax.pts_data[1])
        self.ystart_edit.setValue(y_ax.pts_data[0])
        self.yend_edit.setValue(y_ax.pts_data[1])
        # Update log/lin radio buttons.
        self.btn_lin_x.setChecked(not x_ax.log_scale())
        self.btn_log_x.setChecked(x_ax.log_scale())
        self.btn_lin_y.setChecked(not y_ax.log_scale())
        self.btn_log_y.setChecked(y_ax.log_scale())
        # Pick axes points buttons
        self.btn_pick_x.set_green(x_ax.valid_pts_px())
        self.btn_pick_y.set_green(y_ax.valid_pts_px())
        # Store config button
        self.btn_store_config.setChecked(self.model.persistent_storage())

    def set_green_x_ax(self, state):
        # Background set to green when model has valid data
        style = "QLineEdit { background-color: Palegreen; }" if state else ""
        self.group_x.setStyleSheet(style)

    def set_green_y_ax(self, state):
        # Background set to green when model has valid data
        style = "QLineEdit { background-color: Palegreen; }" if state else ""
        self.group_y.setStyleSheet(style)

    def _set_layout(self):
        # Group X layout
        group_x_layout = QHBoxLayout(self.group_x)
        group_x_layout.addWidget(self.xstart_edit)
        group_x_layout.addWidget(self.xend_edit)
        group_x_layout.addWidget(self.btn_lin_x)
        group_x_layout.addWidget(self.btn_log_x)
        group_x_layout.addWidget(self.btn_pick_x)        
        # Group Y layout
        group_y_layout = QHBoxLayout(self.group_y)
        group_y_layout.addWidget(self.ystart_edit)
        group_y_layout.addWidget(self.yend_edit)
        group_y_layout.addWidget(self.btn_lin_y)
        group_y_layout.addWidget(self.btn_log_y)
        group_y_layout.addWidget(self.btn_pick_y)
        # This is all input boxes plus label
        axconfw_layout = QHBoxLayout(self)
        axconfw_layout.addWidget(self.group_x)
        axconfw_layout.addWidget(self.group_y)
        axconfw_layout.addWidget(self.btn_store_config)


########## Custom Widgets Used Above
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
    def set_green(self, state=True):
        style = "background-color: Palegreen" if state else ""
        self.setStyleSheet(style)


class NumberedButton(StyledButton):
    """This subclass of QPushButton adds a number index property and
    emits a corresponding signal emitting index and new state.
    """
    i_toggled = pyqtSignal(int, bool)
    def __init__(self, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = index
        self.toggled.connect(
                partial(self.i_toggled.emit, index)
                )

    def index(self) ->int:
        return self._index

    @pyqtSlot(int)
    def setIndex(self, index: int):
        self._index = index


class SciLineEdit(QLineEdit):
    """QLineEdit with added validator for scientific notation input.
    Also, this takes a number as preset value instead of a string
    and holds a validated number property.
    """
    valid_number_entered = pyqtSignal(float)
    def __init__(
            self,
            preset_value=NaN,
            placeholderText="",
            num_fmt=".6G",
            *args,
            **kwargs
            ):
        text = "" if isnan(preset_value) else f"{preset_value:{num_fmt}}"
        super().__init__(
                text,
                *args,
                placeholderText=placeholderText,
                **kwargs
                )
        self.editingFinished.connect(self._update_value)
        # Defaults to notation=QDoubleValidator.ScientificNotation
        validator = QDoubleValidator(self)
        validator.setLocale(QLocale("en_US"))
        self.setValidator(validator)
        self._num_fmt = num_fmt
        self._value = preset_value
    
    def value(self) -> float:
        return self._value
    @pyqtSlot(float)
    def setValue(self, value: float):
        self._value = value
        text = "" if isnan(value) else f"{value:{self._num_fmt}}"
        self.setText(text)

    @pyqtSlot()
    def _update_value(self):
        self._value = float(self.text().replace(",", "."))
        self.valid_number_entered.emit(self._value)


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
