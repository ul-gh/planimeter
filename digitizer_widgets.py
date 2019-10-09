#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Digitizer Widgets

License: GPL version 3
2019-07-29 Ulrich Lukas
"""
import logging
logger = logging.getLogger(__name__)

from functools import partial
import numpy as np
from numpy import NaN, isnan

from PyQt5.QtCore import Qt, QLocale, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QIcon
from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit,
        QGroupBox, QLabel, QPushButton, QRadioButton, QCheckBox, QComboBox,
        QTableWidget, QTableWidgetItem, QToolBar,
        QStyle, QAction, 
        )

from upylib.pyqt_debug import logExceptionSlot


class TraceDataModelTab(QWidget):
    def __init__(self, digitizer, model, mplw):
        super().__init__(digitizer)
        self.digitizer = digitizer
        self.model = model
        self.mplw = mplw

        self.table_tr_conf = TraceConfTable(digitizer, model, mplw)


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
        # Row 3: Trace conf table
        layout.addWidget(self.table_tr_conf, 3, 0, 1, 3)


class CoordinateSystemTab(QWidget):
    def __init__(self, digitizer, model, mplw):
        super().__init__(digitizer)
        self.axconfw = AxConfWidget(digitizer, model, mplw)
        self.canvas_extents_box = CanvasExtentsBox(digitizer, model, mplw)
        layout = QVBoxLayout(self)
        layout.addWidget(self.axconfw)
        layout.addWidget(self.canvas_extents_box)
        layout.addStretch(1)


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
                model.x_ax.sect_data[0],
                "X Axis Start Value",
                model.num_fmt)
        self.xend_edit = SciLineEdit(
                model.x_ax.sect_data[1],
                "X Axis End Value",
                model.num_fmt)
        self.btn_lin_x = QRadioButton("Lin")
        self.btn_log_x = QRadioButton("Log")
        #### Group box for Y Coordinate picker and input boxes
        self.group_y = QGroupBox("Enter Y Axis Start and End Values")
        self.btn_pick_y = StyledButton("Pick Points", self)
        self.ystart_edit = SciLineEdit(
                model.y_ax.sect_data[0],
                "Y Axis Start Value",
                model.num_fmt)
        self.yend_edit = SciLineEdit(
                model.y_ax.sect_data[1],
                "Y Axis End Value",
                model.num_fmt)
        self.btn_lin_y = QRadioButton("Lin")
        self.btn_log_y = QRadioButton("Log")
        ##### Common settings
        # Enable make axes points mouse-pickable if set
        self.btn_drag_axes = QCheckBox("Axes Draggable")
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
        self.btn_drag_axes.toggled.connect(mplw.set_drag_axes)
        self.btn_store_config.toggled.connect(model.set_wants_persistent_storage)
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
        self.btn_drag_axes.setChecked(self.mplw._drag_axes)

    # Updates buttons and input boxes to represent the data model state
    # and also the new and current matplotlib widget operation mode.
    @logExceptionSlot()
    def update_model_view(self):
        x_ax = self.model.x_ax
        y_ax = self.model.y_ax
        self.set_green_x_ax(x_ax.is_complete)
        self.set_green_y_ax(y_ax.is_complete)
        # Update axis section value input boxes
        self.xstart_edit.setValue(x_ax.sect_data[0])
        self.xend_edit.setValue(x_ax.sect_data[1])
        self.ystart_edit.setValue(y_ax.sect_data[0])
        self.yend_edit.setValue(y_ax.sect_data[1])
        # Update log/lin radio buttons.
        self.btn_lin_x.setChecked(not x_ax.log_scale)
        self.btn_log_x.setChecked(x_ax.log_scale)
        self.btn_lin_y.setChecked(not y_ax.log_scale)
        self.btn_log_y.setChecked(y_ax.log_scale)
        # Pick axes points buttons
        self.btn_pick_x.set_green(x_ax.pts_px_valid)
        self.btn_pick_y.set_green(y_ax.pts_px_valid)
        # Store config button
        self.btn_store_config.setChecked(self.model.wants_persistent_storage)

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
        group_x_layout.setContentsMargins(6, 0, 6, 6)
        # Group Y layout
        group_y_layout = QHBoxLayout(self.group_y)
        group_y_layout.addWidget(self.ystart_edit)
        group_y_layout.addWidget(self.yend_edit)
        group_y_layout.addWidget(self.btn_lin_y)
        group_y_layout.addWidget(self.btn_log_y)
        group_y_layout.addWidget(self.btn_pick_y)
        group_y_layout.setContentsMargins(6, 0, 6, 6)
        # Common setings checkboxes
        common_btns_layout = QHBoxLayout()
        common_btns_layout.addWidget(self.btn_store_config)
        common_btns_layout.addWidget(self.btn_drag_axes)
        # This is all input boxes plus label
        axconfw_layout = QVBoxLayout(self)
        axconfw_layout.addWidget(self.group_x)
        axconfw_layout.addWidget(self.group_y)
        axconfw_layout.addLayout(common_btns_layout)
        axconfw_layout.setContentsMargins(0, 0, 0, 0)


class CanvasExtentsBox(QGroupBox):
    def __init__(self, digitizer, model, mplw):
        super().__init__("Data Coordinate System", digitizer)
        self.digitizer = digitizer
        self.model = model
        self.mplw = mplw

        ########## Widgets setup
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
        self.x_min_edit.valid_number_entered.connect(self._set_canvas_extents)
        self.x_max_edit.valid_number_entered.connect(self._set_canvas_extents)
        self.y_min_edit.valid_number_entered.connect(self._set_canvas_extents)
        self.y_max_edit.valid_number_entered.connect(self._set_canvas_extents)

        ########## Connect foreign signals
        #model.coordinate_system_changed.connect(self.update_model_view)
        # Update when matplotlib widget changes operating mode
        mplw.canvas_rescaled.connect(self.update_mplw_view)

    @logExceptionSlot()
    def update_model_view(self):
        pass

    @logExceptionSlot(int)
    def update_mplw_view(self, op_mode):
        xb = self.mplw.mpl_ax.get_xbound()
        yb = self.mplw.mpl_ax.get_ybound()
        bounds_px = np.concatenate((xb, yb)).reshape(-1, 2, order="F")
        bounds_data = self.model.px_to_data(bounds_px)
        if bounds_data.shape[0] == 0:
            return
        (x_min, y_min), (x_max, y_max) = bounds_data
        self.x_min_edit.setValue(x_min)
        self.x_max_edit.setValue(x_max)
        self.y_min_edit.setValue(y_min)
        self.y_max_edit.setValue(y_max)

    @logExceptionSlot(float)
    def _set_canvas_extents(self, _): # Signal value not needed
        xy_min = self.x_min_edit.value(), self.y_min_edit.value()
        xy_max = self.x_max_edit.value(), self.y_max_edit.value()
        xy_min_max = np.concatenate((xy_min, xy_max)).reshape(-1, 2)
        # Displays error message box for invalid data
        if not self.model.validate_data_pts(xy_min_max):
            return
        bounds_px = self.model.data_to_px(xy_min_max)
        self.mplw.set_canvas_extents(bounds_px)

    def _set_layout(self):
        layout = QGridLayout(self)
        # self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.x_range_label, 0, 0)
        layout.addWidget(self.x_min_edit, 0, 1)
        layout.addWidget(self.x_max_edit, 0, 2)
        layout.addWidget(self.y_range_label, 1, 0)
        layout.addWidget(self.y_min_edit, 1, 1)
        layout.addWidget(self.y_max_edit, 1, 2)


class TraceConfTable(QTableWidget):
    def __init__(self, digitizer, model, mplw):
        self.model = model
        self.mplw = mplw
        headers = ["Name", "Pick Points", "Enable", "Export",
                   "X Start", "X End"]
        self.col_xstart = headers.index("X Start")
        self.col_xend = headers.index("X End")
        n_traces = len(model.traces)
        n_headers = len(headers)
        self.btns_pick_trace = []
        self.cbs_export = []
        self.cbs_enable = []
        super().__init__(n_traces, n_headers, digitizer)
#        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        self.setHorizontalHeaderLabels(headers)
        # Data Export options
        ###

        ###
        i_types = {"linear": "Linear", "cubic": "C-Splines"}
        for row, tr in enumerate(self.model.traces):
            name = QTableWidgetItem(tr.name)
            btn_pick_trace = NumberedButton(row, "Pick!", self)
            self.btns_pick_trace.append(btn_pick_trace)
            cb_export = NumberedCenteredCheckbox(row, self)
            self.cbs_export.append(cb_export)
            cb_enable = NumberedCenteredCheckbox(row, self)
            self.cbs_enable.append(cb_enable)
            #x_start = QTableWidgetItem(f"{tr.x_start_export}")
            #x_end = QTableWidgetItem(f"{tr.x_end_export}")
            combo_i_type = QComboBox(self)
            for key, value in i_types.items():
                combo_i_type.addItem(value, key)
            self.setItem(row, 0, name)
            self.setCellWidget(row, 1, btn_pick_trace)
            self.setCellWidget(row, 2, combo_i_type)
            self.setCellWidget(row, 3, cb_enable)
            self.setCellWidget(row, 4, cb_export)
            #self.setItem(row, 4, x_start)
            #self.setItem(row, 5, x_end)
            ##### Signals
            cb_export.i_toggled.connect(self.set_export)
            cb_enable.i_toggled.connect(self.mplw.enable_trace)

        ########## Initialise view from model
        self.update_model_view()
        self.update_mplw_view(mplw.MODE_DEFAULT)

        ########## Connect own and sub-widget signals
        #self.itemSelectionChanged.connect(self._handle_selection)
        for btn in self.btns_pick_trace:
            btn.i_toggled.connect(mplw.set_mode_add_trace_pts)

        ########## Connect foreign signals
        # Update when trace config changes, e.g. if traces are added or renamed
        model.tr_conf_changed.connect(self.update_model_view)
        # Update when matplotlib widget changes operating mode
        mplw.mode_sw.connect(self.update_mplw_view)

    @logExceptionSlot()
    def update_model_view(self):
        for i, trace in enumerate(self.model.traces):
            self.cbs_export[i].setChecked(trace.export)

    @logExceptionSlot(int)
    def update_mplw_view(self, op_mode):
        for i, trace in enumerate(self.model.traces):
            self.btns_pick_trace[i].setChecked(
                    i == self.mplw.curr_trace_no
                    and op_mode == self.mplw.MODE_ADD_TRACE_PTS)
            self.cbs_enable[i].setChecked(self.mplw.is_enabled_trace(i))

    @logExceptionSlot(int, bool)
    def set_export(self, trace_no, state=True):
        trace = self.model.traces[trace_no]
        trace.export = state

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


class ExportSettingsTab(QWidget):
    def __init__(self, digitizer, model, mplw):
        super().__init__(digitizer)
        ######### Shortcuts to the data model
        self.model = model
        self.mplw = mplw

        ######### Setup widgets
        ##### Upper grid layout: inputs + labels
        self.btn_preview = StyledButton("Preview Points")
        self.lbl_extrapolation = QLabel("Extrapolation",
                                        alignment=Qt.AlignCenter)
        self.combo_extrapolation = QComboBox()
        self.combo_extrapolation.addItems(["Trend", "Constant", "None"])

        self.lbl_traces_select = QLabel("Export Traces",
                                        alignment=Qt.AlignCenter)
        self.combo_traces_select = QComboBox()
        self.combo_traces_select.addItems(
                ["All Selected", "Trace 1", "Trace 2"])
        self.lbl_grid_type = QLabel("Grid Type", alignment=Qt.AlignCenter)
        self.combo_grid_type = QComboBox()
        self.combo_grid_type.addItems(
                ["Adaptive Individual", "Lin Fixed N", "Lin Fixed Step",
                 "Log Fixed N/dec"]
                )

        self.lbl_n_pts = QLabel("Total N Points", alignment=Qt.AlignCenter)
        self.combo_n_pts = QComboBox()
        self.combo_n_pts.setEditable(True)
        self.combo_n_pts.setValidator(QIntValidator())
        self.combo_n_pts.addItems(["10", "15", "20", "35", "50", "100"])
        self.lbl_grid_parameter = QLabel("Step Size / N/dec / Q",
                                         alignment=Qt.AlignCenter)
        self.edit_grid_parameter = SciLineEdit(
                0.1, "Step Size", self.model.num_fmt)

        ##### Lower grid layout: Definition range display and export range edit
        self.lbl_definition_range = QLabel("Definition Range Start/End:")
        self.lbl_export_range = QLabel("Export Range Start/End:")

        self.edit_definition_range_start = SciLineEdit(
                0.0, "Lower Limit", self.model.num_fmt)
        self.edit_definition_range_start.setReadOnly(True)
        self.edit_definition_range_start.setStyleSheet(
                "background-color: LightGrey")
        self.edit_x_start_export = SciLineEdit(
                self.model.x_start_export,
                "X Axis Start Value",
                self.model.num_fmt
                )

        self.edit_definition_range_end = SciLineEdit(
                10.0, "Upper Limmit", self.model.num_fmt)
        self.edit_definition_range_end.setReadOnly(True)
        self.edit_definition_range_end.setStyleSheet(
                "background-color: LightGrey")
        self.edit_x_end_export = SciLineEdit(
                self.model.x_end_export,
                "X Axis End Value",
                self.model.num_fmt
                )
        # Setup layout
        self._set_layout()
        
        ########## Initialise view from model
        self.update_model_view()
        self.update_mplw_view(mplw.MODE_DEFAULT)

        ########## Connect own and sub-widget signals
        # self.btn_export.toggled.connect(self.wip_do_export)

        ########## Connect foreign signals
        model.export_settings_changed.connect(self.update_model_view)
        # Update when matplotlib widget changes operating mode
        mplw.mode_sw.connect(self.update_mplw_view)

    @logExceptionSlot()
    def update_model_view(self):
        #self.btn_lin_export.setChecked(not self.model.x_log_scale_export)
        #self.btn_log_export.setChecked(self.model.x_log_scale_export)
        pass

    @logExceptionSlot(int)
    def update_mplw_view(self, op_mode):
        pass

    def _set_layout(self):
        ##### Upper Grid
        l_upper = QGridLayout()
        l_upper.setColumnStretch(1, 3)
        l_upper.setColumnStretch(2, 2)
        # Row 0
        l_upper.addWidget(self.lbl_traces_select, 0, 1)
        l_upper.addWidget(self.lbl_n_pts, 0, 2)
        # Row 1
        l_upper.addWidget(self.btn_preview, 1, 0)
        l_upper.addWidget(self.combo_traces_select, 1, 1)
        l_upper.addWidget(self.combo_n_pts, 1, 2)
        # Row 2
        l_upper.addWidget(self.lbl_extrapolation, 2, 0)
        l_upper.addWidget(self.lbl_grid_type, 2, 1)
        l_upper.addWidget(self.lbl_grid_parameter, 2, 2)
        # Row 3
        l_upper.addWidget(self.combo_extrapolation, 3, 0)
        l_upper.addWidget(self.combo_grid_type, 3, 1)
        l_upper.addWidget(self.edit_grid_parameter, 3, 2)
        ##### Lower Grid
        l_lower = QGridLayout()
        # Row 0
        l_lower.addWidget(self.lbl_definition_range, 0, 0)
        l_lower.addWidget(self.edit_definition_range_start, 0, 1)
        l_lower.addWidget(self.edit_definition_range_end, 0, 2)
        # Rows 1
        l_lower.addWidget(self.lbl_export_range, 1, 0)
        l_lower.addWidget(self.edit_x_start_export, 1, 1)
        l_lower.addWidget(self.edit_x_end_export, 1, 2)
        ##### Complete Layout
        l_outer = QVBoxLayout(self)
        l_outer.addLayout(l_upper)
        l_outer.addLayout(l_lower)
        # Fill up empty space to the bottom
        l_outer.addStretch(1)

########## Digitizer Toolbar
class DigitizerToolBar(QToolBar):
    def __init__(self, parent):
        super().__init__(parent)
        ########## Define new actions
        icon_export = QIcon.fromTheme(
                "document-save",
                self.style().standardIcon(QStyle.SP_DialogSaveButton))
        icon_send = QIcon.fromTheme(
                "document-send",
                self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.act_put_clipboard = QAction(
                icon_send,
                "Put to Clipboard",
                self)
        self.act_export_xlsx = QAction(
                icon_export,
                "Export data as XLSX",
                self)
        self.act_export_csv = QAction(
                icon_export,
                "Export data as CSV",
                self)
        # Dict of our new actions plus text
        self._actions = {self.act_export_csv: "Export{}CSV",
                         self.act_export_xlsx: "Export{}XLSX", 
                         self.act_put_clipboard: "Put Into{}Clipboard",
                         }
        ########## Add new actions to the toolbar
        self.addActions(self._actions.keys())
        ########## Connect own signals
        self.orientationChanged.connect(self._on_orientationChanged)
        ########## Initial view setup
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._on_orientationChanged(Qt.Horizontal)
        #self.setIconSize(self.iconSize() * 0.8)
        self.setStyleSheet("spacing:2px")

    
    @pyqtSlot(Qt.Orientation)
    def _on_orientationChanged(self, new_orientation):
        logger.debug("Digitizer Toolbar orientation change handler called")
        if new_orientation == Qt.Horizontal:
            self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            # Set text without line break
            for act, text in self._actions.items():
                act.setIconText(text.format(" "))
        else:
            self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            # Set text with line break
            for act, text in self._actions.items():
                act.setIconText(text.format("\n"))


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
        self.toggled.connect(partial(self.i_toggled.emit, index))

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


class NumberedCenteredCheckbox(QWidget):
    # This is a checkbox inside a QWidget to get centered alignment - see:
    # https://bugreports.qt.io/browse/QTBUG-5368
    i_toggled = pyqtSignal(int, bool)

    def __init__(self, index, parent, *args, **kwargs):
        super().__init__(parent)
        self._index = index
        self.cbox = QCheckBox(self, *args, **kwargs)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cbox)
        self.cbox.toggled.connect(partial(self.i_toggled.emit, index))

    def isChecked(self):
        return self.cbox.isChecked()
    
    @pyqtSlot(bool)
    def setChecked(self, state):
        self.cbox.setChecked(state)

    def index(self) ->int:
        return self._index

    @pyqtSlot(int)
    def setIndex(self, index: int):
        self._index = index
