#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Modeling Tool Digitizer Widget

License: GPL version 3
"""
import logging
logger = logging.getLogger(__name__)

import io
import os
import tempfile
import numpy as np
from numpy import NaN, isnan
from typing import Optional

from PyQt5.QtCore import Qt, QDir, QSize, QMimeData, QTimer, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QIcon, QIntValidator
from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QGroupBox, QLabel, QRadioButton, QCheckBox, QComboBox, QFileDialog,
        QTableWidget, QTableWidgetItem, QStyle, QAction, QTabWidget, QSplitter,
         QPushButton, QLineEdit)
from custom_standard_widgets import (
        SciLineEdit, SmallSciLineEdit, StyledButton, NumberedButton,
        NumberedCenteredCheckbox, CustomisedMessageBox,
        )

import matplotlib.figure
import matplotlib.image
#from matplotlib.patches import Polygon
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT

from plot_model import Trace, Axis, DescriptiveIDs
from upylib.pyqt_debug import logExceptionSlot


class Digitizer(QWidget):
    def __init__(self, plot_model_assistant, plot_model, index, conf):
        super().__init__(plot_model_assistant)
        self.conf = plot_model_assistant.conf
        ##### Back reference to MultiPlotAssistant instance
        self.plot_model_assistant = plot_model_assistant
        self.set_wdir(conf.app_conf.wdir)
        ##### Data Model for one plot of one or more traces
        self.plot_model = plot_model
        #### Index conforming with index of plots in a multi-plot model
        self.index = index
        ##### Add widgets
        # Messagebox for confirming points deletion etc.
        self.messagebox = CustomisedMessageBox(self)
        # Splitter dividing the horizontal space in two halves
        self.splitter = QSplitter()
        # Right side shows single-plot configuration tabs
        self.tabs = QTabWidget()
        # Central Matplotlib Widget
        self.mpl_widget = MplWidget(self, plot_model)
        logger.debug(f"Added matplotlib widget: {self.mpl_widget}")
        ########## Toolbar to be later added to Main Window
        self.toolbar = DigitizerToolBar(self.mpl_widget.canvas_qt, self)
        # System clipboard instance already used by MplWidget
        self.clipboard = self.mpl_widget.clipboard
        # Push buttons and axis value input fields widget.
        self.tab_coordinate_system = CoordinateSystemTab(self.mpl_widget, plot_model)
        # Trace Data Model tab
        self.tab_trace_conf = TraceConfTab(self.mpl_widget, plot_model)
        # Export options box
        self.tab_export_settings = ExportSettingsTab(self.mpl_widget, plot_model)
        # Add all tabs to QTabWidget
        self.tabs.addTab(self.tab_coordinate_system, "Coordinate System")
        self.tabs.addTab(self.tab_trace_conf, "Traces")
        self.tabs.addTab(self.tab_export_settings, "Export Settings")
        # Custom Dialogs for file selection
        self.dlg_open_image_file = QFileDialog(
                self, "Open Source Image", self.wdir, "Images (*.png *.jpg *.jpeg)")
        self.dlg_export_csv = QFileDialog(
                self, "Export CSV", self.wdir, "Text/CSV (*.csv *.txt)")
        self.dlg_export_xlsx = QFileDialog(
                self, "Export XLS/XLSX", self.wdir, "Excel (*.xlsx)")
        self._set_layout()
        
        ##### Connect own and sub-widget signals
        # Mplwidget dialog box signals
        self.dlg_open_image_file.directoryEntered.connect(self.set_wdir)
        # Own Dialog box signals
        self.dlg_open_image_file.fileSelected.connect(
                self.mpl_widget.load_image_file)
        self.dlg_export_csv.fileSelected.connect(
                self.export_csv)
        self.dlg_export_xlsx.fileSelected.connect(
                lambda _: self.messagebox.show_warning("Not yet implemented!"))
        # ToolBar signals
        self.toolbar.act_open_file.triggered.connect(self.dlg_open_image_file.open)
        self.toolbar.act_load_clipboard.triggered.connect(
                self.mpl_widget.load_clipboard_image)
        self.toolbar.act_export_csv.triggered.connect(self.dlg_export_csv.open)
        self.toolbar.act_export_xlsx.triggered.connect(self.dlg_export_xlsx.open)
        self.toolbar.act_put_clipboard.triggered.connect(self.put_clipboard)


    @pyqtSlot(str)
    def set_wdir(self, abs_path):
        # Set working directory to last opened file directory
        self.wdir = abs_path if os.path.isdir(abs_path) else QDir.homePath()
        self.plot_model_assistant.set_wdir(self.wdir)

    # This is connected to from the main window toolbar!
    @logExceptionSlot(str)
    def export_csv(self, filename):
        """Export CSV textstring to file
        """
        trace = self.curr_plot.traces[self.curr_digitizer.curr_trace_no]
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
            self.messagebox.show_error(e)

    @logExceptionSlot(bool)
    def put_clipboard(self, state=True, pts_data=None):
        trace = self.curr_plot.traces[self.curr_digitizer.curr_trace_no]
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

    def sizeHint(self):
        return self.splitter.sizeHint()

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

    # Layout is two columns of widgets, arranged by movable splitter widgets
    def _set_layout(self):
        # Horizontal splitter layout is left and right side combined
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.mpl_widget)
        self.splitter.addWidget(self.tabs)
        logger.debug(f"DE_DEDE_DEBUG BEFORE Layout: {self.splitter.widget(0)}")
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        # All combined
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)
        logger.debug(f"DE_DEDE_DEBUG After Layout: {self.splitter.widget(0)}")

    def __repr__(self):
        return f"<Digitizer for model: {self.plot_model.name}>"


class MplWidget(QWidget):
    """This is the core graphic widget based on a matplotlib Qt backend.

    This displays the input image on the canvas using an unscaled pixel
    coordinate system.

    Using pixel coordinate based mouse events, a data coordinate system
    is defined by the user, this data is entered into the interactive
    data model for calculating the corresponding forward and inverse
    coordinate transformation matrix.
    
    When the data coordinate transformation is thus set up, plot traces
    can be entered which the model transforms into data space for output
    and further processing, e.g. an interpolated or fitted version is
    backplotted on the pixel canvas.

    2019-07-29 Ulrich Lukas
    """
    ########## Operation modes
    MODE_DEFAULT = 0
    MODE_SETUP_X_AXIS = 1
    MODE_SETUP_Y_AXIS = 2
    MODE_ADD_TRACE_PTS = 3
    MODE_DRAG_OBJ = 4
    ########## Custom Qt signals for interaction with the input widget
    # Emitted on mode switch, represents the new mode and is used to update
    # the dependand widgets, i.e. state display in configuration boxes.
    mode_sw = pyqtSignal(int)
    # Emitted when the figure canvas is loaded initially or rescaled to inform
    # about bounding box size etc. Int argument is the current operation mode.
    canvas_rescaled = pyqtSignal()
    # Emitts new X and Y coords in data space when pointer is inside mpl canvas.
    mouse_coordinates_updated = pyqtSignal(float, float)

    def __init__(self, digitizer, model):
        super().__init__(digitizer)
        self.digitizer = digitizer
        self.conf = digitizer.conf
        # System clipboard access
        self.clipboard = QApplication.instance().clipboard()
        # Messagebox for confirming points deletion etc.
        self.messagebox = CustomisedMessageBox(self)
        ########## Access to the plot data model
        self.model = model
        ########## View-Model-Mapping
        # Mapping of individual view objects (lines, points) to the associated
        # data model components. Dict keys are pyplot.Line2D objects.
        self._view_model_map = {}

        ########## Operation state
        # What happens when the plot canvas is clicked..
        self._op_mode = self.MODE_DEFAULT
        self.curr_trace_no = 0
        # Enabling axes picking can lead to difficulties picking trace points
        # if these are close to the axis lines. Thus defaulting to False.
        self._drag_axes = False
        # This stores the pyplot.Lines2D object when a plot item was picked
        self._picked_obj = None
        # Model component with view associated data for a mouse-picked object
        self._picked_obj_submodel = None
        # Index of a single picked point inside the view object
        self._picked_obj_pt_index = 0
        # All Lines2D objects that have to be redrawn on model updates
        self._blit_view_objs = []
        # Flag indicating if a full redraw is needed
        self._blit_buffer_stale = True

        ########## Matplotlib figure and axes setup
        self.fig = matplotlib.figure.Figure()

        ########## Qt widget setup
        # FigureCanvas factory of matplotlib Qt Agg backend takes the Figure
        # object and returns the matplotlib canvas as a QWidget instance.
        self.canvas_qt = FigureCanvas(self.fig)
        # QTimer used for delayed screen updates in case the canvas is resized
        self._redraw_timer = QTimer(self, interval=500, singleShot=True)
        # Data coordinate display box displayed above plot canvas
        self._setup_coordinates_display()
        # Add canvas_qt to own layout
        self._set_layout()
        
        # One matplotlib AxesSubplot instance is used
        self.mpl_ax = self.fig.add_subplot(111, autoscale_on=False)
        self.mpl_ax.xaxis.set_visible(False)
        self.mpl_ax.yaxis.set_visible(False)
        # Set matplotlib layout to fill space, with some margin for borders
        self.fig.subplots_adjust(0.001, 0.002, 0.999, 0.999)

        ########## Initialise view from model
        self.update_model_view()
       
        ########## Connect own signals
        # Matplotlib signals
        self.canvas_qt.mpl_connect("key_press_event", self._on_key_press)
        self.canvas_qt.mpl_connect("figure_enter_event", self._on_figure_enter)
        self.canvas_qt.mpl_connect("figure_leave_event", self._on_figure_leave)
        self.canvas_qt.mpl_connect("button_press_event", self._on_button_press)
        self.canvas_qt.mpl_connect("button_release_event", self._on_button_release)
        self.canvas_qt.mpl_connect("motion_notify_event", self._on_motion_notify)
        self.canvas_qt.mpl_connect("pick_event", self._on_pick)
        self.canvas_qt.mpl_connect("resize_event", self._on_resize)
        # QTimer event updates the blit buffer after screen resizing
        self._redraw_timer.timeout.connect(self._do_blit_redraw)

        ########## Connect foreign signals
        # Complete traces update
        model.tr_conf_changed.connect(self.update_model_view)
        # Update plot view displaying axes points and origin
        model.ax_conf_changed.connect(self.update_model_view_axes)
        # Re-display pixel-space input points when model has updated data.
        model.output_data_changed.connect(self.update_model_view_traces_data)
        model.output_data_changed[int].connect(self.update_model_view_traces_data)


    ########## Configuration
    @pyqtSlot(bool)
    def set_drag_axes(self, state):
        self._drag_axes = state
    
    def set_canvas_extents(self, bounds_px: np.array):
        bounds_px = np.array(bounds_px, copy=False)
        if not isnan(bounds_px).any() and bounds_px.shape == (2, 2):
            self.mpl_ax.set_xbound(bounds_px[:,0])
            self.mpl_ax.set_ybound(bounds_px[:,1])
            self._blit_buffer_stale = True
            self._do_blit_redraw()

    ########## Data Input Methods
    @logExceptionSlot(int, bool)
    def enable_trace(self, trace_no, state=True):
        trace = self.model.traces[trace_no]
        trace.pts_view_obj.set_visible(state)
        trace.pts_i_view_obj.set_visible(state)
        self._blit_buffer_stale = True
        self.update_model_view_traces_data(trace_no)

    @logExceptionSlot()
    def update_model_view(self):
        """Complete update of axes and traces features from model data
        """
        self.update_model_view_axes()
        for tr in self.model.traces:
            if tr.pts_view_obj is not None:
                del self._view_model_map[tr.pts_view_obj]
                tr.pts_view_obj.remove()
                tr.pts_view_obj = None
            if tr.pts_i_view_obj is not None:
                del self._view_model_map[tr.pts_i_view_obj]
                tr.pts_i_view_obj.remove()
                tr.pts_i_view_obj = None
        self._blit_buffer_stale = True
        self.update_model_view_traces_data()
    
    @logExceptionSlot()
    def update_model_view_axes(self):
        """Updates axes model features displayed in plot widget,
        including origin point only if it fits inside the canvas.

        This also registers the view objects back into the model
        and emits a signal informing when the canvas has been
        re-drawn.
        """
        #logger.debug("update_model_view_axes called")
        model = self.model
        ########## X and Y axis:
        for ax in model.x_ax, model.y_ax:
            if ax.pts_view_obj is not None:
                ax.pts_view_obj.set_data(*ax.pts_px.T)
            else:
                ax.pts_view_obj, = self.mpl_ax.plot(*ax.pts_px.T, **ax.pts_fmt)
                self._view_model_map[ax.pts_view_obj] = ax
            ax.pts_view_obj.set_label(f"Pixel Section Defining the {ax.name}")
        ########## Origin:
        # Containment check via numpy elementwise operators
        if (    isnan(model.origin_px).any()
                or (model.origin_px > self.fig.bbox.size).any()
                or (model.origin_px < 0.0).any()
                ):
            if model.origin_view_obj is not None:
                del self._view_model_map[model.origin_view_obj]
                model.origin_view_obj.remove()
                model.origin_view_obj = None
        else:
            if model.origin_view_obj is None:
                model.origin_view_obj, = self.mpl_ax.plot(
                        *model.origin_px, **model.origin_fmt)
                # The origin point is calculated and not supposed to be
                # subjected to drag-and-drop etc.: registering it as None
                self._view_model_map[model.origin_view_obj] = model.origin_view_obj
            else:
                model.origin_view_obj.set_data(*model.origin_px)
            model.origin_view_obj.set_label("Data Axes Origin")
        ##### Redraw axes and origin view objects
        self._do_blit_redraw()

    @pyqtSlot()
    @pyqtSlot(int)
    def update_model_view_traces_data(self, trace_no=None):
        """Draw or redraw a trace from the data model.
        
        If the argument is None, update all traces.
        This also registers the view objects back into the model.
        """
        #logger.debug(f"update_model_view_traces_data called for trace: {trace_no}")
        model = self.model
        traces = model.traces if trace_no is None else [model.traces[trace_no]]
        self._tr_view_objs = []
        ########## Update interpolated trace if available
        for tr in traces:
            ##### STEP A: Draw or update raw pixel points
            if tr.pts_view_obj is None:
                tr.pts_view_obj, = self.mpl_ax.plot(*tr.pts_px.T, **tr.pts_fmt)
                self._view_model_map[tr.pts_view_obj] = tr
            else:
                tr.pts_view_obj.set_data(*tr.pts_px.T)
            tr.pts_view_obj.set_label(f"Raw Points for {tr.name}")
            self._tr_view_objs.append(tr.pts_view_obj)
            ##### STEP B: Draw or update interpolated pixel points
            # Backtransform trace to pixel data coordinate system
            pts_i_px = model.linscale_to_px(tr.pts_linscale_i)
            if tr.pts_i_view_obj is None:
                # Draw trace on matplotlib widget
                tr.pts_i_view_obj, = self.mpl_ax.plot(*pts_i_px.T, **tr.pts_i_fmt)
                self._view_model_map[tr.pts_i_view_obj] = tr.pts_i_view_obj
            else:
                # Trace handle for pts_linscale_i exists. Update data.
                tr.pts_i_view_obj.set_data(*pts_i_px.T)
            tr.pts_i_view_obj.set_label(f"Interpolated Points for {tr.name}")
            self._tr_view_objs.append(tr.pts_i_view_obj)
        ##### Redraw traces view objects
        self._do_blit_redraw()


    @logExceptionSlot(str)
    def load_image_file(self, filename):
        """Load source/input image for digitizing.
        """
        logger.debug(f"load_image called with argument: {filename}")
        # Remove existing image from plot canvas if present
        if hasattr(self, "_mpl_axes_image"):
            self._mpl_axes_image.remove()
        try:
            if os.path.isfile(filename):
                image = matplotlib.image.imread(filename)
            else:
                self.messagebox.show_warning("Cannot open file. Not an image?")
                return
        except Exception as e:
            self.messagebox.show_error(e)
            return
        #all_except_image = self._blit_view_objs + self._blit_view_objs
        #for obj in all_except_image:
        #    obj.set_visible(False)
        self._mpl_axes_image = self.mpl_ax.imshow(
                image[-1::-1],
                interpolation=self.conf.app_conf.img_interpolation,
                origin="lower",
                zorder=0,
                )
        self.mpl_ax.autoscale(enable=True)
        # Complete canvas redraw
        self.canvas_qt.draw()
        self.mpl_ax.autoscale(enable=False)
        self.canvas_rescaled.emit()
        self.update_model_view()


    @logExceptionSlot()
    def load_clipboard_image(self, state):
        # Filename for temporary storage of clipboard images
        self.temp_filename = os.path.join(
                tempfile.gettempdir(),
                f"Clipboard Paste Image for {self.model.name}.png")
        image = self.clipboard.image()
        if image.isNull():
            self.messagebox.show_warning("No image data found in clipboard!")
            return
        image.save(self.temp_filename, format="png")
        self.load_image_file(self.temp_filename)


    ########## Data Output Methods
    def is_enabled_trace(self, trace_no: int) -> bool:
        tr = self.model.traces[trace_no]
        if None in (tr.pts_view_obj, tr.pts_i_view_obj):
            return False
        return tr.pts_view_obj.get_visible() or tr.pts_i_view_obj.get_visible()

    def print_model_view_items(self):
        """Prints info about the currently plotted view items
        """
        view_items_text = ",\n".join([str(i) for i in self.mpl_ax.lines])
        print(view_items_text)

    ########## State Machine Methods
    @logExceptionSlot(bool)
    def set_mode_setup_x_axis(self, state=True):
        if state:
            self._blit_buffer_stale = True
            self.set_mode(self.MODE_SETUP_X_AXIS)
        else:            
            self.set_mode(self.MODE_DEFAULT)

    @logExceptionSlot(bool)
    def set_mode_setup_y_axis(self, state=True):
        if state:
            self.set_mode(self.MODE_SETUP_Y_AXIS)
        else:
            self.set_mode(self.MODE_DEFAULT)

    @logExceptionSlot(int, bool)
    def set_mode_add_trace_pts(self, trace_no, state=True):
        self.curr_trace_no = trace_no
        if state:
            self.set_mode(self.MODE_ADD_TRACE_PTS)
        else:
            self.set_mode(self.MODE_DEFAULT)


    @logExceptionSlot(int)
    def set_mode(self, new_mode: int):
        logger.debug(f"set_mode called with argument: {new_mode}.")
        # Prevent recursive calls when external widgets are updated
        if new_mode == self._op_mode:
            return
        # Leave old mode
        previous_mode = self._op_mode
        logger.debug(f"Leaving operation mode: {previous_mode}.")
        # Call handlers for leaving each operation mode
        if previous_mode == self.MODE_ADD_TRACE_PTS:
            self._leave_mode_add_trace_pts()
        # Enter new mode
        self._op_mode = new_mode
        logger.debug(f"Entering new operation mode: {new_mode}.")
        # Call hanlers setting up each operation mode
        if new_mode == self.MODE_SETUP_X_AXIS:
            self._enter_mode_setup_x_axis()
        elif new_mode == self.MODE_SETUP_Y_AXIS:
            self._enter_mode_setup_y_axis()
        elif new_mode == self.MODE_ADD_TRACE_PTS:
            self._enter_mode_add_trace_pts()
        elif new_mode == self.MODE_DRAG_OBJ:
            self._enter_mode_drag_obj()
        else:
            self._op_mode = self.MODE_DEFAULT
            self._enter_mode_default()
        self.mode_sw.emit(self._op_mode)

    def sizeHint(self):
        if hasattr(self, "_mpl_axes_image"):
            img_height, img_width = self._mpl_axes_image.get_size()
        else:
            img_height, img_width = 100, 100
        coords_height = self.cursor_x_display.sizeHint().height()
        return QSize(img_width+3, coords_height+img_height+2)
        


    ##### Handlers Performing the Operation Mode Transitions
    def _leave_mode_add_trace_pts(self):
        trace = self.model.traces[self.curr_trace_no]
        # When hitting a button to exit trace points add mode, the last
        # selected point is likely not supposed to be included
        trace.delete_pt_px(trace.pts_px.shape[0] - 1)
        trace.sort_remove_duplicate_pts()
        
    def _enter_mode_setup_x_axis(self):
        logger.info("Pick X axis points!")
        self._blit_buffer_stale = True
        # Assuming the cursor is outside the figure anyways,initialise with NaN
        self._add_and_pick_point(self.model.x_ax, np.full(2, NaN))

    def _enter_mode_setup_y_axis(self):
        logger.info("Pick Y axis points!")
        self._blit_buffer_stale = True
        self._add_and_pick_point(self.model.y_ax, np.full(2, NaN))

    def _enter_mode_add_trace_pts(self):
        self._blit_buffer_stale = True
        trace = self.model.traces[self.curr_trace_no]
        # If trace points have already been selected, ask whether to
        # delete them first before adding new points.
        if trace.pts_px.shape[0] > 0 and self.messagebox.confirm_delete(
                "<b>There are trace points already selected.\n"
                "Discard or save and add more Points?</b>"):
            # Clears data objects of curr_trace and triggers a view update
            trace.clear_trace()
        logger.info(f"Add points mode for trace {self.curr_trace_no + 1}!")
        self._add_and_pick_point(trace, (NaN, NaN))
        if not self.model.axes_setup_is_complete:
            text = "You must configure the axes first!"
            logger.info(text)
            self.messagebox.show_warning(text)
            self.set_mode(self.MODE_DEFAULT)

    def _enter_mode_drag_obj(self):
        self._blit_buffer_stale = True
        logger.info("Drag the picked object!")
        # Actual movement happens in mouse motion notify event handler

    def _enter_mode_default(self):
        logger.info("Switching back to default mode")
        self._blit_buffer_stale = True
        self._picked_obj = None
        self._do_blit_redraw()

    # Adds point to model, causes a model-view update and sets picked obj
    def _add_and_pick_point(self, submodel, px_xy):
        logger.debug(
                f"_add_and_pick_point called. Submodel name: {submodel.name}"
                f" and pixel coordinates: {px_xy}")
        # The model returns an array index for the point inside the trace.
        self._picked_obj_pt_index = submodel.add_pt_px(px_xy)
        self._picked_obj = submodel.pts_view_obj
        self._picked_obj_submodel = submodel

    ########## Matplotlib Canvas Event Handlers
    def _on_figure_enter(self, event):
        # Set Qt keyboard input focus to the matplotlib canvas
        # in order to receive key press events
        self.canvas_qt.setFocus()

    def _on_figure_leave(self, event):
        self.mouse_coordinates_updated.emit(NaN, NaN)

    def _on_key_press(self, event):
        logger.debug(f"Event key pressed is: {event.key}")
        ##### Escape key switches back to default
        if event.key == "escape":
            self.set_mode(self.MODE_DEFAULT)

    def _on_button_press(self, event):
        ##### Right mouse button switches back to default
        if event.button == 3: # matplotlib.backend_bases.MouseButton.RIGHT = 3
            if self._op_mode != self.MODE_DEFAULT:
                self.set_mode(self.MODE_DEFAULT)
            return
        model = self.model
        px_xy = (event.xdata, event.ydata)
        # Ignore invalid coordinates (when clicked outside of plot canvas)
        if None in px_xy:
            return
        ##### Add X-axis point
        if self._op_mode == self.MODE_SETUP_X_AXIS:
            if isnan(model.x_ax.pts_px).any():
                # First point was added before when MODE_SETUP_X_AXIS was set.
                # Add new point to the model and continue.
                self._add_and_pick_point(model.x_ax, px_xy)
            else:
                # Two X-axis points set. Validate and reset op mode if valid
                if model.x_ax.pts_px_valid:
                    self.set_mode(self.MODE_DEFAULT)
            return
        ##### Add Y-axis point
        if self._op_mode == self.MODE_SETUP_Y_AXIS:
            if isnan(model.y_ax.pts_px).any():
                # First point was added before when MODE_SETUP_X_AXIS was set.
                # Add new point to the model and continue.
                self._add_and_pick_point(model.y_ax, px_xy)
            else:
                # Two X-axis points set. Validate and reset op mode if valid
                if model.y_ax.pts_px_valid:
                    self.set_mode(self.MODE_DEFAULT)
            return
        ##### Add trace point
        if self._op_mode == self.MODE_ADD_TRACE_PTS:
            trace = model.traces[self.curr_trace_no]
            # Add new point to the model at current mouse coordinates
            self._add_and_pick_point(trace, px_xy)
            return
        return

    # Mouse pick event handling. This sets MODE_DRAG_OBJ
    def _on_pick(self, event):
        logger.debug("Mouse pick event received!")
        # Picking is only enabled in MODE_DEFAULT
        if self._op_mode != self.MODE_DEFAULT:
            return
        picked_obj = event.artist
        if picked_obj in self._view_model_map:
            # Model components with view associated data for each picked object
            # are looked up from the view-model mapping.
            picked_obj_submodel = self._view_model_map[picked_obj]
            if isinstance(picked_obj_submodel, Trace):
                self.trace_no = picked_obj_submodel.trace_no
                # Interpolated lines are not draggable, only activate the trace
                if picked_obj is picked_obj_submodel.pts_i_view_obj:
                    return
            elif isinstance(picked_obj_submodel, Axis):
                if not self._drag_axes:
                    return
            else:
                # Origin or other objects are not pickable
                return
        else:
            # Not found in mapping, view object does not belong here
            return
        if event.mouseevent.button == 3:
            picked_obj_submodel.delete_pt_px(event.ind[0])
            # Stay in MODE_DEFAULT
            return
        # Object is pickable: Set instance attributes to select them
        self._picked_obj = picked_obj
        self._picked_obj_pt_index = event.ind[0]
        self._picked_obj_submodel = picked_obj_submodel
        logger.debug(f"Picked object: {self._picked_obj}")
        logger.debug(f"Picked object index: {self._picked_obj_pt_index}")
        logger.debug(f"Picked from model: {self._picked_obj_submodel.name}")
        # Actual movement/dragging of objects occurs in motion notify handler
        self.set_mode(self.MODE_DRAG_OBJ)

    def _on_button_release(self, event):
        # Mouse button release event ends MODE_DRAG_OBJ
        if self._op_mode == self.MODE_DRAG_OBJ:
            self.set_mode(self.MODE_DEFAULT)

    def _on_motion_notify(self, event):
        xy_px = event.xdata, event.ydata
        # Cursor outside of canvas returns none coordiantes, these are ignored
        if None in xy_px:
            return
        if self._picked_obj is not None:
            if isinstance(self._picked_obj_submodel, Trace):
                # Move trace point
                if self._op_mode == self.MODE_DRAG_OBJ:
                    self._picked_obj_submodel.move_restricted_pt_px(
                            xy_px, self._picked_obj_pt_index)
                elif self._op_mode == self.MODE_ADD_TRACE_PTS:
                    self._picked_obj_submodel.update_pt_px(
                            xy_px, self._picked_obj_pt_index)
            elif isinstance(self._picked_obj_submodel, Axis):
                # Moe axis point
                self._picked_obj_submodel.update_pt_px(
                        xy_px, self._picked_obj_pt_index)
        # Anyways, update coordinates display etc.
        xy_data = self.model.px_to_data(xy_px)
        if xy_data.shape[0] > 0:
            self.mouse_coordinates_updated.emit(*xy_data)

    def _on_resize(self, _):
        # Delayed screen redraw and blit buffer update when canvas is resized
        self._blit_buffer_stale = True
        self._redraw_timer.start()

    ########## 2D Graphics Acceleration using BLIT Methods
    # Qt AGG backend required:
    # Select view objects for mouse dragging and prepare block image transfer
    # by capturing the canvas background excluding selected view objects.
    # For this, selected objs are temporarily disabled and canvas is redrawn.
    def _capture_objs_background(self):
        if self._op_mode == self.MODE_DRAG_OBJ:
            if isinstance(self._picked_obj_submodel, Trace):
                self._blit_view_objs = [
                        self._picked_obj_submodel.pts_view_obj,
                        self._picked_obj_submodel.pts_i_view_obj]
            elif isinstance(self._picked_obj_submodel, Axis):
                # Changing axes points changes all traces.
                # We need to capture all visible traces.
                self._blit_view_objs = [
                        obj for obj in self._view_model_map.keys()
                        if obj.get_visible()
                        ]
            else:
                self._blit_view_objs = [self._picked_obj]
        else:
            self._blit_view_objs = [
                        obj for obj in self._view_model_map.keys()
                        if obj.get_visible()
                        ]
        for obj in self._blit_view_objs:
            obj.set_visible(False)
        self.canvas_qt.draw()
        self._blit_bg = self.canvas_qt.copy_from_bbox(self.mpl_ax.bbox)
        for obj in self._blit_view_objs:
            obj.set_visible(True)
        self._blit_buffer_stale = False

    @logExceptionSlot()
    def _do_blit_redraw(self):
        if self._blit_buffer_stale:
            logger.debug("Calling capture for blit redraw")
            self._capture_objs_background()
        # Restores captured object background
        self.canvas_qt.restore_region(self._blit_bg)
        # Redraws object using cached renderer
        for obj in self._blit_view_objs:
            self.mpl_ax.draw_artist(obj)
        # Blitting final step does seem to also update the Qt widget,
        # this seems to implement double-buffering
        self.canvas_qt.blit(self.mpl_ax.bbox)


    def _setup_coordinates_display(self):
        self.cursor_xy_label = QLabel("Cursor X and Y data coordinates:")
        self.cursor_x_display = SmallSciLineEdit(readOnly=True)
        self.cursor_y_display = SmallSciLineEdit(readOnly=True)
        #height = self.cursor_x_display.fontMetrics().height()
        #self.setStyleSheet(
        #        "QLineEdit { border: 1px solid black; padding-bottom: 2px; "
        #        f"max-height: {height}px; background-color: LightGrey; }}")
        self.mouse_coordinates_updated.connect(self._update_xy_display)

    @logExceptionSlot(float, float)
    def _update_xy_display(self, px_x: float, px_y: float):
        self.cursor_x_display.setValue(px_x)
        self.cursor_y_display.setValue(px_y)

    # Layout has only the matplotlib Qt AGG backend as a widget (canvas_qt)
    def _set_layout(self):
        self.setMinimumHeight(self.conf.app_conf.min_plotwin_height)
        coords_hbox = coords_hbox = QHBoxLayout()
        coords_hbox.addStretch(1)
        coords_hbox.addWidget(self.cursor_xy_label)
        coords_hbox.addWidget(self.cursor_x_display)
        coords_hbox.addWidget(self.cursor_y_display)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 2, 1)
        layout.setSpacing(2)
        layout.addLayout(coords_hbox)
        layout.addWidget(self.canvas_qt)


class MplToolBar(NavigationToolbar2QT): 
    def __init__(self, mpl_canvas, parent):
        # When coordinates display is enabled, this influences the toolbar
        # sizeHint, causing trouble in case of vertical toolbar orientation.
        # Also, we use an own coordinate display box. Thus setting to False.
        super().__init__(mpl_canvas, parent, coordinates=False)
        ########## Patch original API action buttons with new text etc:
        for act in self.actions():
            text = act.text()
            # We want to insert our buttons before the external matplotlib
            # API buttons where the "Home" is the leftmost
            if text == "Home":
                act.setText("Reset Zoom")
                self.act_api_home = act
            # The matplotlib save button only saves a screenshot thus it should
            # be appropriately renamed
            elif text == "Save":
                act.setText("Save as Image")
                act_api_save = act
            elif text == "Customize":
                act.setText("Figure Options")
                act_api_customize = act            
            elif text in ("Back", "Forward", "Subplots"):
                self.removeAction(act)
        api_actions = {self.act_api_home: "Reset{}Zoom",
                       act_api_save: "Save{}Image",
                       act_api_customize: "Figure{}Options",
                       }
        ########## Define new actions
        icon_open = QIcon.fromTheme(
                "document-open",
                self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.act_open_file = QAction(
                icon_open,
                "Open an image file",
                self)
        self.act_load_clipboard = QAction(
                icon_open,
                "Load Image from Clipboard",
                self)
        # Dict of our new actions plus text
        self._custom_actions = {self.act_load_clipboard: "From{}Clipboard",
                                self.act_open_file: "Open File",
                                }
        # Separator before first external API buttons
        sep = self.insertSeparator(self.act_api_home)
        ########## Add new actions to the toolbar
        self.insertActions(sep, self._custom_actions.keys())
        ########## Add original buttons to the dict of all custom actions
        self._custom_actions.update(api_actions)
        ########## Connect own signals
        self.orientationChanged.connect(self._on_orientationChanged)
        ########## Initial view setup
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._on_orientationChanged(Qt.Horizontal)
        #self.setIconSize(self.iconSize() * 0.8)
        self.setStyleSheet("spacing:2px")

    def sizeHint(self):
        # Matplotlib returns a minimum of 48 by default, we don't want this
        # size.setHeight(max(48, size.height()))
        return super(NavigationToolbar2QT, self).sizeHint()
    
    @pyqtSlot(Qt.Orientation)
    def _on_orientationChanged(self, new_orientation):
        logger.debug("Main Toolbar orientation change handler called")
        if new_orientation == Qt.Horizontal:
            self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            # Set text without line break
            for act, text in self._custom_actions.items():
                act.setIconText(text.format(" "))
        else:
            self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            # Set text with line break
            for act, text in self._custom_actions.items():
                act.setIconText(text.format("\n"))


class DigitizerToolBar(MplToolBar):
    def __init__(self, mpl_canvas, digitizer):
        super().__init__(mpl_canvas, digitizer)
        self.setWindowTitle(digitizer.plot_model.name)
        ########## Define new actions
        icon_export = QIcon.fromTheme(
                "document-save",
                self.style().standardIcon(QStyle.SP_DialogSaveButton))
        icon_send = QIcon.fromTheme(
                "document-send",
                self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.act_put_clipboard = QAction(
                icon_send,
                "Put Into Clipboard",
                self)
        self.act_export_xlsx = QAction(
                icon_export,
                "Export XLSX",
                self)
        self.act_export_csv = QAction(
                icon_export,
                "Export CSV",
                self)
        # Dict of our new actions plus text
        new_actions = {self.act_export_csv: "Export{}CSV",
                       self.act_export_xlsx: "Export{}XLSX", 
                       self.act_put_clipboard: "Put Into{}Clipboard",
                       }
        self._custom_actions.update(new_actions)
        ########## Add new actions to the toolbar
        # Separator before first external API buttons
        sep = self.insertSeparator(self.act_api_home)
        self.insertActions(sep, new_actions.keys())
        ##### Initial View Update
        self._on_orientationChanged(Qt.Horizontal)
        


class CoordinateSystemTab(QWidget):
    def __init__(self, mpl_widget, plot_model):
        super().__init__(mpl_widget)
        self.axconfw = AxConfWidget(mpl_widget, plot_model)
        self.canvas_extents_box = CanvasExtentsBox(mpl_widget, plot_model)
        layout = QVBoxLayout(self)
        layout.addWidget(self.axconfw)
        layout.addWidget(self.canvas_extents_box)
        layout.addStretch(1)


class TraceConfTab(QWidget):
    def __init__(self, mpl_widget, plot_model):
        super().__init__(mpl_widget)

        ########## Widgets setup
        self.table_tr_conf = TracesTable(mpl_widget, plot_model)
        self._set_layout()

        ########## Initialise view from plot_model
        self.update_plot_model_view()

        ########## Connect own and sub-widget signals

        ########## Connect foreign signals

    @logExceptionSlot()
    def update_plot_model_view(self):
        pass

    def _set_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self.table_tr_conf)


class ExportSettingsTab(QWidget):
    def __init__(self, mpl_widget, plot_model):
        super().__init__(mpl_widget)
        self.plot_model = plot_model
        self.exporter = plot_model.exporter
        # Reentrancy guard
        self._inhibit_updates = False
        ######### Setup widgets
        self.setStyleSheet(
                "QLineEdit {background-color: White}\n"
                "QLineEdit:read-only {background-color: LightGrey}\n"
                "QLabel {qproperty-alignment: AlignHCenter}")
        ##### Upper grid layout: inputs + labels
        self.lbl_extrapolation = QLabel("Extrapolation")
        self.combo_extrapolation = QComboBox()
        self.combo_extrapolation.addItems(self.exporter.extrapolation_types)
        self.lbl_traces_select = QLabel("Export Traces")
        self.combo_traces_select = QComboBox()
        self.combo_traces_select.addItems(
                ["All Selected"] + self.plot_model.trace_names)
        self.lbl_grid_type = QLabel("Grid Type")
        self.combo_grid_type = QComboBox()
        self.combo_grid_type.addItems(self.exporter.grid_types.long)
        self.lbl_n_pts_target = QLabel("Target N Points")
        self.combo_n_pts_target = QComboBox(editable=True)
        self.combo_n_pts_target.setValidator(QIntValidator())
        self.combo_n_pts_target.addItems(["10", "15", "20", "35", "50", "100"])
        self.lbl_n_pts_actual = QLabel("Generated N pts")
        self.edit_n_pts_actual = QLineEdit(readOnly=True)
        self.lbl_grid_parameter = QLabel("Step Size / N/dec / Tol")
        self.edit_grid_parameter = SciLineEdit(
                0.1, "Grid Parameter", self.plot_model.num_fmt)
        self.btn_generate_grid = QPushButton("Generate Grid")
        self.btn_calculate_points = QPushButton("Calculate Points")
        self.btn_preview = QPushButton("Preview Plot")
        self.lbl_warning = QLabel("Warnings:")
        self.edit_warning_display = QLineEdit()

        ##### Lower grid layout: Definition range display and export range edit
        self.lbl_x_start = QLabel("Start")
        self.lbl_x_end = QLabel("End")
        self.lbl_definition_range = QLabel("Common Definition Range:")
        self.cb_autorange = QCheckBox("Export Range Automatic: ",
                                      layoutDirection=Qt.RightToLeft)
        self.edit_definition_range_start = SciLineEdit(
                self.exporter.x_common_range_start,
                "Lower Limit",
                self.plot_model.num_fmt,
                readOnly=True,
                )
        self.edit_definition_range_end = SciLineEdit(
                self.exporter.x_common_range_end,
                "Upper Limit", 
                self.plot_model.num_fmt,
                readOnly=True,
                )
        self.edit_x_export_start = SciLineEdit(
                self.exporter.x_export_start,
                "X Axis Start Value",
                self.plot_model.num_fmt,
                )
        self.edit_x_export_end = SciLineEdit(
                self.exporter.x_export_start,
                "X Axis End Value",
                self.plot_model.num_fmt,
                )
        # Setup layout
        self._set_layout()
        
        ########## Initialise view from plot_model
        self.update_plot_model_view()
        self.update_mpl_widget_view(mpl_widget.MODE_DEFAULT)

        ########## Connect own and sub-widget signals
        self.combo_traces_select.currentIndexChanged[int].connect(
                self._update_active_traces)
        self.combo_n_pts_target.currentTextChanged.connect(
                self.exporter.set_n_pts_target)
        self.combo_grid_type.currentIndexChanged.connect(
                self._set_grid_type)
        self.edit_grid_parameter.valid_number_entered.connect(
                self._set_grid_parameter)
        self.combo_extrapolation.currentIndexChanged.connect(
                self.exporter.set_extrapolation_type)
        self.cb_autorange.toggled.connect(
                self._set_autorange)
        self.edit_x_export_start.valid_number_entered.connect(
                self.exporter.set_x_export_start)
        self.edit_x_export_end.valid_number_entered.connect(
                self.exporter.set_x_export_end)
        self.btn_generate_grid.clicked.connect(
                self.exporter.generate_grid)
        self.btn_calculate_points.clicked.connect(
                self.exporter.calculate_points)
        self.btn_preview.clicked.connect(
                self.do_preview_plot)

        ########## Connect foreign signals
        plot_model.output_data_changed.connect(self.update_plot_model_view)
        plot_model.output_data_changed[int].connect(self.update_plot_model_view)
        plot_model.export_settings_changed.connect(self.update_plot_model_view)
        # Update list of active traces
        plot_model.export_settings_changed.connect(self._update_active_traces)
        # Update when matplotlib widget changes operating mode
        mpl_widget.mode_sw.connect(self.update_mpl_widget_view)
        plot_model.export_range_warning.connect(self.edit_warning_display.setText)

    @logExceptionSlot()
    def update_plot_model_view(self, _=None):
        # Prevent recursive updates when called via signal from model
        if self._inhibit_updates:
            return
        self.combo_n_pts_target.setCurrentText(f"{self.exporter.n_pts_target}")
        actual_pts_text = ("" if self.exporter.x_grid_export is None
                           else f"{self.exporter.x_grid_export.shape[0]}")
        type_str = self.exporter.grid_type
        type_index = self.exporter.grid_types.short.index(type_str)
        if type_str == "lin_fixed_n":
            self.lbl_grid_parameter.setText("Step Size")
            self.edit_grid_parameter.setReadOnly(True)
            self.edit_grid_parameter.setValue(self.exporter.x_step)
            self.combo_n_pts_target.setEnabled(True)
            self.edit_n_pts_actual.setText(actual_pts_text)
        elif type_str == "lin_fixed_step":
            self.lbl_grid_parameter.setText("Step Size")
            self.edit_grid_parameter.setReadOnly(False)
            self.edit_grid_parameter.setValue(self.exporter.x_step)
            self.combo_n_pts_target.setEnabled(False)
            self.edit_n_pts_actual.setText(actual_pts_text)
        elif type_str == "log_fixed_n_dec":
            self.lbl_grid_parameter.setText("Points/Decade")
            self.edit_grid_parameter.setReadOnly(False)
            self.edit_grid_parameter.setValue(self.exporter.n_pts_dec)
            self.combo_n_pts_target.setEnabled(False)
            self.edit_n_pts_actual.setText(actual_pts_text)
        elif type_str == "log_fixed_n":
            self.lbl_grid_parameter.setText("Points/Decade")
            self.edit_grid_parameter.setReadOnly(True)
            self.edit_grid_parameter.setValue(self.exporter.n_pts_dec)
            self.combo_n_pts_target.setEnabled(True)
            self.edit_n_pts_actual.setText(actual_pts_text)
        elif type_str == "adaptive":
            self.lbl_grid_parameter.setText("Grid Tolerance")
            self.edit_grid_parameter.setReadOnly(False)
            self.edit_grid_parameter.setValue(self.exporter.grid_tol)
            self.combo_n_pts_target.setEnabled(True)
            self.edit_n_pts_actual.setText("Individual")
        self.combo_grid_type.setCurrentIndex(type_index)
        self.cb_autorange.setChecked(self.exporter.autorange_export)
        self.edit_definition_range_start.setValue(self.exporter.x_common_range_start)
        self.edit_definition_range_end.setValue(self.exporter.x_common_range_end)
        self.edit_x_export_start.setValue(self.exporter.x_export_start)
        self.edit_x_export_end.setValue(self.exporter.x_export_end)
        self.edit_x_export_start.setReadOnly(self.exporter.autorange_export)
        self.edit_x_export_end.setReadOnly(self.exporter.autorange_export)
        self._inhibit_updates = False

    @logExceptionSlot()
    def _update_active_traces(self, _=None):
        # Prevent recursive updates when called via signal from model
        if self._inhibit_updates:
            return
        self._inhibit_updates = True
        index = self.combo_traces_select.currentIndex()
        if index == 0:
            # Selects all traces marked for export
            self.exporter.set_traces()
        else:
            self.exporter.set_traces(index - 1)
        self._inhibit_updates = False
            
    @logExceptionSlot(int)
    def update_mpl_widget_view(self, op_mode: int):
        pass

    @logExceptionSlot(bool)
    def do_preview_plot(self, state: bool):
        logger.info("FIXME: Preview not yet implemented!")

    @logExceptionSlot(float)
    def _set_grid_parameter(self, value: float):
        if self.exporter.grid_type == "adaptive":
            self.exporter.set_grid_tol(value)
        elif self.exporter.grid_type == "log_fixed_n_dec":
            self.exporter.set_n_pts_dec(value)
        else:
            self.exporter.set_x_step(value)
    
    @logExceptionSlot(int)
    def _set_grid_type(self, index: int):
        type_str = self.exporter.grid_types.short[index]
        self.exporter.set_grid_type(type_str)

    @logExceptionSlot(bool)
    def _set_autorange(self, state):
        self.exporter.set_autorange_export(state)
        self.edit_x_export_start.setReadOnly(state)
        self.edit_x_export_end.setReadOnly(state)

    def _set_layout(self):
        ##### Upper Grid
        l_upper = QGridLayout()
        l_upper.setColumnStretch(0, 3)
        l_upper.setColumnStretch(1, 2)
        l_upper.setColumnStretch(2, 2)
        # Row 0
        l_upper.addWidget(self.lbl_traces_select, 0, 0)
        l_upper.addWidget(self.lbl_n_pts_target, 0, 1)
        l_upper.addWidget(self.lbl_n_pts_actual, 0, 2)
        # Row 1
        l_upper.addWidget(self.combo_traces_select, 1, 0)
        l_upper.addWidget(self.combo_n_pts_target, 1, 1)
        l_upper.addWidget(self.edit_n_pts_actual, 1, 2)
        # Row 2
        l_upper.addWidget(self.lbl_grid_type, 2, 0)
        l_upper.addWidget(self.lbl_grid_parameter, 2, 1)
        l_upper.addWidget(self.lbl_extrapolation, 2, 2)
        # Row 3
        l_upper.addWidget(self.combo_grid_type, 3, 0)
        l_upper.addWidget(self.edit_grid_parameter, 3, 1)
        l_upper.addWidget(self.combo_extrapolation, 3, 2)
        ##### Middle Grid
        l_middle = QGridLayout()
        # Row 0
        l_middle.addWidget(self.lbl_x_start, 0, 1)
        l_middle.addWidget(self.lbl_x_end, 0, 2)
        # Row 1
        l_middle.addWidget(self.lbl_definition_range, 1, 0)
        l_middle.addWidget(self.edit_definition_range_start, 1, 1)
        l_middle.addWidget(self.edit_definition_range_end, 1, 2)
        # Rows 2
        l_middle.addWidget(self.cb_autorange, 2, 0)
        l_middle.addWidget(self.edit_x_export_start, 2, 1)
        l_middle.addWidget(self.edit_x_export_end, 2, 2)
        ##### Lower Grid
        l_lower = QGridLayout()
        l_lower.addWidget(self.btn_generate_grid, 0, 0)
        l_lower.addWidget(self.btn_calculate_points, 0, 1)
        l_lower.addWidget(self.btn_preview, 0, 2)
        l_lower.addWidget(self.lbl_warning, 1, 1)
        l_lower.addWidget(self.edit_warning_display, 2, 0, 1, 3)
        ##### Complete Layout
        l_outer = QVBoxLayout(self)
        l_outer.addLayout(l_upper)
        l_outer.addLayout(l_middle)
        l_outer.addLayout(l_lower)
        # Fill up empty space to the bottom
        l_outer.addStretch(1)


class AxConfWidget(QWidget):
    def __init__(self, mpl_widget, plot_model):
        super().__init__(mpl_widget)
        self.plot_model = plot_model
        self.mpl_widget = mpl_widget
        ######### Qt widget setup
        #### Group box for X Coordinate picker and input boxes
        self.group_x = QGroupBox("Enter X Axis Start and End Values")
        self.btn_pick_x = StyledButton("Pick Points", self)
        self.xstart_edit = SciLineEdit(
                plot_model.x_ax.sect_data[0],
                "X Axis Start Value",
                plot_model.num_fmt)
        self.xend_edit = SciLineEdit(
                plot_model.x_ax.sect_data[1],
                "X Axis End Value",
                plot_model.num_fmt)
        self.btn_lin_x = QRadioButton("Lin")
        self.btn_log_x = QRadioButton("Log")
        #### Group box for Y Coordinate picker and input boxes
        self.group_y = QGroupBox("Enter Y Axis Start and End Values")
        self.btn_pick_y = StyledButton("Pick Points", self)
        self.ystart_edit = SciLineEdit(
                plot_model.y_ax.sect_data[0],
                "Y Axis Start Value",
                plot_model.num_fmt)
        self.yend_edit = SciLineEdit(
                plot_model.y_ax.sect_data[1],
                "Y Axis End Value",
                plot_model.num_fmt)
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
        self.update_plot_model_view()
        self.update_mpl_widget_view(mpl_widget.MODE_DEFAULT)

        ########## Connect own and sub-widget signals
        self.btn_pick_x.toggled.connect(mpl_widget.set_mode_setup_x_axis)
        self.btn_pick_y.toggled.connect(mpl_widget.set_mode_setup_y_axis)
        self.btn_log_x.toggled.connect(plot_model.x_ax.set_log_scale)
        self.btn_log_y.toggled.connect(plot_model.y_ax.set_log_scale)
        self.btn_drag_axes.toggled.connect(mpl_widget.set_drag_axes)
        self.btn_store_config.toggled.connect(plot_model.set_wants_persistent_storage)
        # Number input boxes emit float signals.
        self.xstart_edit.valid_number_entered.connect(plot_model.x_ax.set_ax_start)
        self.ystart_edit.valid_number_entered.connect(plot_model.y_ax.set_ax_start)
        self.xend_edit.valid_number_entered.connect(plot_model.x_ax.set_ax_end)
        self.yend_edit.valid_number_entered.connect(plot_model.y_ax.set_ax_end)

        ########## Connect foreign signals
        # Update when axes config changes
        plot_model.ax_conf_changed.connect(self.update_plot_model_view)
        # Update when matplotlib widget changes operating mode
        mpl_widget.mode_sw.connect(self.update_mpl_widget_view)

    ########## Slots
    # Updates state of the Matplotlib widget display by setting down the
    # the buttons when each mode is active
    @logExceptionSlot(int)
    def update_mpl_widget_view(self, op_mode):
        self.btn_pick_x.setChecked(op_mode == self.mpl_widget.MODE_SETUP_X_AXIS)
        self.btn_pick_y.setChecked(op_mode == self.mpl_widget.MODE_SETUP_Y_AXIS)
        self.btn_drag_axes.setChecked(self.mpl_widget._drag_axes)

    # Updates buttons and input boxes to represent the data model state
    # and also the new and current matplotlib widget operation mode.
    @logExceptionSlot()
    def update_plot_model_view(self):
        x_ax = self.plot_model.x_ax
        y_ax = self.plot_model.y_ax
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
        self.btn_store_config.setChecked(self.plot_model.wants_persistent_storage)

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
    def __init__(self, mpl_widget, plot_model):
        super().__init__("Data Coordinate System", mpl_widget)
        self.plot_model = plot_model
        self.mpl_widget = mpl_widget

        ########## Widgets setup
        self.x_range_label = QLabel("Canvas X Data Extent:")
        self.x_min_edit = SciLineEdit()
        self.x_max_edit = SciLineEdit()
        self.y_range_label = QLabel("Canvas Y Data Extent:")
        self.y_min_edit = SciLineEdit()
        self.y_max_edit = SciLineEdit()
        # Layout setup
        self._set_layout()
        
        ########## Initialise view from plot_model
        self.update_plot_model_view()
        self.update_mpl_widget_view(mpl_widget.MODE_DEFAULT)

        ########## Connect own and sub-widget signals
        self.x_min_edit.valid_number_entered.connect(self._set_canvas_extents)
        self.x_max_edit.valid_number_entered.connect(self._set_canvas_extents)
        self.y_min_edit.valid_number_entered.connect(self._set_canvas_extents)
        self.y_max_edit.valid_number_entered.connect(self._set_canvas_extents)

        ########## Connect foreign signals
        #model.coordinate_system_changed.connect(self.update_plot_model_view)
        # Update when matplotlib widget changes operating mode
        mpl_widget.canvas_rescaled.connect(self.update_mpl_widget_view)
        plot_model.ax_conf_changed.connect(self.update_mpl_widget_view)

    @logExceptionSlot()
    def update_plot_model_view(self):
        pass

    @logExceptionSlot()
    @logExceptionSlot(int)
    def update_mpl_widget_view(self, op_mode=MplWidget.MODE_DEFAULT):
        xb = self.mpl_widget.mpl_ax.get_xbound()
        yb = self.mpl_widget.mpl_ax.get_ybound()
        bounds_px = np.concatenate((xb, yb)).reshape(-1, 2, order="F")
        bounds_data = self.plot_model.px_to_data(bounds_px)
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
        if not self.plot_model.validate_data_pts(xy_min_max):
            return
        bounds_px = self.plot_model.data_to_px(xy_min_max)
        self.mpl_widget.set_canvas_extents(bounds_px)

    def _set_layout(self):
        layout = QGridLayout(self)
        # self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.x_range_label, 0, 0)
        layout.addWidget(self.x_min_edit, 0, 1)
        layout.addWidget(self.x_max_edit, 0, 2)
        layout.addWidget(self.y_range_label, 1, 0)
        layout.addWidget(self.y_min_edit, 1, 1)
        layout.addWidget(self.y_max_edit, 1, 2)


class TracesTable(QTableWidget):
    def __init__(self, mpl_widget, plot_model):
        super().__init__(mpl_widget)
        # Prevent (recursive) updates when set
        self._inhibit_updates = False
        self.plot_model = plot_model
        self.mpl_widget = mpl_widget
        headers = ["Name", "Points", "Interpolation", "Enable",
                   "Export", "Color", "X Start", "X End"]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.col_xstart = headers.index("X Start")
        self.col_xend = headers.index("X End")
        self.btns_pick_trace = []
        self.cbs_enable = []
        self.interp_types = ("C-Splines", "Linear")
        self.colors = DescriptiveIDs(
                ["c", "m", "y", "r", "g", "b", "k"],
                ["Cyan", "Magenta", "Yellow", "Red", "Green", "Blue", "Black"])

        ########## Initialise view from plot_model
        self.update_plot_model_view()
        self.update_mpl_widget_view()

        ########## Connect own and sub-widget signals
        self.cellChanged.connect(self.on_cell_changed)
        #self.itemSelectionChanged.connect(self._handle_selection)

        ########## Connect foreign signals
        # Update when trace config changes, e.g. if traces are added or renamed
        plot_model.tr_conf_changed.connect(self.update_plot_model_view)
        # Update when matplotlib widget changes operating mode
        mpl_widget.mode_sw.connect(self.update_mpl_widget_view)

    @logExceptionSlot()
    def update_plot_model_view(self):
        logger.debug("Update plot_model_view for traces table called.")
        # Prevent recursive updates
        self._inhibit_updates = True
        # Delete and disconnect items from old rows
        self.clearContents()
        self.clearSpans()
        # Populate table
        self.setRowCount(1 + len(self.plot_model.traces))
        for row, tr in enumerate(self.plot_model.traces):
            name = QTableWidgetItem(tr.name)
            btn_pick = NumberedButton(row, "Pick!")
            combo_interp_type = QComboBox()
            combo_interp_type.addItems(self.interp_types)
            cb_enable = NumberedCenteredCheckbox(row)
            cb_export = NumberedCenteredCheckbox(row)
            cb_export.setChecked(tr.export)
            color_index = self.colors.short.index(tr.pts_fmt["color"])
            combo_color = QComboBox()
            combo_color.addItems(self.colors.long)
            combo_color.setCurrentIndex(color_index)
            x_start = QTableWidgetItem("0 %")
            x_end = QTableWidgetItem("100 %")
            self.setItem(row, 0, name)
            self.setCellWidget(row, 1, btn_pick)
            self.setCellWidget(row, 2, combo_interp_type)
            self.setCellWidget(row, 3, cb_enable)
            self.setCellWidget(row, 4, cb_export)
            self.setCellWidget(row, 5, combo_color)
            self.setItem(row, 6, x_start)
            self.setItem(row, 7, x_end)
            ##### Signals
            btn_pick.i_toggled.connect(self.mpl_widget.set_mode_add_trace_pts)
            combo_interp_type.currentIndexChanged[str].connect(self.set_i_type)
            cb_enable.i_toggled.connect(self.mpl_widget.enable_trace)
            cb_export.i_toggled.connect(self.set_export)
            combo_color.currentIndexChanged[int].connect(self.set_color)
        # Add placeholder for adding new traces
        row += 1
        self.itm_add_new = QTableWidgetItem("Add New!")
        itm_remarks = QTableWidgetItem(
                "<--Enter name to add trace.  Delete any name to remove.")
        itm_remarks.setFlags(itm_remarks.flags() & ~Qt.ItemIsEnabled)
        self.setItem(row, 0, self.itm_add_new)
        self.resizeColumnsToContents()
        self.setItem(row, 1, itm_remarks)
        self.setSpan(row, 1, 1, self.columnCount()-1)
        self._inhibit_updates = False
        self.update_mpl_widget_view()

    @logExceptionSlot()
    @logExceptionSlot(int)
    def update_mpl_widget_view(self, op_mode=MplWidget.MODE_DEFAULT):     
        # Prevent recursive updates
        if self._inhibit_updates:
            return
        for row, tr in enumerate(self.plot_model.traces):
            btn_pick = self.cellWidget(row, 1)
            cb_enable = self.cellWidget(row, 3)
            btn_pick.setChecked(
                    row == self.mpl_widget.curr_trace_no
                    and op_mode == self.mpl_widget.MODE_ADD_TRACE_PTS)
            cb_enable.setChecked(self.mpl_widget.is_enabled_trace(row))

    @logExceptionSlot(str)
    def on_cell_changed(self, row, column):
        # Prevent recursive updates
        if self._inhibit_updates:
            return
        self._inhibit_updates = True
        itm = self.item(row, column)
        if itm is self.itm_add_new:
            name = itm.text().lstrip().rstrip()
            if name == "Add New!":
                return
            if name != "" and name.isprintable():
                logger.debug(
                        f'Adding trace "{name}" for "{self.plot_model.name}"')
                self.plot_model.add_trace(name)
            else:
                self.plot_model.value_error.emit("Invalid Name!")
                itm.setText("Add New!")
        elif column == 0:
            # Changing trace name
            name = itm.text().lstrip().rstrip()
            trace = self.plot_model.traces[row]
            if name == "":
                self.plot_model.remove_trace(row)
            elif name.isprintable():
                logger.debug(
                        f'Changing name "{name}" for "{self.plot_model.name}"')
                trace.set_name(name)
            else:
                self.plot_model.value_error.emit("Invalid Name!")
                itm.setText(trace.name)
        self._inhibit_updates = False


    @logExceptionSlot(int, bool)
    def set_export(self, trace_no, state=True):
        trace = self.plot_model.traces[trace_no]
        trace.set_export(state)

    @logExceptionSlot(str)
    def set_i_type(self, interp_type: str):
        trace = self.plot_model.traces[self.currentRow()]
        trace.set_interp_type(interp_type)

    @logExceptionSlot(int)
    def set_color(self, index: int):
        trace = self.plot_model.traces[self.currentRow()]
        color_code = self.colors.short[index]
        trace.set_color(color_code)
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
#                self.plot_model.traces[i].x_start_export = x_start
#                self.plot_model.traces[i].x_end_export = x_end
#                self.x_start_export = x_start
#                self.x_end_export = x_end
#            #self.show_xrange.emit(True)
#        elif self._show_xrange:
#            self._show_xrange = False
#            #self.show_xrange.emit(False)
