#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench MPL View Widget

License: GPL version 3
"""
import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from numpy import NaN, isnan

from PyQt5.QtCore import pyqtSlot, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QLabel)

import matplotlib.figure
import matplotlib.image
#from matplotlib.patches import Polygon
import matplotlib.backends.backend_qt5agg as mpl_backend_qt

from plot_model import Trace, Axis
from digitizer_widgets import SciLineEdit
from upylib.pyqt_debug import logExceptionSlot


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
    canvas_rescaled = pyqtSignal(int)
    # Emitts new X and Y coords in data space when pointer is inside mpl canvas.
    mouse_coordinates_updated = pyqtSignal(float, float)

    def __init__(self, digitizer, model):
        super().__init__(digitizer)
        self.digitizer = digitizer
        self.conf = digitizer.conf
        
        ########## Access to the data model
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
        self.canvas_qt = mpl_backend_qt.FigureCanvas(self.fig)
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
        # After removing axis visibility, reset matplotlib layout to fill space
        self.fig.tight_layout(pad=0, rect=(0.001, 0.002, 0.999, 0.999))

        ########## Initialise view from model
        self.update_model_view_axes()
        if model.coordinate_transformation_defined:
            self.update_model_view_traces()
       
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
        # Update plot view displaying axes points and origin
        model.ax_input_data_changed.connect(self.update_model_view_axes)
        # Re-display pixel-space input points when model has updated data.
        model.output_data_changed.connect(self.update_model_view_traces)
        model.output_data_changed[int].connect(self.update_model_view_traces)


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
        self.update_model_view_traces(trace_no)

    @logExceptionSlot()
    def update_model_view(self):
        """Complete update of axes and traces features from model data
        """
        self.update_model_view_axes()
        self._blit_buffer_stale = True
        self.update_model_view_traces()
    
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
    def update_model_view_traces(self, trace_no=None):
        """Draw or redraw a trace from the data model.
        
        If the argument is None, update all traces.
        This also registers the view objects back into the model.
        """
        #logger.debug(f"update_model_view_traces called for trace: {trace_no}")
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
    def load_image(self, filename):
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
                self.digitizer.show_text("Cannot open file. Not an image?")
                return
        except Exception as e:
            self.digitizer.show_error(e)
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
        self.canvas_rescaled.emit(self._op_mode)
        self.update_model_view()


    ########## Data Output Methods
    def is_enabled_trace(self, trace_no: int) -> bool:
        tr = self.model.traces[trace_no]
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
        if not self.model.axes_setup_is_complete:
            text = "You must configure the axes first!"
            logger.info(text)
            self.digitizer.show_text(text)
            self.set_mode(self.MODE_DEFAULT)
            return
        self._blit_buffer_stale = True
        trace = self.model.traces[self.curr_trace_no]
        # If trace points have already been selected, ask whether to
        # delete them first before adding new points.
        if trace.pts_px.shape[0] > 0 and self._confirm_delete():
            # Clears data objects of curr_trace and triggers a view update
            trace.clear_trace()
        logger.info(f"Add points mode for trace {self.curr_trace_no + 1}!")
        self._add_and_pick_point(trace, (NaN, NaN))

    def _enter_mode_drag_obj(self):
        self._blit_buffer_stale = True
        logger.info("Drag the picked object!")
        # Actual movement happens in mouse motion notify event handler

    def _enter_mode_default(self):
        logger.info("Switching back to default mode")
        self._blit_buffer_stale = True
        self._picked_obj = None

    # Adds point to model, causes a model-view update and sets picked obj
    def _add_and_pick_point(self, submodel, px_xy):
        logger.debug(
                f"_add_and_pick_point called. Submodel name: {submodel.name}"
                f" and pixel coordinates: {px_xy}")
        # The model returns an array index for the point inside the trace.
        self._picked_obj_pt_index = submodel.add_pt_px(px_xy)
        self._picked_obj = submodel.pts_view_obj
        self._picked_obj_submodel = submodel

    # Blocking messagebox confirming points deletion
    def _confirm_delete(self):
        messagebox = self.digitizer.messagebox
        messagebox.setIcon(QMessageBox.Warning)
        messagebox.setText(
            "<b>There are trace points already selected.\n"
            "Discard or save and add more Points?</b>"
            )
        messagebox.setWindowTitle("Confirm Delete")
        messagebox.setStandardButtons(
                QMessageBox.Save | QMessageBox.Discard)
        return messagebox.exec_() == QMessageBox.Discard

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
        self.cursor_x_display = SciLineEdit()
        self.cursor_y_display = SciLineEdit()
        self.cursor_x_display.setReadOnly(True)
        self.cursor_y_display.setReadOnly(True)
        self.cursor_x_display.setStyleSheet("background-color: LightGrey")
        self.cursor_y_display.setStyleSheet("background-color: LightGrey")
        self.mouse_coordinates_updated.connect(self._update_xy_display)

    @logExceptionSlot(float, float)
    def _update_xy_display(self, px_x: float, px_y: float):
        self.cursor_x_display.setValue(px_x)
        self.cursor_y_display.setValue(px_y)

    # Layout has only the matplotlib Qt AGG backend as a widget (canvas_qt)
    def _set_layout(self):
        self.setMinimumHeight(self.conf.app_conf.min_plotwin_height)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 6, 0)
        coords_hbox = QHBoxLayout()
        coords_hbox.addWidget(self.cursor_xy_label)
        coords_hbox.addWidget(self.cursor_x_display)
        coords_hbox.addWidget(self.cursor_y_display)
        layout.addLayout(coords_hbox)
        layout.addWidget(self.canvas_qt)