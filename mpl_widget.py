#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench MPL View Widget

License: GPL version 3
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
from numpy import NaN, isnan

from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMessageBox

import matplotlib.figure
import matplotlib.image
#from matplotlib.patches import Polygon
import matplotlib.backends.backend_qt5agg as mpl_backend_qt

from plot_model import Trace, Axis
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
        # This stores the pyplot.Lines2D object when a plot item was picked
        self._picked_obj = None
        # Model component with view associated data for a mouse-picked object
        self._picked_obj_model_attr = None
        # Index of a single picked point inside the view object
        self._picked_obj_index = 0
        # All matplotlib Lines2D objects for axes, origin and traces
        self._ax_view_objs = []
        self._tr_view_objs = []

        ########## Matplotlib figure and axes setup
        self.fig = matplotlib.figure.Figure()

        ########## Qt widget setup
        # FigureCanvas factory of matplotlib Qt Agg backend takes the Figure
        # object and returns the matplotlib canvas as a QWidget instance.
        self.canvas_qt = mpl_backend_qt.FigureCanvas(self.fig)
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
        if model.coordinate_transformation_defined():
            self.update_model_view_traces()
       
        ########## Connect own signals
        self.canvas_qt.mpl_connect("key_press_event", self._on_key_press)
        self.canvas_qt.mpl_connect("figure_enter_event", self._on_figure_enter)
        self.canvas_qt.mpl_connect("button_press_event", self._on_button_press)
        self.canvas_qt.mpl_connect("button_release_event",
                                   self._on_button_release)
        self.canvas_qt.mpl_connect("motion_notify_event", self._on_motion_notify)
        self.canvas_qt.mpl_connect("pick_event", self._on_pick)

        ########## Connect foreign signals
        # Update plot view displaying axes points and origin
        model.ax_input_data_changed.connect(self.update_model_view_axes)
        # Re-display pixel-space input points when model has updated data.
        model.output_data_changed.connect(self.update_model_view_traces)
        model.output_data_changed[int].connect(self.update_model_view_traces)


    @logExceptionSlot()
    def update_model_view_axes(self):
        """Updates axes model features displayed in plot widget,
        including origin only if it fits inside the canvas.

        This also registers the view objects back into the model
        and emits a signal informing when the canvas has been
        re-drawn.
        """
        #logger.debug("update_model_view_axes called")
        model = self.model
        ax_view_objs = []
        ########## X and Y axis:
        for ax in model.x_ax, model.y_ax:
            if ax.pts_view_obj is not None:
                ax.pts_view_obj.set_data(*ax.pts_px.T)
            else:
                ax.pts_view_obj, = self.mpl_ax.plot(*ax.pts_px.T, **ax.pts_fmt)
                self._view_model_map[ax.pts_view_obj] = ax
            ax_view_objs.append(ax.pts_view_obj)
        ########## Origin:
        if isnan(model.origin_px).any():
            if model.origin_view_obj is not None:
                model.origin_view_obj.remove()
                model.origin_view_obj = None
        else:
            # Containment check via numpy elementwise operators
            if (    (model.origin_px < self.fig.bbox.size).all()
                    and (model.origin_px > 0).all()
                    ):
                if model.origin_view_obj is None:
                    view_obj, = self.mpl_ax.plot(
                        *model.origin_px, **model.origin_fmt)
                    # The origin point is calculated and not supposed to be
                    # subjected to drag-and-drop etc.: registering it as None
                    self._view_model_map[view_obj] = None
                    model.origin_view_obj = view_obj
                else:
                    model.origin_view_obj.set_data(*model.origin_px)
                ax_view_objs.append(model.origin_view_obj)
        ##### Redraw axes and origin view objects
        # For modes other than MODE_DEFAULT, background was captured before,
        # otherwise we need to do the blitting capture here
        if self._op_mode == self.MODE_DEFAULT:
            self._capture_objs_background(ax_view_objs)
        self._blit_background_redraw_objs(ax_view_objs)
        self._ax_view_objs = ax_view_objs


    @pyqtSlot()
    @pyqtSlot(int)
    def update_model_view_traces(self, trace_no=None):
        """Draw or redraw a trace from the data model.
        
        If the argument is None, update all traces.
        This also registers the view objects back into the model.
        """
        #logger.debug("update_model_view_traces called")
        model = self.model
        traces = model.traces if trace_no is None else [model.traces[trace_no]]
        tr_view_objs = []
        ########## Update interpolated trace if available
        for tr in traces:
            ##### STEP A: Draw or update raw pixel points
            view_obj = tr.pts_view_obj
            if view_obj is None:
                view_obj, = self.mpl_ax.plot(*tr.pts_px.T, **tr.pts_fmt)
                self._view_model_map[view_obj] = tr
            else:
                view_obj.set_data(*tr.pts_px.T)
            tr_view_objs.append(view_obj)
            ##### STEP B: Draw or update interpolated pixel points
            # Backtransform trace to pixel data coordinate system
            pts_i_px = model.get_pts_lin_i_px_coords(tr)
            view_obj = tr.pts_i_view_obj
            if view_obj is None:
                # Draw trace on matplotlib widget
                view_obj, = self.mpl_ax.plot(*pts_i_px.T, **tr.pts_i_fmt)
                # The origin point is calculated and not supposed to be
                # subjected to drag-and-drop etc.: registering it as None
                self._view_model_map[view_obj] = None
                tr.pts_i_view_obj = view_obj
            else:
                # Trace handle for pts_lin_i exists. Update data.
                view_obj.set_data(*pts_i_px.T)
            tr_view_objs.append(view_obj)
        ##### Redraw traces view objects
        # For modes other than MODE_DEFAULT, background was captured before,
        # otherwise we need to do the blitting capture here
        if self._op_mode == self.MODE_DEFAULT:
            self._capture_objs_background(tr_view_objs)
        self._blit_background_redraw_objs(tr_view_objs)
        self._tr_view_objs = tr_view_objs


    @logExceptionSlot(str)
    def load_image(self, filename):
        """Load source/input image for digitizing"""
        # Remove existing image from plot canvas if present
        if hasattr(self, "_mpl_axes_image"):
            self._mpl_axes_image.remove()
        try:
            image = matplotlib.image.imread(filename)
        except Exception as e:
            self.digitizer.show_error(e)
            return
        #all_except_image = self._ax_view_objs + self._tr_view_objs
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
        self.update_model_view_axes()
        self.update_model_view_traces()
        self.canvas_rescaled.emit(self._op_mode)

    ########## Mplwidget operation mode state machine transitions
    @logExceptionSlot(int)
    def set_mode(self, new_mode: int):
        logger.debug(f"set_mode called with argument: {new_mode}.")
        # Prevent recursive calls when external widgets are updated
        if new_mode == self._op_mode:
            return
        logger.debug(f"Entering new operation mode: {new_mode}.")
        self._op_mode = new_mode
        self._based_on_op_mode_capture_background()
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

    @logExceptionSlot(bool)
    def set_mode_setup_x_axis(self, state=True):
        if state:
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


    ########## Handlers performing the operation mode transitions
    #          Each called from self.set_mode()
    def _enter_mode_setup_x_axis(self):
        logger.info("Pick X axis points!")
        self._add_point_and_select(self.model.x_ax, np.full(2, NaN))

    def _enter_mode_setup_y_axis(self):
        logger.info("Pick Y axis points!")
        self._add_point_and_select(self.model.y_ax, np.full(2, NaN))

    def _enter_mode_add_trace_pts(self):
        if not self.model.axes_setup_is_complete():
            text = "You must configure the axes first!"
            logger.info(text)
            self.digitizer.show_text(text)
            self.set_mode(self.MODE_DEFAULT)
            return
        trace = self.model.traces[self.curr_trace_no]
        # If trace points have already been selected, ask whether to
        # delete them first before adding new points.
        if trace.pts_px.shape[0] > 0 and self._confirm_delete():
            # Clears data objects of curr_trace and triggers a view update
            trace.init_data()
        logger.info(f"Add points mode for trace {self.curr_trace_no + 1}!")
        # Add point to the trace, assuming the cursor is outside the figure
        # canvas, point data is initialised with NaN coordinates.
        self._add_point_and_select(trace, np.full(2, NaN))

    def _enter_mode_drag_obj(self):
        logger.info("Drag the picked object!")

    def _enter_mode_default(self):
        logger.info("Switching back to default mode")
        self._picked_obj = None


    def _add_point_and_select(self, model_attr, pt_px):
        # The model returns an array index for the point inside the trace.
        self._picked_obj_index = model_attr.add_pt_px(pt_px)
        self._picked_obj = model_attr.pts_view_obj
        self._picked_obj_model_attr = model_attr

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

    ########## Matplotlib canvas event handlers
    def _on_figure_enter(self, event):
        # Set Qt keyboard input focus to the matplotlib canvas
        # in order to receive key press events
        self.canvas_qt.setFocus()

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
            ax = model.x_ax
            if isnan(ax.pts_px).any():
                # First point was added before when MODE_SETUP_X_AXIS was set.
                # Add new point to the model and continue.
                self._add_point_and_select(ax, px_xy)
            else:
                # Two X-axis points set. Validate and reset op mode if valid
                if ax.valid_pts_px():
                    self.set_mode(self.MODE_DEFAULT)
            return
        ##### Add Y-axis point
        if self._op_mode == self.MODE_SETUP_Y_AXIS:
            ax = model.y_ax
            if isnan(ax.pts_px).any():
                # First point was added before when MODE_SETUP_X_AXIS was set.
                # Add new point to the model and continue.
                self._add_point_and_select(ax, px_xy)
            else:
                # Two X-axis points set. Validate and reset op mode if valid
                if ax.valid_pts_px():
                    self.set_mode(self.MODE_DEFAULT)
            return
        ##### Add trace point
        if self._op_mode == self.MODE_ADD_TRACE_PTS:
            trace = model.traces[self.curr_trace_no]
            # Add new point to the model at current mouse coordinates
            self._add_point_and_select(trace, px_xy)
            return
        return

    # Mouse pick event handling. This sets MODE_DRAG_OBJ
    def _on_pick(self, event):
        logger.debug("Mouse pick event received!")
        # Picking is only enabled in MODE_DEFAULT
        if self._op_mode != self.MODE_DEFAULT:
            return
        picked_obj = event.artist
        # Model components with view associated data for each picked object
        # are looked up from the view-model mapping.
        picked_obj_model_attr = self._view_model_map[picked_obj]
        # For non-mouse-pickable view objects, a None is stored in the mapping
        if picked_obj_model_attr is None:
            # Not found in mapping, view object is thus not mouse-pickable
            return
        # Object is pickable: Set instance attributes to select them
        self._picked_obj = picked_obj
        self._picked_obj_index = event.ind[0] if hasattr(event, "ind") else None
        self._picked_obj_model_attr = picked_obj_model_attr
        logger.debug(f"Picked object: {self._picked_obj}")
        logger.debug(f"Picked object index: {self._picked_obj_index}")
        logger.debug(f"Picked from model: {self._picked_obj_model_attr.name}")
        # Further select ependent objects for operation the blitting function
        if isinstance(picked_obj_model_attr, Trace):
            self._picked_and_dependent_objs = self._tr_view_objs
        elif isinstance(picked_obj_model_attr, Axis):
            self._picked_and_dependent_objs = self._ax_view_objs
        else:
            self._picked_and_dependent_objs = [picked_obj]
        # Actual movement/dragging of objects occurs in motion notify handler
        self.set_mode(self.MODE_DRAG_OBJ)

    def _on_button_release(self, event):
        # Mouse button release event ends MODE_DRAG_OBJ
        if self._op_mode == self.MODE_DRAG_OBJ:
            self.set_mode(self.MODE_DEFAULT)

    def _on_motion_notify(self, event):
        xydata = event.xdata, event.ydata
        # Cursor outside of canvas returns none coordiantes, these are ignored
        if None in xydata:
            return
        picked_obj = self._picked_obj
        if picked_obj is not None:
            index = self._picked_obj_index
            if index is not None:
                # Move normal points
                self._picked_obj_model_attr.update_pt_px(xydata, index)
            else:
                # FIXME: Not implemented, move polygons along X etc.
                pass
        else:
            # FIXME: Update coordinates display etc
            pass


    ########## Canvas object redrawing implements background capture blit
    # Qt AGG backend required:
    # Select view objects for mouse dragging and prepare block image transfer
    # by capturing the canvas background excluding selected view objects.
    # For this, selected objs are temporarily disabled and canvas is redrawn.
    def _capture_objs_background(self, view_objs):
        for obj in view_objs:
            obj.set_visible(False)
        self.canvas_qt.draw()
        self._blit_bg = self.canvas_qt.copy_from_bbox(self.mpl_ax.bbox)
        for obj in view_objs:
            obj.set_visible(True)

    def _blit_background_redraw_objs(self, view_objs):
        self.canvas_qt.restore_region(self._blit_bg)
        # Redraws object using cached renderer
        for obj in view_objs:
            self.mpl_ax.draw_artist(obj)
        # Blitting final step does seem to also update the Qt widget.
        self.canvas_qt.blit(self.mpl_ax.bbox)

    def _based_on_op_mode_capture_background(self):
        # Prepare blitting operation for modes other than MODE_DEFAULT
        # (for default mode, blitting is always done on model view updates)
        if self._op_mode in (self.MODE_SETUP_X_AXIS, self.MODE_SETUP_Y_AXIS):
            self._capture_objs_background(self._ax_view_objs)
        elif self._op_mode == self.MODE_ADD_TRACE_PTS:
            self._capture_objs_background(self._tr_view_objs)
        elif self._op_mode == self.MODE_DRAG_OBJ:
            self._capture_objs_background(self._picked_and_dependent_objs)

    # Layout has only the matplotlib Qt AGG backend as a widget (canvas_qt)
    def _set_layout(self):
        self.setMinimumHeight(self.conf.app_conf.min_plotwin_height)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.addWidget(self.canvas_qt)