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
    MODE_DRAG = 4
    ########## Custom Qt signals for interaction with the input widget
    # Emitted on mode switch, represents the new mode and is used to update
    # the dependand widgets, i.e. state display in configuration boxes.
    mode_sw = pyqtSignal(int)
    # Emits True when config OK, False when NOK
    valid_x_axis_setup = pyqtSignal(bool)
    valid_y_axis_setup = pyqtSignal(bool)


    def __init__(self, digitizer):
        super().__init__(digitizer)
        ########## Operation state
        # What happens when the plot canvas is clicked..
        self.op_mode = self.MODE_DEFAULT
        # Prevents recursive updates when True
        self.inhibit_model_input_data_updates = False
        self.curr_trace_no = 0
        # This stores the pyplot.Lines2D object when a plot item was picked
        self.picked_obj = None
        # Model component with view associated data for a mouse-picked object
        self.picked_obj_model = None
        # Index of a single picked point inside the view object
        self.picked_obj_index = 0
        # App configuration
        self.conf = digitizer.conf

        ########## Access to the data model
        self.model = digitizer.model

        ########## View-Model-Map
        # Mapping of individual view objects (lines, points) to the
        # associated data model components. Dict keys are pyplot.Line2D objects.
        self.view_model_map = {}

        ########## Qt widget setup
        self.setMinimumHeight(100)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 6, 0)
        # Message box popup used for various purposes, e.g. points deletion
        self.messagebox = QMessageBox(self)

        ########## Matplotlib figure and axes setup
        self.fig = matplotlib.figure.Figure()
        # Matplotlib figure instance passed to FigureCanvas of matplotlib
        # Qt5Agg backend. This returns the matplotlib canvas as a Qt widget.
        self.canvas_qt = mpl_backend_qt.FigureCanvas(self.fig)
        # Only one matplotlib AxesSubplot instance is used
        self.mpl_ax = self.fig.add_subplot(111, autoscale_on=False)
        self.mpl_ax.xaxis.set_visible(False)
        self.mpl_ax.yaxis.set_visible(False)
        # After removing axis visibility, reset matplotlib layout to fill space
        self.fig.tight_layout(pad=0, rect=(0.001, 0.002, 0.999, 0.999))
        # Add matplotlib widget to this widgets layout
        layout.addWidget(self.canvas_qt)

        ########## Connect non-public events and signals
        self.canvas_qt.mpl_connect("key_press_event", self.on_key_press)
        self.canvas_qt.mpl_connect("figure_enter_event", self.on_figure_enter)
        self.canvas_qt.mpl_connect("button_press_event", self.on_button_press)
        self.canvas_qt.mpl_connect(
                "button_release_event", self.on_button_release)
        self.canvas_qt.mpl_connect("motion_notify_event", self.on_motion_notify)
        self.canvas_qt.mpl_connect("pick_event", self.on_pick)

        ########## Initial view update
        self.using_model_redraw_ax_pts_px()
        self.using_model_redraw_tr_pts_px()


    # Mplwidget operation state transitions
    def set_mode(self, new_mode: int):
        logger.debug(f"Entering new mode: {new_mode}.")
        # Prevent recursive updates for most modes
        self.inhibit_model_input_data_updates = True
        if new_mode == self.MODE_SETUP_X_AXIS:
            logger.info("Pick X axis points!")
        elif new_mode == self.MODE_SETUP_Y_AXIS:
            logger.info("Pick Y axis points!")
        elif new_mode == self.MODE_ADD_TRACE_PTS:
            logger.info(f"Add points for trace {self.curr_trace_no + 1}!")
        elif new_mode == self.MODE_DRAG:
            logger.info("Drag the picked object!")
        else:
            logger.info("Switching back to default mode")
            new_mode = self.MODE_DEFAULT
            self.picked_obj = None
            # Default mode displays data updates from outside this widget
            self.inhibit_model_input_data_updates = False
        self.op_mode = new_mode
        self.mode_sw.emit(new_mode)


    @logExceptionSlot(str)
    def load_image(self, filename):
        """Load source/input image for digitizing"""
        # Remove existing image from plot canvas if present
        if hasattr(self, "img_handle"):
            self.img_handle.remove()
        try:
            image = matplotlib.image.imread(filename)
        except Exception as e:
            self.show_error(e)
        #self.resize(image.shape[0]+200, image.shape[1]+200)
        self.img_handle = self.mpl_ax.imshow(
                image[-1::-1],
                interpolation=self.conf.app_conf.img_interpolation,
                origin="lower",
                zorder=0,
                )
        self.mpl_ax.autoscale(enable=True)
        self.canvas_qt.draw_idle()
        self.mpl_ax.autoscale(enable=False)


    @logExceptionSlot()
    def toggle_setup_x_axis_mode(self):
        x_ax = self.model.x_ax
        # When already in SETUP_X_AXIS mode, switch back to default mode
        if self.op_mode == self.MODE_SETUP_X_AXIS:
            self.set_mode(self.MODE_DEFAULT)
        else:
            self.set_mode(self.MODE_SETUP_X_AXIS)
            # Point is actually only displayed when a mouse move event occurs
            # and point data is updated from valid mouse coordinates.
            pt_index = x_ax.add_pt_px(np.full(2, NaN))
            self.pick_and_blit(x_ax.pts_view_obj, pt_index)
 
    @logExceptionSlot()
    def toggle_setup_y_axis_mode(self):
        y_ax = self.model.y_ax
        # When already in SETUP_Y_AXIS mode, switch back to default mode
        if self.op_mode == self.MODE_SETUP_Y_AXIS:
            self.set_mode(self.MODE_DEFAULT)
        else:
            self.set_mode(self.MODE_SETUP_Y_AXIS)
            # Point is actually only displayed when a mouse move event occurs
            # and point data is updated from valid mouse coordinates.
            pt_index = y_ax.add_pt_px(np.full(2, NaN))
            self.pick_and_blit(y_ax.pts_view_obj, pt_index)

    @logExceptionSlot(int)
    def toggle_add_trace_pts_mode(self, trace_no):
        # When already in ADD_TRACE_PTS mode:
        # when same button was clicked, toggle back to default mode.
        # Otherwise, given valid axes setup, set current trace according to
        # the index number and continue selecting trace points. 
        if (self.op_mode == self.MODE_ADD_TRACE_PTS
            and trace_no == self.curr_trace_no
            ):
            self.set_mode(self.MODE_DEFAULT)
        elif not self.model.axes_setup_is_complete():
            self.show_text("You must configure the axes first!")
            self.set_mode(self.MODE_DEFAULT)
        else:
            self.curr_trace_no = trace_no
            tr = self.model.traces[trace_no]
            # If trace points have already been selected, ask whether to
            # delete them first before adding new points.
            if tr.pts_px.shape[0] > 0 and self.confirm_delete():
                # Clears data objects of curr_trace and triggers a view update
                tr.init_data()
            pt_index = tr.add_pt_px(np.full(2, NaN))
            # View from model update, this is necessary because
            # self.inhibit_model_input_updates is set
            tr.pts_view_obj.set_data(*tr.pts_px.T)
            self.pick_and_blit(tr.pts_view_obj, pt_index)
            # Enter or stay in add trace points mode
            self.set_mode(self.MODE_ADD_TRACE_PTS)

    @logExceptionSlot()
    def using_model_redraw_ax_pts_px(self):
        """Updates axes model features displayed in plot widget,
        including origin only if it fits inside the canvas.

        This also registers the view objects back into the model.
        """
        logger.debug("using_model_redraw_ax_pt_px called")
        # Prevent recursive updates
        if self.inhibit_model_input_data_updates:
            return
        model = self.model
        ########## X and Y axis:
        for ax in model.x_ax, model.y_ax:
            if ax.pts_view_obj is not None:
                ax.pts_view_obj.set_data(*ax.pts_px.T)
            else:
                ax.pts_view_obj, = self.mpl_ax.plot(*ax.pts_px.T, **ax.pts_fmt)
                self.view_model_map[ax.pts_view_obj] = ax
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
                    self.view_model_map[view_obj] = None
                    model.origin_view_obj = view_obj
                else:
                    model.origin_view_obj.set_data(*model.origin_px)
        ##### Anyways, after updating axes point markers, redraw plot canvas:
        # Autoscale and fit axes limits in case points are outside image limits
        self.mpl_ax.relim()
        self.mpl_ax.autoscale(None)
        self.canvas_qt.draw_idle()


    @logExceptionSlot()
    @logExceptionSlot(int)
    def using_model_redraw_tr_pts_px(self, trace_no=None):
        """Draw or redraw trace raw pts_px from the data model.
        If the argument is None, update all traces.

        This also registers the view objects back into the model.
        """
        logger.debug("using_model_redraw_tr_pts_px called")
        # Prevent recursive updates
        if self.inhibit_model_input_data_updates:
            return
        model = self.model
        traces = model.traces if trace_no is None else [model.traces[trace_no]]
        for tr in traces:
            ########## Update raw pixel points
            if tr.pts_view_obj is not None:
                tr.pts_view_obj.set_data(*tr.pts_px.T)
            else:
                tr.pts_view_obj, = self.mpl_ax.plot(*tr.pts_px.T, **tr.pts_fmt)
                self.view_model_map[tr.pts_view_obj] = tr
        # Anyways, after updating point markers, redraw plot canvas:
        self.canvas_qt.draw_idle()


    @logExceptionSlot()
    @logExceptionSlot(int)
    def update_output_view(self, trace_no=None):
        """Draw or redraw a trace from the data model.
        If the argument is None, update all traces.
        """
        model = self.model
        traces = model.traces if trace_no is None else [model.traces[trace_no]]
        ########## Update interpolated trace if available
        for tr in traces:
            # Draw or update data on an interpolated plot line.
            # Backtransform trace to pixel data coordinate system
            pts_i_px = model.get_pts_lin_i_px_coords(tr)
            view_obj = tr.pts_i_view_obj
            if view_obj is None:
                # Draw trace on matplotlib widget
                view_obj, = self.mpl_ax.plot(*pts_i_px.T, **tr.pts_i_fmt)
                # The origin point is calculated and not supposed to be
                # subjected to drag-and-drop etc.: registering it as None
                self.view_model_map[view_obj] = None
                tr.pts_i_view_obj = view_obj
            else:
                # Trace handle for pts_lin_i exists. Update data.
                view_obj.set_data(*pts_i_px.T)
        ########## After updating all traces and markers, redraw plot canvas
        self.mpl_ax.relim()
        self.mpl_ax.autoscale(None)
        self.canvas_qt.draw_idle()


    def on_figure_enter(self, event):
        # Set Qt keyboard input focus to the matplotlib canvas
        # in order to receive key press events
        self.canvas_qt.setFocus()

    def on_key_press(self, event):
        logger.debug(f"Event key pressed is: {event.key}")
        if event.key == "escape":
            self.set_mode(self.MODE_DEFAULT)


    def on_button_press(self, event):
        ########## Mouse click event handler
        # event: Matplotlib event object
        # matplotlib.backend_bases.MouseButton.RIGHT = 3
        if event.button == 3:
            if self.op_mode != self.MODE_DEFAULT:
                self.set_mode(self.MODE_DEFAULT)
            return
        model = self.model
        px_xy = (event.xdata, event.ydata)
        # Ignore invalid coordinates (when clicked outside of plot canvas)
        if None in px_xy:
            return
        ##### Add X-axis point
        if self.op_mode == self.MODE_SETUP_X_AXIS:
            ax = model.x_ax
            ax.update_pt_px(px_xy, self.picked_obj_index)
            if isnan(ax.pts_px).any():
                # First point was added before when MODE_SETUP_X_AXIS was set.
                # Add new point to the model and continue. Point is actually
                # only first displayed when a mouse move event occurs
                # and point data is updated from valid mouse coordinates.
                pt_index = ax.add_pt_px(px_xy)
                self.pick_and_blit(ax.pts_view_obj, pt_index)
            else:
                logger.debug(
                        "This should now be complete and switch to default!")
                # Two X-axis points set. Validate and reset op mode if valid
                if ax.valid_pts_px():
                    logger.info("X axis points complete")
                    self.valid_x_axis_setup.emit(True)
                    self.set_mode(self.MODE_DEFAULT)
            return
        ##### Add Y-axis point
        if self.op_mode == self.MODE_SETUP_Y_AXIS:
            ax = model.y_ax
            ax.update_pt_px(px_xy, self.picked_obj_index)
            if isnan(ax.pts_px).any():
                # First point was added before when MODE_SETUP_Y_AXIS was set.
                # Add new point to the model and continue. Point is actually
                # only first displayed when a mouse move event occurs
                # and point data is updated from valid mouse coordinates.
                pt_index = ax.add_pt_px(px_xy)
                self.pick_and_blit(ax.pts_view_obj, pt_index)
            else:
                # Two X-axis points set. Validate and reset op mode if valid
                if ax.valid_pts_px():
                    logger.info("Y axis points complete")
                    self.valid_y_axis_setup.emit(True)
                    self.set_mode(self.MODE_DEFAULT)
            return
        ##### Add trace point
        if self.op_mode == self.MODE_ADD_TRACE_PTS:
            tr = model.traces[self.curr_trace_no]
            # Add new point to the model at current mouse coordinates
            pt_index = tr.add_pt_px(px_xy)
            self.pick_and_blit(tr.pts_view_obj, pt_index)
            return
        

    def on_pick(self, event):
        # Mouse pick event handling
        # Prevent recursive updates
        self.inhibit_model_input_data_updates = True
        index = event.ind if hasattr(event, "ind") else None
        #self.pick_origin_x = event.mouseevent.xdata
        self.pick_and_blit(event.artist, index)
        self.canvas_qt.draw_idle()
        

    def on_button_release(self, event):
        # The mouse button release event is only used for object drag-mode
        if self.op_mode != self.MODE_DRAG:
            return
        picked_obj = self.picked_obj
        if picked_obj is None:
            return
        # Model data is updated from the view
        index = self.picked_obj_index
        xydata = picked_obj.get_xydata()
        # This also emits the pts_changed signal
        self.picked_obj_model.update_pt_px(xydata[index], index)
        # Restore default state
        self.set_mode(self.MODE_DEFAULT)
        #self.picked_obj = None


    def on_motion_notify(self, event):
        picked_obj = self.picked_obj
        if picked_obj is not None:
            index = self.picked_obj_index
            if index is not None:
                # Move normal points
                xydata = picked_obj.get_xydata()
                # Implicit cast to np.ndarray
                xydata[index] = event.xdata, event.ydata
                # Set data only in view component
                # (model is updated on button release)
                picked_obj.set_data(*xydata.T)
                ### Option:
                # Set data also in model when dragging existing points.
                # Causes re-calculation.
                self.picked_obj_model.update_pt_px(xydata[index], index)
                ###
                # Blitting operation
                self.canvas_qt.restore_region(self.background)
                # Redraws object using cached Agg renderer
                self.mpl_ax.draw_artist(picked_obj)
                # Blitting final step does seem to also update the Qt widget.
                self.canvas_qt.blit(self.mpl_ax.bbox)
            else:
                # FIXME: Not implemented, move polygons along X etc.
                pass


    def pick_and_blit(self, view_obj, index):
        # Select the view object for mouse dragging and prepare bit blitting
        # operation by capturing the canvas background after temporarily
        # disabling the selected view object and re-enabling it afterwards.
        #
        # Model component with view associated data for a mouse-picked object
        # is looked up from the view-model mapping.
        picked_obj_model = self.view_model_map[view_obj]
        # For non-mouse-pickable view objects, a None is stored in the mapping
        if picked_obj_model is None:
            # Not found in mapping, view object is thus not mouse-pickable
            return
        # View_obj is mouse-pickable; selected view object is added to instance
        self.picked_obj = view_obj
        # also the associated data model component
        self.picked_obj_model = picked_obj_model
        # Index of a single picked point inside the view object
        self.picked_obj_index = index
        ##### Setting up bit blitting for smooth drag and drop operation
        view_obj.set_visible(False)
        self.canvas_qt.draw()
        self.background = self.canvas_qt.copy_from_bbox(self.mpl_ax.bbox)
        view_obj.set_visible(True)
        

    def confirm_delete(self):
        self.messagebox.setIcon(QMessageBox.Warning)
        self.messagebox.setText(
            "<b>There are trace points already selected.\n"
            "Discard or save and add more Points?</b>"
            )
        self.messagebox.setWindowTitle("Confirm Delete")
        self.messagebox.setStandardButtons(
                QMessageBox.Save | QMessageBox.Discard)
        return self.messagebox.exec_() == QMessageBox.Discard


    def show_error(self, error):
        self.messagebox.setIcon(QMessageBox.Warning)
        self.messagebox.setText("<b>Error!</b>")
        self.messagebox.setInformativeText(error.args[0])
        self.messagebox.setWindowTitle("Input Error")
        self.messagebox.exec_()

    def show_text(self, text):
        self.messagebox.setIcon(QMessageBox.Warning)
        self.messagebox.setText("<b>Please note:</b>")
        self.messagebox.setInformativeText(text)
        self.messagebox.setWindowTitle("Plot Workbench Notification")
        self.messagebox.exec_()
