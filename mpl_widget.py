#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench MPL View Widget

License: GPL version 3
"""
from collections import namedtuple
from numpy import NaN, isnan

from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMessageBox

import matplotlib.figure
import matplotlib.image as mpimg
import matplotlib.backends.backend_qt5agg as mpl_backend_qt

# Constants definition for modal mouse interaction with the plot canvas
ClickModes = namedtuple(
    "MODES",
    "DEFAULT  SETUP_X_AXIS  SETUP_Y_AXIS  ADD_TRACE_PTS")
MODES = ClickModes(0, 1, 2, 3)


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
    ########## Custom Qt signals for interaction with the input widget
    # Emitted on mode switch
    mode_sw_default = pyqtSignal()
    mode_sw_setup_x_axis = pyqtSignal()
    mode_sw_setup_y_axis = pyqtSignal()
    # Same but int parameter tells which trace number is requested
    mode_sw_add_trace_pts = pyqtSignal(int)
    # Emits True when config OK, False when NOK
    valid_x_axis_setup = pyqtSignal(bool)
    valid_y_axis_setup = pyqtSignal(bool)

    def __init__(self, parent, model, conf):
        super().__init__(parent)
        ########## Operation state
        # What happens when the plot canvas is clicked..
        self.click_mode = MODES.DEFAULT
        self.curr_trace_no = 0
        # This stores the pyplot.Lines2D object when a plot item was picked
        self.picked_obj = None
        # Model component with view associated data for a mouse-picked object
        self.picked_obj_model = None
        # Index of a single picked point inside the view object
        self.picked_obj_index = 0
        # App configuration
        self.conf = conf

        ########## Access to the data model
        self.model = model

        ########## View-Model-Map
        # Mapping of individual view objects (lines, points) to the
        # associated data model components. Dict keys are pyplot.Line2D objects.
        self.view_model_map = {}

        ########## Qt widget setup
        self.setMinimumHeight(100)
        vbox = QVBoxLayout(self)
        self.setLayout(vbox)
        # Message box popup used for various purposes, e.g. points deletion
        self.messagebox = QMessageBox(self)

        ########## Matplotlib figure and axes setup
        self.fig = matplotlib.figure.Figure()
        # Matplotlib figure instance passed to FigureCanvas of matplotlib
        # Qt5Agg backend. This returns the matplotlib canvas as a Qt widget.
        self.canvas_qt = mpl_backend_qt.FigureCanvas(self.fig)
        # Only one matplotlib AxesSubplot instance is used
        self.mpl_ax = self.fig.add_subplot(111)
        self.mpl_ax.xaxis.set_visible(False)
        self.mpl_ax.yaxis.set_visible(False)
        # After removing axis visibility, reset matplotlib layout to fill space
        self.fig.tight_layout(pad=0, rect=(0.001, 0.002, 0.999, 0.999))
        # Add matplotlib widget to this widgets layout
        vbox.addWidget(self.canvas_qt)

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

    
    @pyqtSlot()
    def toggle_setup_x_axis(self):
        if self.click_mode != MODES.SETUP_X_AXIS:
            # Entering X-axis picking mode.
            self.click_mode = MODES.SETUP_X_AXIS
            # Qt signal sets the input widget button to reflect new state
            self.mode_sw_setup_x_axis.emit()
            print("Pick X axis points!")
            if isnan(self.model.x_ax.pts_px[1]).any():
                # First axis point already set, select second one
                self.pick_and_blit(self.model.x_ax, 1)
            else:
                # Begin with first axis point
                self.pick_and_blit(self.model.x_ax, 0)
        else:
            # When already in SETUP_X_AXIS mode, this switches back to default
            self.click_mode = MODES.DEFAULT
            self.mode_sw_default.emit()
    
    @pyqtSlot()
    def toggle_setup_y_axis(self):
        if self.click_mode != MODES.SETUP_Y_AXIS:
            # Entering Y-axis picking mode.
            self.click_mode = MODES.SETUP_Y_AXIS
            # Qt signal sets the input widget button to reflect new state
            self.mode_sw_setup_y_axis.emit()
            print("Pick Y axis points!")
            if isnan(self.model.y_ax.pts_px[1]).any():
                # First axis point already set, select second one
                self.pick_and_blit(self.model.y_ax.pts_view_obj, 1)
            else:
                # Begin with first axis point
                self.pick_and_blit(self.model.y_ax.pts_view_obj, 0)
        else:
            # When already in SETUP_Y_AXIS mode, this switches back to default
            self.click_mode = MODES.DEFAULT
            self.mode_sw_default.emit()

    @pyqtSlot(int)
    def toggle_add_trace_pts_mode(self, trace_no):
        # When already in ADD_TRACE_PTS mode:
        # when same button was clicked, toggle back to default mode.
        # Any other case: Set current trace according to the index number
        # and continue selecting trace points. 
        if (    self.click_mode == MODES.ADD_TRACE_PTS
                and trace_no == self.curr_trace_no):
            self.click_mode = MODES.DEFAULT
            self.mode_sw_default.emit()
        elif not model.axes_setup_is_complete():
            self.show_text("You must configure the axes first!")
            self.click_mode = MODES.DEFAULT
            self.mode_sw_default.emit()
        else:
            self.curr_trace_no = trace_no
            # If trace points have already been selected, ask whether to
            # delete them first before adding new points.
            curr_trace = self.model.traces[trace_no]
            if curr_trace.pts_px.shape[0] > 0 and self.confirm_delete():
                # Clears the data objects of curr_trace
                curr_trace.init_data()
                curr_trace.redraw_pts_px.emit()
                curr_trace.input_changed.emit()
            # Enter or stay in add trace points mode
            self.click_mode = MODES.ADD_TRACE_PTS
            # Qt signal reflecting the new state and trace number
            self.mode_sw_add_trace_pts.emit(trace_no)
            print(f"Pick trace {trace_no + 1} points!")
            curr_trace.add_pt_px((NaN, NaN))
            self.pick_and_blit(
                curr_trace.pts_view_obj, curr_trace.pts_px.shape[0]-1)


    @pyqtSlot(str)
    def load_image(self, filename):
        """Load source/input image for digitizing"""
        # Remove existing image from plot canvas if present
        if hasattr(self, "img_handle"):
            self.img_handle.remove()
        try:
            image = mpimg.imread(filename)
        except Exception as e:
            self.show_error(e)
        #self.resize(image.shape[0]+200, image.shape[1]+200)
        self.img_handle = self.mpl_ax.imshow(
            image[-1::-1], interpolation=self.conf.app_conf.img_interpolation,
            origin="lower", zorder=0,)
        self.canvas_qt.draw_idle()


    @pyqtSlot()
    def using_model_redraw_ax_pts_px(self):
        """Updates axes model features displayed in plot widget,
        including origin only if it fits inside the canvas.

        This also registers the view objects back into the model.
        """
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
        self.mpl_ax.relim()
        self.mpl_ax.autoscale()
        self.canvas_qt.draw_idle()


    @pyqtSlot()
    @pyqtSlot(int)
    def using_model_redraw_tr_pts_px(self, trace_no=None):
        """Draw or redraw trace raw pts_px from the data model.
        If the argument is None, update all traces.

        This also registers the view objects back into the model.
        """
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
        self.mpl_ax.relim()
        self.mpl_ax.autoscale()
        self.canvas_qt.draw_idle()


    @pyqtSlot()
    @pyqtSlot(int)
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
        self.mpl_ax.autoscale()
        self.canvas_qt.draw_idle()


    def on_key_press(self, event):
        print("Event key pressed is: ", event.key)
        if event.key == "escape":
            print("Switching back to default mode")
            self.click_mode = MODES.DEFAULT
            self.mode_sw_default.emit()



    def on_button_press(self, event):
        ########## Add points etc.
        # event: Matplotlib event object
        # matplotlib.backend_bases.MouseButton.RIGHT = 3
        if event.button == 3:
            if self.click_mode != MODES.DEFAULT:
                print("Switching back to default mode")
                self.click_mode = MODES.DEFAULT
                self.mode_sw_default.emit()

        xydata = (event.xdata, event.ydata)
        model = self.model
        # Ignore invalid coordinates (when clicked outside of plot canvas)
        if None in xydata:
            return

        if self.click_mode == MODES.SETUP_X_AXIS:
            # Add point to the model. Emits error signal for invalid data
            model.x_ax.add_pt_px(xydata)

            if isnan(model.x_ax.pts_px[1]).any():
                # Normal case, first pixel was just set, pick second one
                self.pick_and_blit(model.x_ax.pts_view_obj, 1)
            else:
                # Two X-axis points complete. Reset Operating mode.
                self.picked_obj = None
                self.valid_x_axis_setup.emit(True)
                print("X axis points complete")
                self.click_mode = MODES.DEFAULT
                self.mode_sw_default.emit()

        if self.click_mode == MODES.SETUP_Y_AXIS:
            # Add point to the model. Emits error signal for invalid data
            model.y_ax.add_pt_px(xydata)
            
            if isnan(model.y_ax.pts_px[1]).any():
                # Normal case, first pixel was just set, pick second one
                self.pick_and_blit(model.y_ax.pts_view_obj, 1)
            else:
                # Two Y-axis points complete. Reset Operating mode.
                self.picked_obj = None
                self.valid_y_axis_setup.emit(True)
                print("Y axis points complete")
                self.click_mode = MODES.DEFAULT
                self.mode_sw_default.emit()

        if self.click_mode == MODES.ADD_TRACE_PTS:
            # Add point to the model. Emits error signal for invalid data
            tr = model.traces[self.curr_trace_no]
            tr.add_pt_px(xydata)


    def pick_and_blit(self, view_obj, index):
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


    def on_pick(self, event):
        # Mouse pick event handling
        self.pick_and_blit(event.artist, event.ind)
        self.canvas_qt.draw_idle()

    def on_button_release(self, event):
        picked_obj = self.picked_obj
        if picked_obj is None:
            return
        # Model data is updated from the view
        index = self.picked_obj_index
        xydata = picked_obj.get_xydata()
        self.picked_obj_model.pts_px[index] = xydata[index]
        # Restore default state
        self.picked_obj = None
        self.picked_obj_model.redraw_pts_px.emit()
        self.picked_obj_model.input_changed.emit()

    def on_motion_notify(self, event):
        picked_obj = self.picked_obj
        if picked_obj is None:
            return
        index = self.picked_obj_index
        xydata = picked_obj.get_xydata()
        # Implicit cast to np.ndarray
        xydata[index] = (event.xdata, event.ydata)
        # Writing changed pts_px back into the model
        self.picked_obj_model.pts_px[index] = xydata[index]
        picked_obj.set_data(*xydata.T)
        # Blitting operation
        self.canvas_qt.restore_region(self.background)
        # Redraws object using cached Agg renderer
        self.mpl_ax.draw_artist(picked_obj)
        # Blitting final step does seem to also update the Qt widget.
        self.canvas_qt.blit(self.mpl_ax.bbox)
        #self.picked_obj_model.redraw_pts_px.emit()
        #self.picked_obj_model.input_changed.emit()

    def on_figure_enter(self, event):
        self.canvas_qt.setFocus()


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
