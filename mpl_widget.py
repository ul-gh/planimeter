#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple

from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMessageBox

import matplotlib.figure
import matplotlib.image as mpimg
import matplotlib.backends.backend_qt5agg as mpl_backend_qt

# Constants definition, representing the operation modes of the graphic widget
ClickModes = namedtuple(
    "MODES",
    "DEFAULT  SETUP_X_AXIS  SETUP_Y_AXIS  ADD_TRACE_PTS  DEL_TRACE_PTS")
MODES = ClickModes(0, 1, 2, 3, 4)


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

    def __init__(self, parent, model):
        super().__init__(parent)
        ########## Operation state
        # What happens when the plot canvas is clicked..
        self.click_mode = MODES.DEFAULT
        self.curr_trace_no = 0

        ########## Access to the data model
        self.model = model

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
        self.canvas_qt.mpl_connect("button_press_event", self._plot_clicked)
        #self.canvas_qt.mpl_connect("motion_notify_event", self.onMouseMove)
        #self.canvas_qt.mpl_connect("pick_event", self.itemPicked)

        self.canvas_qt.draw_idle()
        # FIXME: testing only:
        #self.display_image("tmp/cossnew.png")
        #self.display_image("tmp/i_d.png")

    
    @pyqtSlot()
    def toggle_setup_x_axis(self):
        if self.click_mode != MODES.SETUP_X_AXIS:
            # Entering X-axis picking mode.
            self.click_mode = MODES.SETUP_X_AXIS
            # Qt signal sets the input widget button to reflect new state
            self.mode_sw_setup_x_axis.emit()
#            # Set pick index to zero to pick first of two points
#            self.pt_index = 0
            print("Pick two X axis section points!")
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
#            # Set pick index to zero to pick first of two points
#            self.pt_index = 0
            print("Pick two Y axis section points!")
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
        else:
            self.curr_trace_no = trace_no
            # If trace points have already been selected, ask whether to
            # delete them first before adding new points.
            curr_trace = self.model.traces[trace_no]
            if len(curr_trace.pts_px) > 0 and self.confirm_delete():
                curr_trace.pts_px = []
                curr_trace.redraw_pts_px.emit()
                curr_trace.input_changed.emit()
            # Enter or stay in add trace points mode
            self.click_mode = MODES.ADD_TRACE_PTS
            # Qt signal reflecting the new state and trace number
            self.mode_sw_add_trace_pts.emit(trace_no)
            print(f"Pick trace {trace_no + 1} points!")



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
            image[-1::-1], interpolation="lanczos", origin="lower", zorder=0)
        self.canvas_qt.draw_idle()

    @pyqtSlot()
    def using_model_redraw_ax_pts_px(self):
        """Updates axes model features displayed in plot widget,
        including origin only if it fits inside the canvas.

        This also registers the view objects back into the model.
        """
        model = self.model
        # X and Y axis:
        for ax in model.x_ax, model.y_ax:
            for i, artist in enumerate(ax.view_objs):
                if artist is not None:
                    if ax.pts_px[i] is not None:
                        artist.set_data(*ax.pts_px[i])
                    else:
                        artist.remove()
                        ax.view_objs[i] = None
                        del model.view_model_map[artist]
                else:
                    # artist is None, draw new point
                    artist, = self.mpl_ax.plot(*ax.pts_px[i], **ax.pts_fmt)
                    ax.view_objs[i] = artist
                    model.view_model_map[artist] = ax

        # Origin:
        if model.origin_px is None:
            if model.origin_view_obj is not None:
                model.origin_view_obj.remove()
                model.origin_view_obj = None
        else:
            # Containment check via numpy elementwise operators
            if (    (model.origin_px < self.fig.bbox.size).all()
                    and (model.origin_px > 0).all()
                    ):
                if model.origin_view_obj is None:
                    artist, = self.mpl_ax.plot(
                        *model.origin_px, **model.origin_fmt)
                    model.origin_view_obj = artist
                    # The origin is not registered into the data model as this
                    # is a calculated property, not input data.
                else:
                    model.origin_view_obj.set_data(*model.origin_px)
        # Anyways, after updating axes point markers, redraw plot canvas:
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
            n_pts = len(tr.pts_px)
            n_view_objs = len(tr.view_objs)
            print(  "new debug: trace_no, n_pts, n_view_objs",
                    model.traces.index(tr), n_pts, n_view_objs)
            if n_view_objs > n_pts:
                # There is more markers than data. Delete excess markers.
                for i in range(n_pts, n_view_objs):
                    artist = tr.view_objs[i]
                    #if obj in model.view_model_map:
                    #    del model.view_model_map[obj]
                    del model.view_model_map[artist]
                    artist.remove()
                    del tr.view_objs[i]
            # Anyways, after removing excess markers:
            # There is enough or more data to update markers (or no markers).
            # Update existing maker handles, then extend list of handles
            for i in range(n_view_objs):
                tr.view_objs[i].set_data(*tr.pts_px[i])
            for i in range(n_view_objs, n_pts):
                artist, = self.mpl_ax.plot(*tr.pts_px[i], **tr.pts_fmt)
                tr.view_objs.append(artist)
                model.view_model_map[artist] = tr.pts_px
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
            if tr.pts_lin_i is not None:
                # Draw or update data on an interpolated plot line.
                # Backtransform trace to pixel data coordinate system
                pts_i_px = model.get_pts_lin_i_px_coords(tr)
                artist = tr.pts_i_view_obj
                if artist is None:
                    # Draw trace on matplotlib widget
                    artist, = self.mpl_ax.plot(*pts_i_px.T, **tr.pts_i_fmt)
                    tr.pts_i_view_obj = artist
                else:
                    # Trace handle for pts_lin_i exists. Update data.
                    artist.set_data(*pts_i_px.T)
            elif tr.pts_i_view_obj is not None:
                # No data in model but trace handle exists. Delete trace.
                tr.pts_i_view_obj.remove()
                tr.pts_i_view_obj = None
        ########## After updating all traces and markers, redraw plot canvas
        self.mpl_ax.relim()
        self.mpl_ax.autoscale()
        self.canvas_qt.draw_idle()


    def _plot_clicked(self, event):
        ########## Add points etc.
        # event: Matplotlib event object
        xydata = (event.xdata, event.ydata)
        model = self.model
        # Ignore invalid coordinates (when clicked outside of plot canvas)
        if None in xydata:
            return

        if self.click_mode == MODES.SETUP_X_AXIS:
            artist, = self.mpl_ax.plot(*xydata, **model.x_ax.pts_fmt)
            self.canvas_qt.draw_idle()
            # Add point to the model. In case the value is invalid,
            # emits an error message
            model.using_view_add_x_ax_pt_px(artist)
            if None not in model.x_ax.pts_px:
                # Two X-axis points complete. Reset Operating mode.
                self.valid_x_axis_setup.emit(True)
                print("X axis points complete")
                self.click_mode = MODES.DEFAULT
                self.mode_sw_default.emit()

        if self.click_mode == MODES.SETUP_Y_AXIS:
            artist, = self.mpl_ax.plot(*xydata, **model.y_ax.pts_fmt)
            self.canvas_qt.draw_idle()
            # Add point to the model. In case the value is invalid,
            # emits an error message
            model.using_view_add_y_ax_pt_px(artist)
            if None not in model.y_ax.pts_px:
                # Two Y-axis points complete. Reset Operating mode.
                self.valid_y_axis_setup.emit(True)
                print("Y axis points complete")
                self.click_mode = MODES.DEFAULT
                self.mode_sw_default.emit()

        if self.click_mode == MODES.ADD_TRACE_PTS:
            if not model.axes_setup_is_complete():
                self.show_text("You must configure the axes first!")
                self.click_mode = MODES.DEFAULT
                self.mode_sw_default.emit()
            else:
                curr_trace = model.traces[self.curr_trace_no]
                artist, = self.mpl_ax.plot(*xydata, **curr_trace.pts_fmt)
                self.canvas_qt.draw_idle()
                # Add point to the model. In case the value is invalid,
                # emits an error message
                model.using_view_add_trace_pt_px(artist, self.curr_trace_no)


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