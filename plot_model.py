#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Data Model

License: GPL version 3
"""
import logging
logger = logging.getLogger(__name__)

from functools import partial

import numpy as np
from numpy import NaN, isnan, isclose
from scipy.interpolate import interp1d

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class DataModel(QObject):
    """DataModel
    
    UI/data interactive model for digitizing a graph plot using
    linear or logarithmic scale cartesic or skew (e.g. from perspective)
    affine-linear coordinate system on an (orthogonal) pixel image space.

    Data coordinate X and Y axes can be /offset/, /skew/ and /rotated/
    arbitrarily and each is defined by setting coinciding start and end
    points of an axis section both in image pixel coordinates and their
    associated data coordinate values.

    When logarithmic scale data coordinates are configured, a linearised
    data space representation is used. The log values of the data coor-
    dinate axes section start and end point values are calculated
    during axes configuration and coordinate transformation is performed
    as is the case for linear coordinates. Only the result values are
    transformed back to logarithmic space by exponentiation.

    Data coordinate axes configuration is performed by accessing
    instances of class Axis, see definition further below in
    this module.

    Individual plot traces are stored in a list of Trace type instances.
    
    2019-07-29 Ulrich Lukas
    """
    ########## Qt signals
    # This overloaded signal triggers a GUI update when the model has updated
    # trace data. With trace number as argument, that trace is updated.
    # Without argument, all traces are updated.
    # Since the input data is normally set by the view itself, a redraw of raw
    # input data is not performed when this signal is emitted.
    output_data_changed = pyqtSignal([], [int])
    # Mainly used for configuring export options
    export_options_changed = pyqtSignal()
    # This is re-emitted from the traces and triggers a re-display of the raw
    # (pixel-space) input points. This can be indexed with a trace number.
    redraw_tr_pts_px = pyqtSignal([], [int])
    # Same for axes points
    redraw_ax_pts_px = pyqtSignal()
    # GUI error feedback when invalid data was entered
    value_error = pyqtSignal(str)
    # GUI feedback when export range settings are outside of points range
    export_range_warning = pyqtSignal()

    def __init__(self, parent, conf):
        super().__init__(parent)
        ########## Plot model composition
        ##### Two axes
        self.x_ax = Axis(self, conf.x_ax_conf)
        self.y_ax = Axis(self, conf.y_ax_conf)
        ##### Origin
        self.origin_px = np.full(2, NaN)
        # Matplotlib format code
        self.origin_fmt = conf.model_conf.origin_fmt
        self.origin_view_obj = None
        ##### Arbitrary number of traces
        # Three traces are default, see main.DefaultConfig
        self.traces = [
            Trace(self, conf.trace_conf, name, color)
            for name, color
            in zip(conf.model_conf.traces_names, conf.model_conf.traces_colors)
            ]

        ##### Trace export options
        # X-axis range used for interpolating traces export data.
        # In order to avoid strange results due to uncalled-for extrapolation
        # when no extrapolation function is defined, this is later set to the 
        # intersection of definition ranges of all traces marked for export.
        self.x_start_export = NaN
        self.x_end_export = NaN
        self.x_export_log_scale = False
        # Number of X-axis points per decade in case of log X grid
        self.n_pts_dec_export = conf.model_conf.n_pts_dec_export
        # Maximum number of export points for user input verification
        self.n_pts_export_max = conf.model_conf.n_pts_export_max
        # Number of X-axis points of a linear interpolation grid
        self.update_n_pts_export(conf.model_conf.n_pts_export)
        self.update_x_step_export(conf.model_conf.x_step_export)
        ##### X-axis grid in data coordinates used for export
        self.x_grid_export = None # Optional: np.ndarray

        ##### Common settings
        # Python string format code for display of numbers
        self.num_fmt = conf.app_conf.num_fmt_gui
        # Absolute tolerance for testing if values are close to zero
        self.atol = conf.model_conf.atol
        # Store axes configuration persistently on disk when set
        self.store_ax_conf = conf.model_conf.store_ax_conf


        ########## Restore data model configuration and state from stored data
        if conf.x_ax_state is not None:
            vars(self.x_ax).update(conf.x_ax_state)
        if conf.y_ax_state is not None:
            vars(self.y_ax).update(conf.y_ax_state)

        ########## Connect signals from the axes and traces components
        for ax in (self.x_ax, self.y_ax):
            # Axis points movement triggers a recalculation of the transform
            ax.pts_changed.connect(self._calculate_coordinate_transformation)
            # So does adding or removing points, butt also causes re-display
            ax.pts_added_deleted.connect(
                self._calculate_coordinate_transformation)
            # So do configuration changes
            ax.config_changed.connect(self._calculate_coordinate_transformation)
            # Error signals re-emitted here, later connected to message pop-up
            ax.value_error.connect(self.value_error)
        # Trace data changes trigger a model update for that trace only.
        for i, tr in enumerate(self.traces):
            # When points are moved, update outputs but no sorting is needed
            tr.pts_changed.connect(
                partial(self.calculate_live_outputs, i, False))
            # Export range update
            tr.pts_changed.connect(self.update_export_settings)
            # When points are added or deleted, not only update outputs but
            # re-sort the points and update the view of the raw points
            tr.pts_added_deleted.connect(
                partial(self.calculate_live_outputs, i, True))
            tr.pts_added_deleted.connect(
                partial(self.redraw_tr_pts_px[int].emit, i))
            # Export range update
            tr.pts_added_deleted.connect(self.update_export_settings)
            # The errors are also re-emitted.
            tr.value_error.connect(self.value_error)

        ########## Initialise model outputs if axes are configured
        if self.axes_setup_is_complete():
            self._calculate_coordinate_transformation()

    ########## GUI scope and public methods
    @pyqtSlot()
    def export_traces(self):
        """Interpolate data from the given trace numbers using a common
        interpolation grid with n_interp X-axis points, spaced evenly
        in linearised data coordinate units between x_start and x_end
        and return the resulting data as rows.
        
        Parameters from DataModel instance:
            self.traces : Trace[*]
                One or more traces (zero-indexed) for which to
                interpolate and export data
            self.x_grid_export : np.ndarray
                Common X axis values for all traces
        
        Outputs:
            self.output_arr : np.ndarray
                Output trace data with X axis values in first row,
                traces data in following rows
        
        Note:
        For generating the export data, the cubic spline interpolation
        polynomial representation of each trace is used by default.
        This function needs at least four data points from each trace.

        If no interpolation data is available for a trace, a column of
        NaN values is returned for that trace.
        """
        ######### Generate export X axis grid
        if self.x_export_log_scale:
            # Grid is specified by number of points per decade; thus using log10
            x_start_lin = np.log10(self.x_start_export)
            x_end_lin = np.log10(self.x_end_export)
            x_grid_export = np.logspace(
                    x_start_lin, x_end_lin, self.n_pts_export)
        else:
            # self.x_step_export is pre-calculated by the update_export_range
            # method or set by the user.
            x_grid_export = np.arange(
                    self.x_start_export, self.x_end_export, self.x_step_export)
        # Anyways:
        self.x_grid_export = x_grid_export

        ########## Calculate export traces Y data using interpolation function
        n_interp = x_grid_export.shape[0]
        interp_funcs = [tr.f_interp for tr in self.traces if tr.export]
        output_arr = np.empty((1+len(trace_funcs), n_interp))
        output_arr[0] = x_grid_export
        for row, f_interp in enumerate(interp_funcs, start=1):
            output_arr[row] = f_interp(x_grid_export)
        self.output_arr = output_arr

    @pyqtSlot(float)
    def set_x_start_export(self, x_start):
        if isclose(self.x_end_export - x_start, 0.0, atol=self.atol):
            self.value_error.emit("X axis section must not be zero length")
            return
        self.autorange_export = False
        self.x_start_export = x_start
        self.update_export_range()

    @pyqtSlot(float)
    def set_x_end_export(self, x_end):
        if isclose(x_end - self.x_start_export, 0.0, atol=self.atol):
            self.value_error.emit("X axis section must not be zero length")
            return
        self.autorange_export = False
        self.x_end_export = x_end
        self.update_export_range()

    @pyqtSlot(bool)
    def set_autorange_export(self, state=True):
        self.autorange_export = state
        self.update_export_range()

    def update_export_range(self):
        """Set or check export X values range.

        Parameters from DataModel instance:
            self.x_start_export : float
                Start of the data range on the X-axis used for export
            self.x_end_export : float
                End of the data range on the X-axis used for export
            self.x_export_log_scale : bool
                True when output with log-scale X axis is requested
            
            FIXME: incomplete...

        Output:
            self.x_grid_export : np.ndarray
               Evenly spaced points in linear or logarithmic data space

        When self.autorange_export is set (default), the X axis value
        range is automatically determined as the intersection of
        definition ranges of all traces marked for export.

        In other words, data is not extrapolated.

        When manual limits are entered, this emits an
        export_range_warning if the range results in extrapolation.
        """
        ########## Calculate interpolation limits
        try:
            x_start_lin_limit = max(
                    # max() takes a generator expression
                    tr.pts_lin[0,0] for tr in self.traces if tr.export
                    )
            x_end_lin_limit = min(
                    tr.pts_lin[-1,0] for tr in self.traces if tr.export
                    )
        except ValueError as e:
            logger.info(f"Got error: {e.args[0]}. No trace marked for export?")
            return
        # Anti-log treatment for X axis
        if self.x_ax.log_scale:
            x_start_export_limit = self.x_ax.log_base ** x_start_lin_limit
            x_end_export_limit = self.x_ax.log_base ** x_end_lin_limit
        else:
            x_start_export_limit = self.x_start_lin_limit
            x_end_export_limit = self.x_end_lin_limit

        ########## Autoranging if activated, else range checking only
        if self.autorange_export:
            # Set class attribute to enable GUI display of limits
            self.x_start_export = x_start_export_limit
            self.x_end_export = x_end_export_limit
        else:
            if (    self.x_start_export < x_start_export_limit
                    or self.x_end_export > x_end_export_limit
                    ):
                self.export_range_warning.emit()

        ########## Update total number of output points
        # FIXME: This prefers keeping the linear X spacing and modifying
        # the total number of points when changing export limits.
        # Perhaps add a config option?
        self.update_n_pts_export()

    def update_n_pts_export(self):
        # Update total number of output points
        if self.x_export_log_scale:
            x_start_lin = np.log10(self.x_start_export)
            x_end_lin = np.log10(self.x_end_export)
            n_dec = x_end_lin - x_start_lin
            N_tot = n_dec * self.n_pts_dec_export
            self.n_pts_export = min(
                    self.n_pts_export_max,
                    N_tot
                    )
        else:
            x_step = self.x_step_export
            self.n_pts_export = min(
                    self.n_pts_export_max,
                    1 + int((self.x_end_export - self.x_start_export) / x_step)
                    )
        self.export_options_changed.emit()

    def update_n_pts_dec_export(self):
        x_start_lin = np.log10(self.x_start_export)
        x_end_lin = np.log10(self.x_end_export)
        n_dec = x_end_lin - x_start_lin
        self.n_pts_dec_export = self.n_pts_export / n_dec
        self.export_options_changed.emit()


    def update_x_step_export(self):
        x_step = (self.x_end_export - self.x_start_export) / self.n_pts_export
        self.export_options_changed.emit()

    @pyqtSlot(int)
    def set_n_pts_export(self, n_pts=None):
        if n_pts is None or n_pts > self.n_pts_export_max:
            n_pts = self.n_pts_export_max
        if n_pts < 1:
            n_pts = 1
        if x_export_log_scale:
            self.update_n_pts_dec_export()
        else:
            self.update_x_step_export()

    @pyqtSlot(float)
    def set_x_step_export(self, x_step):
        # Only to be called in linear export mode
        x_step_min = (self.x_end_export - self.x_start_export
                      ) / self.n_pts_export_max
        self.x_step_export = max(x_step_min, x_step)
        self.update_n_pts_export()

    @pyqtSlot(float)
    def set_n_pts_dec_export(self, n_pts_dec):
        # Only to be called in log export mode
        x_start_lin = np.log10(self.x_start_export)
        x_end_lin = np.log10(self.x_end_export)
        n_dec = x_end_lin - x_start_lin
        N_tot = n_dec * n_pts_dec
        self.n_pts_dec_export = min(N_tot, self.n_pts_export_max)
        self.update_n_pts_export()


    def axes_setup_is_complete(self) -> bool:
        """Returns True if axes configuration is all complete and valid
        """
        x_ax = self.x_ax
        y_ax = self.y_ax
        if None in x_ax.pts_data or None in y_ax.pts_data:
            return False
        if isnan(np.concatenate((x_ax.pts_px, y_ax.pts_px))).any():
            return False
        invalid_x = x_ax.log_scale and isclose(
            x_ax.pts_data, 0.0, atol=x_ax.atol).any()
        invalid_y = y_ax.log_scale and isclose(
            y_ax.pts_data, 0.0, atol=y_ax.atol).any()
        if invalid_x or invalid_y:
            return False
        return True

    def get_pts_lin_px_coords(self, trace) -> np.ndarray:
        """Returns graph mouse selected points in linearised pixel data
        coordinate system. Used for backplotting the transformed points.
        """
        # Transformation matrix applied to points as column vectors, plus offset
        return (self._data_to_px_m @ trace.pts_lin.T).T + self.origin_px
    
    def get_pts_lin_i_px_coords(self, trace) -> np.ndarray:
        """Returns graph interpolated points in linearised pixel data
        coordinate system. Used for backplotting the interpolated points.
        """
        # Transformation matrix applied to points as column vectors, plus offset
        return (self._data_to_px_m @ trace.pts_lin_i.T).T + self.origin_px

    @pyqtSlot()
    @pyqtSlot(int, bool)
    def calculate_live_outputs(self, trace_no=None, sorting_needed=True):
        """Performs coordinate transformation, interpolation, curve fitting,
        integration and any additional transformations and calculations on the
        data model.
        """
        # Transform points and calculate original values for logarithmic scale
        # if needed. "None" really means none selected, i.e. update all traces.
        traces = self.traces if trace_no is None else [self.traces[trace_no]]
        for tr in traces:
            if tr.pts_px.shape[0] == 0:
                # If no points are set or if these have been deleted, reset
                # model properties to initial values
                tr._init_data()
            else:
                # These calls do the heavy work
                tr._px_to_linear_data_coords(self._px_to_data_m, self.origin_px)
                if sorting_needed:
                    tr._sort_pts()
                # Anyways:
                tr._interpolate_view_data()
                tr._handle_log_scale(self.x_ax, self.y_ax)
        # Emit signals informing of updated trace data
        if trace_no is None:
            self.output_data_changed.emit()
        else:
            self.output_data_changed[int].emit(trace_no)

    @pyqtSlot(bool)
    def set_store_config(self, state):
        """Sets flag to request axes configuration to be saved and restored
        when the application is closed
        """
        self.store_ax_conf = state

    ########## Model-specific implementation part
    @pyqtSlot()
    def _calculate_coordinate_transformation(self) -> None:
        """Calculates data axes origin point offset in pixel coordinates and
        the coordinate transformation matrix including axes scale.

        Axes scale can be logarithmic: For log axes, linearised values
        are used and later transformed back to log scale.

        For backplotting the result, inverse transformation matrix is also
        calculated.
        """
        # This can be called before all axes points have been set
        if not self.axes_setup_is_complete():
            return
        x_ax = self.x_ax
        y_ax = self.y_ax
        ########## Two points in pixel coordinates
        # For each axis, ax_px_near is the pixel coordinates of the axis
        # section point which is closer to the origin than ax_px_far
        x_ax_px_near, x_ax_px_far = x_ax.pts_px
        y_ax_px_near, y_ax_px_far = y_ax.pts_px

        ########## Two points in data coordinates
        # For logarithmic axes, their values are linearised.
        x_ax.pts_data_lin = (
                np.log(x_ax.pts_data) / np.log(x_ax.log_base)
                if x_ax.log_scale
                else x_ax.pts_data
                )
        y_ax.pts_data_lin = (
                np.log(y_ax.pts_data) / np.log(y_ax.log_base)
                if y_ax.log_scale
                else y_ax.pts_data
                )
        x_ax_data_near, x_ax_data_far = x_ax.pts_data_lin
        y_ax_data_near, y_ax_data_far = y_ax.pts_data_lin

        ########## Axes section vectors
        x_ax_vect = x_ax_px_far - x_ax_px_near
        y_ax_vect = y_ax_px_far - y_ax_px_near

        ########## Each axis origin and axes intersection
        # Calculate data axes origin in pixel coordinates for both axes.
        # This is done by extrapolating axes sections down to zero value.
        origin_xax = x_ax_px_near - x_ax_vect * (
                x_ax_data_near / (x_ax_data_far - x_ax_data_near)
                )
        origin_yax = y_ax_px_near - y_ax_vect * (
                y_ax_data_near / (y_ax_data_far - y_ax_data_near)
                )
        # Calculate intersection point of the possibly shifted data coordinate
        # axes.
        ax_intersection = self._lines_intersection(x_ax.pts_px, y_ax.pts_px)

        ######### Origin point of data coordinate system in pixel coordinates
        # This is later used for transforming all points
        self.origin_px = origin_yax + (origin_xax - ax_intersection)
        
        ######### Coordinate transformation matrix
        # Scale factor between data axes values and length in pixel coordinates
        x_scale = np.linalg.norm(x_ax_vect) / (x_ax_data_far - x_ax_data_near) 
        y_scale = np.linalg.norm(y_ax_vect) / (y_ax_data_far - y_ax_data_near)
        # Matrix representation of scale
        scale_m = np.diag((x_scale, y_scale))
        # Inverse scale for opposite transformation direction
        inv_scale_m = np.diag((1/x_scale, 1/y_scale))

        # Axes unit base vectors, unit length in pixel coordinates
        x_ax_uvect = x_ax_vect / np.linalg.norm(x_ax_vect)
        y_ax_uvect = y_ax_vect / np.linalg.norm(y_ax_vect) 
        # Matrix representation of base vectors
        data_unit_base = np.stack((x_ax_uvect, y_ax_uvect), axis=1)
        
        # Inverse of base matrix transforms offset pixel coordinates
        # (i.e. vectors from data axes origin point to graph point)
        # into multiples of the data coordinate base vectors.
        # That is, the wanted data space values, but still scaled
        # in pixel space length units.
        data_inv_base = np.linalg.inv(data_unit_base)

        # Left-multiplication with inverse scale matrix yields the
        # matrix transforming offset pixel coordinates into data units.
        # "@" is the numpy matrix product operator.
        self._px_to_data_m = inv_scale_m @ data_inv_base
        # Also calculating inverse transformation matrix for backplotting
        # interpolated values onto the pixel plane.
        # Scale must be multiplied from the right-hand side.
        self._data_to_px_m = data_unit_base @ scale_m

        # Affine-linear coordinate transformation is now defined, trigger an
        # update of all plot traces in case trace data is already available.
        self.calculate_live_outputs()
        self.redraw_ax_pts_px.emit()

    @staticmethod
    def _lines_intersection(line1_pts, line2_pts):
        """Calculates intersection of two lines defined by two points each.

        line1_pts, line2_pts: For each line, two points in rows of a 2D array
        returns: intersection point, 1D array
        """
        x1, y1 = line1_pts[0] # Line 1, point 1
        x2, y2 = line1_pts[1] # Line 1, point 2
        x3, y3 = line2_pts[0] # Line 2, point 1
        x4, y4 = line2_pts[1] # Line 2, point 2
        # Explicit solution for intersection point of two non-parallel lines
        # each defined by two points with coordinates (xi, yi).
        denominator = (y4-y3)*(x2-x1) - (y2-y1)*(x4-x3)
        num_xs = (x4-x3)*(x2*y1 - x1*y2) - (x2-x1)*(x4*y3 - x3*y4)
        num_ys = (y1-y2)*(x4*y3 - x3*y4) - (y3-y4)*(x2*y1 - x1*y2)
        return np.array((num_xs, num_ys)) / denominator


class Trace(QObject):
    """Data representing a single plot trace, including individual
    configuration.
    """
    ########## Qt signals
    # This signal triggers an update of model output data and the associated
    # GUI view when this trace data or configuration was changed.
    # Since the input data is normally set by the view itself, a redraw of raw
    # input data is not performed when this signal is emitted.
    pts_changed = pyqtSignal()
    # This causes not only an update of model output data but also a re-sorting
    # of the input points and a full re-display of the pixel-space input points
    pts_added_deleted = pyqtSignal()
    # This updates the configuration boxes and buttons
    config_changed = pyqtSignal()
    # GUI error feedback when invalid data was entered
    value_error = pyqtSignal(str)

    def __init__(self, parent, tr_conf, name: str, color: str):
        super().__init__(parent)
        ########## Plot trace configuration
        self.name = name
        # Marks this trace for export
        self.export = tr_conf.export
        # Keyword options for plotting. The instances can have different
        # colors, thus using a copy from conf obj with updated color attribute.
        self.pts_fmt = dict(tr_conf.pts_fmt, **{"color": color})
        self.pts_i_fmt = {"color": color}
        # Number of X-axis interpolation points for GUI display only
        self.n_pts_i_view = tr_conf.n_pts_i_view
        # Kind of interpolation function used for GUI display
        self.interp_type = tr_conf.interp_type
        # Plot data initial state, see below
        self._init_data()
        ########## Associated view objects
        # For raw pts
        self.pts_view_obj = None # Optional: pyplot.Line2D
        # For pts_i curve
        self.pts_i_view_obj = None


    def _init_data(self):
        ########## Plot data layout
        # Data containers are numpy.ndarrays initialised with zero length.
        #
        # pts_px is array of image pixel coordinates with X and Y in rows
        self.pts_px = np.empty((0, 2))
        # pts_lin is array of x,y-tuples of linear or linearised
        # data coordinates. These are calculated by transformation
        # of image pixel vector into data coordinate system.
        # These are also used for the interactive plot.
        self.pts_lin = np.empty((0, 2))
        # pts_lin_i is array of x,y-tuples of the same graph
        # with user-defined x grid. Y-values are interpolated.
        self.pts_lin_i = np.empty((0, 2))
        # pts and pts_i are final result to be output.
        # For linear axes, these are copies of pts_lin and pts_lin_i.
        # For log axes, the values are calculated.
        self.pts = np.empty((0, 2))
        self.pts_i = np.empty((0, 2))
        # This is overwritten with interpolation function
        # representation of trace data when at least four data points
        # are available and Trace._interpolate_view_data is called.
        self.f_interp_lin = lambda arr: np.full(arr.shape[0], NaN)
        # In case of log X axis, this is the above interpolation function
        # with its output values exponentiated.
        # In case of lin X axis, this is a copy.
        self.f_interp = lambda arr: np.full(arr.shape[0], NaN)


    ########## GUI scope and public methods
    def clear_trace(self):
        """Clears this trace
        """
        self._init_data()
        # Trigger view update
        self.pts_added_deleted.emit()

    def add_pt_px(self, xydata):
        self.pts_px = np.concatenate((self.pts_px, (xydata,)), axis=0)
        # Trigger a full update of the model and view of inputs and outputs
        self.pts_added_deleted.emit()

    def update_pt_px(self, xydata, index: int):
        # Assuming this is called from the view only thus raw points need not
        # be redrawn
        self.pts_px[index] = xydata
        # Update of the model outputs plus update of outputs view
        self.pts_changed.emit()

    def delete_pt_px(self, index: int):
        self.pts_px = np.delete(self.pts_px, index, axis=0)
        # Trigger a full update of the model and view of inputs and outputs
        self.pts_added_deleted.emit()

    ########## Model-specific implementation part
    def _px_to_linear_data_coords(self, transform_matrix, origin_px) -> None:
        """Transform image pixel coordinates to linear or
        linearized (in case of log axes) data coordinate system.
        """
        # Offset raw points by pixel coordinate offset of data axes origin.
        # pts_px is array of x, y in rows.
        pts_shifted = self.pts_px - origin_px
        # Transform into data coodinates using pre-calculated matrix.
        # T is a property alias for the numpy.ndarray.transpose method.
        self.pts_lin = (transform_matrix @ pts_shifted.T).T

    def _sort_pts(self) -> None:
        # Sort trace points along the first axis and
        # remove duplicate rows.
        #
        # Because the orientation of the data coordinate system in pixel
        # coordinates is also first known after transformation, the sort
        # indices are also used to sort the input points in the same
        # order as the output. That step is especially useful for plotting
        # when points are added out-of-order.
        self.pts_lin, unique_ids = np.unique(
            self.pts_lin, axis=0, return_index=True)
        self.pts_px = self.pts_px[unique_ids]

    def _interpolate_view_data(self) -> None:
        # Acts on linear coordinates.
        # Needs at least four data points for interpolation.
        pts = self.pts_lin
        if pts.shape[0] < 4:
            return
        # Scipy interpolate generates an interpolation function which is added
        # to this instance attriutes
        self.f_interp_lin = interp1d(*pts.T, kind=self.interp_type)
        # Generate finer grid
        xgrid = np.linspace(pts[0,0], pts[-1,0], num=self.n_pts_i_view)
        yvals = self.f_interp_lin(xgrid)
        self.pts_lin_i = np.concatenate(
                (xgrid.reshape(-1, 1), yvals.reshape(-1, 1)),
                axis=1
                )


    def _handle_log_scale(self, x_ax, y_ax) -> None:
        # For each trace this:
        #
        # ==> Defines an individual interpolation function to be directly used
        # with coordinates in data space no matter if the model internally
        # works with linearised values in case of log scale or not.
        #
        # ==> Transforms non-interpolated linearised trace points to
        # logarithmic scale by exponentiation if needed for each axis.
        #
        # Anyways, makes a copy to keep the original linearised coordinates.
        if x_ax.log_scale:
            if y_ax.log_scale:
                ##### CASE 1: dual logarithmic scale
                # Transform points Y coordinates back to log scale
                self.pts = (x_ax.log_base, y_ax.log_base) ** self.pts_lin
                # Interpolation function applied to logarithmised X values
                # and also post-exponentiated
                self.f_interp = lambda x: x_ax.log_base ** (
                        self.f_interp_lin(np.log(x) / np.log(x_ax.log_base))
                        )
            else:
                ##### CASE 2: X axis only logarithmic scale
                # Transform points X coordinates back to log scale
                self.pts = self.pts_lin.copy()
                self.pts[:,0] = np.power(x_ax.log_base, self.pts_lin[:,0])
                # Interpolation function applied to logarithmised X values
                self.f_interp = lambda x: self.f_interp_lin(
                        # This seems to perform better than np.emath.logn
                        np.log(x) / np.log(x_ax.log_base)
                        )
        else:
            if y_ax.log_scale:
                ##### CASE 3: Y axis only logarithmic scale
                # Transform points Y coordinates back to log scale
                self.pts = self.pts_lin.copy()
                self.pts[:,1] = y_ax.log_base ** self.pts_lin
                # Interpolation function post-exponentiation only
                self.f_interp = lambda x: y_ax.log_base ** self.f_interp_lin(x)
            else:
                ##### CASE 4: no logarithmic scale
                self.pts = self.pts_lin.copy()
                self.f_interp = self.f_interp_lin


class Axis(QObject):
    """Plot axis

    This class includes access methods for GUI/model interactive
    axis configuration or re-configuration.
    """
    # This signal triggers an update of model output data and the associated
    # GUI view when this trace data or configuration was changed.
    # Since the input data is normally set by the view itself, a redraw of raw
    # input data is not performed when this signal is emitted.
    pts_changed = pyqtSignal()
    # This causes not only an update of model output data but also a re-sorting
    # of the input points and a full re-display of the pixel-space input points
    pts_added_deleted = pyqtSignal()
    # This updates the input widget configuration boxes and buttons
    config_changed = pyqtSignal()
    # When invalid data was entered, e.g. zero value for logarithmic scale
    value_error = pyqtSignal(str)

    def __init__(self, parent, ax_conf):
        """Initial axis configuration. These attributes can be overwritten
        with stored configuration after this is instantiated in DataModel
        """
        super().__init__(parent)
        ########## Axis default configuration
        # Keyword options for point markers.
        self.pts_fmt = ax_conf.pts_fmt
        self.log_scale = ax_conf.log_scale
        self.log_base = ax_conf.log_base
        self.atol = ax_conf.atol

        ########## Axis data
        # Two points defining an axis section in pixel coordinate space
        self.pts_px = np.full((2, 2), NaN)
        # Axis section values in data coordinates
        self.pts_data = np.array((NaN, NaN))
        # Axis section values, linearised in case of log scale, copy otherwise
        self.pts_data_lin = np.array((NaN, NaN))

        ########## Associated view object
        self.pts_view_obj = None # Optional: pyplot.Line2D
 

    @pyqtSlot(float)
    def set_ax_start(self, value):
        if self.log_scale and isclose(value, 0.0, atol=self.atol):
            self.value_error.emit("X axis values must not be zero for log axes")
        elif self.pts_data[1] and (
                isclose(value, self.pts_data[1], atol=self.atol)):
            self.value_error.emit("X axis values must be numerically different")
        else:
            self.pts_data[0] = value
        # Updates input widget, model, outputs and also sets self.pts_data_lin
        self.config_changed.emit()

    @pyqtSlot(float)
    def set_ax_end(self, value):
        if self.log_scale and isclose(value, 0.0, atol=self.atol):
            self.value_error.emit("X axis values must not be zero for log axes")
        elif self.pts_data[0] and (
                isclose(value, self.pts_data[0], atol=self.atol)):
            self.value_error.emit("X axis values must be numerically different")
        else:
            self.pts_data[1] = value
        # Updates input widget, model, outputs and also sets self.pts_data_lin
        self.config_changed.emit()

    @pyqtSlot(bool)
    def set_log_scale(self, state):
        log_scale = bool(state)
        # Prevent setting logarithmic scale when axes values contain zero
        if log_scale and isclose(self.pts_data, 0.0, atol=self.atol).any():
            log_scale = False
            self.value_error.emit("X axis values must not be zero for log axes")
        self.log_scale = log_scale
        # Updates input widget, model, outputs and also sets self.pts_data_lin
        self.config_changed.emit()

    def add_pt_px(self, xydata):
        """Add a point defining an axis section
        
        xydata: (x, y)-tuple or np.array shape (2, )

        If both points are unset, set the first one.
        If both points are set, delete second and set first one.
        If only one point remains, i.e. this is the second one, do a
        validity check and set second one only if input data is valid.

        Emits error message signal if invalid, emits view update triggers
        """
        pts_px = self.pts_px
        # Results are each True when point is unset..
        unset_1st, unset_2nd = isnan(pts_px).any(axis=1)
        pt_index = 0
        if not unset_1st and not unset_2nd:
            # Both points set. Invalidate second point, so that it will be set
            # in next call. NaN values make matplotlib not plot this point.
            self.pts_px[1] = (NaN, NaN)
        elif not unset_1st or not unset_2nd:
            # Only one point is still unset. (Both not set was covered above)
            # Set index according to the remaining unset point:
            if unset_2nd:
                pt_index = 1
            # Check if input point is too close to other point, emit
            # error message and return if this is the case
            pts_distance = np.linalg.norm(pts_px[pt_index-1] - xydata)
            if isclose(pts_distance, 0.0, atol=self.atol):
                self.value_error.emit("X axis section must not be zero length")
                return
        # Point validated or no validity check necessary
        pts_px[pt_index] = xydata
        # Model plus view update
        self.pts_added_deleted.emit()


    def update_pt_px(self, xydata, index: int):
        """Update axis point in pixel coordinates 
        
        Leave values untouched and emit error message if
        data would yield a zero-length axis section
        """
        # Check validity if other point is already set
        other_pt = self.pts_px[index-1]
        if None not in other_pt:
            pts_distance = np.linalg.norm(other_pt - xydata)
            if isclose(pts_distance, 0.0, atol=self.atol):
                self.value_error.emit("X axis section must not be zero length")
                return
        # Validity check passed or skipped
        self.pts_px[index] = xydata
        self.pts_changed.emit()

    def delete_pt_px(self, index: int):
        """Delete axis point in pixel coordinates 
        """
        self.pts_px[index] = (NaN, NaN)
        # Model plus view update
        self.pts_added_deleted.emit()


    def _get_state(self):
        """Returns the axis configuration attributes as a dictionary
        Used for persistent storage.
        """
        state = vars(self).copy()
        # The view object belongs to the view component and cannot be restored
        # into a new context.
        del state["pts_view_obj"]
        return state


