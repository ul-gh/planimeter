#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Data Model

License: GPL version 3
"""
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
    # This is re-emitted from the traces and triggers a re-display of the raw
    # (pixel-space) input points. This can be indexed with a trace number.
    redraw_tr_pts_px = pyqtSignal([], [int])
    # Same for axes points
    redraw_ax_pts_px = pyqtSignal()
    # GUI error feedback when invalid data was entered
    value_error = pyqtSignal(str)

    def __init__(self, parent, conf):
        super().__init__(parent)
        ########## Data Model Composition
        self.x_ax = Axis(self, conf.x_ax_conf)
        self.y_ax = Axis(self, conf.y_ax_conf)
        self.origin_px = np.full(2, NaN)
        # Matplotlib format code
        self.origin_fmt = conf.model_conf.origin_fmt
        self.origin_view_obj = None
        # Three traces are default, see main.DefaultConfig
        self.traces = [
            Trace(self, conf.trace_conf, name, color)
            for name, color
            in zip(conf.model_conf.traces_names, conf.model_conf.traces_colors)
            ]
        # Python string format code for display of numbers
        self.num_fmt = conf.app_conf.num_fmt_gui
        # Absolute tolerance for testing if values are close to zero
        self.atol = conf.model_conf.atol
        # Store axes configuration persistently on disk when set
        self.store_ax_conf = conf.model_conf.store_ax_conf
        # X-axis range used for exporting traces data
        self.x_start_export = conf.model_conf.x_start_export
        self.x_end_export = conf.model_conf.x_end_export
        # Number of X-axis interpolation points for data export
        self.n_pts_i_export = conf.model_conf.n_pts_i_export

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
                partial(self.calculate_outputs, i, False))
            # When points are added or deleted, not only update outputs but
            # re-sort the points and update the view of the raw points
            tr.pts_added_deleted.connect(
                partial(self.calculate_outputs, i, True))
            tr.pts_added_deleted.connect(
                partial(self.redraw_tr_pts_px[int].emit, i))
            # The errors are also re-emitted.
            tr.value_error.connect(self.value_error)

        ########## Initialise model outputs if axes are configured
        if self.axes_setup_is_complete():
            self._calculate_coordinate_transformation()

    ########## GUI and Public part
    def export_traces(
            self, *trace_nums, n_interp=None, x_start=None, x_end=None):
        """Interpolate data from the given trace numbers using a common
        interpolation grid with n points, spaced evenly between x_start
        and x_end and return everything with axes and traces data in
        columns of length n. First column: X-axis interpolation values.
        
        Parameters:
        * trace_nums : int
            One or more trace numbers (zero-indexed) used for export data
        n_interp : int, optional
            Number of evenly spaced interpolation points on the common X-axis
        x_start : float, optional
            Start of the data range on the X-axis used for export
        x_end : float, optional
            End of the data range on the X-axis used for export
        
        When number of points, start or end values are not specified,
        instance data set by the GUI or from config file is used.
        """
        if n_interp is None:
            n_interp = self.n_pts_i_export
        if x_start is None:
            x_start = self.x_start_export
        if x_end is None:
            x_end = self.x_end_export

        x_grid_export = np.linspace(x_start, x_end, num=n_interp)
        
        traces = [self.traces[i] for i in trace_nums]
        
        traces_output = np.array([tr.f_interp(x_grid_export) for tr in traces])
        
        pts = self.pts_lin
        if pts.shape[0] < 4:
            return
        # Scipy interpolate generates an interpolation function which is added
        # to this instance attriutes
        self.f_interp = interp1d(*pts.T, kind="cubic")
        # Generate finer grid
        xgrid = np.linspace(pts[0,0], pts[-1,0], num=self.n_pts_i_view)
        yvals = self.f_interp(xgrid)
        self.pts_lin_i = np.concatenate(
            (xgrid.reshape(-1,1), yvals.reshape(-1,1)), axis=1)

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
    def calculate_outputs(self, trace_no=None, sorting_needed=True) -> None:
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
                tr._interpolate_cubic_splines()
                # Transform result data coordinates back to log scale if needed
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
        # 2D arrays, x and y pixel coordinates are in rows
        x_ax_px = self.x_ax.pts_px
        y_ax_px = self.y_ax.pts_px
        # For logarithmic axes, its values are linearised.
        x_ax_data = (np.log(self.x_ax.pts_data) / np.log(self.x_ax.log_base)
                     if self.x_ax.log_scale
                     else np.array(self.x_ax.pts_data))
        y_ax_data = (np.log(self.y_ax.pts_data) / np.log(self.y_ax.log_base)
                     if self.y_ax.log_scale
                     else np.array(self.y_ax.pts_data))

        # Axes section vectors
        x_ax_vect = x_ax_px[1] - x_ax_px[0]
        y_ax_vect = y_ax_px[1] - y_ax_px[0]

        # Calculate data axes origin in pixel coordinates for both axes.
        # This is done by extrapolating axes sections down to zero value.
        origin_xax = x_ax_px[0] - x_ax_vect * (
            x_ax_data[0] / (x_ax_data[1] - x_ax_data[0])
            )
        origin_yax = y_ax_px[0] - y_ax_vect * (
            y_ax_data[0] / (y_ax_data[1] - y_ax_data[0])
            )
        # Calculate intersection point of the possibly shifted data coordinate
        # axes.
        ax_intersection = self._lines_intersection(x_ax_px, y_ax_px)
        # Result data axes position is later used for transforming all points
        self.origin_px = origin_yax + (origin_xax - ax_intersection)
        
        # Scale factor between data axes values and length in pixel coordinates
        x_scale = np.linalg.norm(x_ax_vect) / (x_ax_data[1] - x_ax_data[0]) 
        y_scale = np.linalg.norm(y_ax_vect) / (y_ax_data[1] - y_ax_data[0])
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
        self.calculate_outputs()
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
    configuration. Common configuration is stored in Axes instance.
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
        ########## Data plot trace configuration
        self.name = name
        # Keyword options for plotting. The instances can have different
        # colors, thus using a copy from conf obj with updated color attribute.
        self.pts_fmt = dict(tr_conf.pts_fmt, **{"color": color})
        self.pts_i_fmt = {"color": color}
        # Number of X-axis interpolation points for GUI display only
        self.n_pts_i_view = tr_conf.n_pts_i_view
        # Data containers initial state, see below
        self._init_data()
        ########## Associated view objects
        # For raw pts
        self.pts_view_obj = None # Optional: pyplot.Line2D
        # For pts_i curve
        self.pts_i_view_obj = None
        # Cubic spline interpolation function if available
        self.f_interp = lambda x: raise ValueError(
                f"{self.name}: No Data Available for this trace!")

    ########## GUI and public part
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
    def _init_data(self):
        ########## Plot Data
        # Data containers are np.ndarrays and initialised with zero length
        # pts_px is array of image pixel coordinates, x and y in rows
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

    def _interpolate_cubic_splines(self) -> None:
        # Acts on linear coordinates.
        # Needs at least four data points for interpolation.
        pts = self.pts_lin
        if pts.shape[0] < 4:
            return
        # Scipy interpolate generates an interpolation function which is added
        # to this instance attriutes
        self.f_interp = interp1d(*pts.T, kind="cubic")
        # Generate finer grid
        xgrid = np.linspace(pts[0,0], pts[-1,0], num=self.n_pts_i_view)
        yvals = self.f_interp(xgrid)
        self.pts_lin_i = np.concatenate(
            (xgrid.reshape(-1,1), yvals.reshape(-1,1)), axis=1)

    def _handle_log_scale(self, x_ax, y_ax) -> None:
        # For log axes, the linearised coordinates are transformed
        # back to original logarithmic axes scale. No action for lin axes.
        #
        # Make a copy to keep the original linearised coordinates
        pts = self.pts_lin.copy()
        pts_i = self.pts_lin_i.copy()
        if x_ax.log_scale:
            pts[:,0] = np.power(x_ax.log_base, pts[:,0])
            pts_i[:,0] = np.power(x_ax.log_base, pts_i[:,0])
        if y_ax.log_scale:
            pts[:,1] = np.power(y_ax.log_base, pts[:,1])
            pts_i[:,1] = np.power(y_ax.log_base, pts_i[:,1])
        self.pts = pts
        self.pts_i = pts_i


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
        # Axes section values in data space
        self.pts_data = [0.0, None] # Optional: [Float, Float]

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
        # Updates input widget, model and outputs
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
        # Updates input widget, model and outputs
        self.config_changed.emit()

    @pyqtSlot(bool)
    def set_log_scale(self, state):
        log_scale = bool(state)
        # Prevent setting logarithmic scale when axes values contain zero
        if log_scale and isclose(self.pts_data, 0.0, atol=self.atol).any():
            log_scale = False
            self.value_error.emit("X axis values must not be zero for log axes")
        self.log_scale = log_scale
        # Updates input widget, model and outputs
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
                self.value_error.emit(
                    "X axis section must not be zero length")
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


