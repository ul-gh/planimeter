#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    """
    ########## Qt signals
    # This overloaded signal triggers a GUI update when the model has updated
    # trace data. With trace number as argument, that trace is updated.
    # Without argument, all traces are updated.
    # Since the input data is normally set by the view itself, a redraw of raw
    # input data is not performed when this signal is emitted.
    output_data_changed = pyqtSignal([], [int])
    # This updates the mpl_widget display of the origin
    affine_transformation_defined = pyqtSignal()
    # This is re-emitted from the traces and triggers a re-display of the raw
    # (pixel-space) input points. This can be indexed with a trace number.
    redraw_tr_pts_px = pyqtSignal([], [int])
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
        self.num_format = conf.app_conf.num_format
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
            # Axes input data changes trigger a complete model recalculation
            ax.input_changed.connect(
                self._calculate_coordinate_transformation)
            # Error signals re-emitted here, later connected to message pop-up
            ax.value_error.connect(self.value_error)
        # Trace data changes trigger a model update for that trace only.
        for i, tr in enumerate(self.traces):
            tr.input_changed.connect(partial(self.calculate_outputs, i))
            # Re-emit with trace number index
            tr.redraw_pts_px.connect(
                partial(self.redraw_tr_pts_px[int].emit, i))
            # The errors are also re-emitted.
            tr.value_error.connect(self.value_error)
        # When affine-linear coordinate transformation is defined or changed by
        # configuring the axes, trigger an update of all plot traces in case
        # trace data is already available
        self.affine_transformation_defined.connect(self.calculate_outputs)

        ########## Initialise model outputs if axes are configured
        if self.axes_setup_is_complete():
            self._calculate_coordinate_transformation()

    ########## Internals
    @pyqtSlot()
    def _calculate_coordinate_transformation(self) -> None:
        """Calculates data axes origin point offset in pixel coordinates and
        the coordinate transformation matrix including axes scale.

        Axes scale can be logarithmic, points need to be linear: For log axes,
        linearised values are used and later transformed back to log scale.

        For backplotting the result, inverse transformation matrix is also
        calculated.

        This reads from and manipulates the input Trace instance.
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

        # Calculate pixel offset of data axes origin point from both axes.
        # This is done by extrapolating axes sections down to zero value.
        # Because of limited accuracy of graphic point selection, the finally
        # used origin offset is averaged between both axes.
        origin_xax = x_ax_px[0] - x_ax_vect * (
            x_ax_data[0] / (x_ax_data[1] - x_ax_data[0])
            )
        origin_yax = y_ax_px[0] - y_ax_vect * (
            y_ax_data[0] / (y_ax_data[1] - y_ax_data[0])
            )
        # The result axes offset is later used for transformating all points
        self.origin_px = (origin_xax + origin_yax) / 2
        
        print(  f"Debug: calculated origins:\n"
                f"       From X-axis: {origin_xax};\n"
                f"       from Y-axis: {origin_yax};\n"
                f"       origin mean: {self.origin_px}"
                )
        
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

        # Emit signal to inform about processed axes data
        self.affine_transformation_defined.emit()

    def _px_to_linear_data_coords(self, trace) -> None:
        """Transform image pixel coordinates to linear or
        linearized (in case of log axes) data coordinates.
        This reads from and manipulates the input Trace instance.
        """
        # Offset raw points by pixel coordinate offset of data axes origin.
        # pts_px is list of x,y-tuples
        pts_shifted = trace.pts_px - self.origin_px
        # Transform into data coodinates using pre-calculated matrix.
        # T is a property alias for the numpy.ndarray.transpose method.
        pts_lin = (self._px_to_data_m @ pts_shifted.T).T
        # Sort with increasing X values and remove duplicate rows. This is
        # necessary for interpolation and likely desirable for export data
        trace.pts_lin = np.unique(pts_lin, axis=0)

    def _interpolate_cubic_splines(self, trace) -> None:
        """Acts on linear coordinates.
        Needs at least four data points for interpolation."""
        pts = trace.pts_lin
        if pts.shape[0] < 4:
            return
        # Scipy interpolate
        f_interp = interp1d(*pts.T, kind="cubic")
        # Generate finer grid
        xgrid = np.linspace(pts[0,0], pts[-1,0], num=trace.n_pts_interpolation)
        yvals = f_interp(xgrid)
        trace.pts_lin_i = np.stack((xgrid, yvals), axis=1)

    def _handle_log_scale(self, trace) -> None:
        """For log axes, the linearised coordinates are transformed
        back to original logarithmic axes scale. No action for lin axes.
        """
        # Make a copy to keep the original linearised coordinates
        pts = trace.pts_lin.copy()
        pts_i = trace.pts_lin_i.copy()
        if self.x_ax.log_scale:
            pts[:,0] = np.power(self.x_ax.log_base, pts[:,0])
            pts_i[:,0] = np.power(self.x_ax.log_base, pts_i[:,0])
        if self.y_ax.log_scale:
            pts[:,1] = np.power(self.y_ax.log_base, pts[:,1])
            pts_i[:,1] = np.power(self.y_ax.log_base, pts_i[:,1])
        trace.pts = pts
        trace.pts_i = pts_i


    ########## Public methods
    def axes_setup_is_complete(self) -> bool:
        """Returns True if axes configuration is all complete and valid
        """
        x_ax = self.x_ax
        y_ax = self.y_ax
        if None in x_ax.pts_data or None in y_ax.pts_data:
            return False
        if isnan((x_ax.pts_px, y_ax.pts_px)).any():
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
    @pyqtSlot(int)
    def calculate_outputs(self, trace_no=None) -> None:
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
                tr.init_data()
            else:
                # These calls do the heavy work
                self._px_to_linear_data_coords(tr)
                self._interpolate_cubic_splines(tr)
                # Transform result data coordinates back to log scale if needed
                self._handle_log_scale(tr)
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


class ViewRegisterModel(DataModel):
    """Subclass ViewRegisterModel of DataModel
    
    This adds to the data model a mapping of individual objects of the
    graphical view presentation to the associated data model objects.
    
    This is used to retrieve and manipulate the data model items
    that belong to the graphic objects.
    
    For the mapping, a dictionary is used for which most python objects
    as are valid indexing keys, e.g. pyplot.Line2D objects.

    Also implemented here are the respective access functions for
    registering the view objects and for updating the data model using
    view object data.
    """
    __doc__ += f"\n\nFrom inherited class:\n\n{DataModel.__doc__}"

    def __init__(self, parent, conf):
        super().__init__(parent, conf)
        # This dict does the mapping of the view objects to the model
        self.view_model_map = {}

    def using_view_add_x_ax_pt_px(self, view_obj):
        """Register a view object in the view-to-model map and in the
        model component, set model input data from view object
        and trigger a model update.
        The view objects must have a "get_xydata" attribute returning the pixel
        coordinates as a 2D np.array.
        """
        self.view_model_map[view_obj] = self.x_ax
        xydata = view_obj.get_xydata().tolist()[0]
        # This also registers the view object in the model component and 
        # emits the input_changed signal
        self.x_ax.add_pt_px(xydata, view_obj)

    def using_view_add_y_ax_pt_px(self, view_obj):
        """Register a view object in the view-to-model map and in the
        model component, set model input data from view object
        and trigger a model update.
        The view objects must have a "get_xydata" attribute returning the pixel
        coordinates as a 2D np.array.
        """
        self.view_model_map[view_obj] = self.y_ax
        xydata = view_obj.get_xydata().tolist()[0]
        # This also registers the view object in the model component and 
        # emits the input_changed signal
        self.y_ax.add_pt_px(xydata, view_obj)

    def using_view_add_trace_pt_px(self, view_obj, trace_no):
        """Register a view object in the view-to-model map and in the
        model component, set model input data from view object
        and trigger a model update. Traces require an index number.
        The view objects must have a "get_xydata" attribute returning the pixel
        coordinates as a 2D np.array.
        """
        trace = self.traces[trace_no]
        self.view_model_map[view_obj] = trace
        xydata = view_obj.get_xydata().tolist()[0]
        # This also registers the view object in the model component and 
        # emits the input_changed signal
        trace.add_pt_px(xydata, view_obj)

    def using_view_update_pt_px(self, view_obj):
        """Update the data model when an already registered view_obj
        has changed data.

        This triggers a model re-calculation.
        
        The view objects must have a "get_xydata" attribute returning the pixel
        coordinates as a 2D np.array.

        This is called when a point was moved in the view component
        """
        model_comp = self.view_model_map[view_obj]
        index = model_comp.view_objs.index(view_obj)
        xydata = view_obj.get_xydata().tolist()[0]
        # This also emits the input_changed signal
        model_comp.update_pt_px(index, xydata)

    def using_view_delete_pt_px(self, view_obj):
        """Un-register a view object, delete data from model and
        trigger a model re-calculation.
        """
        model_comp = self.view_model_map.pop(view_obj)
        index = model_comp.view_objs.index(view_obj)
        # This also deletes the view object from the model component
        # and emits the input_changed signal
        model_comp.delete_pt_px(index)


class Trace(QObject):
    """Data representing a single plot trace, including individual
    configuration. Common configuration is stored in Axes instance.
    """
    ########## Qt signals
    # This signal triggers an update of model output data and the associated
    # GUI view when this trace data or configuration was changed.
    # Since the input data is normally set by the view itself, a redraw of raw
    # input data is not performed when this signal is emitted.
    input_changed = pyqtSignal()
    # This explicitly triggers a re-display of the pixel-space input points
    redraw_pts_px = pyqtSignal()
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
        # Number of X-axis interpolation points for export and display
        self.n_pts_interpolation = tr_conf.n_pts_interpolation
        # Data containers initial state, see below
        self.init_data()

        ########## Associated view objects
        # For raw pts
        self.pts_view_obj = None # Optional: pyplot.Line2D
        # For pts_i curve
        self.pts_i_view_obj = None


    def init_data(self):
        # This is also called to clear existing traces
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


    def add_pt_px(self, xydata):
        self.pts_px = np.vstack((self.pts_px, xydata))
        # Trigger an update of the view
        self.redraw_pts_px.emit()
        # Update of the model results plus view update of results
        self.input_changed.emit()

    def update_pt_px(self, index: int, xydata):
        # Assuming this is called from the view only thus raw points need not
        # be redrawn
        self.pts_px[index] = xydata
        # Update of the model results plus view update of results
        self.input_changed.emit()

    def delete_pt_px(self, index: int):
        self.pts_px = np.delete(self.pts_px, index, axis=0)
        # Trigger an update of the view
        self.redraw_pts_px.emit()
        # Update of the model results plus view update of results
        self.input_changed.emit()


class Axis(QObject):
    """Plot axis

    This class includes access methods for GUI/model interactive
    axis configuration or re-configuration.
    """
    # This signal triggers an update of model output data and the associated
    # GUI view when this trace data or configuration was changed.
    # Since the input data is normally set by the view itself, a redraw of raw
    # input data is not performed when this signal is emitted.
    input_changed = pyqtSignal()
    # This explicitly triggers a re-display of the pixel-space input points
    redraw_pts_px = pyqtSignal()
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
        # Updates input widget
        self.config_changed.emit()
        # Updates outputs
        self.input_changed.emit()

    @pyqtSlot(float)
    def set_ax_end(self, value):
        if self.log_scale and isclose(value, 0.0, atol=self.atol):
            self.value_error.emit("X axis values must not be zero for log axes")
        elif self.pts_data[0] and (
                isclose(value, self.pts_data[0], atol=self.atol)):
            self.value_error.emit("X axis values must be numerically different")
        else:
            self.pts_data[1] = value
        self.config_changed.emit()
        self.input_changed.emit()

    @pyqtSlot(bool)
    def set_log_scale(self, state):
        log_scale = bool(state)
        # Prevent setting logarithmic scale when axes values contain zero
        if log_scale and isclose(self.pts_data, 0.0, atol=self.atol).any():
            log_scale = False
            self.value_error.emit("X axis values must not be zero for log axes")
        self.log_scale = log_scale
        # Updates input widget
        self.config_changed.emit()
        # Updates outputs
        self.input_changed.emit()

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
        both_unset = isnan(pts_px[0]).any() and isnan(pts_px[1]).any()
        none_unset = not isnan(pts_px).any()
        second_unset = isnan(pts_px[1]).any()
        pt_index = 0
        if none_unset:
            # Invalidate second point, so that it will be set in next call
            # Also, this makes matplotlib not plot this point
            self.pts_px[1] = (NaN, NaN)
        elif not both_unset:
            # Only one point is still unset.
            if second_unset:
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
        # If no view object was supplied, trigger a view redraw of raw points,
        # otherwise register the supplied object with this point
        self.redraw_pts_px.emit()
        self.input_changed.emit()


    def update_pt_px(self, index: int, xydata):
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
        self.redraw_pts_px.emit()
        self.input_changed.emit()

    def delete_pt_px(self, index: int):
        """Delete axis point in pixel coordinates 
        """
        self.pts_px = np.delete(self.pts_px, index, axis=0)
        # Trigger an update of the view
        self.redraw_pts_px.emit()
        # Update of the model results plus view update of results
        self.input_changed.emit()

    def get_state(self):
        """Returns the axis configuration attributes as a dictionary
        Used for persistent storage.
        """
        state = vars(self).copy()
        # The view object belongs to the view component and cannot be restored
        # into a new context.
        del state["pts_view_obj"]
        return state


