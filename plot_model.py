#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Workbench Data Model

License: GPL version 3
"""
import logging
logger = logging.getLogger(__name__)

from functools import partial

from typing import Iterable

import numpy as np
from numpy import NaN, isnan, isclose
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import scipy.misc as misc

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

import matplotlib.pyplot as plt
import upylib.u_plot_format as u_format

from upylib.pyqt_debug import logExceptionSlot

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
    # Emitted from the traces objects and triggers a re-display of the raw
    # (pixel-space) input points. Can address a single trace by number index.
    # When a trace is added or deleted, the signal is emitted without index.
    tr_input_data_changed = pyqtSignal([], [int])
    # Emitted from the axes objects, updates all view outputs
    ax_input_data_changed = pyqtSignal()
    # Triggers updates of coordinate settings box
    coordinate_system_changed = pyqtSignal()
    # Mainly used for configuring export options
    export_settings_changed = pyqtSignal()
    # GUI error feedback when invalid data was entered
    value_error = pyqtSignal(str)
    # GUI feedback when export range settings are outside of points range
    export_range_warning = pyqtSignal(str)

    def __init__(self, parent, conf):
        super().__init__(parent)
        ########## Plot model composition

        ##### Two axes
        self.x_ax = Axis(self, conf.x_ax_conf, "X Axis")
        self.y_ax = Axis(self, conf.y_ax_conf, "Y Axis")
        ##### Origin
        self.origin_px = np.full(2, NaN)
        # Matplotlib format code
        self.origin_fmt = conf.model_conf.origin_fmt
        self.origin_view_obj = None # Optional: matplotlib lines2D instance
        ##### Coordinate transformation matrices
        self.data_to_px_mat = None # Optional: np.ndarray
        self.px_to_data_mat = None # Optional: np.ndarray
        
        ##### Arbitrary number of traces
        # Three traces are default, see main.DefaultConfig
        self.traces = [
            Trace(self, conf.trace_conf, trace_no, name, color)
            for trace_no, name, color
            in zip( # trace_no: enumeration from zero in order of traces_names
                    range(len(conf.model_conf.traces_names)),
                    conf.model_conf.traces_names,
                    conf.model_conf.traces_colors,
                    )
            ]

        ##### Trace export options
        # X-axis range used for interpolating traces export data.
        # In order to avoid strange results due to uncalled-for extrapolation
        # when no extrapolation function is defined, this is later set to the 
        # intersection of definition ranges of all traces marked for export.
        self.autorange_export = True
        self.x_start_export = NaN
        self.x_end_export = NaN
        # Fixed step size for export range is optional
        self.fixed_n_pts_export = conf.model_conf.fixed_n_pts_export
        self.x_step_export = conf.model_conf.x_step_export
        # Alternative definition of export range by total number of points
        self.n_pts_export = conf.model_conf.n_pts_export
        # Number of X-axis points per decade in case of log X grid
        self.n_pts_dec_export = conf.model_conf.n_pts_dec_export
        # Maximum number of export points for user input verification
        self.n_pts_export_max = conf.model_conf.n_pts_export_max
        # Export grid can be logarithmic independent from original axes scale
        self.x_log_scale_export = False

        ##### Generated X-axis grid in data coordinates used for export
        self.x_grid_export = None # Optional: np.ndarray

        ##### Common settings
        # Python string format code for display of numbers
        self.num_fmt = conf.app_conf.num_fmt_gui
        # Absolute tolerance for testing if values are close to zero
        self.atol = conf.model_conf.atol
        # Store axes configuration persistently on disk when set
        self._store_ax_conf = conf.model_conf.store_ax_conf

        ########## Restore data model configuration and state from stored data
        if conf.x_ax_state is not None:
            vars(self.x_ax).update(conf.x_ax_state)
        if conf.y_ax_state is not None:
            vars(self.y_ax).update(conf.y_ax_state)

        ########## Initialise model outputs if axes are configured
        if self.axes_setup_is_complete():
            self._calc_coordinate_transformation()
            
        ########## Connect axes input changes to own calculation
        self.ax_input_data_changed.connect(self._calc_coordinate_transformation)
        
        ########## Connect trace input changes to own outputs calculation
        self.tr_input_data_changed.connect(self._process_tr_input_data)
        self.tr_input_data_changed[int].connect(self._process_tr_input_data)


    ########## Export Related Methods
    def wip_export(self, tr):
        # FIXME: Temporary solution
        grid = np.linspace(tr.pts[0,0], tr.pts[-1,0], 100)
        y = tr.f_interp(grid)*1e-12
        y_int = integrate.cumtrapz(y, grid, initial=0.0)
        x_times_y = grid * y
        x_times_y_int = integrate.cumtrapz(x_times_y, grid, initial=0.0)
        tr.pts_export = np.stack((grid, y), axis=1)
        tr.pts_export_int = np.stack((grid, y_int), axis=1)
        tr.pts_export_xy_int = np.stack((grid, x_times_y_int), axis=1)
        
    def wip_plot_cap_charge_e_stored(self,tr):
        self.fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        self.fig.set_size_inches(18, 6.0)
        # Plot ax1
        u_format.plot_lin_lin_engineering(
                ax1,
                *tr.pts_export.T,
                title=r"Differential Capacitance C(u)",
                xlabel=r"U /Volts",
                ylabel=r"C(u) /Farads",
                #xlim=(1, 450),
                #ylim=(0, 0.6),
                )
        # Plot ax2
        u_format.plot_lin_lin_engineering(
                ax2,
                *tr.pts_export_int.T,
                title=r"Charge Q(u)",
                xlabel=r"U /Volts",
                ylabel=r"Q(u) /Coulombs",
                #xlim=(1, 450),
                #ylim=(0, 0.6),
                )
        ax2_text = r"$Q(u) = \int C(u) \: du$"
        ax2.text(0.05, 0.9, ax2_text, fontsize=15, transform=ax2.transAxes)
        # Plot ax3
        u_format.plot_lin_lin_engineering(
                ax3,
                *tr.pts_export_xy_int.T,
                title=r"C Stored Energy E(u)",
                xlabel=r"U /Volts",
                ylabel=r"E(u) /Joules",
                #xlim=(1, 450),
                #ylim=(0, 0.6),
                )
        ax3_text = r"$E(u) = \int u \cdot C(u) \: du$"
        ax3.text(0.05, 0.9, ax3_text, fontsize=15, transform=ax3.transAxes)
        self.fig.tight_layout()

    @logExceptionSlot()
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
        For generating the export data on any grid, an the generated
        interpolation function is used in any case.
        By default, the cubic spline polynomial interpolation is used.
        This function needs at least four data points from each trace.
        For linear interpolation, at least two data points are needed.

        If no interpolation data is available for a trace, a column of
        NaN values is returned for that trace.
        """
        ######### Generate export X axis grid
        if self.x_log_scale_export:
            # Grid is specified by number of points per decade; thus using log10
            x_grid_export = np.geomspace(
                    self.x_start_export, self.x_end_export, self.n_pts_export)
        else:
            if self.fixed_n_pts_export:
                x_grid_export = np.linspace(
                    self.x_start_export, self.x_end_export, self.n_pts_export)
            else:
                x_grid_export = np.arange(
                    self.x_start_export, self.x_end_export, self.x_step_export)
        # Anyways:
        self.x_grid_export = x_grid_export

        ########## Calculate export traces Y data using data output function
        n_interp = x_grid_export.shape[0]
        output_funcs = [
                tr.output_funcs[tr.post_processing]
                for tr in self.traces
                if tr.f_interp is not None and tr.export
                ]
        output_arr = np.empty((1+len(output_funcs), n_interp))
        output_arr[0] = x_grid_export
        for row, output_func in enumerate(output_funcs, start=1):
            output_arr[row] = output_func(x_grid_export)
        self.output_arr = output_arr


    @logExceptionSlot(float)
    def set_x_start_export(self, x_start: float):
        if isclose(self.x_end_export - x_start, 0.0, atol=self.atol):
            self.value_error.emit("X axis section must not be zero length")
            return
        self.autorange_export = False
        self.x_start_export = x_start
        self._check_or_update_export_range()

    @logExceptionSlot(float)
    def set_x_end_export(self, x_end: float):
        if isclose(x_end - self.x_start_export, 0.0, atol=self.atol):
            self.value_error.emit("X axis section must not be zero length")
            return
        self.autorange_export = False
        self.x_end_export = x_end
        self._check_or_update_export_range()

    @logExceptionSlot(bool)
    def set_autorange_export(self, state=True):
        self.autorange_export = state
        self._check_or_update_export_range()

    @logExceptionSlot(int)
    def set_n_pts_export(self, n_pts: int = None):
        # Step size is now artibrary
        self.fixed_n_pts_export = True
        if n_pts is None or n_pts > self.n_pts_export_max:
            n_pts = self.n_pts_export_max
            self.export_range_warning.emit(
                "Value set to maximum number of points limit!")
        if n_pts < 1:
            n_pts = 1
            self.export_range_warning.emit(
                "Value changed to one point minimum")
        # Calculation only possible when export range is defined
        if not isnan((self.x_start_export, self.x_end_export)).any():
            if self.x_log_scale_export:
                self._update_n_pts_dec_export()
            else:
                self._update_x_step_export()
        self.n_pts_export = n_pts
        self.export_settings_changed.emit()

    @logExceptionSlot(float)
    def set_x_step_export(self, x_step: float):
        self.fixed_n_pts_export = False
        # If called, this assumes linear scale export is desired:
        self.x_log_scale_export = False
        # Validation only possible when export range is defined
        if isnan((self.x_start_export, self.x_end_export)).any():
            # Without validation
            self.x_step_export = x_step
            return
        # Validation: If the selected X step results in too many points,
        # limit to max. value and emit a warning
        x_step_min = (
                self.x_end_export - self.x_start_export
                ) / self.n_pts_export_max
        if x_step < x_step_min:
            self.x_step_export = x_step_min
            self.export_range_warning.emit(
                    "Step size changed to satisfy maximum points limit!")
        else:
            self.x_step_export = x_step
        # Update total number of points
        self.n_pts_export = 1 + int(
                (self.x_end_export - self.x_start_export) / x_step
                )
        self.export_settings_changed.emit()

    @logExceptionSlot(float)
    def set_n_pts_dec_export(self, n_pts_dec: float):
        # If called, this assumes log scale export is desired:
        self.x_log_scale_export = True
        # Validation only possible when export range is defined
        if isnan((self.x_start_export, self.x_end_export)).any():
            # Without validation
            self.n_pts_dec_export = n_pts_dec
            return
        # Validation: If the selected X step results in too many points,
        # limit to max. value and emit a warning
        x_start_lin = np.log10(self.x_start_export)
        x_end_lin = np.log10(self.x_end_export)
        n_dec = x_end_lin - x_start_lin
        N_tot = int(n_dec * n_pts_dec)
        if N_tot > self.n_pts_export_max:
            self.n_pts_dec_export = self.n_pts_export_max / n_dec
            N_tot = self.n_pts_export_max
            self.export_range_warning.emit(
                "Value changed to satisfy maximum number of points limit!")
        else:
            self.n_pts_dec_export = n_pts_dec
        # Update total number of points
        self.n_pts_export = N_tot
        self.export_settings_changed.emit()


    ########## Configuration and Validation Methods
    def axes_setup_is_complete(self) -> bool:
        """Returns True if both axes configuration is complete and valid
        """
        return self.x_ax.is_complete() and self.y_ax.is_complete()

    def coordinate_transformation_defined(self) -> bool:
        """Returns True if coordinate system transformation matrices for
        both pixel-to-data direction and reverse direction are defined
        """
        return (self.data_to_px_mat is not None
                and self.px_to_data_mat is not None)

    def persistent_storage(self):
        """Returns true if persistent storage is selected
        """
        return self._store_ax_conf

    @logExceptionSlot(bool)
    def set_persistent_storage(self, state):
        """Sets flag to request axes configuration to be saved and restored
        when the application is closed
        """
        self._store_ax_conf = state


    ########## Intermediate Data Related Methods
    def data_to_px_coords(self, lin_data_pts: np.ndarray) -> np.ndarray:
        """Returns transformation from linear(!) data coordinates
        into image pixel coordinates.
        
        ! Data points must already be in linear or linearised space !

        Output points are offset by pixel coordinates of data axes
        origin, i.e. the output is absolute.
        
        Input and output are 2D arrays of X and Y in rows.
        """
        if self.coordinate_transformation_defined():
            # Transformation matrix @ points as column vectors + offset
            return (self.data_to_px_mat @ lin_data_pts.T).T + self.origin_px
        else:
            return np.empty((0, 2))

    def px_to_data_coords(self, px_pts: np.ndarray) -> np.ndarray:
        """Returns transformation of image pixel coordinates into
        linear or linearized (in case of log axes) data coordinates.

        Input points are assumed to be absolute, i.e. the pixel offset
        of data axes origin is subtracted first.
        
        Input and output are 2D arrays of X and Y in rows.
        """
        if self.coordinate_transformation_defined():
            pts_shifted = px_pts - self.origin_px
            # Transformation matrix @ points as column vectors + offset
            return (self.px_to_data_mat @ pts_shifted.T).T
        else:
            return np.empty((0, 2))

    def pts_lin_i_px_coords(self, trace) -> np.ndarray:
        """Returns graph interpolated points in (linear) pixel
        coordinate system. Used for backplotting the interpolated points.
        """
        return self.data_to_px_coords(trace.pts_lin_i)

    def px_from_data_bounds(self, x_min_max, y_min_max):
        if not self.axes_setup_is_complete():
            self.value_error.emit("You must configure both axes first!")
            return None
        if self.x_ax._log_scale:
            if True in [i < self.atol for i in x_min_max]:
                self.value_error.emit("Value must be greater 0 for log axes")
                return None
            xmin, xmax = np.log((x_min_max)) / np.log(self.x_ax.log_base)
        else:
            xmin, xmax = x_min_max
        if self.y_ax._log_scale:
            if True in [i < self.atol for i in y_min_max]:
                self.value_error.emit("Value must be greater 0 for log axes")
                return None
            ymin, ymax = np.log((y_min_max)) / np.log(self.y_ax.log_base)
        else:
            ymin, ymax = y_min_max
        bbox_data = np.array(((xmin, ymin),
                              (xmin, ymax),
                              (xmax, ymin),
                              (xmax, ymax)))
        bbox_px = self.origin_px + (self.data_to_px_mat @ bbox_data.T).T
        xmin_px, ymin_px = np.min(bbox_px, axis=0)
        xmax_px, ymax_px = np.max(bbox_px, axis=0)
        return (xmin_px, xmax_px), (ymin_px, ymax_px)


    ########## Coordinate Transformation Related Private Methods
    def _calc_coordinate_transformation(self) -> None:
        """Calculates data axes origin point offset in pixel coordinates and
        the coordinate transformation matrix including axes scale.

        Axes scale can be logarithmic: For log axes, linearised values
        are used and later transformed back to log scale.

        For backplotting the result, inverse transformation matrix is also
        calculated.
        """
        #logger.debug(f"DataModel._calc_coordinate_transformation called")
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
                if x_ax._log_scale
                else x_ax.pts_data
                )
        y_ax.pts_data_lin = (
                np.log(y_ax.pts_data) / np.log(y_ax.log_base)
                if y_ax._log_scale
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
        # axes. Returns None if a division by a close-to-zero value occurs,
        # which happens if axes sections are almost parallel or too short.
        ax_intersection = self._lines_intersection(x_ax.pts_px, y_ax.pts_px)
        if ax_intersection is None:
            return

        ######### Origin point of data coordinate system in pixel coordinates
        # This is later used for transforming all points
        self.origin_px = origin_yax + (origin_xax - ax_intersection)
        
        ######### Coordinate transformation matrix
        # Scale factor between data axes values and length in pixel coordinates
        x_scale = np.linalg.norm(x_ax_vect) / (x_ax_data_far - x_ax_data_near) 
        y_scale = np.linalg.norm(y_ax_vect) / (y_ax_data_far - y_ax_data_near)
        # Matrix representation of scale
        scale_mat = np.diag((x_scale, y_scale))
        # Inverse scale for opposite transformation direction
        inv_scale_mat = np.diag((1/x_scale, 1/y_scale))

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
        self.px_to_data_mat = inv_scale_mat @ data_inv_base
        # Also calculating inverse transformation matrix for backplotting
        # interpolated values onto the pixel plane.
        # Scale must be multiplied from the right-hand side.
        self.data_to_px_mat = data_unit_base @ scale_mat

        # Affine-linear coordinate transformation is now defined, trigger an
        # update of all plot traces in case trace data is already available.
        self._process_tr_input_data() # Emits the "output_data_changed" signal.
        self.coordinate_system_changed.emit()

    @logExceptionSlot()
    @logExceptionSlot(int)
    def _process_tr_input_data(self, trace_no=None, sorting_needed=False):
        """Performs coordinate transformation, sorting, interpolation 
        and linearisation of log axes on the data model.
        """
        if not self.coordinate_transformation_defined():
            text = "Tried processing trace points but transform not defined!"
            logger.warn(text)
            return
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
                tr.pts_lin = self.px_to_data_coords(tr.pts_px)
                # Anyways:
                if sorting_needed:
                    tr._sort_pts()
                tr._interpolate_view_data()
                tr._handle_log_scale(self.x_ax, self.y_ax)
        # What the name says..
        self._check_or_update_export_range()
        # Emit signals informing of updated trace data
        if trace_no is None:
            self.output_data_changed.emit()
        else:
            self.output_data_changed[int].emit(trace_no)


    ##### Exporting Related Private Methods
    # Called from self._process_tr_input_data.
    # Set export range limits on the common X axis such that all traces can
    # be exported by using interpolation, i.e. no extrapolation takes place.
    # An exception is made for traces with less than two points selected,
    # these are not taken into account.
    def _check_or_update_export_range(self):
        # Calculate interpolation limits
        tr_start_lin = [tr.pts_lin[0,0] for tr in self.traces
                        if tr.export and tr.pts_lin.shape[0] > 1]
        tr_end_lin = [tr.pts_lin[-1,0] for tr in self.traces
                      if tr.export and tr.pts_lin.shape[0] > 1]
        if not tr_start_lin or not tr_end_lin:
            #logger.debug(f"No export range. No traces marked for export?")
            return
        x_start_lin_limit = max(tr_start_lin)
        x_end_lin_limit = min(tr_end_lin)
        # Anti-log treatment for X axis. This is not to be confused with log
        # scale export setting - these are independent.
        if self.x_ax._log_scale:
            x_start_export_limit = self.x_ax.log_base ** x_start_lin_limit
            x_end_export_limit = self.x_ax.log_base ** x_end_lin_limit
        else:
            x_start_export_limit = x_start_lin_limit
            x_end_export_limit = x_end_lin_limit

        if self.autorange_export:
            # Set class attribute
            self.x_start_export = x_start_export_limit
            self.x_end_export = x_end_export_limit
            # Update dependent attributes
            if self.fixed_n_pts_export:
                if self.x_log_scale_export:
                    self._update_n_pts_dec_export()
                else:
                    self._update_x_step_export()
            else:
                self._update_n_pts_export()
        else:
            # Range check only
            if (    self.x_start_export < x_start_export_limit
                    or self.x_end_export > x_end_export_limit
                    ):
                logger.warn("Export range is extrapolated!")
                self.export_range_warning.emit()

    # Update total number of output points when export range is changed
    def _update_n_pts_export(self):
        if self.x_log_scale_export:
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

    def _update_n_pts_dec_export(self):
        x_start_lin = np.log10(self.x_start_export)
        x_end_lin = np.log10(self.x_end_export)
        n_dec = x_end_lin - x_start_lin
        self.n_pts_dec_export = self.n_pts_export / n_dec

    def _update_x_step_export(self):
        self.x_step_export = (self.x_end_export - self.x_start_export
                       ) / self.n_pts_export

    #Calculates intersection of two lines defined by two points each.
    # line1_pts, line2_pts: Each two points in rows of a 2D array
    # returns: Intersection point, 1D array or None value for invalid data
    def _lines_intersection(self, line1_pts, line2_pts):
        x1, y1 = line1_pts[0] # Line 1, point 1
        x2, y2 = line1_pts[1] # Line 1, point 2
        x3, y3 = line2_pts[0] # Line 2, point 1
        x4, y4 = line2_pts[1] # Line 2, point 2
        # Explicit solution for intersection point of two non-parallel lines
        # each defined by two points with coordinates (xi, yi).
        denominator = (y4-y3)*(x2-x1) - (y2-y1)*(x4-x3)
        num_xs = (x4-x3)*(x2*y1 - x1*y2) - (x2-x1)*(x4*y3 - x3*y4)
        num_ys = (y1-y2)*(x4*y3 - x3*y4) - (y3-y4)*(x2*y1 - x1*y2)
        if isclose(denominator, 0.0, self.atol):
            return None
        return np.array((num_xs, num_ys)) / denominator


class Trace(QObject):
    """Data representing a single plot trace, including individual
    configuration.
    """
    def __init__(self, model, tr_conf, trace_no: int, name: str, color: str):
        super().__init__(model)
        ########## Connection to the containing data model
        self.model = model
        ########## Plot trace configuration
        self.name = name
        self.trace_no = trace_no
        # Marks this trace for export
        self.export = tr_conf.export
        # Keyword options for plotting. The instances can have different
        # colors, thus using a copy from conf obj with updated color attribute.
        self.pts_fmt = dict(tr_conf.pts_fmt, **{"color": color})
        self.pts_i_fmt = {"color": color}
        # Number of X-axis interpolation points for GUI display only
        self.n_pts_i_view = tr_conf.n_pts_i_view
        # Default interpolation type can be "linear" or "cubic"
        self.interp_type = tr_conf.interp_type
        # Interpolation or output function currently selected
        self.post_processing = tr_conf.post_processing
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
        self.f_interp_lin = None
        # In case of log X axis, this is the above interpolation function
        # with its output values exponentiated.
        # In case of lin X axis, this is a copy.
        self.f_interp = None
        # Post processing functions for output data
        self.output_funcs = {
                "plain": self.f_interp,
                "integral": partial(integrate.quad, self.f_interp),
                "derivative": partial(misc.derivative, self.f_interp),
                }


    def clear_trace(self):
        """Clears this trace and re-initialises with zero presets
        """
        self._init_data()
        # Trigger a full update of the model and view of inputs and outputs
        self.model._process_tr_input_data(self.trace_no)
        self.model.tr_input_data_changed[int].emit(self.trace_no)

    def add_pt_px(self, xy_px: Iterable[float]) -> int:
        """Add a point in pixel coordinates to the trace.
        
        This triggers a sorting of the points along the (transformed)
        data coordinate axis as well as a model and view update.
        """
        logger.debug(f"Call Trace.add_pt_px with data: {xy_px}")
        if self.pts_px.shape[0] > 0 and np.isnan(self.pts_px).any():
            # There are NaN values in at least one data point. Set these first.
            pt_index = np.isnan(self.pts_px).any(axis=1).nonzero()[0][0]
            self.pts_px[pt_index] = xy_px
            logger.warning(f"NaN value occurred in model!\n"
                           f"Setting point at index {pt_index}")
        else:
            # Append new point with given coordinates
            self.pts_px = np.concatenate((self.pts_px, (xy_px,)), axis=0)
        # Trigger a full update of the model and view of inputs and outputs
        if not self.model.axes_setup_is_complete():
            logger.warning("No live updates: Coordinate system not defined.")
            return self.pts_px.shape[0] - 1
        # Sorting also puts NaN values at the end
        self.model._process_tr_input_data(self.trace_no, sorting_needed=True)
        pt_index = self.pts_px.shape[0] - 1
        logger.debug(f"Setting point at index {pt_index}")
        self.model.tr_input_data_changed[int].emit(self.trace_no)
        return pt_index

    def update_pt_px(self, xy_px: Iterable[float], index: int):
        """Moves a point to the specified position in pixel coordinates.
        
        Moves only if this does not result in duplicate points,
        otherwise, no action takes place.
        """
        xy_data = self.model.px_to_data_coords(xy_px)
        if self.pts_lin.shape[0] > 1:
            if index == self.pts_lin.shape[0] - 1:
                # This is the last point. Restrict movement to the left
                if xy_data[0] - self.pts_lin[index-1,0] < self.model.atol:
                    return
            elif index > 0:
                # Middlle point
                if (    xy_data[0] - self.pts_lin[index-1,0] < self.model.atol
                        or self.pts_lin[index+1,0] - xy_data[0] < self.model.atol):
                    return
            else:
                # First point
                if self.pts_lin[index+1,0] - xy_data[0] < self.model.atol:
                    return
        #if not isclose(
                #xy_data[0], self.pts_lin[:,0], atol=self.model.atol).any():
        self.pts_px[index] = xy_px
        # This emits a notify signal for the output data
        self.model._process_tr_input_data(self.trace_no)

    def delete_pt_px(self, index: int):
        self._delete_pt_px(index)
        # Trigger a full update of the model and view of inputs and outputs
        self.model._process_tr_input_data(self.trace_no)
        self.model.tr_input_data_changed[int].emit(self.trace_no)

    def _delete_pt_px(self, index: int):
        logger.debug(f"Call Trace.delete_pt_px with index: {index}")
        self.pts_px = np.delete(self.pts_px, index, axis=0)

    def sort_pts(self):
        self._sort_pts()
        self.model._process_tr_input_data(self.trace_no)
        self.model.tr_input_data_changed[int].emit(self.trace_no)

    def _sort_pts(self):
        # Sort trace points along the first axis.
        #
        # This sorts raw pixel space input points in the order as
        # they correspond to the sorted data space points.
        if self.pts_lin.shape != self.pts_px.shape:
            logger.critical("Sorting error: Array shape is not compatible")
            return
        ids = np.argsort(self.pts_lin[:,0])
        logger.debug(f"Sorting IDs: {ids}")
        self.pts_lin = self.pts_lin[ids]
        self.pts_px = self.pts_px[ids]

    def sort_remove_duplicate_pts(self):
        self._sort_remove_duplicate_pts()
        self.model._process_tr_input_data(self.trace_no)
        self.model.tr_input_data_changed[int].emit(self.trace_no)

    def _sort_remove_duplicate_pts(self) -> None:
        # Sort trace points along the first axis and
        # remove duplicate rows.
        #
        # This sorts raw pixel space input points in the order as
        # they correspond to the sorted data space points.
        if self.pts_lin.shape != self.pts_px.shape:
            logger.critical("Sorting error: Array shape is not compatible")
            return
        self.pts_lin, unique_ids = np.unique(
                self.pts_lin, axis=0, return_index=True)
        self.pts_px = self.pts_px[unique_ids]
        logger.debug(f"UNIQUE IDs: {unique_ids}")

    def _interpolate_view_data(self) -> None:
        # Acts on linear coordinates.
        # Needs at least four data points for interpolation.
        pts = self.pts_lin
        if pts.shape[0] < 4:
            return
        try:
            # Scipy interpolate generates an interpolation function which is added
            # to this instance attriutes
            self.f_interp_lin = interp1d(
                    *pts.T,
                    kind=self.interp_type,
                    fill_value="extrapolate",
                    assume_sorted=True,
                    )
            # Generate finer grid
            xgrid = np.linspace(pts[0,0], pts[-1,0], num=self.n_pts_i_view)
            yvals = self.f_interp_lin(xgrid)
            self.pts_lin_i = np.concatenate(
                    (xgrid.reshape(-1, 1), yvals.reshape(-1, 1)),
                    axis=1
                    )
            #self._interpolation_valid = True
        except ValueError as e:
            #self._interpolation_valid = False
            print("Interpolation exception: ", e)
            pass


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
        if self.f_interp_lin is None:
            return
        if x_ax._log_scale:
            if y_ax._log_scale:
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
                self.pts[:,0] = x_ax.log_base ** self.pts_lin[:,0]
                # Interpolation function applied to logarithmised X values
                self.f_interp = lambda x: self.f_interp_lin(
                        # This seems to perform better than np.emath.logn
                        np.log(x) / np.log(x_ax.log_base)
                        )
        else:
            if y_ax._log_scale:
                ##### CASE 3: Y axis only logarithmic scale
                # Transform points Y coordinates back to log scale
                self.pts = self.pts_lin.copy()
                self.pts[:,1] = y_ax.log_base ** self.pts_lin[:,1]
                # Interpolation function post-exponentiation only
                self.f_interp = lambda x: y_ax.log_base ** self.f_interp_lin(x)
            else:
                ##### CASE 4: no logarithmic scale
                self.pts = self.pts_lin.copy()
                self.f_interp = self.f_interp_lin

    def __str__(self):
        return self.name


class Axis(QObject):
    """Plot axis

    This class includes access methods for GUI/model interactive
    axis configuration or re-configuration.
    """
    def __init__(self, model, ax_conf, name):
        super().__init__(model)
        # Initial axis configuration. These attributes are overwritten
        # with stored configuration after axis is instantiated in DataModel
        ########## Connection to the containing data model
        self.model = model
        ########## Axis default configuration
        self.name = name
        # Keyword options for point markers.
        self.pts_fmt = ax_conf.pts_fmt
        self.log_base = ax_conf.log_base
        self.atol = ax_conf.atol

        ########## Axis data
        # Two points defining an axis section in pixel coordinate space
        self.pts_px = np.full((2, 2), NaN)
        # Axis section values in data coordinates
        self.pts_data = np.full(2, NaN)
        # Axis section values, linearised in case of log scale, copy otherwise
        self.pts_data_lin = np.full(2, NaN)

        ########## Associated view object
        self.pts_view_obj = None # Optional: pyplot.Line2D

        ########## Private properties
        # These flags are updated when the respective attr setters are called
        self._valid_pts_px = False
        self._valid_pts_data = False
        self._log_scale = ax_conf.log_scale
 
    def is_complete(self) -> bool:
        """Returns True if axis setup is all complete and valid"""
        return self._valid_pts_data and self._valid_pts_px

    def valid_pts_px(self) -> bool:
        """Returns True if axis section has both points set
        and points distance is nonzero length
        """
        return self._valid_pts_px
    
    def valid_pts_data(self) -> bool:
        """Returns True if the data coordinate points defining the axis
        section are both set and valid (i.e. not zero for log axes)
        """
        return self._valid_pts_data

    # Returns axis configuration as a dictionary used for persistent storage.
    def restorable_state(self) -> dict:
        state = vars(self).copy()
        # The view object belongs to the view component and cannot be restored
        # into a new context.
        del state["pts_view_obj"], state["model"]
        return state

    def log_scale(self) -> bool:
        """Returns True if this axis data space scale is logarithmic
        """
        return self._log_scale
    @logExceptionSlot(bool)
    def set_log_scale(self, state=True):
        """Sets logarithmic scale data coordinates for this axis.
                
        For invalid data, an error signal is emitted and model is left
        untouched.
        """
        logger.debug(f"set_log_state called with state: {state}")
        # Prevent recursive or duplicate calls
        if state == self._log_scale:
            return
        # Prevent setting logarithmic scale when axes values contain zero
        if state and isclose(self.pts_data, 0.0, atol=self.atol).any():
            self._log_scale = False
            self.model.value_error.emit(
                    "X axis values must not be zero for log axes")
        else:
            self._log_scale = state
        self.model.ax_input_data_changed.emit()


    @logExceptionSlot(float)
    def set_ax_start(self, value: float):
        """Set the data coordinate space value defining the start of
        the axis section corresponding with the first of two pixel
        coordinate points defining the same axis section in pixel space
        
        For invalid data, an error signal is emitted and model is left
        untouched.
        """
        # Prevent recursive calls and unnecessary updates
        if isclose(self.pts_data[0], value, atol=self.atol):
            return
        if self._log_scale and isclose(value, 0.0, atol=self.atol):
            self.model.value_error.emit(
                    "X axis values must not be zero for log axes")
        elif isclose(value, self.pts_data[1], atol=self.atol):
            self.model.value_error.emit(
                    "X axis values must be numerically different")
        else:
            self.pts_data[0] = value
            self._valid_pts_data = not isnan(self.pts_data).any()
        # Updates model outputs etc. when X and Y axis setup is complete
        self.model.ax_input_data_changed.emit()

    @logExceptionSlot(float)
    def set_ax_end(self, value: float):
        """Set the data coordinate space value defining the end of
        the axis section corresponding with the second of two pixel
        coordinate points defining the same axis section in pixel space
        
        For invalid data, an error signal is emitted and model is left
        untouched.
        """
        # Prevent recursive calls and unnecessary updates
        if isclose(self.pts_data[1], value, atol=self.atol):
            return
        if self._log_scale and isclose(value, 0.0, atol=self.atol):
            self.model.value_error.emit(
                    "X axis values must not be zero for log axes")
        elif isclose(value, self.pts_data[0], atol=self.atol):
            self.model.value_error.emit(
                    "X axis values must be numerically different")
        else:
            self.pts_data[1] = value
            self._valid_pts_data = not isnan(self.pts_data).any()
        # Updates model outputs etc. when X and Y axis setup is complete
        self.model.ax_input_data_changed.emit()

    def add_pt_px(self, xy_px: Iterable[float]) -> int:
        """Add a point defining an axis section

        xy_px: (x, y)-tuple or np.array shape (2, )

        Returns index of newly added point, which is determined
        automatically.

        If both points are unset, set the first one.
        If only one point is unset, set the other one.
        If both points are set, invalidate second and set first one
        to start over.

        Leaves model silently untouched if input data
        would yield a zero-length axis section.
        """
        logger.debug(f"add_pt_px called with xy data: {xy_px}")
        # Results are each True when point is unset..
        unset_1st, unset_2nd = isnan(self.pts_px).any(axis=1)
        index = 0
        if not unset_1st and not unset_2nd:
            # Both points set. Invalidate second point, so that it will be set
            # in next call. NaN values make matplotlib not plot this point.
            self.pts_px[1] = np.full(2, NaN)
        elif not unset_1st or not unset_2nd:
            # Only one point is still unset. (Both not set was covered above)
            # Set index to the remaining unset point:
            if unset_2nd:
                index = 1
            # else index is still 0
        pts_distance = np.linalg.norm(self.pts_px[index-1] - xy_px)
        if isclose(pts_distance, 0.0, atol=self.atol):
            return index
        self.pts_px[index] = xy_px
        self._valid_pts_px = not isnan(self.pts_px).any()
        # Updates model outputs etc. when X and Y axis setup is complete
        self.model.ax_input_data_changed.emit()
        return index

    def update_pt_px(self, xy_px: Iterable[float], index: int):
        """Update axis point in pixel coordinates 

        Leaves model silently untouched if input data
        would yield a zero-length axis section.
        """
        pts_distance = np.linalg.norm(self.pts_px[index-1] - xy_px)
        if isclose(pts_distance, 0.0, atol=self.atol):
            return
        self.pts_px[index] = xy_px
        self._valid_pts_px = not isnan(self.pts_px).any()
        # Updates model outputs etc. when X and Y axis setup is complete
        self.model.ax_input_data_changed.emit()

    def delete_pt_px(self, index: int):
        """Delete axis point in pixel coordinates and invalidates state
        """
        self.pts_px[index] = np.full(2, NaN)
        self._valid_pts_px = False
        # Updates model outputs etc. when X and Y axis setup is complete
        self.model.ax_input_data_changed.emit()
    
    def __str__(self):
        return self.name


