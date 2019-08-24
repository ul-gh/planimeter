# -*- coding: utf-8 -*-
"""Configuration settings for application and modules
"""
from PyQt5.QtCore import QDir

class APPLICATION():
    """Application Default Settings
    """
    config_file_name = "config.pickle"
    # Default working directory for file open etc.
    wdir = QDir.homePath()
    # Default window size
    window_size = (800, 600)

    ### String format codes, see: https://fmt.dev/latest/syntax.html
    # Number string format code for DATA EXPORT
    num_fmt_export = ".6E"
    # Number string format code for GUI display
    num_fmt_gui = ".6G"
    # Decimal point character. Use "system" for auto-detect based on locale
    decimal_chr = "system"
    # Image data rescaling interpolation method.
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/
    # interpolation_methods.html
    img_interpolation = "bilinear"


class DATA_MODEL():
    """Settings for all traces, axes etc
    """
    ########## Export options
    # Number of X-axis interpolation points for data export for linear X grid
    n_pts_i_export = 100
    # Number of X-axis interpolation points for data export for log X grid
    n_pts_i_export_dec = 100
    # Maximum number of export points
    n_pts_i_export_max = 100000
    x_step_export = float("NaN")
    x_step_export_min = 1e-18
    # Absolute tolerance for testing if values are close to zero
    atol = 1e-18
    # Default trace names and colors
    traces_names = "Trace 1", "Trace 2", "Trace 3"
    traces_colors = "r", "g", "b"
    # Matplotlib plot keyword options as a dictionary
    origin_fmt = {"color": "c", "marker": "o", "markersize": 15.0,
                  "markerfacecolor": "none", "markeredgewidth": 2.0}
    # Store axes configuration persistently on disk when set
    store_ax_conf = False


class TRACE():
    """Settings for any new plot trace
    """
    # Default number of X axis points for GUI display interpolation only
    n_pts_i_view = 100
    # Kind of interpolation function used for a new trace by default
    interp_type = "cubic"
    # Trace raw points format, same for all traces
    pts_fmt = {"picker": 10.0, "linestyle": ":",
               "marker": "x", "markersize": 10.0}
    # Mark new traces for export
    export = True

class X_AXIS():
    """Settings for X-Axis
    """
    pts_fmt = {"color": "y", "picker": 15.0, "marker": "s", "markersize": 15.0,
               "markerfacecolor": "none", "markeredgewidth": 2.0}
    log_scale = False
    log_base = 10
    # Absolute tolerance, same as for data model
    atol = DATA_MODEL.atol

class Y_AXIS():
    """Settings for Y-Axis
    """
    pts_fmt = {"color": "y", "picker": 15.0, "marker": "s", "markersize": 15.0,
               "markerfacecolor": "none", "markeredgewidth": 2.0}
    log_scale = False
    log_base = 10
    # Absolute tolerance, same as for data model
    atol = DATA_MODEL.atol
