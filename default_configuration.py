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

    # Python string format code for GUI number formatting
    num_format = "G"


class DATA_MODEL():
    """Settings for all traces, axes etc
    """
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
    # Default number of X axis points for interpolation and data export
    n_pts_interpolation = 100
    # Trace raw points format, same for all traces
    pts_fmt = {"picker": 5.0, "marker": "x", "markersize": 10.0}

class X_AXIS():
    """Settings for X-Axis
    """
    pts_fmt = {"color": "y", "picker": 5.0, "marker": "s", "markersize": 15.0,
               "markerfacecolor": "none", "markeredgewidth": 2.0}
    log_scale = False
    log_base = 10
    # Absolute tolerance, same as for data model
    atol = DATA_MODEL.atol

class Y_AXIS():
    """Settings for Y-Axis
    """
    pts_fmt = {"color": "y", "picker": 5.0, "marker": "s", "markersize": 15.0,
               "markerfacecolor": "none", "markeredgewidth": 2.0}
    log_scale = False
    log_base = 10
    # Absolute tolerance, same as for data model
    atol = DATA_MODEL.atol
