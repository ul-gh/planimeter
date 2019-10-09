# -*- coding: utf-8 -*-
"""Physical Data Models for Plot Model Workbench
"""
from plot_model import PlotModel

class MosfetDynamic():
    """Physical model for Spice simulation of MOSFET dynamic properties
    in switching power application
    """
    # Display name, used by the GUI
    name = "Mosfet Dynamic"

    def __init__(self, digitizer, conf):
        confs = [conf, conf, conf]
        self.plots = plots = [PlotModel(digitizer, conf) for conf in confs]
        plots[0].name = "Capacitances"
        plots[1].name = "Transconductance"
        plots[2].name = "Reverse Diode Conductance"

class Custom():
    """Custom model, defaulting to three empty traces
    """
    # Display name, used by the GUI
    name = "Custom"

    def __init__(self, digitizer, conf):
        confs = [conf, conf, conf]
        self.plots = plots = [PlotModel(digitizer, conf) for conf in confs]
        plots[0].name = "Plot 1"
        plots[1].name = "Plot 2"
        plots[2].name = "Plot 3"
        