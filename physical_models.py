# -*- coding: utf-8 -*-
"""Physical Data Models for Plot Model Workbench
"""

class MosfetDynamic():
    """Physical model for Spice simulation of MOSFET dynamic properties
    in switching power application
    """
    # Display name, used by the GUI
    name = "Mosfet Dynamic"

    def __init__(self):
        print("foo")


class Custom():
    """Custom model, defaulting to three empty traces
    """
    # Display name, used by the GUI
    name = "Custom"

    def __init__(self):
        self.n_traces = 3
        self.names = ["Trace 1", "Trace 2", "Trace 3"]
        