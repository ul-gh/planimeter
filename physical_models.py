# -*- coding: utf-8 -*-
"""Physical Data Models for Plot Model Assistant
"""
from plot_model import PlotModel, PhysModelABC
from exporters import LTSpice_MOSFET, Redmont_XLSX

class MosfetDynamic(PhysModelABC):
    """Physical model for Spice simulation of MOSFET dynamic properties
    in switching power application
    """
    # Display name, used by the GUI
    name = "Mosfet Dynamic"

    def __init__(self, pma):
        super().__init__(pma)
        # Define three plots and their traces
        plot_capacitances = PlotModel(
                pma,
                name="Capacitances",
                trace_names=("C_ISS", "C_OSS", "C_RSS"),
                colors=("r", "g", "b"),
                )
        plot_conductances = PlotModel(
                pma,
                name="Conductances",
                trace_names=("Transconductance", "Channel Modulation"),
                colors=("r", "g"),
                )
        plot_diode = PlotModel(
                pma,
                name="Reverse Diode",
                trace_names=("Forward Conductance", "Diffusion Charge"),
                colors=("g", "b"),
                )
        # Put above into list "plots" of this instance
        self.plots = [plot_capacitances, plot_conductances, plot_diode]
        
        ltspice = LTSpice_MOSFET(self)
        xlsx = Redmont_XLSX(self)
        self.exporters = {"LTSpice": ltspice, "Redmont XLSX": xlsx}
        self.curr_exporter = ltspice


class Default(PhysModelABC):
    """Default model of two plots each having three traces
    """
    # Display name, used by the GUI
    name = "Default"

    def __init__(self, pma):
        super().__init__(pma)
        plot_0 = PlotModel(
                pma,
                name="Plot 0",
                trace_names=("Trace 1", "Trace 2", "Trace 3"),
                colors=("r", "g", "b"),
                )
        plot_1 = PlotModel(
                pma,
                name="Plot 1",
                trace_names=("Trace 1", "Trace 2", "Trace 3"),
                colors=("r", "g", "b"),
                )
        self.plots = [plot_0, plot_1]
        # self.curr_exporter = DefaultExporter(self)
        # self.exporters = {"Default": self.curr_exporter}