# -*- coding: utf-8 -*-
"""Physical Data Models for Plot Model Assistant
"""
from plot_model import PlotModel
from exporters import LTSpice_MOSFET, Redmont_XLSX, DefaultExporter

class MosfetDynamic():
    """Physical model for Spice simulation of MOSFET dynamic properties
    in switching power application
    """
    # Display name, used by the GUI
    name = "Mosfet Dynamic"

    def __init__(self, pma):
        self.pma = pma # Plot Model Assistant instance
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
        self.curr_exporter = ltspice
        self.exporters = [ltspice, xlsx]

    def do_export(self):
        filename = self.pma.dlg_save_as.exec()
        self.curr_exporter.export_as(filename)


class Default():
    """Default model of two plots each having three traces
    """
    # Display name, used by the GUI
    name = "Default"

    def __init__(self, pma):
        self.pma = pma # Plot Model Assistant instance
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
        self.curr_exporter = DefaultExporter(self)
        self.exporters = [self.curr_exporter]

    def do_export(self):
        filename = self.pma.dlg_save_as.exec()
        self.curr_exporter.export_as(filename)