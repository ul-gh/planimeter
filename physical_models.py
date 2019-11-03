# -*- coding: utf-8 -*-
"""Physical Data Models for Plot Model Assistant
"""
import numpy as np
import scipy.integrate as integrate

import matplotlib.pyplot as plt
import upylib.u_plot_format as u_format

from plot_model import PlotModel, PhysModelABC

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
                trace_names=["C_ISS", "C_OSS", "C_RSS"],
                trace_colors=["r", "g", "b"],
                )
        plot_conductances = PlotModel(
                pma,
                name="Conductances",
                trace_names=["Transconductance", "Channel Modulation"],
                trace_colors=["r", "g"],
                )
        plot_diode = PlotModel(
                pma,
                name="Reverse Diode",
                trace_names=["Forward Conductance", "Diffusion Charge"],
                trace_colors=["g", "b"],
                )
        # Put above into list "plots" of this instance
        self.plots = [plot_capacitances, plot_conductances, plot_diode]

        self.exportfuncs.update({"LTSpice": self.export_ltspice_mosfet})
        self.curr_exportfunc = self.export_ltspice_mosfet

    def export_ltspice_mosfet(self, filename):
        pass
     
    def wip_plot_cap_charge_e_stored(self, tr):
        # FIXME: Temporary solution
        grid = np.linspace(tr.pts[0,0], tr.pts[-1,0], 100)
        y = tr.f_interp(grid)*1e-12
        y_int = integrate.cumtrapz(y, grid, initial=0.0)
        x_times_y = grid * y
        x_times_y_int = integrate.cumtrapz(x_times_y, grid, initial=0.0)
        pts_export = np.stack((grid, y), axis=1)
        pts_export_int = np.stack((grid, y_int), axis=1)
        pts_export_xy_int = np.stack((grid, x_times_y_int), axis=1)
        
        self.fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        self.fig.set_size_inches(18, 6.0)
        # Plot ax1
        u_format.plot_lin_lin_engineering(
                ax1,
                *pts_export.T,
                title=r"Differential Capacitance C(u)",
                xlabel=r"U /Volts",
                ylabel=r"C(u) /Farads",
                #xlim=(1, 450),
                #ylim=(0, 0.6),
                )
        # Plot ax2
        u_format.plot_lin_lin_engineering(
                ax2,
                *pts_export_int.T,
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
                *pts_export_xy_int.T,
                title=r"C Stored Energy E(u)",
                xlabel=r"U /Volts",
                ylabel=r"E(u) /Joules",
                #xlim=(1, 450),
                #ylim=(0, 0.6),
                )
        ax3_text = r"$E(u) = \int u \cdot C(u) \: du$"
        ax3.text(0.05, 0.9, ax3_text, fontsize=15, transform=ax3.transAxes)
        self.fig.tight_layout()


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
                trace_names=["Trace 1", "Trace 2", "Trace 3"],
                trace_colors=["r", "g", "b"],
                )
        plot_1 = PlotModel(
                pma,
                name="Plot 1",
                trace_names=["Trace 1", "Trace 2", "Trace 3"],
                trace_colors=["r", "g", "b"],
                )
        self.plots = [plot_0, plot_1]
        # self.curr_exporter = DefaultExporter(self)
        # self.exporters = {"Default": self.curr_exporter}