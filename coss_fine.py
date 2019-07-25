#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from  numpy import genfromtxt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import upylib.u_plot_format as u_format

# When running in ipython, set up matplotlib to use separate Qt5 plot window.
if "get_ipython" in globals():
    get_ipython().run_line_magic("matplotlib", "qt5")

# Data from CSV file, first line is assumed to be header and is skipped.
# File format: Tuples of x in volts, y in picofarads
coss_raw = genfromtxt("coss.csv", delimiter=",", skip_header=1)
v_ds, c_oss = coss_raw.transpose()
# First column is assumed to be picofarads, factoring this in:
c_oss *= 1e-12


# equivalent: ax1 = plt.subplot(2, 1, 1)
#             ax2 = plt.subplot(2, 1, 2) etc.
# also possible to get Axis object via: ax1 = plt.gca()
fig, ((ax1, ax1b), (ax2, ax3)) = plt.subplots(2, 2)
fig.set_size_inches(12.8, 8.0)

# Extrapolated capacity C_OSS
v_ds_extra = np.append(v_ds, 400)
c_oss_extra = np.append(c_oss, c_oss[-1])

# Cubic spline or piecewise linear interpolation of intermediate values
c_oss_interp = interp1d(v_ds_extra, c_oss_extra, kind="linear")
v_ds_fine = np.linspace(v_ds_extra[0], v_ds_extra[-1], 500)
c_oss_fine = c_oss_interp(v_ds_fine)


# Plot C_OSS
u_format.plot_log_log_engineering(
    ax1,
    v_ds, c_oss, "b-",
    v_ds_extra[-2:], c_oss_extra[-2:], "r-",
    title=r"$C_{OSS}$ differenziell, extrapoliert",
    xlabel=r"$U_{DS}$ /Volt",
    ylabel=r"$C_{OSS}$ /Farad",
    xlim=(1, 450),
    #ylim=(10e-12, 1000e-12),
    )


# Plot C_OSS, interpolated version
u_format.plot_log_log_engineering(
    ax1b,
    v_ds_fine, c_oss_fine, "b-",
    title=r"$C_{OSS}$ differenziell, Interpolation (PWL oder Splines)",
    xlabel=r"$U_{DS}$ /Volt",
    ylabel=r"$C_{OSS}$ /Farad",
    xlim=(1, 450),
#    ylim=(10e-12, 1000e-12),
    )


# Calculate Q_OSS by integration of C_OSS
q_oss = cumtrapz(c_oss_fine, v_ds_fine, initial=0)

# Plot Q_OSS
u_format.plot_log_log_engineering(
    ax2,
    v_ds_fine, q_oss,
    title=r"$Q_{OSS}$ kumulativ",
    xlabel=r"$U_{DS}$ /Volt",
    ylabel=r"$Q_{OSS}$ /Coulomb",
    xlim=(1, 450),
    )


# Calculate P_v for given combinations of V_DS and cumulative Q_OSS
# At 125 kHz
f_sw = 125000
# This is elementwise. Indexes must be corresponding to the same V_DS.
p_v = q_oss * v_ds_fine * f_sw

# Plot P_v
u_format.plot_log_lin_engineering(
    ax3,
    v_ds_fine, p_v,
    title=r"$P_{V}$ über Spannung" + f" bei {f_sw/1000} kHz",
    xlabel=r"$U_{DS}$ /Volt",
    ylabel=r"$P_{V}$ /Watt",
    xlim=(1, 450),
    ylim=(0, 0.6),
    )



# Auto-Adjust layout spacing
fig.tight_layout(pad=0.5)
# Manual adjusting:
# fig.subplots_adjust(left=0.15, top=0.95, hspace=0.35)

if __name__ == "__main__":
    plt.show()


