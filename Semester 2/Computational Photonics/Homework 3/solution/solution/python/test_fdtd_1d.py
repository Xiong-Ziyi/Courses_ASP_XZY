"""Test script for task I (1D-FDTD) of the assignment of
Computational Photonics seminar 6 - FDTD.
"""

import numpy as np
from ftdt_functions import fdtd_1d, Fdtd1DAnimation, Timer
from matplotlib import pyplot as plt
import sys
# dark bluered colormap, registers automatically with matplotlib on import
# noinspection PyUnresolvedReferences
# import bluered_dark

plt.rcParams.update({
    'figure.figsize': (12 / 2.54, 9 / 2.54),
    'figure.subplot.bottom': 0.15,
    'figure.subplot.left': 0.165,
    'figure.subplot.right': 0.90,
    'figure.subplot.top': 0.9,
    'axes.grid': False,
    'image.cmap': 'bluered_dark',
})

# save figures to disk
save_figures = False
# save movie to disk (requires Ffmpeg)
save_movie = False

plt.close('all')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# constants
c = 2.99792458e8  # speed of light [m/s]
mu0 = 4 * np.pi * 1e-7  # vacuum permeability [Vs/(Am)]
eps0 = 1 / (mu0 * c ** 2)  # vacuum permittivity [As/(Vm)]
Z0 = np.sqrt(mu0 / eps0)  # vacuum impedance [Ohm]

# geometry parameters
x_span = 18e-6  # width of computatinal domain [m]
n1 = 1  # refractive index in front of interface
n2 = 2  # refractive index behind interface
x_interface = x_span / 4  # postion of dielectric interface

# simulation parameters
dx = 30e-9 / max(n1, n2)  # grid spacing [m]
time_span = 60e-15  # duration of simulation [s]

Nx = int(round(x_span / dx)) + 1  # number of grid points

# source parameters
source_frequency = 500e12  # [Hz]
source_position = 0  # [m]
source_pulse_length = 1e-15  # [s]

# %% run simulations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# timer to measure the execution time
timer = Timer()

# simulate homogeneous medium
eps_rel = np.ones((Nx,)) * n1 ** 2
Ez_ref, Hy_ref, x, t = fdtd_1d(eps_rel, dx, time_span,
                               source_frequency, source_position,
                               source_pulse_length)

# simulate dielectric interface
eps_rel[x >= x_interface] = n2 ** 2
timer.tic()
Ez, Hy, x, _ = fdtd_1d(eps_rel, dx, time_span,
                       source_frequency, source_position,
                       source_pulse_length)
print('time: {0:g}s'.format(timer.toc()))

# %% make video %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fps = 25
step = t[-1] / fps / 30
ani = Fdtd1DAnimation(x, t, Ez, Hy, x_interface=x_interface,
                      step=step, fps=fps)
plt.show()

# requires Ffmpeg
if save_movie:
    try:
        print('saving movie to disk')
        ani.save("video_FDTD1D.mp4", bitrate=256, fps=fps, dpi=120)
    except Exception as e:
        print('saving movie failed', file=sys.stderr)
        print(e, file=sys.stderr)

# %% plot time traces of electric field and Poynting vector %%%%%%%%%%%%%%%%%%%
extent = [t[0] * c * 1e6, t[-1] * c * 1e6, x[-1] * 1e6, x[0] * 1e6]

# plot electric field
vmax = max(np.max(np.abs(Ez_ref)), np.max(np.abs(Ez))) * 1e6
fig = plt.figure()
plt.imshow(Ez_ref.real.T * 1e6, extent=extent, vmin=-vmax, vmax=vmax)
cb = plt.colorbar()
plt.xlabel('$ct$ [µm]')
plt.ylabel('$x$ [µm]')
cb.set_label('real part of $E_z$ [µV/m]')
if save_figures:
    fig.savefig('Ez_ref_1D.pdf', dpi=300)

fig = plt.figure()
plt.imshow(Ez.real.T * 1e6, extent=extent, vmin=-vmax, vmax=vmax)
cb = plt.colorbar()
plt.xlabel('$ct$ [µm]')
plt.ylabel('$x$ [µm]')
cb.set_label('real part of $E_z$ [µV/m]')
if save_figures:
    fig.savefig('Ez_1D.pdf', dpi=300)

# calculate Poynting vector
Sx_ref = -0.5 * np.real(Ez_ref * np.conj(Hy_ref))
Sx = -0.5 * np.real(Ez * np.conj(Hy))

# plot power flow
vmax = np.abs(Sx_ref).max()
fig = plt.figure()
plt.imshow(Sx_ref.T / vmax, extent=extent, vmin=-1, vmax=1)
cb = plt.colorbar()
plt.xlabel('$ct$ [µm]')
plt.ylabel('$x$ [µm]')
cb.set_label('normalize Poynting vector')
cb.set_ticks(np.linspace(-1, 1, 11))
if save_figures:
    fig.savefig('Sx_ref_1D.pdf', dpi=300)

fig = plt.figure()
plt.imshow(Sx.T / vmax, extent=extent, vmin=-1, vmax=1)
cb = plt.colorbar()
plt.xlabel('$ct$ [µm]')
plt.ylabel('$x$ [µm]')
cb.set_label('normalize Poynting vector')
cb.set_ticks(np.linspace(-1, 1, 11))
plt.show()

if save_figures:
    fig.savefig('Sx_1D.pdf', dpi=300)
