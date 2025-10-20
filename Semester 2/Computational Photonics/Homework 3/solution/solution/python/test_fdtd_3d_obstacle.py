"""Additional test script for tak II (3D-FDTD) of the assignment of
Computational Photonics seminar 5 - FDTD.

The script simulates the effect of a dielectric obstacle.
"""

import numpy as np
from ftdt_functions import fdtd_3d, Fdtd3DAnimation
from matplotlib import pyplot as plt
from scipy.interpolate import interpn
import sys
# dark bluered colormap, registers automatically with matplotlib on import
# noinspection PyUnresolvedReferences
import bluered_dark

plt.rcParams.update({
    'figure.figsize': (12 / 2.54, 9 / 2.54),
    'figure.subplot.bottom': 0.15,
    'figure.subplot.left': 0.165,
    'figure.subplot.right': 0.90,
    'figure.subplot.top': 0.9,
    'axes.grid': False,
    'image.cmap': 'bluered_dark',
})

# save movie to disk (requires Ffmpeg)
save_movie = False

plt.close('all')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# constants
c = 2.99792458e8  # speed of light [m/s]
mu0 = 4 * np.pi * 1e-7  # vacuum permeability [Vs/(Am)]
eps0 = 1 / (mu0 * c ** 2)  # vacuum permittivity [As/(Vm)]
Z0 = np.sqrt(mu0 / eps0)  # vacuum impedance [Ohm]

# simulation parameters
Nx = 199  # nuber of grid points in x-direction
Ny = 201  # nuber of grid points in y-direction
Nz = 5  # nuber of grid points in z-direction
dr = 30e-9  # grid spacing in [m]
time_span = 15e-15  # duration of simulation [s]

# x coordinates
x = np.arange(-int(np.ceil((Nx - 1) / 2)), int(np.floor((Nx - 1) / 2)) + 1) * dr
# y coordinates
y = np.arange(-int(np.ceil((Ny - 1) / 2)), int(np.floor((Ny - 1) / 2)) + 1) * dr

r_obstacle = 0.75 * 1e-6
n_obstacle = 1.5
x0 = -1.5 * 1e-6
y0 = -1.5 * 1e-6
ind = (x[:, np.newaxis] - x0) ** 2 + (y[np.newaxis, :] - y0) ** 2 <= r_obstacle ** 2
ind = np.tile(ind[:, :, np.newaxis], (1, 1, Nz))
# relative permittivity distribution
eps_rel = np.ones((Nx, Ny, Nz), dtype=np.float32)
eps_rel[ind] = n_obstacle ** 2
# source parameters
freq = 500e12  # pulse [Hz]
tau = 1e-15  # pulse width [s]
source_width = 2  # width of Gaussian current dist. [grid points]

# current distribution
midx = int(np.ceil((Nx - 1) / 2))
midy = int(np.ceil((Ny - 1) / 2))
midz = int(np.ceil((Nz - 1) / 2))
jx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
jy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
# Gaussion distribution in the xy-plane, constant along z
jz = np.tile(np.exp(-((np.arange(Nx)[:, np.newaxis, np.newaxis]
                       - midx) / source_width) ** 2)
             * np.exp(-((np.arange(Ny)[np.newaxis, :, np.newaxis]
                         - midy) / source_width) ** 2),
             (1, 1, Nz))

# output parameters
z_ind = midz  # z-index of field output
output_step = 4  # time steps between field output

# %% run simulations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
field_component = 'ez'
print('Calculating Ez')
# noinspection PyTypeChecker
Ez, t = fdtd_3d(eps_rel, dr, time_span, freq, tau,
                jx, jy, jz, field_component, z_ind, output_step)

# %% movie of Ez %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F = Ez * 1e6
titlestr = 'z-Component of Electric Field'
cb_label = '$\\Re\\{E_z\\}$ [µV/m]'
rel_color_range = 1 / 3
fps = 10

# noinspection PyTypeChecker
ani = Fdtd3DAnimation(x, y, t, F, titlestr, cb_label, rel_color_range, fps)
plt.show()

# requires Ffmpeg
if save_movie:
    try:
        print('saving movie to disk')
        ani.save("video_FDTD3D_Ez_obstacle.mp4", bitrate=1024, fps=fps, dpi=120)
    except Exception as e:
        print('saving movie failed', file=sys.stderr)
        print(e, file=sys.stderr)
