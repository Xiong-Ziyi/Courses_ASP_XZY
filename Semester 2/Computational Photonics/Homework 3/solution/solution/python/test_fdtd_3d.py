"""Test script for tak II (3D-FDTD) of the assignment of
Computational Photonics seminar 6 - FDTD.
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

# simulation parameters
Nx = 199  # nuber of grid points in x-direction
Ny = 201  # nuber of grid points in y-direction
Nz = 5  # nuber of grid points in z-direction
dr = 30e-9  # grid spacing in [m]
time_span = 10e-15  # duration of simulation [s]

# x coordinates
x = np.arange(-int(np.ceil((Nx - 1) / 2)), int(np.floor((Nx - 1) / 2)) + 1) * dr
# y coordinates
y = np.arange(-int(np.ceil((Ny - 1) / 2)), int(np.floor((Ny - 1) / 2)) + 1) * dr

# relative permittivity distribution
eps_rel = np.ones((Nx, Ny, Nz), dtype=np.float32)

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
field_component = 'hx'
print('Calculating Hx')
# noinspection PyTypeChecker
Hx, t = fdtd_3d(eps_rel, dr, time_span, freq, tau,
                jx, jy, jz, field_component, z_ind, output_step)

field_component = 'hy'
print('Calculating Hy')
# noinspection PyTypeChecker
Hy, _ = fdtd_3d(eps_rel, dr, time_span, freq, tau,
                jx, jy, jz, field_component, z_ind, output_step)

field_component = 'ez'
print('Calculating Ez')
# noinspection PyTypeChecker
Ez, _ = fdtd_3d(eps_rel, dr, time_span, freq, tau,
                jx, jy, jz, field_component, z_ind, output_step)

# %% movie of Hx %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F = Hx * Z0 * 1e6
titlestr = 'x-Component of Magnetic Field'
cb_label = '$\\Re\\{Z_0H_x\\}$ [µV/m]'
rel_color_range = 1 / 3
fps = 10

ani = Fdtd3DAnimation(x, y, t, F, titlestr, cb_label, rel_color_range, fps)
plt.show()

# requires Ffmpeg
if save_movie:
    try:
        print('saving movie to disk')
        ani.save("video_FDTD3D_Hx.mp4", bitrate=1024, fps=fps, dpi=120)
    except Exception as e:
        print('saving movie failed', file=sys.stderr)
        print(e, file=sys.stderr)

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
        ani.save("video_FDTD3D_Ez.mp4", bitrate=1024, fps=fps, dpi=120)
    except Exception as e:
        print('saving movie failed', file=sys.stderr)
        print(e, file=sys.stderr)

# %% plot fields at end of simulation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = plt.figure()

# plot electric field
# image data must be transposed because Ez  has x as row coordinate
# but imshow expects x as column coordinate
res = Ez[-1, :].T
color_range = np.max(np.abs(res)) * 1e6
phw = 0.5 * (x[1] - x[0])  # pixel half-width
extent = ((x[0] - phw) * 1e6, (x[-1] + phw) * 1e6,
          (y[-1] + phw) * 1e6, (y[0] - phw) * 1e6)
plt.imshow(res.real * 1e6, vmin=-color_range, vmax=color_range,
           extent=extent)
cb = plt.colorbar()
plt.gca().invert_yaxis()

# interpolate real part of magnetic field to polar grid for better plotting
R = dr * np.arange(int(round(max(x) * 3 / 8 / dr)) - 1,
                   int(round(max(x) / dr)) + 1, 2)
phi = np.deg2rad(np.arange(72) * 5)
xr = R[:, np.newaxis] * np.cos(phi[np.newaxis, :])
yr = R[:, np.newaxis] * np.sin(phi[np.newaxis, :])
X, Y = np.meshgrid(x, y)
interp_points = np.hstack((xr.reshape((-1, 1)), yr.reshape((-1, 1))))
U = interpn((x, y), Hx[-1, :, :].real, interp_points)
U = U.reshape(xr.shape)
V = interpn((x, y), Hy[-1, :, :].real, interp_points)
V = V.reshape(xr.shape)

# normalize vectors
norm = np.sqrt(U ** 2 + V ** 2).max()
U /= norm
V /= norm

# plot magnetic field as vectors
plt.quiver(xr * 1e6, yr * 1e6, U, V, scale=1e-6 / dr, angles='xy',
           color='w', label='magnetic field')
l = plt.legend(frameon=False, loc='lower left')
for text in l.get_texts():
    text.set_color('w')
plt.title('$ct$ = {0:1.2f}µm'.format(t[-1] * c * 1e6))
plt.xlabel('x position [µm]')
plt.ylabel('y position [µm]')
cb.set_label('$\\Re\\{E_z\\}$ [µV/m]')
plt.show()

if save_figures:
    fig.savefig('fields_at_end_3D.pdf', dpi=300)
