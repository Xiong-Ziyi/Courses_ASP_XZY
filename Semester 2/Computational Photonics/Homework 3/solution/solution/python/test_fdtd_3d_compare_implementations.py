"""Additional test script for tak II (3D-FDTD) of the assignment of
Computational Photonics seminar 5 - FDTD.

The script compares the speed and accuracy of two iplementations of the
3D-FDTD algorithm.
"""

import numpy as np
from ftdt_functions import fdtd_3d, fdtd_3d_naive_accelerated

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
field_component = 'Hx'
print('Calculating Hx with vectorized implementation')
Hx, t = fdtd_3d(eps_rel, dr, time_span, freq, tau,
                jx, jy, jz, field_component, z_ind, output_step)

print('Calculating Hx with jit accelerated implementation')
Hx1, t = fdtd_3d_naive_accelerated(eps_rel, dr, time_span, freq, tau,
                                   jx, jy, jz, field_component, z_ind, output_step)

print('abs error of Hx with respect to vectorired implementation: '
      '{0}'.format(np.abs(Hx - Hx1).max()))
