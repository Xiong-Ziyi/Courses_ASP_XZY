"""Additional test script for tak I (1D-FDTD) of the assignment of
Computational Photonics seminar 5 - FDTD.

The script compares the speed and accuracy of three iplementations of the
1D-FDTD algorithm.
"""

import numpy as np
from ftdt_functions import (
    fdtd_1d, fdtd_1d_naive, fdtd_1d_naive_accelerated, Timer)
import sys

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
for i in range(3):
    timer.tic()
    Ez, Hy, x, t = fdtd_1d(eps_rel, dx, time_span,
                           source_frequency, source_position,
                           source_pulse_length)
    print('time for vectorized implementation: {0:g}s'.format(timer.toc()))

# measure execution time of alternative implementations
for i in range(3):
    timer.tic()
    Ez1, Hy1, x, t = fdtd_1d_naive(eps_rel, dx, time_span,
                                   source_frequency, source_position,
                                   source_pulse_length)
    print('time for naive implementation: {0:g}s'.format(timer.toc()))
print('abs error of Ez with respect to vectorired implementation: '
      '{0}'.format(np.abs(Ez - Ez1).max()))
print('abs error of Hy with respect to vectorired implementation: '
      '{0}'.format(np.abs(Hy - Hy1).max()))
try:
    for i in range(3):
        timer.tic()
        Ez2, Hy2, x, t = fdtd_1d_naive_accelerated(eps_rel, dx, time_span,
                                                   source_frequency, source_position,
                                                   source_pulse_length)
        print('time for naive implementation (jit accelerated): '
              '{0:g}s'.format(timer.toc()))
    print('abs error of Ez with respect to vectorired implementation: '
          '{0}'.format(np.abs(Ez - Ez2).max()))
    print('abs error of Hy with respect to vectorired implementation: '
          '{0}'.format(np.abs(Hy - Hy2).max()))
except Exception as e:
    print('Cannot test jit accelerated implementation', file=sys.stderr)
    print(e, file=sys.stderr)
