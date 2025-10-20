'''Test script for Homework 3, Computational Photonics, SS 2020:  FDTD method.
'''


import numpy as np
from function_headers_fdtd import fdtd_1d, Fdtd1DAnimation
from matplotlib import pyplot as plt

# dark bluered colormap, registers automatically with matplotlib on import
import bluered_dark

plt.rcParams.update({
        'figure.figsize': (12/2.54, 9/2.54),
        'figure.subplot.bottom': 0.15,
        'figure.subplot.left': 0.165,
        'figure.subplot.right': 0.90,
        'figure.subplot.top': 0.9,
        'axes.grid': False,
        'image.cmap': 'bluered_dark',
})

plt.close('all')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# constants
c = 2.99792458e8 # speed of light [m/s]
mu0 = 4*np.pi*1e-7 # vacuum permeability [Vs/(Am)]
eps0 = 1/(mu0*c**2) # vacuum permittivity [As/(Vm)]
Z0 = np.sqrt(mu0/eps0) # vacuum impedance [Ohm]

# geometry parameters
x_span = 18e-6 # width of computatinal domain [m]
n1 = 1 # refractive index in front of interface
n2 = 2 # refractive index behind interface
x_interface = x_span/4 #postion of dielectric interface

# simulation parameters
dx = 15e-9 # grid spacing [m]
time_span = 60e-15 # duration of simulation [s]
dt=dx/(2*c)
Nx = int(round(x_span/dx)) + 1 # number of grid points
Nt=int(round(time_span/dt))
# source parameters
source_frequency = 500e12 # [Hz]
source_position = 0 # [m]
source_pulse_length = 1e-15 # [s]

# %% create permittivity distribution and run simulation %%%%%%%%%%%%%%%%%%%%%%

eps_rel=np.ones(Nx) * n1**2
interface_idx=int(x_interface/dx)
eps_rel[interface_idx:]=n2**2

x, t ,Ez, Hy=fdtd_1d(eps_rel, dx, time_span, source_frequency, source_position,
            source_pulse_length)


# Create a time-space plot for Ez
plt.figure()
plt.imshow(np.real(Ez).T, extent=extent,  aspect='auto', origin='lower')
plt.colorbar(label='Real part of Ez')
plt.title('Electric Field Ez over Time and Space')
plt.xlabel('Time (s)')
plt.ylabel('Space (m)')
plt.show()

# Create a time-space plot for Hy
plt.figure()
plt.imshow(np.real(Hy).T, extent=extent, aspect='auto', origin='lower')
plt.colorbar(label='Real part of Hy')
plt.title('Magnetic Field Hy over Time and Space')
plt.xlabel('Time (s)')
plt.ylabel('Space (m)')
plt.show()

# %% make video %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fps = 25
step = t[-1]/fps/30
ani = Fdtd1DAnimation(x, t, Ez, Hy, x_interface=x_interface,
                       step=step, fps=fps)
plt.show()

# %% create representative figures of the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# please add your code here
