'''Homework 2, Computational Photonics, SS 2020:  Beam propagation method.
'''

import numpy as np
from Homework_2_function_headers import waveguide, gauss, beamprop_CN
from matplotlib import pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla



# dark bluered colormap, registers automatically with matplotlib on import
import bluered_dark

plt.rcParams.update({
        'figure.figsize': (12/2.54, 9/2.54),
        'figure.subplot.bottom': 0.15,
        'figure.subplot.left': 0.165,
        'figure.subplot.right': 0.925,
        'figure.subplot.top': 0.9,
        'axes.grid': True,
})
plt.close('all')

# %%
# computational parameters
z_end   = 100       # propagation distance
lam     = 1         # wavelength
nd      = 1.455     # reference index
xa      = 50        # size of computational window
Nx      = 251       # number of transverse points
dx      = xa/(Nx-1) # transverse step size

# waveguide parameters
xb      = 2.0       # size of waveguide
n_cladding  = 1.45      # cladding index
n_core  = 1.46      # core refr. index

# source width
w       = 5.0       # Gaussian beam width

# propagation step size
dz = 0.5
output_step = round(1.0/dz)

# create index distribution
n, x = waveguide(xa, xb, Nx, n_cladding, n_core)

# create initial field
v_in, x     = gauss(xa, Nx, w)
v_in        = v_in/np.sqrt(np.sum(np.abs(v_in)**2)) # normalize power to unity

# calculation
v_out, z = beamprop_CN(v_in, lam, dx, n, nd, z_end, dz, output_step)

plt.figure()
plt.imshow(np.abs(v_out)**2., extent=[min(x), max(x), min(z), max(z)], origin='lower', cmap='bluered_dark')
plt.title('Intensity Distribution')
plt.xlabel(r'x[$\mu$m]')
plt.ylabel(r'z[$\mu$m]')
plt.colorbar()
plt.show()