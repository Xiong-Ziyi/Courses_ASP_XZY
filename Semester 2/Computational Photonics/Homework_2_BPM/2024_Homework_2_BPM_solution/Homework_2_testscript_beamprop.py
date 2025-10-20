'''Test script for homework 2 (Crank-Nicolson scheme)
'''

import numpy as np
from Homework_2_solution import waveguide, gauss, beamprop_CN, beamprop_explicit, beamprop_implicit
from matplotlib import pyplot as plt
import time

# mode solver for comparison with waveguide mode
from Homework_1_solution import guided_modes_1DTE

# dark bluered colormap, registers automatically with matplotlib on import
#import bluered_dark

plt.rcParams.update({
        'figure.figsize': (12/2.54, 9/2.54),
        'figure.subplot.bottom': 0.15,
        'figure.subplot.left': 0.165,
        'figure.subplot.right': 0.925,
        'figure.subplot.top': 0.9,
        'axes.grid': True,
})
plt.close('all')
save_figures = False

# %%
# computational parameters
z_end   = 100       # propagation distance
lam     = 1         # wavelength
nd      = 1.455     # referecne index
xa      = 50        # size of computational window
Nx      = 251       # number of transverse points
dx      = xa/(Nx-1) # transverse step size

# waveguide parameters
xb      = 2.0       # size of waveguide
n_clad  = 1.45      # cladding index
n_core  = 1.46      # core refr. index

# source width
w       = 5.0       # Gaussian beam width

# choose which schemes to calculate
# (explicit scheme may cause problems due to limited number space)
calc_CN = True
calc_ex = True
calc_im = True

# propagation step size
dz = 0.5
output_step = int(np.round(1/dz))

# create index distribution
n, x = waveguide(xa, xb, Nx, n_clad, n_core)

# create initial field
v_in, x = gauss(xa, Nx, w)
v_in = v_in/np.sqrt(np.sum(np.abs(v_in)**2)) # normalize power to unity

# calculate fundamental waveguide mode for comparison
eff_eps, guided = guided_modes_1DTE(n**2,2*np.pi/lam, dx)
v_ref = guided[:,0]
ind = np.argmax(np.abs(v_ref)) # find index of max
v_ref = v_ref*np.exp(-1j*np.angle(v_ref[ind])) # set phase of max to zero
v_ref = v_ref/np.sqrt(np.sum(np.abs(v_ref)**2)) # normalize power to unity

overlap = np.sum(v_in*np.conj(v_ref))
print('Fundamental waveguide mode:\n'
      '\tn_eff = {0:g}\n\toverlapp = {1:g}'.format(
              np.sqrt(eff_eps[0]), np.abs(overlap)))

# %% plot initial field, fundamental waveguide mode and index distribution
delta_n = n_core - n_clad
fig = plt.figure()
plt.plot(x, np.real(v_in)/np.max(np.abs(v_ref))*delta_n,
         label='input field')
plt.plot(x, np.real(v_ref)/np.max(np.abs(v_ref))*delta_n,
         label='fundamental\nwaveguide mode')
plt.plot(x, n - n_clad,'k-',
         label='index difference')
plt.xlim(x[[0, -1]])
plt.ylim(np.array([-0.05, 1.05])*(n_core - n_clad))
plt.grid(True)
plt.xlabel('lateral position [µm]')
plt.ylabel(' ')
plt.legend(frameon=False)
plt.show()

if save_figures:
    fig.savefig('waveguide.png', dpi=1000)

# %% propagate field with CN scheme
if calc_CN:
    t = time.time()
    v_out_CN, z = beamprop_CN(v_in, lam, dx, n, nd, z_end, dz, output_step)
    print('elapsed time: {0:g}'.format(time.time() - t))
    
    fig = plt.figure()
    # value range for colorbar
    v = np.abs(v_out_CN).max()
    # meshgrid for contourf
    X,Z = np.meshgrid(x,z)
    plt.contourf(Z, X, v_out_CN.real, np.linspace(-v , v, 101),
                 cmap='bluered_dark', zorder=-5)
    cb = plt.colorbar()
    plt.xlabel('propagation distance [µm]')
    plt.ylabel('lateral position [µm]')
    plt.title('Crank-Nicolson scheme')
    cb.set_label('real part of field')
    #cb.set_ticks(np.linspace(-0.3, 0.3, 7))
    plt.show()
    
    if save_figures:
        fig.savefig('BPM_CN.png', dpi=1000)
    
# %% propagate field with explicit scheme
if calc_ex:
    t = time.time()
    v_out_ex, z = beamprop_explicit(v_in, lam, dx, n, nd, z_end, dz, output_step)
    print('elapsed time: {0:g}'.format(time.time() - t))
    
    fig = plt.figure()
    # value range for colorbar
    v = np.abs(v_out_ex).max()
    # meshgrid for contourf
    X,Z = np.meshgrid(x,z)
    plt.contourf(Z, X, v_out_ex.real, np.linspace(-v , v, 101),
                 cmap='bluered_dark', zorder=-5)
    
    cb = plt.colorbar()
    plt.xlabel('propagation distance [µm]')
    plt.ylabel('lateral position [µm]')
    plt.title('explicit FTCS scheme')
    cb.set_label('real part of field')
    #cb.set_ticks(np.linspace(-0.3, 0.3, 7))
    plt.show()
    
    if save_figures:
        fig.savefig('BPM_explicit.png', dpi=1000)
    
# %% propagate field with implicit scheme
if calc_im:
    t = time.time()
    v_out_im, z = beamprop_implicit(v_in, lam, dx, n, nd,
                                      z_end, dz, output_step)
    print('elapsed time: {0:g}'.format(time.time() - t))
    
    fig = plt.figure()
    # value range for colorbar
    v = np.abs(v_out_im).max()
    # meshgrid for contourf
    X,Z = np.meshgrid(x,z)
    plt.contourf(Z, X, v_out_im.real, np.linspace(-v , v, 101),
                 cmap='bluered_dark', zorder=-5)
    
    cb = plt.colorbar()
    plt.xlabel('propagation distance [µm]')
    plt.ylabel('lateral position [µm]')
    plt.title('implicit BTCS scheme')
    cb.set_label('real part of field')
    #cb.set_ticks(np.linspace(-0.3, 0.3, 7))
    plt.show()
    
    if save_figures:
        fig.savefig('BPM_implicit.png', dpi=1000)
    
# %% plot energy over propagation distance z
fig = plt.figure()
if calc_CN:
    plt.plot(z, np.sum(np.abs(v_out_CN)**2,1),
         label='Crank-Nicolson')
if calc_ex:
    plt.plot(z, np.sum(np.abs(v_out_ex)**2,1),
         label='explicit')
if calc_im:
    plt.plot(z, np.sum(np.abs(v_out_im)**2,1),
         label='implicit')
plt.xlim(np.array([0, z_end]))
plt.ylim(np.array([0.9995, 1.0005]))
plt.grid(True)
plt.xlabel('Propagation distance [µm]')
plt.ylabel('Energy')
plt.legend(frameon=False)
plt.show()

if save_figures:
    fig.savefig('BPM_energy.png', dpi=1000)
    