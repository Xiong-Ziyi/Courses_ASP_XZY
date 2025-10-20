'''Homework 2, Computational Photonics, SS 2020:  Beam propagation method.
'''
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def waveguide(xa, xb, Nx, n_cladding, n_core):
    '''Generates the refractive index distribution of a slab waveguide
    with step profile centered around the origin of the coordinate
    system with a refractive index of n_core in the waveguide region
    and n_cladding in the surrounding cladding area.
    All lengths have to be specified in µm.

    Parameters
    ----------
        xa : float
            Width of calculation window
        xb : float
            Width of waveguide
        Nx : int
            Number of grid points
        n_cladding : float
            Refractive index of cladding
        n_core : float
            Refractive index of core

    Returns
    -------
        n : 1d-array
            Generated refractive index distribution
        x : 1d-array
            Generated coordinate vector
    '''
    # Creat coordinates
    x=np.linspace(-xa/2, xa/2, Nx)

    #Creat n distribution
    n=np.ones_like(x)*n_cladding
    n[(x>=-xb/2) & (x<=xb/2)]=n_core

    return n, x






def gauss(xa, Nx, w):
    '''Generates a Gaussian field distribution v = exp(-x^2/w^2) centered
    around the origin of the coordinate system and having a width of w.
    All lengths have to be specified in µm.

    Parameters
    ----------
        xa : float
            Width of calculation window
        Nx : int
            Number of grid points
        w  : float
            Width of Gaussian field

    Returns
    -------
        v : 1d-array
            Generated field distribution
        x : 1d-array
            Generated coordinate vector
    '''
    x = np.linspace(-xa / 2, xa / 2, Nx)
    v=np.exp(-x**2/(w**2))

    return v, x





def beamprop_CN(v_in, lam, dx, n, nd,  z_end, dz, output_step):
    '''Propagates an initial field over a given distance based on the
    solution of the paraxial wave equation in an inhomogeneous
    refractive index distribution using the explicit-implicit
    Crank-Nicolson scheme. All lengths have to be specified in µm.

    Parameters
    ----------
        v_in : 1d-array
            Initial field
        lam : float
            Wavelength
        dx : float
            Transverse step size
        n : 1d-array
            Refractive index distribution
        nd : float
            Reference refractive index
        z_end : float
            Propagation distance
        dz : float
            Step size in propagation direction
        output_step : int
            Number of steps between field outputs

    Returns
    -------
        v_out : 2d-array
            Propagated field
        z : 1d-array
            z-coordinates of field output
    '''

    kx=(2 * np.pi / lam) * n
    k_bar=(2 * np.pi / lam) * nd

    wx=(kx ** 2 - k_bar ** 2) / 2 * k_bar

    main_diag=np.ones(len(wx))*-2
    off_diag=np.ones(len(wx)-1)

    L1=sp.diags([main_diag, off_diag, off_diag], [0, -1, 1], format='csr')
    L1=(1j/(2 * k_bar * dx**2))*L1

    L2=sp.diags(wx, 0, format='csr')
    L2=1j*L2

    L=L1+L2

    A=sp.eye(len(wx), format='csr') - (dz/2) * L
    B=sp.eye(len(wx), format='csr') + (dz/2) * L

    z_steps=int(z_end/dz)
    v_out=np.zeros((z_steps // output_step + 1, len(v_in) ), dtype=complex)
    z=np.zeros(z_steps // output_step + 1)

    v=v_in
    v_out[0, :]=v
    z[0]=0

    for i in range(1, z_steps+1):
        v=spla.spsolve(A, B.dot(v))

        if i % output_step == 0:
            v_out[i//output_step, :]=v
            z[i//output_step]=i*dz

    return v_out, z