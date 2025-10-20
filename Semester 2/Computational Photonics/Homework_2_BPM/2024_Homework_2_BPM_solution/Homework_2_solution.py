"""Solution to the Homework 2 - BPM.
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg


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

    x = np.linspace(-0.5*xa, 0.5*xa, Nx)
    n = np.ones_like(x)*n_cladding
    n[np.abs(x) <= 0.5*xb] = n_core
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
    x = np.linspace(-xa/2, xa/2, Nx)
    v = np.exp(-x**2/w**2)

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
    Nx = len(v_in)
    Nz = round(z_end/dz)
    v = np.zeros((Nz, Nx), dtype=np.complex128)
    v[0, :] = v_in

    k0 = 2*np.pi/lam

    kbar = k0*nd
    one = np.ones((Nx,))
    l1 = sps.spdiags([one, one*-2, one],
                     [-1,       0,   1],
                     Nx, Nx)

    kj = k0*n
    W_diag = (kj**2-kbar**2)/(2*kbar)
    l2 = sps.spdiags(W_diag, 0, Nx, Nx)

    L = 1j/(2*kbar*dx**2) * l1 + 1j * l2

    A = sps.eye(Nx) - 0.5*dz*L
    B = sps.eye(Nx) + 0.5*dz*L

    for m in range(Nz-1):
        v[m+1, :] = sps.linalg.spsolve(A, B.dot(v[m, :]))

    v_out = v[::output_step]
    z = np.arange(0, z_end, dz*output_step)

    return v_out, z


def beamprop_explicit(v_in, lam, dx, n, nd,  z_end, dz, output_step):
    '''Propagates an initial field over a given distance based on the
    solution of the paraxial wave equation in an inhomogeneous
    refractive index distribution using the explicit scheme. 
    All lengths have to be specified in µm.

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
    Nx = len(v_in)
    Nz = round(z_end/dz)
    v = np.zeros((Nz, Nx), dtype=np.complex128)
    v[0, :] = v_in

    k0 = 2*np.pi/lam

    kbar = k0*nd
    one = np.ones((Nx,))
    l1 = sps.spdiags([one, one*-2, one],
                     [-1,       0,   1],
                     Nx, Nx)

    kj = k0*n
    W_diag = (kj**2-kbar**2)/(2*kbar)
    l2 = sps.spdiags(W_diag, 0, Nx, Nx)

    L = 1j/(2*kbar*dx**2) * l1 + 1j * l2

    A = sps.eye(Nx) + dz*L

    for m in range(Nz-1):
        v[m+1, :] = A.dot(v[m, :])

    v_out = v[::output_step]
    z = np.arange(0, z_end, dz*output_step)

    return v_out, z


def beamprop_implicit(v_in, lam, dx, n, nd,  z_end, dz, output_step):
    '''Propagates an initial field over a given distance based on the
    solution of the paraxial wave equation in an inhomogeneous
    refractive index distribution using the implicit scheme. 
    All lengths have to be specified in µm.

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
    Nx = len(v_in)
    Nz = round(z_end/dz)
    v = np.zeros((Nz, Nx), dtype=np.complex128)
    v[0, :] = v_in

    k0 = 2*np.pi/lam

    kbar = k0*nd
    one = np.ones((Nx,))
    l1 = sps.spdiags([one, one*-2, one],
                     [-1,       0,   1],
                     Nx, Nx)

    kj = k0*n
    W_diag = (kj**2-kbar**2)/(2*kbar)
    l2 = sps.spdiags(W_diag, 0, Nx, Nx)

    L = 1j/(2*kbar*dx**2) * l1 + 1j * l2

    A = sps.eye(Nx) - dz*L

    for m in range(Nz-1):
        v[m+1, :] = sps.linalg.spsolve(A, v[m, :])

    v_out = v[::output_step]
    z = np.arange(0, z_end, dz*output_step)

    return v_out, z
