'''Homework 1, Computational Photonics, SS 2024:  FD mode solver.
'''
import numpy as np
import time
import scipy.sparse as sps

# Task 1: Slab Waveguide

Start_time = time.time()
# 1D Example Parameters
grid_size = 120
number_points = 601
h = grid_size / (number_points - 1)
lam = 0.78
k0 = 2 * np.pi / lam
e_substrate = 2.25
delta_e = 1.5e-2
w = 15.0
xx = np.linspace(-grid_size / 2, grid_size / 2, number_points)
prm = e_substrate + delta_e * np.exp(-(xx / w) ** 2)


def guided_modes_1DTE(prm, k0, h):
    """Computes the effective permittivity of a TE polarized guided eigenmode.
    All dimensions are in µm.
    Note that modes are filtered to match the requirement that
    their effective permittivity is larger than the substrate (cladding).

    Parameters
    ----------
    prm : 1d-array
        Dielectric permittivity in the x-direction
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization

    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated modes
    guided : 2d-array
        Field distributions of the guided eigenmodes
    """
    prm_max = np.max(prm)  # Find the Maximum of prm
    dts = -2 / h ** 2 + k0 ** 2 * prm  # Calculation Diagonal Terms
    adts = 1 / h ** 2  # Calculation adjacent Diagonal Terms

    Dif_Mat = np.zeros((number_points, number_points))  # Creat a Differentiation Matrix with only Zeros
    np.fill_diagonal(Dif_Mat, dts)  # Fill the Diagonal Terms with dts
    np.fill_diagonal(Dif_Mat[1:, :], adts)  # Fill the Adjacent Diagonal Terms below the Main Diagonal with adts
    np.fill_diagonal(Dif_Mat[:, 1:], adts)  # Fill the Adjacent Diagonal Terms above the Main Diagonal with adts

    eigenvalues, eigenvectors = np.linalg.eig(Dif_Mat)  # Calculate the Eigenvalues and Eigenvectors of the Matrix

    eff_eps = eigenvalues[
        (eigenvalues < prm_max) & (eigenvalues > e_substrate)]  # Find the needed effective permittivities
    guided = eigenvectors  # Guided Eigenmodes

    return eff_eps, guided


Effective_permittivities, Field_Distributions = guided_modes_1DTE(prm, k0, h)
print('Effective Permittivities:', Effective_permittivities)
print('Field Distributions:', Field_Distributions)

End_time = time.time()
Running_time = End_time - Start_time
print('Running Time=', Running_time)