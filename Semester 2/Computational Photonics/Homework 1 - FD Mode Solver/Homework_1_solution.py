'''Homework 1, Computational Photonics, SS 2024:  FD mode solver.
'''
import numpy as np
from numpy import linalg 
from matplotlib import pyplot as plt
import scipy.sparse as sps
from scipy.sparse.linalg import eigs
from scipy.sparse import spdiags


#### Task 1: Guided modes in a slab waveguide system
grid_size     = 120
number_points = 601
h             = grid_size/(number_points - 1)
lam           = 0.78
k0            = 2*np.pi/lam
e_substrate   = 2.25
delta_e       = 1.5e-2
w             = 15.0
xx            = np.linspace( -grid_size/2, grid_size/2, number_points )
prm           = e_substrate + delta_e * np.exp(-(xx/w)**2)

#Calculate maximum value of permittivity
prm_max = np.max(prm)
print("Maximum value of prm:", prm_max)

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
    
    #### Calculation matrix 
    # Defining diagonals
    md = -2/(h**2) + k0**2*prm # matrix diagonal 
    ad = 1/(h**2) # adjacent diagonal

    # Modify the diagonal elements
    
    e = np.diag(md)
    #Create adjacent diagonal matrices with values from ad
    ad_minus_one = np.diag(np.full(number_points - 1, ad), k=-1)
    ad_one = np.diag(np.full(number_points - 1, ad), k=1)

    # Add adjacent diagonals to the diagonal matrix e
    e += ad_minus_one + ad_one
 
    print(e)

    # Calculate eigenvalues, eingenvectors of effective permittivity matrix
    eigenvalues, eigenvectors = linalg.eig(e) #needs to be more specified I think, since most of the eigenvalues are way higher than wanted
    
    # now select eigenvalues for condition e_substrate < eff_prm < prm_max
    eff_eps = eigenvalues[(eigenvalues > e_substrate) & (eigenvalues < prm_max)]

    # field given by eigenvectors
    guided = eigenvectors

    return eff_eps, guided

selected_eigenvalues, field = guided_modes_1DTE(prm, k0, h)
print("Selected eigenvalues:", selected_eigenvalues)


#### Task 2: Guided modes in a strip waveguidesystem system
#2D Example Parameters
grid_size     = 120
number_points = 301
h             = grid_size/(number_points - 1)
lam           = 0.78
k0            = 2*np.pi/lam
e_substrate   = 2.25
delta_e       = 1.5e-2
w             = 15.0
xx            = np.linspace(-grid_size/2-h,grid_size/2+h,number_points+2)
yy            = np.linspace(-grid_size/2,grid_size/2,number_points)
XX,YY         = np.meshgrid(xx,yy)
prm           = e_substrate + delta_e * np.exp(-(XX**2+YY**2)/w**2)


def guided_modes_2D(prm, k0, h, numb):
    """Computes the effective permittivity of a quasi-TE polarized guided 
    eigenmode. All dimensions are in µm.
    
    Parameters
    ----------
    prm  : 2d-array
        Dielectric permittivity in the xy-plane
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization
    numb : int
        Number of eigenmodes to be calculated
    
    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated eigenmodes
    guided : 3d-array
        Field distributions of the guided eigenmodes
    """

    # dummy version of sparse matrix
    number = 4
    # set starting value for diagonals which become different values, 0, +-1 and every nth diagonal is changed
    x = []
    for i in range(-number+1, number):
        n = number * i
        if i == 0:
            x.append(-1)
        if i == 1:
            x.append(1)
        x.append(n)
        
    print(x)

    'defining valuables for  diagonals'
    diags = []
    for i in range(-number, number+1):
        if i == 0:
            diags.append([-4] * (number**2)) # 0th diagonal set to -4
        else:
            diags.append([1] * (number**2)) # every else set to 1
    print(diags)

    # set the nth position of d/dx diagonals (1 & -1 index) to 0 for the edge positions !!! not correct problem with counting!!
    for i in range(1,number):
        diags[number-1][(number)*i] = 0  # Adjacent diagonal -1
        # diags[number+1][number*i] = 0  # Adjacent diagonal 1 


    d = spdiags(diags, x, number**2, number**2).toarray()
    print(d)
    pass
