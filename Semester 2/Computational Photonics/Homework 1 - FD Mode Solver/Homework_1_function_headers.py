'''Homework 1, Computational Photonics, SS 2024:  FD mode solver.
'''
import numpy as np
import time
import matplotlib.pyplot as plt

#Task 1: Slab Waveguide

start_time=time.time()
#1D Example Parameters
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
    prm_max = np.max(prm)              # Find the Maximum of prm
    dts = -2 / (h ** 2 * k0 ** 2) + prm  # Calculation Diagonal Terms
    adts = 1 / (h ** 2 * k0 ** 2)                  # Calculation adjacent Diagonal Terms

    dif_Mat = np.zeros((number_points,number_points)) # Creat a Differentiation Matrix with only Zeros
    np.fill_diagonal(dif_Mat, dts)                    # Fill the Diagonal Terms with dts
    np.fill_diagonal(dif_Mat[1:,:], adts)             # Fill the Adjacent Diagonal Terms below the Main Diagonal with adts
    np.fill_diagonal(dif_Mat[:,1:], adts)             # Fill the Adjacent Diagonal Terms above the Main Diagonal with adts

    eigenvalues, eigenvectors = np.linalg.eig(dif_Mat)  # Calculate the Eigenvalues and Eigenvectors of the Matrix

    eff_eps = eigenvalues[(eigenvalues < prm_max) & (eigenvalues > e_substrate)] # Find the needed effective permittivities
    guided = []
    for i in eff_eps:
        indices = np.where(np.isclose(eigenvalues, i))[0]
        for index in indices:
            guided.append(eigenvectors[:, index])

    return eff_eps, guided

effective_permittivities, field_distributions = guided_modes_1DTE(prm, k0, h) # Use the def to calculate results

#for effective_permittivities, field_distributions in zip(effective_permittivities, field_distributions):
#    print(f"Effective Permittivity: {effective_permittivities}")
#    print("Filed Distribution:", field_distributions)
for field_distribution in field_distributions: # Plot field distributions
    plt.plot(xx, field_distribution)
plt.show()


end_time = time.time()
running_time = end_time - start_time # Calculate the time used to run the codes
print('Running Time=', running_time)


'''
#Task 2: Strip Waveguide

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
    
    Computes the effective permittivity of a quasi-TE polarized guided
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


    Total_points=number_points*number_points # Total number of a single Grid
    dts = -4 / h ** 2 + k0 ** 2 * prm        # Calculation Diagonal Terms
    adts = 1 / h ** 2                        # Calculation adjacent Diagonal Terms

    Dif_Mat = np.zeros((Total_points, Total_points)) # Creat the Differentiation Matrix with Zeros
    for i in range(Total_points):
        for j in range(Total_points):
            if i == j:
                Dif_Mat[i,j] = dts
    '''



