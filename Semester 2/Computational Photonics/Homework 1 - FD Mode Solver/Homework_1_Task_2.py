#Task 2: Strip Waveguide
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt


#2D Example Parameters
grid_size     = 120
number_points = 3
h             = grid_size/(number_points - 1)
lam           = 0.78
k0            = 2*np.pi/lam
e_substrate   = 2.25
delta_e       = 1.5e-2
w             = 15.0
xx            = np.linspace(-grid_size/2-h,grid_size/2+h,number_points)#+2
yy            = np.linspace(-grid_size/2,grid_size/2,number_points)
XX,YY         = np.meshgrid(xx,yy)
prm           = e_substrate + delta_e * np.exp(-(XX**2+YY**2)/w**2)

def guided_modes_2D(prm, k0, h):
    '''
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
    '''
    nps = number_points
    total_points = nps * nps  # Total number of points in the 1d matrix/ Numerber of points in one dimension of the 2D matrix
    dts = -4 #/ (h ** 2 * k0 ** 2) + prm.flatten() # Calculation Diagonal Terms
    adts = 1 #/ (h ** 2 * k0 ** 2)        # Calculation adjacent Diagonal Terms
    prm_max = np.max(prm)

    main_diag=dts


    adj_diag=adts*np.ones(total_points-1)
    adj_diag[np.arange(1, total_points)%nps==0]=0

    adj_grid=adts*np.ones(total_points-nps)

    diagonals=[main_diag, adj_diag, adj_diag, adj_grid, adj_grid]

    lap_mat=diags(diagonals, [0, -1, 1, -nps, nps], shape=(total_points, total_points))

    eigenvalues, eigenvectors=eigs(lap_mat)

    val_indices=(eigenvalues<prm_max)&(eigenvalues>e_substrate)
    eff_eps=eigenvalues[val_indices]
    val_eigvecs=eigenvectors[:, val_indices]

    guided=[]
    for val_eigvec in val_eigvecs.T:
        guided.append(val_eigvec.reshape(nps, nps))

    return eff_eps, guided, lap_mat

effective_permittivities, field_distributions, lap_mat= guided_modes_2D(prm, k0, h)
print(lap_mat.toarray())

#2D
#for field in field_distributions:
#    plt.figure()
#    plt.imshow(np.abs(field), extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap='viridis')
#    plt.show()

#3D

#for index, field in enumerate(field_distributions):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    surf = ax.plot_surface(XX, YY, np.abs(field), cmap='viridis')
#    plt.show()