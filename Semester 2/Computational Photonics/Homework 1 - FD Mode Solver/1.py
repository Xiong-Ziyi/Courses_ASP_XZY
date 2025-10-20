import numpy as np
import scipy.sparse as sp
'''
number_points=6
dts=1
adts=2
Matrix=np.zeros((number_points,number_points))
np.fill_diagonal(Matrix, dts)
np.fill_diagonal(Matrix[1:,:], adts)
#np.fill_diagonal(Matrix[:,1:], adts)

eigenvalues, eigenvectors=np.linalg.eig(Matrix)
print(Matrix)


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

prm_max=np.max(prm)
print(prm_max)


# Define your square matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Compute eigenvalues
eigenvalues, _ = np.linalg.eig(matrix)

# Define the threshold digit
threshold_digit = 20

# Filter eigenvalues smaller than the threshold digit
filtered_eigenvalues = eigenvalues[(eigenvalues < threshold_digit) & (eigenvalues > 0)]

# Print filtered eigenvalues
print("Eigenvalues smaller than", threshold_digit, ":", filtered_eigenvalues)
print(eigenvalues)



def construct_2d_differentiation_matrix(n, h):
    """
    Constructs a 2D differentiation matrix for a square grid with n points in each dimension.
    n: Number of points in each dimension
    h: Grid spacing
    """
    # Total number of grid points
    N = n * n

    # Construct the differentiation matrix
    D = np.zeros((N, N))

    # Populate the differentiation matrix
    for i in range(N):
        for j in range(N):
            # Diagonal elements
            if i == j:
                D[i, j] = -4 / (h ** 2)  # Laplacian operator for second derivative
            # Off-diagonal elements for neighboring grid points
            elif (i // n == j // n and abs(i - j) == 1) or (abs(i - j) == n):
                D[i, j] = 1 / (h ** 2)  # Laplacian operator for second derivative

    return D


# Example usage:
n = 3  # Number of points in each dimension
h = 1.0  # Grid spacing
D = construct_2d_differentiation_matrix(n, h)
print("2D Differentiation Matrix:")
print(D)




from scipy.sparse import diags

n = 3
"""
    Create a 2D Laplacian matrix for an n x n grid.

    Args:
    n (int): The number of points in one dimension.

    Returns:
    scipy.sparse.csr_matrix: The n^2 x n^2 Laplacian matrix for the grid.
"""
# Main diagonal values (4 for each grid point, accounting for 4 neighbors)
diagonal = np.ones(n*n) * 4

# Off-diagonal values (-1 for the left, right, up, and down neighbors)
off_diagonal = np.ones(n*n-1)

# Additional diagonal for blocks (-1 for up and down block connections)
far_off_diagonal = np.ones(n*n-n)

# Creation of the sparse matrix in diagonal format
laplacian = diags(
    [diagonal, off_diagonal, off_diagonal, far_off_diagonal, far_off_diagonal],
    [0, -1, 1, -n, n], format="csr")

# Fix the wrap-around points which do not actually connect in a grid
for i in range(1, n):
        laplacian[i*n, i*n-1] = 0
        laplacian[i*n-1, i*n] = 0



# Example of creating a Laplacian for a 5x5 grid
n = 3

print(laplacian.toarray())
'''

'''Homework 1, Computational Photonics, SS 2024: FD mode solver.

import numpy as np
import time
import scipy.sparse as sps

# Task 1: Slab Waveguide

start_time = time.time()  # Variable names in Python should be lowercase

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
prm = e_substrate + delta_e * np.exp(-(xx / w)**2)

def guided_modes_1DTE(prm, k0, h):
    """
    Computes the effective permittivity of a TE polarized guided eigenmode.
    All dimensions are in µm.
    Note that modes are filtered to match the requirement that
    their effective permittivity is larger than the substrate (cladding).
    """
    prm_max = np.max(prm)
    diagonal_terms = -2 / (h**2 * k0**2) + prm  # More descriptive variable names
    off_diagonal_terms = 1 / (h**2 * k0**2)

    # Use sparse matrices for better performance with large grid sizes
    diagonal = np.full(number_points, diagonal_terms)
    off_diagonal = np.full(number_points - 1, off_diagonal_terms)
    Dif_Mat = sps.diags([diagonal, off_diagonal, off_diagonal], [0, -1, 1], format='csr')

    # Compute eigenvalues and eigenvectors using sparse matrix operations
    eigenvalues, eigenvectors = sps.linalg.eigs(Dif_Mat, k=5, which='LR')  # Assuming you need the 5 largest real parts

    eff_eps = eigenvalues.real[(eigenvalues.real < prm_max) & (eigenvalues.real > e_substrate)]
    indices = np.isin(np.round(eigenvalues.real, decimals=8), eff_eps)
    guided = eigenvectors[:, indices]

    return eff_eps, guided

effective_permittivities, field_distributions = guided_modes_1DTE(prm, k0, h)
print('Effective Permittivities:', effective_permittivities)
print('Field Distributions:', field_distributions)

end_time = time.time()
running_time = end_time - start_time  # Descriptive variable naming
print('Running Time =', running_time)

'''

m=np.zeros((5, 5))
np.fill_diagonal(m[:, :], 3)
print(m)