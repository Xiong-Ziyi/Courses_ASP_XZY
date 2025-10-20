'''
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import csr_matrix

nps=3
ttnb=nps**2

main_diag=4 * np.ones(ttnb)
print("main_diag:", main_diag)

side_diag=np.ones(ttnb-1)
side_diag[np.arange(1, ttnb) % (nps+1) == 0] = 0
padded_side_diag = np.concatenate([side_diag, [0]])
print('side_diag:', side_diag)

vert_diag=np.ones(ttnb-nps)
padded_vert_diag = np.concatenate([vert_diag, np.zeros(nps)])
print('vert_diag:', vert_diag)

diagonals = [vert_diag, side_diag, main_diag, side_diag, vert_diag]
offset=[nps, 1, 0, -1, -nps]

laplacian=spdiags(diagonals, [nps, 1, 0, -1, -nps], ttnb, ttnb, format='csr')

print(laplacian.toarray())

'''
import numpy as np
from scipy.sparse import diags


def laplacian_2d_sparse(n):
    # Create the diagonals
    main_diag = -4 * np.ones(n * n)
    side_diag = np.ones(n * n - 1)
    side_diag[np.arange(1, n * n) % n == 0] = 0
    up_down_diag = np.ones(n * n - n)

    # Create the sparse matrix
    diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
    laplacian_sparse = diags(diagonals, [0, -1, 1, -n, n], shape=(n * n, n * n))

    return laplacian_sparse


# Test the function
n = 3
print(laplacian_2d_sparse(n).toarray())




