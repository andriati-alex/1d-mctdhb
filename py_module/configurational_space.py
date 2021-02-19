import numpy as np
from numba import jit, prange, int32


@jit(int32(int32, int32), nopython=True, nogil=True)
def number_configurations(Npar, Norb):
    """ return (Npar + Norb - 1)! / ( (Npar)! x (Norb - 1)! )"""
    n = 1
    j = 2
    if Norb > Npar:
        for i in prange(Npar + Norb - 1, Norb - 1, -1):
            n = n * i
            if n % j == 0 and j <= Npar:
                n = n / j
                j = j + 1
        for k in prange(j, Npar + 1):
            n = n / k
        return n
    else:
        for i in prange(Npar + Norb - 1, Npar, -1):
            n = n * i
            if n % j == 0 and j <= Norb - 1:
                n = n / j
                j = j + 1
        for k in prange(j, Norb):
            n = n / k
        return n


def combinatorial_mat(N, M):
    """
    CALLING:
    (numpy 2D array of ints) = GetNCmat(N,M)
    N: # of particles
    M: # of orbitals
    Returned matrix NC(n,m) = (n + m - 1)! / ( n! (m - 1)! ).
    """
    comb_mat = np.empty([N + 1, M + 1], dtype=np.int32)
    for i in range(N + 1):
        comb_mat[i][0] = 0
        comb_mat[i][1] = 1
        for j in range(2, M + 1):
            comb_mat[i][j] = number_configurations(i, j)
    return comb_mat


@jit((int32, int32, int32, int32[:]), nopython=True, nogil=True)
def index_to_fock(k, N, M, v):
    """
    k : Index of configuration-state coefficient
    N : # of particles
    M : # of orbitals
    v : End up with occupation vector(Fock state) of length M
    """
    x = 0
    m = M - 1
    for i in prange(0, M, 1):
        v[i] = 0
    while k > 0:
        while k - number_configurations(N, m) < 0:
            m = m - 1
        k = k - number_configurations(N, m)
        v[m] = v[m] + 1
        N = N - 1
    if N > 0:
        v[0] = v[0] + N


@jit(int32(int32, int32, int32[:, :], int32[:]), nopython=True, nogil=True)
def fock_to_index(N, M, comb_mat, v):
    """
    Calling: (int) = FockToIndex(N, M, NCmat, v)
    Return Index of Fock Configuration Coeficient of v
    N     : # of particles
    M     : # of orbitals
    comb_mat : see GetNCmat function
    """
    n = 0
    k = 0
    for i in prange(M - 1, 0, -1):
        n = v[i]
        # Number of particles in the orbital
        while n > 0:
            k = k + comb_mat[N][i]
            # number of combinations needed
            N = N - 1
            # decrease the number of particles
            n = n - 1
    return k


def enum_configurations(N, M):
    """
    Calling : (numpy 2D array of ints) = GetFocks(N, M)
    Row k has the occupation vector corresponding to C[k].
    N : # of particles
    M : # of orbitals
    """
    IF = np.empty([number_configurations(N, M), M], dtype=np.int32)
    for k in range(0, number_configurations(N, M)):
        index_to_fock(k, N, M, IF[k])
    return IF
