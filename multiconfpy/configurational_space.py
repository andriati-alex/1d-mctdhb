""" Module for numerically manipulation of multiconfigurational spaces

For numerical approach of a multiconfigurational problem some basic
tools are required to manipulate numerically fock states defined by
configuring particles in single particle states a.k.a orbitals.

From a truncate/finite number of orbitals a set of functions are defined
to enumerate uniquely every possible Fock state known as perfect hashing
function.

Useful references
    [1] 'A perfect Hashing function for exact diagonalization
    of many-body systems of identical particles', Shoudan Liang,
    Computer Physics Communications vol. 92, iss. 1, 1995 pp. 11-15,
    https://doi.org/10.1016/0010-4655(95)00108-R
    [2] 'Hashing algorithms, optimized mappings and massive parallelization
    of multiconfigurational methods for bosons', A. Andriati and A. Gammal,
    https://arxiv.org/abs/2005.13679

Module functions prototypes:

``number_configurations(
    npar -> int,
    norb ->int
)``

``configurational_matrix(
    npar -> int,
    norb -> int
)``

``index_to_fock(
    conf_ind -> int,
    npar -> int,
    norb -> int,
    occ_vec -> numpy.array
)``

``fock_to_index(
    npar -> int,
    norb -> int,
    conf_mat -> numpy.ndarray,
    occ_vec -> numpy.array
)``

``fock_space_matrix(
    npar -> int,
    norb -> int
)``

"""

import numpy as np
from numba import njit, prange, int32, int64


@njit([int64(int64), int32(int32)])
def fac(n):
    """return n!"""
    nfac = 1
    for i in prange(2, n + 1):
        nfac = nfac * i
    return nfac


@njit([int32(int32, int32), int64(int64, int64)])
def number_configurations(npar, norb):
    """
    Return number of configurations that define the Hilbert space
    dimension for `npar` particles in a truncated basis of `norb`
    single particle states a.k.a orbitals
    """
    n = 1
    j = 2
    if norb > npar:
        for i in prange(npar + norb - 1, norb - 1, -1):
            n = n * i
            if n % j == 0 and j <= npar:
                n = n / j
                j = j + 1
        for k in prange(j, npar + 1):
            n = n / k
        return n
    for i in prange(npar + norb - 1, npar, -1):
        n = n * i
        if n % j == 0 and j <= norb - 1:
            n = n / j
            j = j + 1
    for k in prange(j, norb):
        n = n / k
    return n


def configurational_matrix(npar, norb):
    """
    Return the number of configurations (a.k.a Hilbert space dimension)
    for all cases with the number of particles and orbitals constrained
    by `npar` and `Norb` respectively

    Return
    ------
    ``numpy.ndarray[npar + 1, Norb + 1]``
        Matrix of integers with `mat[n, m] = number_configurations(n, m)`
    """
    conf_mat = np.empty([npar + 1, norb + 1], dtype=np.int32)
    for i in range(npar + 1):
        conf_mat[i][0] = 0  # forbidden : no orbitals at all
        conf_mat[i][1] = 1  # single orbital : condensed
        for j in range(2, norb + 1):
            conf_mat[i][j] = number_configurations(i, j)
    return conf_mat


@njit((int32, int32, int32, int32[:]))
def index_to_fock(conf_ind, npar, Norb, occ_vec):
    """
    Set vector with occupation numbers from configuration index

    [1] 'A perfect Hashing function for exact diagonalization
    of many-body systems of identical particles', Shoudan Liang,
    Computer Physics Communications vol. 92, iss. 1, 1995 pp. 11-15,
    https://doi.org/10.1016/0010-4655(95)00108-R
    [2] 'Hashing algorithms, optimized mappings and massive parallelization
    of multiconfigurational methods for bosons', A. Andriati and A. Gammal,
    https://arxiv.org/abs/2005.13679

    Parameters
    ----------
    `ind` : ``int``
        index of the configuration from 0  to
        `number_configurations(npar, Nobr)-1`
    `npar` : ``int``
        Number of particles
    `Norb` : ``int``
        Number of orbitals

    Modify
    ------
    `occ_vec` : ``numpy.array[int32]``
        Array of integers with number of particles in each orbital
        Must have size `Norb`
    """
    m = Norb - 1
    for i in prange(0, Norb):
        occ_vec[i] = 0
    while conf_ind > 0:
        while conf_ind - number_configurations(npar, m) < 0:
            m = m - 1
        conf_ind = conf_ind - number_configurations(npar, m)
        occ_vec[m] = occ_vec[m] + 1
        npar = npar - 1
    if npar > 0:
        occ_vec[0] = occ_vec[0] + npar


@njit(int32(int32, int32, int32[:, :], int32[:]))
def fock_to_index(Npar, Norb, conf_mat, occ_vec):
    """
    Compute the configuration index from vector of occupation numbers

    [1] 'A perfect Hashing function for exact diagonalization
    of many-body systems of identical particles', Shoudan Liang,
    Computer Physics Communications vol. 92, iss. 1, 1995 pp. 11-15,
    https://doi.org/10.1016/0010-4655(95)00108-R
    [2] 'Hashing algorithms, optimized mappings and massive parallelization
    of multiconfigurational methods for bosons', A. Andriati and A. Gammal,
    https://arxiv.org/abs/2005.13679

    Paramters
    ---------
    `npar` : ``int32``
        Number of particles
    `Norb` : ``int32``
        Number of orbitals
    `conf_mat` : ``numpy.ndarray[ npar + 1 , Norb + 1 ]``
        matrix with all configurational results from `configurational_matrix`
    `occ_vec` : ``numpy.array[ Norb ]``
        Array of integers with number of particles in each orbital

    Return
    ------
    ``int32``
        The configuration index correspoding to `occ_vec`
    """
    k = 0
    for i in prange(Norb - 1, 0, -1):
        n = occ_vec[i]
        while n > 0:
            k = k + conf_mat[Npar][i]
            Npar = Npar - 1
            n = n - 1
    return k


def fock_space_matrix(npar, norb):
    """
    Set and return a matrix with all occupation vectors along lines

    Parameters
    ----------
    `npar` : ``int``
        Number of particles
    `Norb` : ``int``
        Number of orbitals

    Return
    ------
    ``numpy.ndarray[ number_configurations(npar, Norb), Norb ]``
        Matrix of integers, where each row provide occupation vector
        corresponding to its configurational index as the row number
    """
    fock_mat = np.empty(
        [number_configurations(npar, norb), norb], dtype=np.int32
    )
    for k in range(number_configurations(npar, norb)):
        index_to_fock(k, npar, norb, fock_mat[k])
    return fock_mat
