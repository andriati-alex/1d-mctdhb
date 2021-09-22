""" Module for high performance computing of density matrices

Density matrices plays a central role in evaluating observables
This module provide performatic routines within numba improving
computations to setup one- and two-body density matrices. These
routines share a common API requiring the expansion coefficients
of the many-body state and some parameters related to conf space

``set_onebody_dm(
    npar     -> int ,
    norb     -> int ,
    conf_mat -> numpy.ndarray([Npar + 1, Norb + 1], dtype=int) ,
    fs       -> numpy.ndarray([nc, Norb], dtype=int) ,
    coef     -> numpy.array(nc, dtype=int) ,
    rho      -> numpy.ndarray([Norb, Norb], dtype=complex)
)``

``set_twobody_dm(
    npar     -> int ,
    norb     -> int ,
    conf_mat -> numpy.ndarray([Npar + 1, Norb + 1], dtype=int) ,
    fs       -> numpy.ndarray([nc, Norb], dtype=int) ,
    coef     -> numpy.array(nc, dtype=int) ,
    rho      -> numpy.array(Norb ** 4, dtype=complex)
)``

"""

from math import sqrt
from numpy import empty
from numba import njit, prange, int32, complex128
from multiconfpy.configurational_space import fock_to_index
from multiconfpy.configurational_space import number_configurations


@njit(
    (int32, int32, int32[:, :], int32[:, :], complex128[:], complex128[:, :])
)
def set_onebody_dm(npar, norb, conf_mat, fs, coef, rho):
    """
    Set the one-body density matrix in the orbital basis

    Paramters
    ---------
    `npar` : ``int32``
        Number of particles
    `norb` : ``int32``
        Number of orbitals
    `conf_mat` : ``numpy.ndarray( [ Npar + 1 , Norb + 1 ] , dtype=int32 )``
        See ``configurational_space.configurational_matrix``
    `fs` : ``numpy.ndarray( [ nc , Norb ] , dtype=int32 )``
        Fock space with all configurations where `nc` is the conf space size
    `coef` : ``numpy.array( nc , dtype=complex128 )``
        coefficients of conf space expansion of the many-body state

    Modified
    --------
    `rho` : ``numpy.ndarray( [ Norb , Norb ] , dtype=complex128 )``
        set the matrix `rho[k, l] = < a*_k, a_l >`
    """
    j = 0
    mod2 = 0.0
    summ = 0.0 + 0.0j
    nc = number_configurations(npar, norb)
    for k in prange(norb):
        summ = 0.0 + 0.0j
        for i in prange(nc):
            mod2 = coef[i].real * coef[i].real + coef[i].imag * coef[i].imag
            summ = summ + mod2 * fs[i][k]
        rho[k][k] = summ
        for s in prange(k + 1, norb, 1):
            summ = 0.0 + 0.0j
            for i in prange(nc):
                if fs[i][k] < 1:
                    continue
                fs[i][k] -= 1
                fs[i][s] += 1
                j = fock_to_index(npar, norb, conf_mat, fs[i])
                fs[i][k] += 1
                fs[i][s] -= 1
                summ = summ + coef[i].conjugate() * coef[j] * sqrt(
                    (fs[i][s] + 1) * fs[i][k]
                )
            rho[k][s] = summ
            rho[s][k] = summ.conjugate()


@njit((int32, int32, int32[:, :], int32[:, :], complex128[:], complex128[:]))
def set_twobody_dm(npar, norb, conf_mat, fs, coef, rho):
    """
    Set the two-body density matrix in the orbital basis.
    A set of rules are required in computing the bosonic
    factor from action of creation/destruction operators.
    Only essential rules are computed and symmetries and
    conjugation properties are used. Check the rules at:

    [1] 'Multiconfigurational time-dependent Hartree method for bosons:
    Many-body dynamics of bosonic systems', Cederbaum et. al, Phys Rev.
    A 77, 033613, doi: http://dx.doi.org/10.1103/PhysRevA.77.033613
    [2] 'Hashing algorithms, optimized mappings and massive parallelization
    of multiconfigurational methods for bosons', A. Andriati and A. Gammal,
    https://arxiv.org/abs/2005.13679

    Paramters
    ---------
    `npar` : ``int32``
        Number of particles
    `norb` : ``int32``
        Number of orbitals
    `conf_mat` : ``numpy.ndarray( [ Npar + 1 , Norb + 1 ] , dtype=int32 )``
        See ``configurational_space.configurational_matrix``
    `fs` : ``numpy.ndarray( [ nc , Norb ] , dtype=int32 )``
        Fock space with all configurations where `nc` is the conf space size
    `coef` : ``numpy.array( nc , dtype=complex128 )``
        coefficients of conf space expansion of the many-body state

    Modified
    --------
    `rho` : ``numpy.array( Norb ** 4 , dtype=complex128 )``
        Vector to set all two-body density matrix entries as:
        `rho[k + s * Norb + q * Norb**2 + l * Norb**3] = < a*_k a*_s a_q a_l >`
        with any of the indexes above in `[ 0 , ... , Norb ]`
    """
    j = 0  # index of new conf after acting with operators
    mod2 = 0.0  # square modulus of coefficients
    bosef = 0.0  # bose factor from operators action
    summ = 0.0 + 0.0j  # summation over the conf space - averages
    v = empty(norb, dtype=int32)
    Norb2 = norb * norb
    Norb3 = norb * norb * norb
    nc = number_configurations(npar, norb)

    # Rule 1: Creation on k k / Annihilation on k k
    for k in prange(norb):
        summ = 0
        for i in prange(nc):
            mod2 = coef[i].real * coef[i].real + coef[i].imag * coef[i].imag
            summ = summ + mod2 * fs[i][k] * (fs[i][k] - 1)
        rho[k + norb * k + Norb2 * k + Norb3 * k] = summ

    # Rule 2: Creation on k s / Annihilation on k s
    for k in prange(norb):
        for s in prange(k + 1, norb):
            summ = 0
            for i in prange(nc):
                mod2 = (
                    coef[i].real * coef[i].real + coef[i].imag * coef[i].imag
                )
                summ += mod2 * fs[i][k] * fs[i][s]
            # commutation of bosonic operators is used
            # to fill elements by exchange  of indexes
            rho[k + s * norb + k * Norb2 + s * Norb3] = summ
            rho[s + k * norb + k * Norb2 + s * Norb3] = summ
            rho[s + k * norb + s * Norb2 + k * Norb3] = summ
            rho[k + s * norb + s * Norb2 + k * Norb3] = summ

    # Rule 3: Creation on k k / Annihilation on q q
    for k in prange(norb):
        for q in prange(k + 1, norb):
            summ = 0
            for i in prange(nc):
                if fs[i][k] < 2:
                    continue
                for t in prange(0, norb, 1):
                    v[t] = fs[i][t]
                bosef = sqrt((v[k] - 1) * v[k] * (v[q] + 1) * (v[q] + 2))
                v[k] -= 2
                v[q] += 2
                j = fock_to_index(npar, norb, conf_mat, v)
                summ += coef[i].conjugate() * coef[j] * bosef
            # Use 2-index-hermiticity
            rho[k + k * norb + q * Norb2 + q * Norb3] = summ
            rho[q + q * norb + k * Norb2 + k * Norb3] = summ.conjugate()

    # Rule 4: Creation on k k / Annihilation on k l
    for k in prange(norb):
        for t in prange(k + 1, norb):
            summ = 0
            for i in prange(nc):
                if fs[i][k] < 2:
                    continue
                for t in prange(norb):
                    v[t] = fs[i][t]
                bosef = (v[k] - 1) * sqrt(v[k] * (v[t] + 1))
                v[k] -= 1
                v[t] += 1
                j = fock_to_index(npar, norb, conf_mat, v)
                summ += coef[i].conjugate() * coef[j] * bosef
            rho[k + k * norb + k * Norb2 + t * Norb3] = summ
            rho[k + k * norb + t * Norb2 + k * Norb3] = summ
            rho[t + k * norb + k * Norb2 + k * Norb3] = summ.conjugate()
            rho[k + t * norb + k * Norb2 + k * Norb3] = summ.conjugate()

    # Rule 5: Creation on k s / Annihilation on s s
    for k in prange(norb):
        for s in prange(k + 1, norb):
            summ = 0
            for i in prange(nc):
                if fs[i][k] < 1 or fs[i][s] < 1:
                    continue
                for t in prange(norb):
                    v[t] = fs[i][t]
                bosef = v[s] * sqrt(v[k] * (v[s] + 1))
                v[k] -= 1
                v[s] += 1
                j = fock_to_index(npar, norb, conf_mat, v)
                summ += coef[i].conjugate() * coef[j] * bosef
            rho[k + s * norb + s * Norb2 + s * Norb3] = summ
            rho[s + k * norb + s * Norb2 + s * Norb3] = summ
            rho[s + s * norb + s * Norb2 + k * Norb3] = summ.conjugate()
            rho[s + s * norb + k * Norb2 + s * Norb3] = summ.conjugate()

    # Rule 6.0: Creation on k k / Annihilation on q l (k < q < l)
    for k in prange(norb):
        for q in prange(k + 1, norb):
            for t in prange(q + 1, norb):
                summ = 0
                for i in prange(nc):
                    if fs[i][k] < 2:
                        continue
                    for t in prange(norb):
                        v[t] = fs[i][t]
                    bosef = sqrt(v[k] * (v[k] - 1) * (v[q] + 1) * (v[t] + 1))
                    v[k] -= 2
                    v[t] += 1
                    v[q] += 1
                    j = fock_to_index(npar, norb, conf_mat, v)
                    summ += coef[i].conjugate() * coef[j] * bosef
                rho[k + k * norb + q * Norb2 + t * Norb3] = summ
                rho[k + k * norb + t * Norb2 + q * Norb3] = summ
                rho[t + q * norb + k * Norb2 + k * Norb3] = summ.conjugate()
                rho[q + t * norb + k * Norb2 + k * Norb3] = summ.conjugate()

    # Rule 6.1: Creation on k k / Annihilation on q l (q < k < l)
    for q in prange(norb):
        for k in prange(q + 1, norb):
            for t in prange(k + 1, norb):
                summ = 0
                for i in prange(nc):
                    if fs[i][k] < 2:
                        continue
                    for t in prange(norb):
                        v[t] = fs[i][t]
                    bosef = sqrt(v[k] * (v[k] - 1) * (v[q] + 1) * (v[t] + 1))
                    v[k] -= 2
                    v[t] += 1
                    v[q] += 1
                    j = fock_to_index(npar, norb, conf_mat, v)
                    summ += coef[i].conjugate() * coef[j] * bosef
                rho[k + k * norb + q * Norb2 + t * Norb3] = summ
                rho[k + k * norb + t * Norb2 + q * Norb3] = summ
                rho[t + q * norb + k * Norb2 + k * Norb3] = summ.conjugate()
                rho[q + t * norb + k * Norb2 + k * Norb3] = summ.conjugate()

    # Rule 6.2: Creation on k k / Annihilation on q l (q < l < k)
    for q in prange(norb):
        for t in prange(q + 1, norb):
            for k in prange(t + 1, norb):
                summ = 0
                for i in prange(nc):
                    if fs[i][k] < 2:
                        continue
                    for t in prange(0, norb, 1):
                        v[t] = fs[i][t]
                    bosef = sqrt(v[k] * (v[k] - 1) * (v[q] + 1) * (v[t] + 1))
                    v[k] -= 2
                    v[t] += 1
                    v[q] += 1
                    j = fock_to_index(npar, norb, conf_mat, v)
                    summ += coef[i].conjugate() * coef[j] * bosef
                rho[k + k * norb + q * Norb2 + t * Norb3] = summ
                rho[k + k * norb + t * Norb2 + q * Norb3] = summ
                rho[t + q * norb + k * Norb2 + k * Norb3] = summ.conjugate()
                rho[q + t * norb + k * Norb2 + k * Norb3] = summ.conjugate()

    # Rule 7.0: Creation on k s / Annihilation on s l (s < k < l)
    for s in prange(norb):
        for k in prange(s + 1, norb):
            for t in prange(k + 1, norb):
                summ = 0
                for i in prange(nc):
                    if fs[i][k] < 1 or fs[i][s] < 1:
                        continue
                    for t in prange(0, norb, 1):
                        v[t] = fs[i][t]
                    bosef = v[s] * sqrt(v[k] * (v[t] + 1))
                    v[k] -= 1
                    v[t] += 1
                    j = fock_to_index(npar, norb, conf_mat, v)
                    summ += coef[i].conjugate() * coef[j] * bosef
                rho[k + s * norb + s * Norb2 + t * Norb3] = summ
                rho[s + k * norb + s * Norb2 + t * Norb3] = summ
                rho[s + k * norb + t * Norb2 + s * Norb3] = summ
                rho[k + s * norb + t * Norb2 + s * Norb3] = summ
                rho[t + s * norb + s * Norb2 + k * Norb3] = summ.conjugate()
                rho[s + t * norb + s * Norb2 + k * Norb3] = summ.conjugate()
                rho[s + t * norb + k * Norb2 + s * Norb3] = summ.conjugate()
                rho[t + s * norb + k * Norb2 + s * Norb3] = summ.conjugate()

    # Rule 7.1: Creation on k s / Annihilation on s l (k < s < l)
    for k in prange(norb):
        for s in prange(k + 1, norb):
            for t in prange(s + 1, norb):
                summ = 0
                for i in prange(nc):
                    if fs[i][k] < 1 or fs[i][s] < 1:
                        continue
                    for t in prange(0, norb, 1):
                        v[t] = fs[i][t]
                    bosef = v[s] * sqrt(v[k] * (v[t] + 1))
                    v[k] -= 1
                    v[t] += 1
                    j = fock_to_index(npar, norb, conf_mat, v)
                    summ += coef[i].conjugate() * coef[j] * bosef
                rho[k + s * norb + s * Norb2 + t * Norb3] = summ
                rho[s + k * norb + s * Norb2 + t * Norb3] = summ
                rho[s + k * norb + t * Norb2 + s * Norb3] = summ
                rho[k + s * norb + t * Norb2 + s * Norb3] = summ
                rho[t + s * norb + s * Norb2 + k * Norb3] = summ.conjugate()
                rho[s + t * norb + s * Norb2 + k * Norb3] = summ.conjugate()
                rho[s + t * norb + k * Norb2 + s * Norb3] = summ.conjugate()
                rho[t + s * norb + k * Norb2 + s * Norb3] = summ.conjugate()

    # Rule 7.2: Creation on k s / Annihilation on s l (k < l < s)
    for k in prange(norb):
        for t in prange(k + 1, norb):
            for s in prange(t + 1, norb):
                summ = 0
                for i in prange(nc):
                    if fs[i][k] < 1 or fs[i][s] < 1:
                        continue
                    for t in prange(0, norb, 1):
                        v[t] = fs[i][t]
                    bosef = v[s] * sqrt(v[k] * (v[t] + 1))
                    v[k] -= 1
                    v[t] += 1
                    j = fock_to_index(npar, norb, conf_mat, v)
                    summ += coef[i].conjugate() * coef[j] * bosef
                rho[k + s * norb + s * Norb2 + t * Norb3] = summ
                rho[s + k * norb + s * Norb2 + t * Norb3] = summ
                rho[s + k * norb + t * Norb2 + s * Norb3] = summ
                rho[k + s * norb + t * Norb2 + s * Norb3] = summ
                rho[t + s * norb + s * Norb2 + k * Norb3] = summ.conjugate()
                rho[s + t * norb + s * Norb2 + k * Norb3] = summ.conjugate()
                rho[s + t * norb + k * Norb2 + s * Norb3] = summ.conjugate()
                rho[t + s * norb + k * Norb2 + s * Norb3] = summ.conjugate()

    # Rule 8: Creation on k s / Annihilation on q l
    for k in prange(norb):
        for s in prange(norb):
            if s == k:
                continue
            for q in prange(norb):
                if q == s or q == k:
                    continue
                for t in prange(norb):
                    summ = 0
                    if t == k or t == s or t == q:
                        continue
                    for i in prange(nc):
                        if fs[i][k] < 1 or fs[i][s] < 1:
                            continue
                        for t in prange(0, norb, 1):
                            v[t] = fs[i][t]
                        bosef = sqrt(v[k] * v[s] * (v[q] + 1) * (v[t] + 1))
                        v[k] -= 1
                        v[s] -= 1
                        v[q] += 1
                        v[t] += 1
                        j = fock_to_index(npar, norb, conf_mat, v)
                        summ += coef[i].conjugate() * coef[j] * bosef
                    rho[k + s * norb + q * Norb2 + t * Norb3] = summ
# Finish two-body density matrix routine --------------------------------------
