"""
=============================================================================


    MODULE DIRECTED TO COLLECT OBSERVABLES FROM MCTDHB OUTPUT
    ---------------------------------------------------------

    The functions contained in this module support the analysis of results
    from (imaginary/real)time propagation. Some of the functions  are  the
    same as those used in C language. Numba package usage may cause  delay
    when imported the first time because it need to compile the functions.


=============================================================================
"""

import numpy as np
from math import sqrt
from numba import jit, prange, int32, complex128
from configurational_space import fock_to_index, number_configurations



@jit(
    (int32, int32, int32[:, :], int32[:, :], complex128[:], complex128[:, :]),
    nopython=True,
    nogil=True,
)
def nb_obrho(N, M, NCmat, IF, C, rho):
    """
    CALLING :
    (void) OBrho(N,M,NCmat,IF,C,rho)
    N     : # of particles
    M     : # of orbitals (also dimension of rho)
    NCmat : see function GetNCmat
    IF    : see function GetFocks
    C     : coeficients of Fock-configuration states
    rho   : Empty M x M matrix. End up configured with values
    """
    # Initialize variables
    j = 0
    mod2 = 0.0
    nc = number_configurations(N, M)
    RHO = 0.0 + 0.0j
    for k in prange(0, M, 1):
        RHO = 0.0 + 0.0j
        for i in prange(0, nc, 1):
            mod2 = C[i].real * C[i].real + C[i].imag * C[i].imag
            RHO = RHO + mod2 * IF[i][k]
        rho[k][k] = RHO
        for l in prange(k + 1, M, 1):
            RHO = 0.0 + 0.0j
            for i in prange(0, nc, 1):
                if IF[i][k] < 1:
                    continue
                IF[i][k] -= 1
                IF[i][l] += 1
                j = fock_to_index(N, M, NCmat, IF[i])
                IF[i][k] += 1
                IF[i][l] -= 1
                RHO = RHO + C[i].conjugate() * C[j] * sqrt(
                    (IF[i][l] + 1) * IF[i][k]
                )
            rho[k][l] = RHO
            rho[l][k] = RHO.conjugate()


@jit(
    (int32, int32, int32[:, :], int32[:, :], complex128[:], complex128[:]),
    nopython=True,
    nogil=True,
)
def nb_tbrho(N, M, NCmat, IF, C, rho):
    """
    CALLING :
    (void) OBrho(N,M,NCmat,IF,C,rho)
    Setup rho argument with two-body density matrix
    N     : # of particles
    M     : # of orbitals (also dimension of rho)
    NCmat : see function GetNCmat
    IF    : see function GetFocks
    C     : coeficients of Fock-configuration states
    rho   : Empty M^4 array . End up configured with values
    """
    # index result of conversion fock_to_index
    j = int(0)
    # occupation vector
    v = np.empty(M, dtype=np.int32)
    mod2 = float(0)
    # |Cj| ^ 2
    sqrtOf = float(0)
    # Factors from the action of creation/annihilation
    RHO = float(0)
    # Auxiliar to memory access of two-body matrix
    M2 = M * M
    M3 = M * M * M
    nc = number_configurations(N, M)

    # Rule 1: Creation on k k / Annihilation on k k
    for k in prange(0, M, 1):
        RHO = 0
        for i in prange(0, nc, 1):
            mod2 = C[i].real * C[i].real + C[i].imag * C[i].imag
            RHO = RHO + mod2 * IF[i][k] * (IF[i][k] - 1)
        rho[k + M * k + M2 * k + M3 * k] = RHO
    # Rule 2: Creation on k s / Annihilation on k s
    for k in prange(0, M, 1):
        for s in prange(k + 1, M, 1):
            RHO = 0
            for i in prange(0, nc, 1):
                mod2 = C[i].real * C[i].real + C[i].imag * C[i].imag
                RHO += mod2 * IF[i][k] * IF[i][s]
            # commutation of bosonic operators is used
            # to fill elements by exchange  of indexes
            rho[k + s * M + k * M2 + s * M3] = RHO
            rho[s + k * M + k * M2 + s * M3] = RHO
            rho[s + k * M + s * M2 + k * M3] = RHO
            rho[k + s * M + s * M2 + k * M3] = RHO
    # Rule 3: Creation on k k / Annihilation on q q
    for k in prange(0, M, 1):
        for q in prange(k + 1, M, 1):
            RHO = 0
            for i in prange(0, nc, 1):
                if IF[i][k] < 2:
                    continue
                for t in prange(0, M, 1):
                    v[t] = IF[i][t]
                sqrtOf = sqrt((v[k] - 1) * v[k] * (v[q] + 1) * (v[q] + 2))
                v[k] -= 2
                v[q] += 2
                j = fock_to_index(N, M, NCmat, v)
                RHO += C[i].conjugate() * C[j] * sqrtOf
            # Use 2-index-'hermiticity'
            rho[k + k * M + q * M2 + q * M3] = RHO
            rho[q + q * M + k * M2 + k * M3] = RHO.conjugate()
    # Rule 4: Creation on k k / Annihilation on k l
    for k in prange(0, M, 1):
        for l in prange(k + 1, M, 1):
            RHO = 0
            for i in prange(0, nc, 1):
                if IF[i][k] < 2:
                    continue
                for t in prange(0, M, 1):
                    v[t] = IF[i][t]
                sqrtOf = (v[k] - 1) * sqrt(v[k] * (v[l] + 1))
                v[k] -= 1
                v[l] += 1
                j = fock_to_index(N, M, NCmat, v)
                RHO += C[i].conjugate() * C[j] * sqrtOf
            rho[k + k * M + k * M2 + l * M3] = RHO
            rho[k + k * M + l * M2 + k * M3] = RHO
            rho[l + k * M + k * M2 + k * M3] = RHO.conjugate()
            rho[k + l * M + k * M2 + k * M3] = RHO.conjugate()
    # Rule 5: Creation on k s / Annihilation on s s
    for k in prange(0, M, 1):
        for s in prange(k + 1, M, 1):
            RHO = 0
            for i in prange(0, nc, 1):
                if IF[i][k] < 1 or IF[i][s] < 1:
                    continue
                for t in prange(0, M, 1):
                    v[t] = IF[i][t]
                sqrtOf = v[s] * sqrt(v[k] * (v[s] + 1))
                v[k] -= 1
                v[s] += 1
                j = fock_to_index(N, M, NCmat, v)
                RHO += C[i].conjugate() * C[j] * sqrtOf
            rho[k + s * M + s * M2 + s * M3] = RHO
            rho[s + k * M + s * M2 + s * M3] = RHO
            rho[s + s * M + s * M2 + k * M3] = RHO.conjugate()
            rho[s + s * M + k * M2 + s * M3] = RHO.conjugate()
    # Rule 6.0: Creation on k k / Annihilation on q l (k < q < l)
    for k in prange(0, M, 1):
        for q in prange(k + 1, M, 1):
            for l in prange(q + 1, M, 1):
                RHO = 0
                for i in prange(0, nc, 1):
                    if IF[i][k] < 2:
                        continue
                    for t in prange(0, M, 1):
                        v[t] = IF[i][t]
                    sqrtOf = sqrt(v[k] * (v[k] - 1) * (v[q] + 1) * (v[l] + 1))
                    v[k] -= 2
                    v[l] += 1
                    v[q] += 1
                    j = fock_to_index(N, M, NCmat, v)
                    RHO += C[i].conjugate() * C[j] * sqrtOf
                rho[k + k * M + q * M2 + l * M3] = RHO
                rho[k + k * M + l * M2 + q * M3] = RHO
                rho[l + q * M + k * M2 + k * M3] = RHO.conjugate()
                rho[q + l * M + k * M2 + k * M3] = RHO.conjugate()
    # Rule 6.1: Creation on k k / Annihilation on q l (q < k < l)
    for q in prange(0, M, 1):
        for k in prange(q + 1, M, 1):
            for l in prange(k + 1, M, 1):
                RHO = 0
                for i in prange(0, nc, 1):
                    if IF[i][k] < 2:
                        continue
                    for t in prange(0, M, 1):
                        v[t] = IF[i][t]
                    sqrtOf = sqrt(v[k] * (v[k] - 1) * (v[q] + 1) * (v[l] + 1))
                    v[k] -= 2
                    v[l] += 1
                    v[q] += 1
                    j = fock_to_index(N, M, NCmat, v)
                    RHO += C[i].conjugate() * C[j] * sqrtOf
                rho[k + k * M + q * M2 + l * M3] = RHO
                rho[k + k * M + l * M2 + q * M3] = RHO
                rho[l + q * M + k * M2 + k * M3] = RHO.conjugate()
                rho[q + l * M + k * M2 + k * M3] = RHO.conjugate()
    # Rule 6.2: Creation on k k / Annihilation on q l (q < l < k)
    for q in prange(0, M, 1):
        for l in prange(q + 1, M, 1):
            for k in prange(l + 1, M, 1):
                RHO = 0
                for i in prange(0, nc, 1):
                    if IF[i][k] < 2:
                        continue
                    for t in prange(0, M, 1):
                        v[t] = IF[i][t]
                    sqrtOf = sqrt(v[k] * (v[k] - 1) * (v[q] + 1) * (v[l] + 1))
                    v[k] -= 2
                    v[l] += 1
                    v[q] += 1
                    j = fock_to_index(N, M, NCmat, v)
                    RHO += C[i].conjugate() * C[j] * sqrtOf
                rho[k + k * M + q * M2 + l * M3] = RHO
                rho[k + k * M + l * M2 + q * M3] = RHO
                rho[l + q * M + k * M2 + k * M3] = RHO.conjugate()
                rho[q + l * M + k * M2 + k * M3] = RHO.conjugate()
    # Rule 7.0: Creation on k s / Annihilation on s l (s < k < l)
    for s in prange(0, M, 1):
        for k in prange(s + 1, M, 1):
            for l in prange(k + 1, M, 1):
                RHO = 0
                for i in prange(0, nc, 1):
                    if IF[i][k] < 1 or IF[i][s] < 1:
                        continue
                    for t in prange(0, M, 1):
                        v[t] = IF[i][t]
                    sqrtOf = v[s] * sqrt(v[k] * (v[l] + 1))
                    v[k] -= 1
                    v[l] += 1
                    j = fock_to_index(N, M, NCmat, v)
                    RHO += C[i].conjugate() * C[j] * sqrtOf
                rho[k + s * M + s * M2 + l * M3] = RHO
                rho[s + k * M + s * M2 + l * M3] = RHO
                rho[s + k * M + l * M2 + s * M3] = RHO
                rho[k + s * M + l * M2 + s * M3] = RHO
                rho[l + s * M + s * M2 + k * M3] = RHO.conjugate()
                rho[s + l * M + s * M2 + k * M3] = RHO.conjugate()
                rho[s + l * M + k * M2 + s * M3] = RHO.conjugate()
                rho[l + s * M + k * M2 + s * M3] = RHO.conjugate()
    # Rule 7.1: Creation on k s / Annihilation on s l (k < s < l)
    for k in prange(0, M, 1):
        for s in prange(k + 1, M, 1):
            for l in prange(s + 1, M, 1):
                RHO = 0
                for i in prange(0, nc, 1):
                    if IF[i][k] < 1 or IF[i][s] < 1:
                        continue
                    for t in prange(0, M, 1):
                        v[t] = IF[i][t]
                    sqrtOf = v[s] * sqrt(v[k] * (v[l] + 1))
                    v[k] -= 1
                    v[l] += 1
                    j = fock_to_index(N, M, NCmat, v)
                    RHO += C[i].conjugate() * C[j] * sqrtOf
                rho[k + s * M + s * M2 + l * M3] = RHO
                rho[s + k * M + s * M2 + l * M3] = RHO
                rho[s + k * M + l * M2 + s * M3] = RHO
                rho[k + s * M + l * M2 + s * M3] = RHO
                rho[l + s * M + s * M2 + k * M3] = RHO.conjugate()
                rho[s + l * M + s * M2 + k * M3] = RHO.conjugate()
                rho[s + l * M + k * M2 + s * M3] = RHO.conjugate()
                rho[l + s * M + k * M2 + s * M3] = RHO.conjugate()
    # Rule 7.2: Creation on k s / Annihilation on s l (k < l < s)
    for k in prange(0, M, 1):
        for l in prange(k + 1, M, 1):
            for s in prange(l + 1, M, 1):
                RHO = 0
                for i in prange(0, nc, 1):
                    if IF[i][k] < 1 or IF[i][s] < 1:
                        continue
                    for t in prange(0, M, 1):
                        v[t] = IF[i][t]
                    sqrtOf = v[s] * sqrt(v[k] * (v[l] + 1))
                    v[k] -= 1
                    v[l] += 1
                    j = fock_to_index(N, M, NCmat, v)
                    RHO += C[i].conjugate() * C[j] * sqrtOf
                rho[k + s * M + s * M2 + l * M3] = RHO
                rho[s + k * M + s * M2 + l * M3] = RHO
                rho[s + k * M + l * M2 + s * M3] = RHO
                rho[k + s * M + l * M2 + s * M3] = RHO
                rho[l + s * M + s * M2 + k * M3] = RHO.conjugate()
                rho[s + l * M + s * M2 + k * M3] = RHO.conjugate()
                rho[s + l * M + k * M2 + s * M3] = RHO.conjugate()
                rho[l + s * M + k * M2 + s * M3] = RHO.conjugate()
    # Rule 8: Creation on k s / Annihilation on q l
    for k in prange(0, M, 1):
        for s in prange(0, M, 1):
            if s == k:
                continue
            for q in prange(0, M, 1):
                if q == s or q == k:
                    continue
                for l in prange(0, M, 1):
                    RHO = 0
                    if l == k or l == s or l == q:
                        continue
                    for i in prange(0, nc, 1):
                        if IF[i][k] < 1 or IF[i][s] < 1:
                            continue
                        for t in prange(0, M, 1):
                            v[t] = IF[i][t]
                        sqrtOf = sqrt(v[k] * v[s] * (v[q] + 1) * (v[l] + 1))
                        v[k] -= 1
                        v[s] -= 1
                        v[q] += 1
                        v[l] += 1
                        j = fock_to_index(N, M, NCmat, v)
                        RHO += C[i].conjugate() * C[j] * sqrtOf
                    rho[k + s * M + q * M2 + l * M3] = RHO
                # Finish l loop
            # Finish q loop
        # Finish s loop
    # Finish k loop


# Finish two-body density matrix routine
