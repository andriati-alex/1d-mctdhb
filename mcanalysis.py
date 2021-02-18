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

import cmath
import numpy as np
import scipy.linalg as la

from math import sqrt, pi
from scipy.integrate import simps
from numba import jit, prange, int32, uint32, uint64, int64, float64, complex128


def dfdx(dx, f):
    """
    CALLING :
    (1D array of size of f) = dfdx(dx,f)
    Use finite differences to compute derivative with
    4-th order scheme with periodic boudaries.
    """
    n = f.size
    dfdx = np.zeros(n, dtype=np.complex128)

    dfdx[0] = (f[n - 3] - f[2] + 8 * (f[1] - f[n - 2])) / (12 * dx)
    dfdx[1] = (f[n - 2] - f[3] + 8 * (f[2] - f[0])) / (12 * dx)
    dfdx[n - 2] = (f[n - 4] - f[1] + 8 * (f[0] - f[n - 3])) / (12 * dx)
    dfdx[n - 1] = dfdx[0]
    # assume last point as the boundary

    for i in range(2, n - 2):
        dfdx[i] = (f[i - 2] - f[i + 2] + 8 * (f[i + 1] - f[i - 1])) / (12 * dx)
    return dfdx


def dfdxFFT(dx, f):
    """
    CALLING :
    -------
    (1D array of size of f) = dfdxFFT(dx,f)
    Use Fourier-Transforms to compute derivative for periodic functions
    """
    n = f.size - 1
    k = 2 * pi * np.fft.fftfreq(n, dx)
    dfdx = np.zeros(f.size, dtype=np.complex128)
    dfdx[:n] = np.fft.fft(f[:n], norm="ortho")
    dfdx[:n] = np.fft.ifft(1.0j * k * dfdx[:n], norm="ortho")
    dfdx[n] = dfdx[0]
    return dfdx


def L2normFFT(dx, func, norm=1):
    """
    CALLING :
    freq_vector , fft_of_func = L2normFFT(dx,func,norm);
    Compute FFT and return it ordered together with  the
    frequency vector. The function in Fourier space that
    is returned is normalized by L^2 norm, i.e,  if  one
    integrate in fourier space the square modulus  get 1
    """
    n = func.size
    k = 2 * pi * np.fft.fftfreq(n, dx)
    dk = k[1] - k[0]
    fftfunc = np.fft.fft(func, norm="ortho")

    # mid point between positive and negative frequencies
    if n % 2 == 0:
        j = int(n / 2) - 1
    else:
        j = int((n - 1) / 2)

    # ordered
    k = np.concatenate([k[j + 1 :], k[: j + 1]])
    fftfunc = np.concatenate([fftfunc[j + 1 :], fftfunc[: j + 1]])

    return k, fftfunc * np.sqrt(norm / simps(abs(fftfunc) ** 2, dx=dk))


def DnormFFT(dx, func, norm=1):
    """
    CALLING
    freq_vector , fft_of_func = DnormFFT(dx,func,norm);
    Give the frequency vector and transformed function
    ordered and normalized by Euclidean norm to 1.
    """
    n = func.size
    k = 2 * pi * np.fft.fftfreq(n, dx)
    fftfunc = np.fft.fft(func, norm="ortho")

    # mid point between positive and negative frequencies
    if n % 2 == 0:
        j = int(n / 2) - 1
    else:
        j = int((n - 1) / 2)

    # ordered
    k = np.concatenate([k[j + 1 :], k[: j + 1]])
    fftfunc = np.concatenate([fftfunc[j + 1 :], fftfunc[: j + 1]])

    return k, fftfunc * sqrt(norm / (abs(fftfunc) ** 2).sum())


@jit(uint64(uint64), nopython=True, nogil=True)
def fac(n):
    """ return n! """
    nfac = 1
    for i in prange(2, n + 1):
        nfac = nfac * i
    return nfac


@jit(uint32(uint32, uint32), nopython=True, nogil=True)
def NC(Npar, Norb):
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


@jit((int32, int32, int32, int32[:]), nopython=True, nogil=True)
def IndexToFock(k, N, M, v):
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
        while k - NC(N, m) < 0:
            m = m - 1
        k = k - NC(N, m)
        v[m] = v[m] + 1
        N = N - 1
    if N > 0:
        v[0] = v[0] + N


@jit(int32(int32, int32, int32[:, :], int32[:]), nopython=True, nogil=True)
def FockToIndex(N, M, NCmat, v):
    """
    Calling: (int) = FockToIndex(N, M, NCmat, v)
    Return Index of Fock Configuration Coeficient of v
    N     : # of particles
    M     : # of orbitals
    NCmat : see GetNCmat function
    """
    n = 0
    k = 0
    for i in prange(M - 1, 0, -1):
        n = v[i]
        # Number of particles in the orbital
        while n > 0:
            k = k + NCmat[N][i]
            # number of combinations needed
            N = N - 1
            # decrease the number of particles
            n = n - 1
    return k


@jit((int32, int32, int32[:, :]), nopython=True, nogil=True)
def MountNCmat(N, M, NCmat):
    """ Auxiliar of GetNCmat """
    for i in prange(0, N + 1, 1):
        NCmat[i][0] = 0
        for j in prange(1, M + 1, 1):
            NCmat[i][j] = NC(i, j)


@jit((int32, int32, int32[:, :]), nopython=True, nogil=True)
def MountFocks(N, M, IF):
    """ Auxiliar of GetFocks """
    for k in prange(0, NC(N, M), 1):
        IndexToFock(k, N, M, IF[k])


def GetNCmat(N, M):
    """
    CALLING:
    (numpy 2D array of ints) = GetNCmat(N,M)
    N: # of particles
    M: # of orbitals
    Returned matrix NC(n,m) = (n + m - 1)! / ( n! (m - 1)! ).
    """
    NCmat = np.empty([N + 1, M + 1], dtype=np.int32)
    for i in range(N + 1):
        NCmat[i][0] = 0
        NCmat[i][1] = 1
        for j in range(2, M + 1):
            NCmat[i][j] = NC(i, j)
    return NCmat


def GetFocks(N, M):
    """
    Calling : (numpy 2D array of ints) = GetFocks(N, M)
    Row k has the occupation vector corresponding to C[k].
    N : # of particles
    M : # of orbitals
    """
    IF = np.empty([NC(N, M), M], dtype=np.int32)
    for k in range(0, NC(N, M)):
        IndexToFock(k, N, M, IF[k])
    return IF


@jit(
    (int32, int32, int32[:, :], int32[:, :], complex128[:], complex128[:, :]),
    nopython=True,
    nogil=True,
)
def OBrho(N, M, NCmat, IF, C, rho):

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
    nc = NC(N, M)
    RHO = 0.0 + 0.0j

    for k in prange(0, M, 1):
        # Compute diagonal elements
        RHO = 0.0 + 0.0j
        for i in prange(0, nc, 1):
            mod2 = C[i].real * C[i].real + C[i].imag * C[i].imag
            RHO = RHO + mod2 * IF[i][k]
        rho[k][k] = RHO

        # compute off-diagonal elements
        for l in prange(k + 1, M, 1):
            RHO = 0.0 + 0.0j
            for i in prange(0, nc, 1):
                if IF[i][k] < 1:
                    continue
                IF[i][k] -= 1
                IF[i][l] += 1
                j = FockToIndex(N, M, NCmat, IF[i])
                IF[i][k] += 1
                IF[i][l] -= 1
                RHO = RHO + C[i].conjugate() * C[j] * sqrt((IF[i][l] + 1) * IF[i][k])
            rho[k][l] = RHO
            rho[l][k] = RHO.conjugate()


# End OBrho routine ------------------------------------------------------


@jit(
    (int32, int32, int32[:, :], int32[:, :], complex128[:], complex128[:]),
    nopython=True,
    nogil=True,
)
def TBrho(N, M, NCmat, IF, C, rho):
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
    # index result of conversion FockToIndex
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
    nc = NC(N, M)

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
                j = FockToIndex(N, M, NCmat, v)
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
                j = FockToIndex(N, M, NCmat, v)
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
                j = FockToIndex(N, M, NCmat, v)
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
                    j = FockToIndex(N, M, NCmat, v)
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
                    j = FockToIndex(N, M, NCmat, v)
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
                    j = FockToIndex(N, M, NCmat, v)
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
                    j = FockToIndex(N, M, NCmat, v)
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
                    j = FockToIndex(N, M, NCmat, v)
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
                    j = FockToIndex(N, M, NCmat, v)
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
                        j = FockToIndex(N, M, NCmat, v)
                        RHO += C[i].conjugate() * C[j] * sqrtOf
                    rho[k + s * M + q * M2 + l * M3] = RHO
                # Finish l loop
            # Finish q loop
        # Finish s loop
    # Finish k loop


# END OF rho^2 ROUTINE -----------------------------------------------------


###########################################################################
#                                                                         #
#                                                                         #
#                      ROUTINES TO COLLECT OBSERVABLES                    #
#                      ===============================                    #
#                                                                         #
#                                                                         #
###########################################################################


def GetOBrho(Npar, Morb, C):
    """
    CALLING : ( 2D array [Morb,Morb] ) = GetOBrho(Npar,Morb,C)
    Npar : # of particles
    Morb : # of orbitals (dimension of the matrix returned)
    C    : coefficients of configuration states basis
    return the one-body density matrix rho[k,l] = < a†ₖ aₗ >
    """
    rho = np.empty([Morb, Morb], dtype=np.complex128)
    NCmat = GetNCmat(Npar, Morb)
    IF = GetFocks(Npar, Morb)
    OBrho(Npar, Morb, NCmat, IF, C, rho)
    return rho


def GetTBrho(Npar, Morb, C):
    """
    CALLING : ( 1D array [ Morb⁴ ] ) = GetOBrho(Npar,Morb,C)
    Npar : # of particles
    Morb : # of orbitals (dimension of the matrix returned)
    C    : coefficients of configuration states basis
    return rho[k + s*Morb + n*Morb² + l*Morb³] = < a†ₖ a†ₛ aₙ aₗ >
    """
    rho = np.empty(Morb * Morb * Morb * Morb, dtype=np.complex128)
    NCmat = GetNCmat(Npar, Morb)
    IF = GetFocks(Npar, Morb)
    TBrho(Npar, Morb, NCmat, IF, C, rho)
    return rho


@jit((int32, float64[:], complex128[:, :]), nopython=True, nogil=True)
def EigSort(Nvals, RHOeigvals, RHOeigvecs):
    """
    CALLING : (void) EigSort(Nvals, RHOeigvals, RHOeigvecs)
    Sort the order of eigenvalues to be  decreasing and the order
    of columns of eigenvectors accordingly so  that the  k column
    keep being the eigenvector of k-th eigenvalue. Auxiliar to be
    used in 'getOccupation' function
    -------------------------------------------------------------------
    Nvals      : dimension of rho = # of orbitals
    RHOeigvals : End up with eigenvalues in decreasing order
    RHOeigvecs : Change the order of columns accordingly to eigenvalues
    """
    auxR = 0.0
    auxC = 0.0
    for i in prange(1, Nvals, 1):
        j = i
        while RHOeigvals[j] > RHOeigvals[j - 1] and j > 0:
            # Sort the vector
            auxR = RHOeigvals[j - 1]
            RHOeigvals[j - 1] = RHOeigvals[j]
            RHOeigvals[j] = auxR
            # Sort the matrix
            for k in prange(0, Nvals, 1):
                auxC = RHOeigvecs[k][j - 1]
                RHOeigvecs[k][j - 1] = RHOeigvecs[k][j]
                RHOeigvecs[k][j] = auxC
            j = j - 1


def GetOccupation(Npar, Morb, C):
    """
    CALLING
    (1D array of size Morb) = GetOccupation(Npar,Morb,C)
    Return natural occupations normalized by the number of particles
    ----------------------------------------------------------------
    Npar : # of particles
    Morb : # of orbitals
    C    : coefficients of configurations
    """
    rho = GetOBrho(Npar, Morb, C)
    eigval, eigvec = la.eig(rho)
    EigSort(Morb, eigval.real, eigvec)
    return eigval.real / Npar


def GetNatOrb(Npar, Morb, C, Orb):
    """
    CALLING
    ( 2D numpy array [Morb x Mpos] ) = GetNatOrb(Npar,Morb,C,Orb)
    Npar : # of particles
    Morb : # of orbitals
    C    : coefficients of many-body state in condiguration basis
    Orb  : Matrix with each row a 'working' orbital
    """
    rho = GetOBrho(Npar, Morb, C)
    eigval, eigvec = la.eig(rho)
    EigSort(Morb, eigval.real, eigvec)
    return np.matmul(eigvec.conj().T, Orb)


def GetNatOrb_FromRho(Morb, rho, Orb):
    """
    CALLING
    ( 2D numpy array [Morb x Mpos] ) = GetNatOrb(Npar,Morb,C,Orb)
    Npar : # of particles
    Morb : # of orbitals
    C    : coefficients of many-body state in condiguration basis
    Orb  : Matrix with each row a 'working' orbital
    """
    eigval, eigvec = la.eig(rho)
    EigSort(Morb, eigval.real, eigvec)
    return np.matmul(eigvec.conj().T, Orb)


def GetGasDensity(NOocc, NO):
    """
    CALLING
    ( 1D np array [Mpos] ) = GetGasDensity(NOocc, NO)
    Give the gas density, i.e, probability distribution to find  a
    particle in a position. To collect natural occupations and the
    natural orbitals see function 'GetOccupation' and 'GetNatOrb'
    --------------------------------------------------------------
    NOocc : natural occupations given by GetOccupation
    NO    : Natural orbitals given by GetNatOrb
    """
    return np.matmul(NOocc, abs(NO) ** 2)


def GetGasMomentumDensity(NOocc, NO, dx, bound="zero"):
    """
    CALLING
    ( array freq, array density ) = GetGasMomentumDensity(NOocc,NO,dx,bound)
    Give the gas density in momentum space,  i.e, probability distribution
    to find a particle with given momentum. To collect natural occupations
    and the  natural orbitals see functions 'GetOccupation' and 'GetNatOrb'
    -----------------------------------------------------------------------
    NOocc : natural occupations given by GetOccupation
    NO    : Natural orbitals given by GetNatOrb
    dx    : grid step
    bound : boundary condition, default is zero
    """
    # grid factor to extent the domain without changing the
    # position grid step, to improve resolution in momentum
    # space(reduce the momentum grid step).
    # Shall be an odd number in order to keep  the symmetry
    gf = 15
    Morb = NO.shape[0]
    Mpos = NO.shape[1]

    if bound == "zero":
        extNO = np.zeros([Morb, gf * Mpos], dtype=np.complex128)
        NOfft = np.zeros([Morb, gf * Mpos], dtype=np.complex128)
        for i in range(Morb):
            l = int((gf - 1) / 2)
            k = int((gf + 1) / 2)
            extNO[i, l * Mpos : k * Mpos] = NO[i]
        for i in range(Morb):
            k, NOfft[i] = L2normFFT(dx, extNO[i])
    else:
        extNO = np.zeros([Morb, gf * (Mpos - 1)], dtype=np.complex128)
        NOfft = np.zeros([Morb, gf * (Mpos - 1)], dtype=np.complex128)
        for i in range(Morb):
            for k in range(gf):
                extNO[i, k * (Mpos - 1) : (k + 1) * (Mpos - 1)] = NO[i, :-1]
        for i in range(Morb):
            k, NOfft[i] = DnormFFT(dx, extNO[i])

    denfft = GetGasDensity(NOocc, NOfft)
    return k, denfft


def VonNeumannS(occ):
    """ (double) = VonNeumannS(natural_occupations) """
    N = occ.sum()
    return -((occ / N) * np.log(occ / N)).sum()


@jit(
    (int32, int32, float64[:], complex128[:, :], complex128[:, :]),
    nopython=True,
    nogil=True,
)
def SpatialOBdensity(Morb, Mpos, NOoccu, NO, n):
    """ Inner loop of GetSpatialOBdensity """
    for i in prange(Mpos):
        for j in prange(Mpos):
            for k in prange(Morb):
                n[i, j] = n[i, j] + NOoccu[k] * NO[k, j].conjugate() * NO[k, i]


def GetSpatialOBdensity(NOoccu, NO):
    """
    CALLING
    -------
    ( 2d array [M,M] ) = SpatialOBdensity(occu, NO)
    NOoccu : occu[k] has # of particles occupying NO[k,:] orbital
    NO     : [Morb,M] matrix with each row being a natural orbital
    given two discretized positions Xi and Xj then the entries of
    the matrix returned (let me call n) represent:
    n[i,j] = <  Ψ†(Xj) Ψ(Xi)  >
    """
    Mpos = NO.shape[1]
    Morb = NO.shape[0]
    n = np.zeros([Mpos, Mpos], dtype=np.complex128)
    SpatialOBdensity(Morb, Mpos, NOoccu, NO, n)
    return n


@jit(
    (int32, int32, float64[:], complex128[:, :], float64[:], float64[:, :]),
    nopython=True,
    nogil=True,
)
def OBcorre(Morb, Mpos, NOoccu, NO, GasDen, g):
    """ Inner loop of GetOBcorrelation """
    OrbSum = 0.0
    for i in prange(Mpos):
        for j in prange(Mpos):
            OrbSum = 0.0
            for k in prange(Morb):
                OrbSum = OrbSum + NOoccu[k] * NO[k, j].conjugate() * NO[k, i]
            g[i, j] = abs(OrbSum) ** 2 / (GasDen[i] * GasDen[j])


@jit(
    (int32, int32, float64[:], complex128[:, :], float64[:], float64[:, :]),
    nopython=True,
    nogil=True,
)
def AvoidZero_OBcorre(Morb, Mpos, NOoccu, NO, GasDen, g):
    """ Inner loop of GetOB_momentum_corr """
    OrbSum = 0.0
    for i in prange(Mpos):
        for j in prange(Mpos):
            OrbSum = 0.0
            for k in prange(Morb):
                OrbSum = OrbSum + NOoccu[k] * NO[k, j].conjugate() * NO[k, i]
            if GasDen[i] * GasDen[j] < 1e-28:
                g[i, j] = 0
            else:
                g[i, j] = abs(OrbSum) ** 2 / (GasDen[i] * GasDen[j])


def GetOBcorrelation(NOocc, NO):
    """
    CALLING
    ( 2d array [M,M] ) = GetOBcorrelation(occu, NO)
    NOoccu : occu[k] has # of particles occupying NO[k,:] orbital
    NO     : [Morb,M] matrix with each row being a natural orbital
    --------------------------------------------------------------
    given two discretized positions Xi and Xj then the entries of
    the matrix returned (let me call g) represent:
    g[i,j] = | <  Ψ†(Xj) Ψ(Xi)  > |^2  /  (Den(Xj) * Den(Xi))
    Where Ψ†(Xj) creates a particle at position Xj.
    """
    GasDensity = np.matmul(NOocc, abs(NO) ** 2)
    Morb = NO.shape[0]
    Mpos = NO.shape[1]
    g = np.empty([Mpos, Mpos], dtype=np.float64)
    OBcorre(Morb, Mpos, NOocc, NO, GasDensity, g)
    return g


def GetOBmomentumCorrelation(NOocc, NO, dx, bound="zero"):
    """
    CALLING
    freq, 2D_corr_img = GetOB_momentum_corr(occu,NO,dx,bound='zero')
    NOoccu : occu[k] has # of particles occupying NO[k,:] orbital
    NO     : [Morb,M] matrix with each row being a natural orbital
    dx     : grid position spacing
    bound  : (optional) 'zero' for trapped systems
    -----------------------------------------------------------------
    given two discretized momenta Ki and Kj then the entries of
    the matrix returned (let me call g) represent:
    g[i,j] = | <  φ†(Kj) φ(Ki)  > |^2  /  Den(Kj) * Den(Ki)
    where φ† creates a particle with certain momenta
    """
    # grid factor to extent the domain without changing the
    # position grid step, to improve resolution in momentum
    # space(reduce the momentum grid step).
    # Shall be an odd number in order to keep  the symmetry
    gf = 15
    Morb = NO.shape[0]
    Mpos = NO.shape[1]

    if bound == "zero":
        extNO = np.zeros([Morb, gf * Mpos], dtype=np.complex128)
        NOfft = np.zeros([Morb, gf * Mpos], dtype=np.complex128)
        for i in range(Morb):
            l = int((gf - 1) / 2)
            k = int((gf + 1) / 2)
            extNO[i, l * Mpos : k * Mpos] = NO[i]
        for i in range(Morb):
            k, NOfft[i] = L2normFFT(dx, extNO[i])
    else:
        extNO = np.zeros([Morb, gf * (Mpos - 1)], dtype=np.complex128)
        NOfft = np.zeros([Morb, gf * (Mpos - 1)], dtype=np.complex128)
        for i in range(Morb):
            for k in range(gf):
                extNO[i, k * (Mpos - 1) : (k + 1) * (Mpos - 1)] = NO[i, :-1]
        for i in range(Morb):
            k, NOfft[i] = DnormFFT(dx, extNO[i])

    denfft = GetGasDensity(NOocc, NOfft)
    g = np.empty([k.size, k.size], dtype=np.float64)
    AvoidZero_OBcorre(Morb, k.size, NOocc, NOfft, denfft, g)
    return k, g


@jit(
    (int32, int32, complex128[:], complex128[:, :], complex128[:, :]),
    nopython=False,
    nogil=True,
)
def MutualProb(Morb, Mpos, rho2, S, mutprob):
    """
    Auxiliar function to GetTBCorrelation, computes the contraction
    of 2-body density matrix with the orbitals.
    """
    M = Morb
    M2 = Morb * Morb
    M3 = Morb * M2
    r2 = complex(0)
    o = complex(0)
    Sum = complex(0)
    for i in prange(Mpos):
        for j in prange(Mpos):
            Sum = 0
            for k in prange(Morb):
                for l in prange(Morb):
                    for q in prange(Morb):
                        for s in prange(Morb):
                            r2 = rho2[k + l * M + q * M2 + s * M3]
                            o = (S[q, j] * S[s, i]).conjugate() * S[k, j] * S[l, i]
                            Sum = Sum + r2 * o
            mutprob[i, j] = Sum


def GetTBcorrelation(Npar, Morb, C, S):
    """
    CALLING
    -------
    ( 2d array [Ngrid,Ngrid] ) = GetTBcorrelation(Npar,Morb,C,S)
    where Ngrid means the number of grid points = S.shape[1]
    The two-body correlation here is the probability to simultaneously
    find two particles at grid points Xi and Xj divided by the
    probability to find independently one Xi and another at Xj.
    -----------------------------------------------------------
    Npar : number of particles
    Morb : number of orbitals
    C    : array of coeficients
    S    : Working orbitals organized by rows
    """
    Mpos = S.shape[1]
    NOocc = GetOccupation(Npar, Morb, C)
    NO = GetNatOrb(Npar, Morb, C, S)
    den = GetGasDensity(NOocc, NO)

    # Normalized 2-body density matrix
    rho2 = GetTBrho(Npar, Morb, C) / Npar / (Npar - 1)
    mutprob = np.zeros([Mpos, Mpos], dtype=np.complex128)
    MutualProb(Morb, Mpos, rho2, S, mutprob)
    # sanity check, it mutual probability must be real
    z = abs(mutprob.imag).sum() / Mpos ** 2
    if z > 1e-12:
        print("\nWARNING : Imag part of g2 = %.10lf" % z)

    g2 = np.zeros([Mpos, Mpos], dtype=np.complex128)
    for i in range(Mpos):
        for j in range(Mpos):
            g2[i, j] = mutprob[i, j] / den[i] / den[j]

    return g2


def GetTBmomentumCorr(Npar, Morb, C, S, dx, bound="zero"):
    """
    CALLING
    freqs, 2d array G2 = GetTB_momentum_corr(Npar,Morb,C,S,dx)
    where 'freqs' output is the frequencies the G2 image corresponds.
    --------------------------------------------------------------------
    Npar : number of particles
    Morb : number of orbitals
    C    : array of coeficients
    S    : Working orbitals organized by rows
    dx   : grid step (sample spacing)
    (optional) bound : can be 'zero' or 'periodic'
    """
    Mpos = S.shape[1]
    NOocc = GetOccupation(Npar, Morb, C)
    NO = GetNatOrb(Npar, Morb, C, S)

    # grid factor to extent the domain without changing the
    # position grid step, to improve resolution in momentum
    # space(reduce the momentum grid step).
    # Shall be an odd number in order to keep  the symmetry
    # After the call of this function, it may be  desirable
    # apply a cutoff for high momenta. See function below.
    gf = 7

    if bound == "zero":
        extS = np.zeros([Morb, gf * Mpos], dtype=np.complex128)
        extNO = np.zeros([Morb, gf * Mpos], dtype=np.complex128)
        for i in range(Morb):
            l = int((gf - 1) / 2)
            k = int((gf + 1) / 2)
            extS[i, l * Mpos : k * Mpos] = S[i]
            extNO[i, l * Mpos : k * Mpos] = NO[i]
        Sfft = np.zeros(extS.shape, dtype=np.complex128)
        NOfft = np.zeros(extNO.shape, dtype=np.complex128)
        for i in range(Morb):
            k, Sfft[i] = L2normFFT(dx, extS[i])
            k, NOfft[i] = L2normFFT(dx, extNO[i])
    else:
        extS = np.zeros([Morb, gf * (Mpos - 1)], dtype=np.complex128)
        extNO = np.zeros([Morb, gf * (Mpos - 1)], dtype=np.complex128)
        for i in range(Morb):
            for k in range(gf):
                init = k * (Mpos - 1)
                final = (k + 1) * (Mpos - 1)
                extS[i, init:final] = S[i, : Mpos - 1]
                extNO[i, init:final] = NO[i, : Mpos - 1]
        Sfft = np.zeros(extS.shape, dtype=np.complex128)
        NOfft = np.zeros(extNO.shape, dtype=np.complex128)
        for i in range(Morb):
            k, Sfft[i] = DnormFFT(dx, extS[i])
            k, NOfft[i] = DnormFFT(dx, extNO[i])

    denfft = GetGasDensity(NOocc, NOfft)

    rho2 = GetTBrho(Npar, Morb, C) / Npar / (Npar - 1)
    mutprob = np.zeros([k.size, k.size], dtype=np.complex128)
    MutualProb(Morb, k.size, rho2, Sfft, mutprob)
    # sanity check, it mutual probability must be real
    z = abs(mutprob.imag).sum() / Mpos ** 2
    if z > 1e-12:
        print("\nWARNING : Imag part of g2 = %.10lf" % z)

    g2 = np.zeros([k.size, k.size], dtype=np.complex128)
    for i in range(k.size):
        for j in range(k.size):
            g2[i, j] = mutprob[i, j]
    return k, g2


def Cutoffmomentum(k, g2, denfft):
    """
    Calling
    freq, corr_img = Cutoffmomentum(k,g2,denfft)
    ----------------------------------------------------------
    Given the correlation function as a matrix for discretized
    values, apply a cutoff for momentum with vanishing density
    """
    likely = denfft.max()

    i = 0
    while denfft[i] / likely < 1e-3:
        i = i + 1

    j = denfft.size - 1
    while denfft[j] / likely < 1e-3:
        j = j - 1

    if abs(k[i]) >= abs(k[j]):
        m = i
        n = k.size - 1
        while k[n] >= abs(k[i]):
            n = n - 1
        n = n + 5
        # Add little extra margin
        m = m - 4
    else:
        n = j
        m = 0
        while abs(k[m]) >= k[n]:
            m = m + 1
        m = m - 5
        n = n + 4

    return k[m:n], g2[m:n, m:n]


def TimeOccupation(Npar, Morb, rhotime):
    """
    CALLING :
    ( 2D numpy array [Nsteps, Morb] ) = TimeOccupation(Morb,Nsteps,rhotime)
    Return the natural occupation normalized by the number of particles
    at each time steps as rows of the output matrix
    -----------------------------------------------------------------------
    Npar    : # of particles
    Morb    : # of orbitals (# of columns in rhotime)
    rhotime : each line has row-major matrix representation (a vector)
              that is the  one-body  densiity matrix at each time-step
    """
    Nsteps = rhotime.shape[0]
    eigval = np.empty([Nsteps, Morb], dtype=np.complex128)
    for i in range(Nsteps):
        eigval[i], eigvec = la.eig(rhotime[i].reshape(Morb, Morb))
        EigSort(Morb, eigval[i].real, eigvec)
    return eigval.real / Npar


def TimeDensity(Morb, Mpos, rhotime, S):
    """
    CALLING :
    ( 2D numpy array [Nsteps, Morb] ) = TimeOccupation(Morb,Nsteps,rhotime)
    Return the natural occupation normalized by the number of particles
    at each time steps as rows of the output matrix
    -----------------------------------------------------------------------
    Npar    : # of particles
    Morb    : # of orbitals (# of columns in rhotime)
    rhotime : each line has row-major matrix representation (a vector)
              that is the  one-body  densiity matrix at each time-step
    """
    Nsteps = rhotime.shape[0]
    den = np.zeros([Nsteps, Mpos], dtype=np.float64)
    for i in range(Nsteps):
        eigval, eigvec = la.eig(rhotime[i].reshape(Morb, Morb))
        eigval = eigval.real
        eigval = eigval / eigval.sum()
        EigSort(Morb, eigval, eigvec)
        NO = np.matmul(eigvec.conj().T, S[i].reshape(Morb, Mpos))
        den[i] = GetGasDensity(eigval, NO)
    return den


def current(Morb, rho, S, dx):
    eigval, eigvec = la.eig(rho)
    eigval = eigval.real
    eigval = eigval / eigval.sum()
    EigSort(Morb, eigval, eigvec)
    NO = np.matmul(eigvec.conj().T, S)
    J = np.zeros(NO.shape[1], dtype=np.complex128)
    for k in range(Morb):
        J = (
            J
            - 1.0j
            * (NO[k].conj() * dfdx(dx, NO[k]) - NO[k] * dfdx(dx, NO[k].conj()))
            * eigval[k]
        )
    return J.real


def Timecurrent(Morb, Mpos, rhotime, S, dx):
    Nsteps = rhotime.shape[0]
    J = np.zeros([Nsteps, Mpos], dtype=np.float64)
    for i in range(Nsteps):
        J[i] = current(
            Morb, rhotime[i].reshape(Morb, Morb), S[i].reshape(Morb, Mpos), dx
        )
    return J


@jit(complex128(int32, complex128[:, :], complex128[:], complex128[:, :]))
def avgP2(Morb, rho1, rho2, mat):
    summ = complex(0.0)
    ind = int(0)
    for m in prange(Morb):
        for s in prange(Morb):
            for n in prange(Morb):
                summ = summ + rho1[m, s] * mat[m, n] * mat[n, s]
                for p in prange(Morb):
                    ind = m + s * Morb + n * Morb * Morb + p * Morb * Morb * Morb
                    summ = summ + rho2[ind] * mat[m, n] * mat[s, p]
    return summ


def avgP(Morb, rho1, mat):
    summ = 0.0 + 0.0j
    for m in range(Morb):
        for s in range(Morb):
            summ = summ + rho1[m, s] * mat[m, s]
    return summ


def MomentumVariance(Morb, rho1, rho2, mat):
    return avgP2(Morb, rho1, rho2, mat) - avgP(Morb, rho1, mat) ** 2


def GetMomentumMat(Morb, rho1, rho2, S, dx):
    mat = np.empty([Morb, Morb], dtype=np.complex128)
    for k in range(Morb):
        mat[k, k] = -1.0j * simps(S[k].conj() * dfdx(dx, S[k]), dx=dx)
        for j in range(k + 1, Morb):
            mat[k, j] = -1.0j * simps(S[k].conj() * dfdx(dx, S[j]), dx=dx)
            mat[j, k] = mat[k, j].conj()
    return mat
