import numpy as np
from math import sqrt, pi
from scipy.integrate import simps
from numba import njit, prange, int32, float64, complex128


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


@njit((int32, float64[:], complex128[:, :]), nogil=True)
def nb_eigen_sort(Nvals, eigvals, eigvecs):
    auxR = 0.0
    auxC = 0.0
    for i in prange(1, Nvals, 1):
        j = i
        while eigvals[j] > eigvals[j - 1] and j > 0:
            # Sort the vector
            auxR = eigvals[j - 1]
            eigvals[j - 1] = eigvals[j]
            eigvals[j] = auxR
            # Sort the matrix
            for k in prange(0, Nvals, 1):
                auxC = eigvecs[k][j - 1]
                eigvecs[k][j - 1] = eigvecs[k][j]
                eigvecs[k][j] = auxC
            j = j - 1


@njit(
    (int32, int32, float64[:], complex128[:, :], complex128[:, :]),
    nogil=True,
)
def nb_position_rdm_sum(norb, npts, nat_occ, nat_orb, rho):
    orb_sum = complex(0, 0)
    for i in prange(npts):
        for j in prange(npts):
            orb_sum = 0.0
            for k in prange(norb):
                orb_sum = (
                    orb_sum
                    + nat_occ[k] * nat_orb[k, j].conjugate() * nat_orb[k, i]
                )
            rho[i, j] = orb_sum
