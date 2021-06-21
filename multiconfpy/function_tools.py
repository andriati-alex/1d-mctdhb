""" Module for basic function operations

Observables often require a set of basic calculus and transformation
operations to be evaluated, as derivatives and Fourier transforms. A
set of tools are provided for these manipulations

``normalize(f -> numpy.array, dx -> float, norm -> float)``

``orthonormalize(fun_set -> numpy.ndarray, dx -> float)``

``overlap_matrix(fun_set -> numpy.ndarray, dx -> float)``

``dfdx_periodic(f -> numpy.array, dx -> float)``

``dfdx_zero(f -> numpy.array, dx -> float)``

``dfdx_periodic_fft(f -> numpy.array, dx -> float)``

``fft_ordered_norm(
    func -> numpy.ndarray,
    dx ->float,
    norm ->float,
    bound -> float
)``

"""

import numpy as np
from math import sqrt, pi
from scipy.integrate import simps

# from numba import njit, prange, int32, float64, complex128


def normalize(f, dx, norm=1):
    """Return `f` normalized to `norm` within grid spacing `dx`"""
    old_norm = sqrt(simps(abs(f) ** 2, dx=dx))
    return f * norm / old_norm


def orthonormalize(fun_set, dx):
    """
    Return matrix with orthonormal set of functions using Gram-Schmidt
    applied to `fun_set` within grid spacing `dx`
    """
    nfun = fun_set.shape[0]
    ortho_set = np.empty(fun_set.shape, dtype=fun_set.dtype)
    ortho_set[0] = normalize(fun_set[0], dx)
    for i in range(1, nfun):
        ortho_set[i] = fun_set[i].copy()
        for j in range(i):
            overlap_ji = simps(ortho_set[j].conj() * fun_set[i], dx=dx)
            ortho_set[i] = ortho_set[i] - overlap_ji * ortho_set[j]
        ortho_set[i] = normalize(ortho_set[i], dx)
    return ortho_set


def overlap_matrix(fun_set, dx):
    """
    Compute all pair-wise scalar products for a set of functions

    Parameters
    ----------
    `fun_set` : ``numpy.ndarray(dtype = complex / float)``
        Matrix with each row holding function values in the grid points
    `dx` : ``float``
        grid spacing

    Return
    ------
    ``numpy.ndarray([fun_set.shape[0], fun_set.shape[0], dtype=complex)``
        Matrix with values of each pair-wise scalar products
        The diagonal are the norms of the functions
    """
    nfun = fun_set.shape[0]
    overlap = np.empty([nfun, nfun], dtype=fun_set.dtype)
    for i in range(nfun):
        for j in range(nfun):
            overlap[i, j] = simps(fun_set[i].conj() * fun_set[j], dx=dx)
    return overlap


def dfdx_periodic(f, dx):
    """
    Return derivative of periodic function `f` within grid spacing `dx`
    The convention for periodicity is `f[f.size - 1] == f[0]`. This can
    also be used to open boundaries since it is a particular case where
    the boundaries are both zero. However will not work for hard-walls,
    since the function can go abruptely to zero

    Return
    ------
    ``numpy.array(f.size, dtype=f.dtype)``
    """
    n = f.size
    dfdx = np.empty(n, dtype=f.dtype)
    dfdx[0] = (f[n - 3] - f[2] + 8 * (f[1] - f[n - 2])) / (12 * dx)
    dfdx[1] = (f[n - 2] - f[3] + 8 * (f[2] - f[0])) / (12 * dx)
    dfdx[n - 2] = (f[n - 4] - f[1] + 8 * (f[0] - f[n - 3])) / (12 * dx)
    dfdx[n - 1] = dfdx[0]
    for i in range(2, n - 2):
        dfdx[i] = (f[i - 2] - f[i + 2] + 8 * (f[i + 1] - f[i - 1])) / (12 * dx)
    return dfdx


def dfdx_zero(f, dx):
    """
    Return derivative of periodic function `f` within grid spacing `dx`
    Consider hard-wall boundaries, that implies `f[0] == f[n - 1] == 0`
    or `f[0]` and `f[n - 1]` as the last nonzero values. Especially at
    the boundaries, right- and left-sided difference schemes are used,
    which show better accuracy over central differences

    Return
    ------
    ``numpy.array(f.size, dtype=f.dtype)``
    """
    n = f.size
    dfdx = np.empty(n, dtype=f.dtype)
    dfdx[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * dx)
    dfdx[1] = (-f[3] + 8 * (f[2] - f[0])) / (12 * dx)
    dfdx[n - 2] = (f[n - 4] + 8 * (f[n - 1] - f[n - 3])) / (12 * dx)
    dfdx[n - 1] = (3 * f[n - 1] - 4 * f[n - 2] + f[n - 3]) / (2 * dx)
    for i in range(2, n - 2):
        dfdx[i] = (f[i - 2] - f[i + 2] + 8 * (f[i + 1] - f[i - 1])) / (12 * dx)
    return dfdx


def dfdx_periodic_fft(f, dx):
    """
    Return derivative of `f` using Fast Fourier Transforms(FFT)
    The convention for periodicity is `f[f.size - 1] == f[0]`.
    It work for open boundary conditions but not for hard wall

    Return
    ------
    ``numpy.array(f.size, dtype=complex)``
        Derivative of `f`. Complex datatype even `f.dtype == float`
    """
    n = f.size - 1
    k = 2 * pi * np.fft.fftfreq(n, dx)
    dfdx = np.zeros(f.size, dtype=np.complex128)
    dfdx[:n] = np.fft.fft(f[:n], norm="ortho")
    dfdx[:n] = np.fft.ifft(1.0j * k * dfdx[:n], norm="ortho")
    dfdx[n] = dfdx[0]
    return dfdx


def fft_ordered_norm(func, dx, norm=1, bound=0):
    """
    Compute normalized ordered FFT and corresponding frequencies
    Use quantum mechanical convention frequency 2 * pi * numpy.fft.fftfreq
    Choose the `bound` according to the boundaires of the functions

    Paramters
    ---------
    `func` : ``numpy.array or numpy.ndarray``
        If `numpy.ndarray` given use each row as a function to be transformed
    `dx` : ``float``
        spatial grid spacing
    `norm`(default = 1) : ``float``
        value to normalize the functions in the Fourier space
    `bound`(default = 0) : ``int``
        Type of the problem indicating
        0 : open boundary conditions     : L2 functions norm
        1 : periodic boundary conditions : Eucliean norm

    Return
    ------
    ``tuple(numpy.array, type(func))``
        (frequency vector, fft_ndarray)
    """
    if len(func.shape) == 1:
        work_func = func.reshape(1, func.size)
    else:
        work_func = func.copy()
    grid_pts = work_func.shape[1] - bound
    k = 2 * pi * np.fft.fftfreq(grid_pts, dx)
    dk = k[1] - k[0]
    fk = np.fft.fft(work_func[:, :grid_pts], axis=1, norm="ortho")
    j = (grid_pts - 1) // 2
    k_ord = np.concatenate([k[j + 1 :], k[: j + 1]])
    fk_ord = np.concatenate([fk[:, j + 1 :], fk[:, : j + 1]], 1)
    if bound == 0:
        old_norms = np.sqrt(simps(abs(fk_ord) ** 2, axis=1, dx=dk))
    else:
        old_norms = np.sqrt((abs(fk_ord) ** 2).sum(1))
    for i in range(old_norms.size):
        fk_ord[i] = fk_ord[i] * norm / old_norms[i]
    if len(func.shape) == 1:
        fk_ord = fk_ord[0]
    return k_ord, fk_ord


def extend_grid(func, bound=0, gf=3):
    if gf % 2 == 0:
        gf = gf + 1
    if len(func.shape) == 1:
        work_func = func.reshape(1, func.size)
    else:
        work_func = func.copy()
    nfun = work_func.shape[0]
    npts = work_func.shape[1]
    ext_grid = gf * (npts - bound) + bound
    ext_func = np.zeros([nfun, ext_grid], dtype=func.dtype)
    if bound == 0:
        l = (gf - 1) // 2
        ext_func[:, l * npts : (l + 1) * npts] = work_func
    else:
        ext_func[:, :-1] = np.concatenate(gf * [work_func[:, : npts - 1]], 1)
        ext_func[:, -1] = work_func[:, npts - 1]
    if len(func.shape) == 1:
        ext_func = ext_func[1]
    return ext_func
