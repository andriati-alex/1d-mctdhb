""" Module with core implementation to compute observables

This module provide the basic functions to compute some of the common
observables requested in MCTDHB analysis. This core implementation is
design to have the simplest possible API to require the most essential
parameters. They are implemented in increasing level of complexity as
the first ones requiring multiconfigurational space information and
many-body state coefficients and orbitals

WARNING
These functions are no meant to be used directly after loading data from
simulations but to provide a common interface for classes that read data
from output files of the main program execution

Parameters abbreviation conventions
-----------------------------------

`npar` : number of particles
`norb` : number of orbitals
`coef` : coefficients of many-body state expansion in configurational basis
`rho1` : one-body density matrix
`rho2` : two-body density matrix
`orbitals` : orbitals
`occ` : occupation in natural orbitals
`natorb` : natural orbitals

"""
import numpy as np
import scipy.linalg as linalg

from numba import prange, njit, int32, complex128
from multiconfpy import configurational_space as cs
from multiconfpy import density_matrices as dm
from multiconfpy import function_tools as ft


def onebody_dm(npar, norb, coef):
    """
    Interface for `density_matrices.set_onebody_dm`

    Return
    ------
    ``numpy.ndarray([norb, norb], dtype=np.complex128)``
        Matrix `rho[k, l] = < a*_k, a_l >`
    """
    rho = np.empty([norb, norb], dtype=np.complex128)
    conf_mat = cs.configurational_matrix(npar, norb)
    ht = cs.fock_space_matrix(npar, norb)
    dm.set_onebody_dm(npar, norb, conf_mat, ht, coef, rho)
    return rho


def twobody_dm(npar, norb, coef):
    """
    Interface for `density_matrices.set_twobody_dm`

    Return
    ------
    ``numpy.array(norb ** 4, dtype=np.complex128)``
        `rho[k + s * Norb + q * Norb**2 + l * Norb**3] = < a*_k a*_s a_q a_l >`
    """
    rho = np.empty(norb ** 4, dtype=np.complex128)
    conf_mat = cs.configurational_matrix(npar, norb)
    ht = cs.fock_space_matrix(npar, norb)
    dm.set_twobody_dm(npar, norb, conf_mat, ht, coef, rho)
    return rho


def natural_occupations(npar, rho1):
    """Return occupation fraction in natural orbitals that sum up to 1"""
    eigval, eigvec = linalg.eig(rho1)
    sort_ind = eigval.real.argsort()[::-1]
    return eigval[sort_ind].real / npar


def natural_orbitals(rho1, orbitals):
    """
    Return
    ------
    ``numpy.ndarray(orbitals.shape)``
        Natural orbitals in rows of a matrix
    """
    eigval, eigvec = linalg.eig(rho1)
    sort_ind = eigval.real.argsort()[::-1]
    eigvec_ord = eigvec[:, sort_ind]
    return np.matmul(eigvec_ord.conj().T, orbitals)


def occupation_entropy(occ):
    """Entropy of natural orbital occupations"""
    return -((occ) * np.log(occ)).sum()


def density(occ, natorb, subset_ind=None):
    """
    Total gas density normalized to 1 unless `subset_ind` is specified
    If `subset_ind` is specified return the density restricted to some
    natural orbitals. Useful to separate excitations from condensate
    `subset_ind` : ``list[int] / numpy.array(dtype=int)``
    `0 <= subset_ind[i] < occ.size` without repetitions if given
    """
    if subset_ind is None:
        subset_ind = np.arange(occ.size, dtype=int)
    frac = occ[subset_ind].sum()
    return np.matmul(occ[subset_ind], abs(natorb[subset_ind, :]) ** 2) / frac


def condensate_density(natorb):
    """Density of condensed atoms normalized to 1"""
    return abs(natorb[0]) ** 2


def momentum_density(
    occ, natorb, dx, kmin=-10, kmax=10, bound=0, gf=7, subset_ind=None
):
    """
    Return normalized density of the gas or of specific subset of orbitals
    Restriction in orbitals are possible using `subset_ind` to select some
    orbitals indexes. By default all orbitals are used

    Parameters
    ----------
    `dx` : ``float``
        grid spacing
    `kmin` : ``float``
        minimum momentum cutoff
    `kmax` : ``float``
        maximum momentum cutoff
    `bound` : ``int {0, 1}``
        boundary information
        0 : open
        1 : periodic
    `gf` : ``int {odd}``
        grid amplification factor to improve momentum resolution
    `subset_ind` : ``list[int] / numpy.array(dtype=int)``
        `0 <= subset_ind[i] < occ.size` without repetitions if provided

    Return
    ------
    ``tuple(numpy.array, numpy.array)``
        Frequency values and density of subset of orbitals in momentum space
    """
    if subset_ind is None:
        subset_ind = np.arange(occ.size, dtype=int)
    frac = occ[subset_ind].sum()
    freq, no_fft = ft.fft_ordered_norm(
        ft.extend_grid(natorb, bound, gf), dx, 1, bound
    )
    momentum_den = (
        np.matmul(occ[subset_ind], abs(no_fft[subset_ind, :]) ** 2) / frac
    )
    slice_ind = (freq - kmin) * (freq - kmax) < 0
    freq_cut = freq[slice_ind]
    momentum_den_cut = momentum_den[slice_ind]
    return freq_cut, momentum_den_cut


def position_rdm(occ, natorb):
    """
    Compute the spatial single particle Reduced Density Matrix (1-RDM)
    The convention is the same as G^(1) as (2.4) of reference:

    "Spatial coherence and density correlations of trapped Bose gases"
    Glauber and Naraschewski, Phys Rev A 59, 4595(1999)
    doi : https://doi.org/10.1103/PhysRevA.59.4595

    Return
    ------
    ``numpy.ndarray([self.npts, self.npts])``
        matrix with 1-RDM values at every grid point pair
    """
    diag_occ = np.diag(occ)
    trans = natorb.T
    return np.matmul(trans, np.matmul(diag_occ, natorb.conj()))


def position_onebody_correlation(occ, natorb):
    """
    One body correlation is the 1-RDM weighted by densities
    The convention is the same as g^(1) as (2.16) of reference:

    "Spatial coherence and density correlations of trapped Bose gases"
    Glauber and Naraschewski, Phys Rev A 59, 4595(1999)
    doi : https://doi.org/10.1103/PhysRevA.59.4595

    Return
    ------
    ``numpy.ndarray([self.npts, self.npts])``
        matrix with 1-body correlation values at every grid point pair
    """
    pos_rdm = position_rdm(occ, natorb)
    den = density(occ, natorb)
    den_rows, den_cols = np.meshgrid(den, den)
    return abs(pos_rdm) ** 2 / (den_rows * den_cols)


def momentum_rdm(occ, natorb, dx, kmin=-10, kmax=10, bound=0, gf=7):
    """
    Equivalent to `position_rdm` but in momentum space

    Paramters
    ---------
    `dx` : ``float``
        grid spacing
    `kmin` : ``float``
        min value for the cutoff applied
    `kmax` : ``float``
        max value of cutoff
    `bound` : ``int``
        Must be either 0 (open boundary) or 1 (periodic boundary)
    `gf` : ``int``
        gird expansion factor. Improve the resolution as larger is `gf`

    Return
    ------
    ``tuple(numpy.array, numpy.ndarray)``
        Frequency values and 1-RDM in momentum space
    """
    diag_occ = np.diag(occ)
    freq, no_fft = ft.fft_ordered_norm(
        ft.extend_grid(natorb, bound, gf), dx, 1, bound
    )
    slice_ind = (freq - kmin) * (freq - kmax) < 0
    freq_cut = freq[slice_ind]
    no_fft = no_fft[:, slice_ind]
    trans = no_fft.T
    return freq_cut, np.matmul(trans, np.matmul(diag_occ, no_fft.conj()))


def momentum_onebody_correlation(
    occ, natorb, dx, kmin=-10, kmax=10, bound=0, gf=7
):
    """
    Equivalent to `position_onebody_correlation` but in momentum space

    Paramters
    ---------
    `dx` : ``float``
        grid spacing
    `kmin` : ``float``
        min value for the cutoff applied
    `kmax` : ``float``
        max value of cutoff
    `bound` : ``int``
        Must be either 0 (open boundary) or 1 (periodic boundary)
    `gf` : ``int``
        gird expansion factor. Improve the resolution as larger is `gf`

    Return
    ------
    ``tuple(numpy.array, numpy.ndarray(ndim=2))``
        Frequency values and one-body correlation in momentum space
    """
    mom_rdm = momentum_rdm(occ, natorb, dx, kmin, kmax, bound, gf)[1]
    freq, den = momentum_density(occ, natorb, dx, kmin, kmax, bound, gf)
    den_rows, den_cols = np.meshgrid(den, den)
    return freq, abs(mom_rdm) ** 2 / (den_rows * den_cols)


@njit((int32, int32, complex128[:], complex128[:, :], complex128[:, :]))
def __set_mutual_probability(norb, grid_size, rho2, orbitals, mutprob):
    """Compiled optimized routine to support mutual probability functions"""
    s1 = norb
    s2 = norb ** 2
    s3 = norb ** 3
    # Some auxiliar variables to evaluate sum over all indexes
    o = complex(0)
    r2 = complex(0)
    contract = complex(0)
    for i in prange(grid_size):
        for j in range(grid_size):
            contract = 0
            for k in range(norb):
                for m in range(norb):
                    for q in range(norb):
                        for s in range(norb):
                            rho2_ind = k + m * s1 + q * s2 + s * s3
                            r2 = rho2[rho2_ind]
                            o = (
                                (orbitals[k, j] * orbitals[m, i]).conjugate()
                                * orbitals[q, j]
                                * orbitals[s, i]
                            )
                            contract = contract + r2 * o
            mutprob[i, j] = contract


def position_mutual_probability(npar, rho2, orbitals):
    """
    Mutual probability of finding two particles in abitrary grid points
    This is a two-body operator since it requires 4-fields coupling

    Return
    ------
    ``numpy.ndarray([orbitals.shape[1], orbitals.shape[1]])``
        Square matrix which row `i` col `j` is the probability
        to find two particles at grid points `x[i]` and `x[j]`
    """
    grid_size = orbitals.shape[1]
    norb = orbitals.shape[0]
    mutprob = np.empty([grid_size, grid_size], dtype=np.complex128)
    __set_mutual_probability(norb, grid_size, rho2, orbitals, mutprob)
    return mutprob.real / npar / (npar - 1)


def momentum_mutual_probability(
    npar, rho2, orbitals, dx, kmin=-10, kmax=10, bound=0, gf=7
):
    """
    Equivalent to `position_mutual_probability` but in momentum space

    Paramters
    ---------
    `dx` : ``float``
        grid spacing
    `kmin` : ``float``
        min value for the cutoff applied
    `kmax` : ``float``
        max value of cutoff
    `bound` : ``int``
        Must be either 0 (open boundary) or 1 (periodic boundary)
    `gf` : ``int``
        gird expansion factor. Improve the resolution as larger is `gf`

    Return
    ------
    ``tuple(numpy.array, numpy.ndarray)``
        Frequency values and mutual probability in momentum space
        Element `[i, j]` of mutual probability matrix returned is
        the probability to simultaneous get particles with momenta
        `k[i]` and `k[j]` of the frequency numpy array
    """
    norb = orbitals.shape[0]
    freq, orb_fft = ft.fft_ordered_norm(
        ft.extend_grid(orbitals, bound, gf), dx, 1, bound
    )
    slice_ind = (freq - kmin) * (freq - kmax) < 0
    freq = freq[slice_ind]
    orb_fft = orb_fft[:, slice_ind]
    grid_size = orb_fft.shape[1]
    mutprob = np.empty([grid_size, grid_size], dtype=np.complex128)
    __set_mutual_probability(norb, grid_size, rho2, orb_fft, mutprob)
    return (freq, mutprob.real / npar / (npar - 1))


def position_twobody_correlation(npar, rho2, orbitals, den):
    """
    Result of ``position_mutual_probability`` weighted by 1-body probability
    Regions where it is greater than 1 are likely to have more than 1 particle

    Parameters
    ----------
    `den` : ``numpy.array(orbitals.shape[1])``
        density probability to find a single particle for each grid point
        Can be compute using ``density``

    Return
    ------
    ``numpy.ndarray([orbitals.shape[1], orbitals.shape[1]])``
        Square matrix which row `i` col `j` is the mutual probability
        to find two particles at grid points `x[i]` and `x[j]` weighted
        by the respective individual probability to find only one particle
    """
    mutprob = position_mutual_probability(npar, rho2, orbitals)
    den_rows, den_cols = np.meshgrid(den, den)
    return mutprob / (den_rows * den_cols)


def momentum_twobody_correlation(
    npar, rho2, orbitals, den, dx, kmin=-10, kmax=10, bound=0, gf=7
):
    """
    Result of ``momentum_mutual_probability`` weighted by 1-body probability
    Regions where it is greater than 1 are likely to have more than 1 particle

    Parameters
    ----------
    `den` : ``numpy.array(orbitals.shape[1])``
        density probability to find a single particle for each grid point
        in momentum space from ``momentum_density``
    `dx` : ``float``
        grid spacing
    `kmin` : ``float``
        min value for the cutoff applied
    `kmax` : ``float``
        max value of cutoff
    `bound` : ``int``
        Must be either 0 (open boundary) or 1 (periodic boundary)
    `gf` : ``int``
        gird expansion factor. Improve the resolution as larger is `gf`

    Return
    ------
    ``numpy.ndarray([orbitals.shape[1], orbitals.shape[1]])``
        Square matrix which row `i` col `j` is the mutual probability
        to find two particles at grid points `x[i]` and `x[j]` weighted
        by the respective individual probability to find only one particle
    """
    k, mutprob = momentum_mutual_probability(
        npar, rho2, orbitals, dx, kmin, kmax, bound, gf
    )
    if k.size != den.size:
        raise ValueError("Invalid `den.size`")
    den_rows, den_cols = np.meshgrid(den, den)
    return (k, mutprob / (den_rows * den_cols))


def average_onebody_operator(occ, natorb, op_action, dx, args=(), ind=None):
    """
    Compute many-body average of a extensive 1-particle operator

    Parameters
    ----------
    `op_action` : ``callable(numpy.array, *args) -> numpy.array``
        function to return action of 1-particle operator over a state
        either take some example from ``multiconfpy.operator_action``
        or provide a new one
    `dx` : ``float``
        grid spacing for integration
    `args` : ``tuple``
        extra arguments specific for `op_action` call
    `ind` : ``list[int] / numpy.array(dtype=int)``
        set of orbitals indexes to restrict the average
        by default no restriction is applied (``None``)
        `0 <= ind[i] < occ.size` without repetitions if provided

    Return
    ------
    ``float``
        many-body expectation value
    """
    if ind is None:
        ind = np.arange(occ.size, dtype=np.int32)
    frac = occ[ind].sum()
    sum_contrib = 0.0
    for i in ind:
        sum_contrib += occ[i] * ft.simps(
            natorb[i].conj() * op_action(natorb[i], *args), dx=dx
        )
    return sum_contrib.real / frac


def manybody_operator_covariance(
    npar,
    ob_denmat,
    tb_denmat,
    raw_orb,
    dx,
    opleft,
    opright,
    args_opleft=(),
    args_opright=(),
):
    """Compute manybody covariances given single particle operator functions

    Covariances are defined as the average of product of two fluctuation
    many-body operators. The fluctuation of an arbitrary operator is
    `dO = O - <O>` where `<>` denote the many-body average. Therefore, a
    covariance of operators `O` and `P` is `<dO dP> = <O P> - <O><P>`.
    For computational purposes the action rule of single particle operators
    counterpart are required and eventually extra arguments. Be careful
    with the order passed, as in this example `O` stands for ``opleft``
    and `P` for ``opright``. The covariance is further divided by the
    number of particles square

    Parameters:
    -----------
    `npar` : ``int`` the number for particles to scale the result
    `ob_denmat` : ``numpy.array`` one-body density matrix
    `tb_denmat` : ``numpy.array`` two-body density matrix
    `raw_orb` : ``numpy.array`` working orbitals as matrix rows
    `dx` : ``float`` grid spacing to evaluate integrals
    `opleft` : ``callable`` function to act as single particle operator(`O`)
    `opright` : ``callable`` function to act as single p operator(`P`)
    `args_opleft` : ``tuple`` used to call `opleft(numpy.array, *args)`
    `args_opright` : ``tuple`` used to call `opright(numpy.array, *args)`

    Return:
    -------
    ``float``
    """
    norb = raw_orb.shape[0]
    overlap_mat_left = ft.overlap_meshgrid(
        raw_orb, np.array([opleft(orb, *args_opleft) for orb in raw_orb]), dx
    )
    overlap_mat_right = ft.overlap_meshgrid(
        raw_orb, np.array([opright(orb, *args_opright) for orb in raw_orb]), dx
    )
    tb_denmat_tensor = tb_denmat.reshape((norb, norb, norb, norb), order="F")
    tb_denmat_contract = np.tensordot(
        np.tensordot(tb_denmat_tensor, overlap_mat_left, ([0, 2], [0, 1])),
        overlap_mat_right,
        ([0, 1], [0, 1]),
    )
    comm_contract = np.tensordot(
        np.tensordot(ob_denmat, overlap_mat_left, (0, 0)),
        overlap_mat_right,
        ([0, 1], [1, 0]),
    )
    avg_prod = np.tensordot(
        ob_denmat, overlap_mat_left, ([0, 1], [0, 1])
    ) * np.tensordot(ob_denmat, overlap_mat_right, ([0, 1], [0, 1]))
    return (comm_contract + tb_denmat_contract - avg_prod) / npar ** 2
