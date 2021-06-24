"""Module with core implementation to compute observables

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


def density(occ, natorb):
    """Total gas density normalized to 1"""
    return np.matmul(occ, abs(natorb) ** 2)


def condensate_density(natorb):
    """Density of condensed atoms normalized to 1"""
    return abs(natorb[0]) ** 2


def momentum_density(occ, natorb, dx, kmin=-10, kmax=10, bound=0, gf=7):
    """
    Momentum density distribution normalized to 1

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

    Return
    ------
    ``tuple(numpy.array, numpy.array)``
        Frequency values and density distribution in momentum space
    """
    freq, no_fft = ft.fft_ordered_norm(
        ft.extend_grid(natorb, bound, gf), dx, 1, bound
    )
    momentum_den = np.matmul(occ, abs(no_fft) ** 2)
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
