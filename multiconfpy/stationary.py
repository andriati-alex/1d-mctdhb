import os
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from multiconfpy import (
    configurational_space as cs,
    density_matrices as dm,
    function_tools as ft,
)


class GroundState:
    """Class to manipulate output data from imag time computations

    This class provide an interface to load data from numerical simulations
    and compute some important observables. Some key quantities, as natural
    occupations and orbitals are computed directly on initialization due to
    their central role in many-body physics analysis. The grid and trap can
    also be consulted through some attributes kept in initialization.

    """

    def __init__(self, files_prefix, dir_path, job_id):
        path_prefix = os.path.join(dir_path, files_prefix)
        self.orbitals = np.loadtxt(
            path_prefix + "_job{}_orb_imagtime.dat".format(job_id),
            dtype=np.complex128,
        ).T
        self.coef = np.loadtxt(
            path_prefix + "_job{}_coef_imagtime.dat".format(job_id),
            dtype=np.complex128,
        )
        self.trap = np.loadtxt(path_prefix + "_job{}_trap.dat".format(job_id))
        eq_setup = np.loadtxt(path_prefix + "_conf.dat")
        if eq_setup.ndim > 1:
            eq_setup = eq_setup[job_id - 1]
        self.npar = int(eq_setup[0])
        self.norb = int(eq_setup[1])
        self.npts = int(eq_setup[2])
        self.xi = eq_setup[3]
        self.xf = eq_setup[4]
        self.dx = (self.xf - self.xi) / (self.npts - 1)
        self.grid = np.linspace(self.xi, self.xf, self.npts)
        self.rho1 = self.get_onebody_dm()
        self.rho2 = self.get_twobody_dm()
        self.occ = self.natural_occupations()
        self.natorb = self.natural_orbitals()

    def get_onebody_dm(self):
        """
        Return one-body density matrix
        Interface for `density_matrices.set_onebody_dm`
        """
        rho = np.empty([self.norb, self.norb], dtype=np.complex128)
        conf_mat = cs.configurational_matrix(self.npar, self.norb)
        ht = cs.fock_space_matrix(self.npar, self.norb)
        dm.set_onebody_dm(self.npar, self.norb, conf_mat, ht, self.coef, rho)
        return rho

    def get_twobody_dm(self):
        """
        Return two-body density matrix in array format
        Interface for `density_matrices.set_onebody_dm`
        """
        rho = np.empty(self.norb ** 4, dtype=np.complex128)
        conf_mat = cs.configurational_matrix(self.npar, self.norb)
        ht = cs.fock_space_matrix(self.npar, self.norb)
        dm.set_twobody_dm(self.npar, self.norb, conf_mat, ht, self.coef, rho)
        return rho

    def natural_occupations(self):
        """Return occupations in natural orbitals"""
        eigval, eigvec = linalg.eig(self.rho1)
        return eigval[::-1].real / self.npar

    def natural_orbitals(self):
        """Return natural orbitals in rows of a matrix"""
        eigval, eigvec = linalg.eig(self.rho1)
        eigvec_ord = eigvec[:, ::-1]
        return np.matmul(eigvec_ord.conj().T, self.orbitals)

    def occupation_entropy(self):
        """Entropy of natural orbital occupations"""
        return -((self.occ) * np.log(self.occ)).sum()

    def density(self):
        """Total gas density normalized to 1"""
        return np.matmul(self.occ, abs(self.natorb) ** 2)

    def condensate_density(self):
        """Density of condensed atoms normalized to 1"""
        return abs(self.natorb[0]) ** 2

    def momentum_density(self, kmin=-10, kmax=10, bound=0, gf=7):
        """
        Momentum density distribution normalized to 1
        Parameters
        ----------
        `kmin` : ``float``
            minimum momentum cutoff
        `kmax` : ``float``
            maximum momentum cutoff
        `bound` : ``int {0, 1}``
            boundary information
            0 : open
            1 : periodic
        `gf` : ``int {odd}`
            grid amplification factor to improve momentum resolution
        """
        freq, no_fft = ft.fft_ordered_norm(
            ft.extend_grid(self.natorb, bound, gf), self.dx, 1, bound
        )
        momentum_den = np.matmul(self.occ, abs(no_fft) ** 2)
        freq_cut = freq[(freq - kmin) * (freq - kmax) < 0]
        momentum_den_cut = momentum_den[(freq - kmin) * (freq - kmax) < 0]
        return freq_cut, momentum_den_cut

    def position_rdm(self):
        diag_occ = np.diag(self.occ)
        natorb_dagg = self.natorb.conj().transpose()
        return np.matmul(natorb_dagg, np.matmul(diag_occ, self.natorb))

    def position_onebody_correlation(self):
        pos_rdm = self.position_rdm()
        den = self.density()
        den_rows, den_cols = np.meshgrid(den, den)
        return abs(pos_rdm) ** 2 / (den_rows * den_cols)

    def momentum_rdm(self, kmin=-10, kmax=10, bound=0, gf=7):
        diag_occ = np.diag(self.occ)
        freq, no_fft = ft.fft_ordered_norm(
            ft.extend_grid(self.natorb, bound, gf), self.dx, 1, bound
        )
        freq_cut = freq[(freq - kmin) * (freq - kmax) < 0]
        no_fft = no_fft[:, (freq - kmin) * (freq - kmax) < 0]
        no_fft_dagg = no_fft.conj().transpose()
        return freq_cut, np.matmul(no_fft_dagg, np.matmul(diag_occ, no_fft))

    def momentum_onebody_correlation(self, kmin=-10, kmax=10, bound=0, gf=7):
        mom_rdm = self.momentum_rdm(kmin, kmax, bound, gf)[1]
        freq, den = self.momentum_density(kmin, kmax, bound, gf)
        den_rows, den_cols = np.meshgrid(den, den)
        return abs(mom_rdm) ** 2 / (den_rows * den_cols)

    def show_density(self, show_trap=False, show_condensate=False):
        den = self.density()
        plt.plot(self.x, den, color="black", label="Gas density")
        ax = plt.gca()
        ax.set_xlabel("position")
        ax.set_ylabel("Density")
        ax.set_xlim(self.xi, self.xf)
        ax.set_ylim(0, den.max() * 1.1)
        if show_trap:
            ax_trap = ax.twinx()
            ax_trap.plot(self.x, self.trap, color="orange")
            ax_trap.tick_params(axis="y", labelcolor="orange")
            ax_trap.set_ylim(0, self.trap.max() * 1.1)
            ax_trap.set_ylabel("Trap Potential")
        if show_condensate:
            condensate_orb = abs(self.natural_orbitals()[0]) ** 2
            ax.plot(self.x, condensate_orb, label="Condensate density")
            ax.legend()
        plt.show()

    def show_momentum_density(self, kmin=-10, kmax=10, bound=0, gf=7):
        k, den = self.momentum_density(kmin, kmax, bound, gf)
        plt.plot(k, den, color="black")
        ax = plt.gca()
        ax.set_xlim(kmin, kmax)
        ax.set_ylim(0, den.max() * 1.1)
        ax.set_ylabel("Momentum distribution")
        ax.set_xlabel("Fourier frequency")
        plt.show()

    def show_position_onebody_correlation(self, cmap="gnuplot"):
        obcorr = self.position_onebody_correlation()
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot()
        im = plt.imshow(
            obcorr,
            extent=[self.xi, self.xf, self.xi, self.xf],
            origin="lower",
            aspect="equal",
            cmap=cmap,
        )
        fig.colorbar(im, ax=ax)
        plt.show()

    def show_momentum_onebody_correlation(
        self, kmin=-10, kmax=10, bound=0, gf=7, cmap="gnuplot"
    ):
        obcorr = self.momentum_onebody_correlation(kmin, kmax, bound, gf)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot()
        im = plt.imshow(
            obcorr,
            extent=[kmin, kmax, kmin, kmax],
            origin="lower",
            aspect="equal",
            cmap=cmap,
        )
        fig.colorbar(im, ax=ax)
        plt.show()
