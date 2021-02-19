import os
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import configurational_space as cs
import density_matrices as dm
import assistant


class GroundState:
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
        self.xi = int(eq_setup[3])
        self.xf = int(eq_setup[4])
        self.dx = (self.xf - self.xi) / (self.npts - 1)
        self.x = np.linspace(self.xi, self.xf, self.npts)
        self.rho1 = self.__one_body_density_matrix()
        self.rho2 = self.__two_body_density_matrix()

    def __one_body_density_matrix(self):
        rho = np.empty([self.norb, self.norb], dtype=np.complex128)
        comb_mat = cs.combinatorial_mat(self.npar, self.norb)
        hashing_table = cs.enum_configurations(self.npar, self.norb)
        dm.nb_obrho(
            self.npar, self.norb, comb_mat, hashing_table, self.coef, rho
        )
        return rho

    def __two_body_density_matrix(self):
        rho = np.empty(self.norb ** 4, dtype=np.complex128)
        comb_mat = cs.combinatorial_mat(self.npar, self.norb)
        hashing_table = cs.enum_configurations(self.npar, self.norb)
        dm.nb_tbrho(
            self.npar, self.norb, comb_mat, hashing_table, self.coef, rho
        )
        return rho

    def __convert_momentum_space(self, orb, bound="zero", gf=7):
        if bound == "zero":
            ext_orb = np.zeros(
                [self.norb, gf * self.npts], dtype=np.complex128
            )
            orb_fft = np.zeros(
                [self.norb, gf * self.npts], dtype=np.complex128
            )
            l = int((gf - 1) / 2)
            k = int((gf + 1) / 2)
            for i in range(self.norb):
                ext_orb[i, l * self.npts : k * self.npts] = orb[i]
            for i in range(self.norb):
                freq, orb_fft[i] = assistant.L2normFFT(self.dx, ext_orb[i])
        else:
            ext_orb = np.zeros(
                [self.norb, gf * (self.npts - 1)], dtype=np.complex128
            )
            orb_fft = np.zeros(
                [self.norb, gf * (self.npts - 1)], dtype=np.complex128
            )
            for i in range(self.norb):
                for k in range(gf):
                    ext_orb[
                        i, k * (self.npts - 1) : (k + 1) * (self.npts - 1)
                    ] = orb[i, :-1]
            for i in range(self.norb):
                freq, orb_fft[i] = assistant.DnormFFT(self.dx, ext_orb[i])
        return freq, orb_fft

    def natural_occupations(self):
        eigval, eigvec = linalg.eig(self.rho1)
        assistant.nb_eigen_sort(self.norb, eigval.real, eigvec)
        return eigval.real / self.npar

    def natural_orbitals(self):
        eigval, eigvec = linalg.eig(self.rho1)
        assistant.nb_eigen_sort(self.norb, eigval.real, eigvec)
        return np.matmul(eigvec.conj().T, self.orbitals)

    def occupation_entropy(self):
        occ = self.natural_occupations()
        return -((occ) * np.log(occ)).sum()

    def density(self):
        nat_occ = self.natural_occupations()
        nat_orb = self.natural_orbitals()
        return np.matmul(nat_occ, abs(nat_orb) ** 2)

    def momentum_density(self, kmin=-15, kmax=15, bound="zero", gf=7):
        nat_occ = self.natural_occupations()
        nat_orb = self.natural_orbitals()
        freq, no_fft = self.__convert_momentum_space(nat_orb, bound, gf)
        momentum_den = np.matmul(nat_occ, abs(no_fft) ** 2)
        freq_cut = freq[(freq - kmin) * (freq - kmax) < 0]
        momentum_den_cut = momentum_den[(freq - kmin) * (freq - kmax) < 0]
        return freq_cut, momentum_den_cut

    def position_rdm(self):
        nat_occ = self.natural_occupations()
        nat_orb = self.natural_orbitals()
        rho = np.zeros([self.npts, self.npts], dtype=np.complex128)
        assistant.nb_position_rdm_sum(
            self.norb, self.npts, nat_occ, nat_orb, rho
        )
        return rho

    def position_onebody_correlation(self):
        pos_rdm = self.position_rdm()
        den = self.density()
        den_rows, den_cols = np.meshgrid(den, den)
        return abs(pos_rdm) ** 2 / (den_rows * den_cols)

    def momentum_rdm(self, kmin=-15, kmax=15, bound="zero", gf=7):
        nat_occ = self.natural_occupations()
        nat_orb = self.natural_orbitals()
        freq, nat_orb_fft = self.__convert_momentum_space(nat_orb, bound, gf)
        freq_cut = freq[(freq - kmin) * (freq - kmax) < 0]
        nat_orb_fft_cut = np.zeros(
            [self.norb, freq_cut.size], dtype=np.complex128
        )
        for i in range(self.norb):
            nat_orb_fft_cut[i] = nat_orb_fft[i][
                (freq - kmin) * (freq - kmax) < 0
            ]
        rho = np.zeros([freq_cut.size, freq_cut.size], dtype=np.complex128)
        assistant.nb_position_rdm_sum(
            self.norb, freq_cut.size, nat_occ, nat_orb_fft_cut, rho
        )
        return rho

    def momentum_onebody_correlation(
        self, kmin=-15, kmax=15, bound="zero", gf=7
    ):
        mom_rdm = self.momentum_rdm(kmin, kmax, bound, gf)
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

    def show_momentum_density(self, kmin=-15, kmax=15, bound="zero", gf=7):
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
        self, kmin=-15, kmax=15, bound="zero", gf=7, cmap="gnuplot"
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
