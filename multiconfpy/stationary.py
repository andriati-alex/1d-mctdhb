import os
import numpy as np
import matplotlib.pyplot as plt

from multiconfpy import observables as obs


class GroundState:
    """Class to manipulate output data from imag time computations

    This class provide an interface to load data from numerical simulations
    and compute some important observables. Some key quantities, as natural
    occupations and orbitals are computed directly on initialization due to
    their central role in many-body physics analysis. The grid and trap can
    also be consulted through some attributes kept in initialization. Below
    some important attributes corresponding to specific job set by method
    ``self.lock_job``

    Main Attributes
    ---------------
    npar - number of particles
    norb - number of orbitals
    npts - number of grid points
    grid - numpy.array with grid points
    rho1 - numpy.ndarray([norb, norb]) raw-orbitals 1-body density matrix
    rho2 - numpy.array(norb ** 4) raw-orbitals 2-body density matrix
    occ  - numpy.array(norb) natural occupation fraction
    natorb - numpy.ndarray([norb, npts]) natural orbitals in rows
    energy - numpy.array(njobs) energy per particle for every job
    njobs - number of jobs found

    A better documentation on routines to compute observables are written
    in `multiconfpy.observables` that provide the core implementation and
    is the backend for this class methods.

    """

    def __init__(self, files_prefix, dir_path):
        """
        Structure for many-body state loading from main program output files

        Parameters
        ----------
        `files_prefix` : ``str``
            string with prefix common to all files
        `dir_path` : ``str``
            string with path containing the files
        """
        self.__dir_path = dir_path
        self.__files_prefix = files_prefix
        self.path_prefix = os.path.join(dir_path, files_prefix)
        with open(self.path_prefix + "_conf.dat") as f:
            info_line = f.readline().strip()
            self.trap_name = info_line.split(":")[1].replace(" ", "")
        eq_setup = np.loadtxt(self.path_prefix + "_conf.dat")
        if eq_setup.ndim == 1:
            eq_setup = eq_setup.reshape(1, eq_setup.size)
        self.eq_setup = eq_setup
        self.energy = (
            np.loadtxt(self.path_prefix + "_energy_imagtime.dat")
            / self.eq_setup[:, 0]
        )
        self.njobs = eq_setup.shape[0]
        self.colormap = "gnuplot"
        self.lock_job(1)

    def lock_job(self, job_id):
        """
        Lock on specific job with number `job_id` loading its data
        that is used to call `multiconfpy.observables` routines
        """
        if job_id > self.njobs:
            print("job {} not available".format(job_id))
            print("Locked in job {} of {}".format(self.job_id, self.njobs))
            return
        self.job_id = job_id
        coef_suffix = "_job{}_coef_imagtime.dat".format(job_id)
        orb_suffix = "_job{}_orb_imagtime.dat".format(job_id)
        trap_suffix = "_job{}_trap.dat".format(job_id)
        self.orbitals = np.loadtxt(
            self.path_prefix + orb_suffix, dtype=np.complex128
        ).T
        self.coef = np.loadtxt(
            self.path_prefix + coef_suffix, dtype=np.complex128
        )
        self.trap = np.loadtxt(self.path_prefix + trap_suffix)
        row = job_id - 1
        self.npar = int(self.eq_setup[row, 0])
        self.norb = int(self.eq_setup[row, 1])
        self.npts = int(self.eq_setup[row, 2])
        self.xi = self.eq_setup[row, 3]
        self.xf = self.eq_setup[row, 4]
        self.dx = (self.xf - self.xi) / (self.npts - 1)
        self.grid = np.linspace(self.xi, self.xf, self.npts)
        self.g = self.eq_setup[row, 7]
        self.trap_params = self.eq_setup[row, 8:]
        self.rho1 = self.onebody_dm()
        self.rho2 = self.twobody_dm()
        self.occ = self.natural_occupations()
        self.natorb = self.natural_orbitals()

    def locked_job_info(self):
        """Print on screen the current job information"""
        print(
            "\nJob setup files: {} at dir {}".format(
                self.__files_prefix, self.__dir_path
            )
        )
        print(
            "Locked on job {} of {} with parameters:".format(
                self.job_id, self.njobs
            )
        )
        print("\tparticles : {}".format(self.npar))
        print("\torbitals  : {}".format(self.norb))
        print("\ttrap      : {}".format(self.trap_name))
        print("\ttrap_par  : {}".format(self.trap_params))
        print("\tg         : {}\n".format(self.g))

    def onebody_dm(self):
        return obs.onebody_dm(self.npar, self.norb, self.coef)

    def twobody_dm(self):
        return obs.twobody_dm(self.npar, self.norb, self.coef)

    def natural_occupations(self):
        return obs.natural_occupations(self.npar, self.rho1)

    def natural_orbitals(self):
        return obs.natural_orbitals(self.rho1, self.orbitals)

    def occupation_entropy(self):
        return obs.occupation_entropy(self.occ)

    def density(self):
        return obs.density(self.occ, self.natorb)

    def condensate_density(self):
        return abs(self.natorb[0]) ** 2

    def momentum_density(self, kmin=-10, kmax=10, bound=0, gf=7):
        return obs.momentum_density(
            self.occ, self.natorb, self.dx, kmin, kmax, bound, gf
        )

    def position_rdm(self):
        return obs.position_rdm(self.occ, self.natorb)

    def position_onebody_correlation(self):
        return obs.position_onebody_correlation(self.occ, self.natorb)

    def momentum_rdm(self, kmin=-10, kmax=10, bound=0, gf=7):
        return obs.momentum_rdm(
            self.occ, self.natorb, self.dx, kmin, kmax, bound, gf
        )

    def momentum_onebody_correlation(self, kmin=-10, kmax=10, bound=0, gf=7):
        return obs.momentum_onebody_correlation(
            self.occ, self.natorb, self.dx, kmin, kmax, bound, gf
        )

    def position_mutual_probability(self):
        return obs.position_mutual_probability(
            self.npar, self.rho2, self.orbitals
        )

    def momentum_mutual_probability(self, kmin=-10, kmax=10, bound=0, gf=7):
        return obs.momentum_mutual_probability(
            self.npar, self.rho2, self.orbitals, self.dx, kmin, kmax, bound, gf
        )

    def position_twobody_correlation(self):
        return obs.position_twobody_correlation(
            self.npar, self.rho2, self.orbitals, self.density()
        )

    def momentum_twobody_correlation(self, kmin=-10, kmax=10, bound=0, gf=7):
        den = self.momentum_density(kmin, kmax, bound, gf)[1]
        extra_args = (den, self.dx, kmin, kmax, bound, gf)
        return obs.momentum_twobody_correlation(
            self.npar, self.rho2, self.orbitals, *extra_args
        )

    def plot_density(self, show_trap=False, show_condensate=False):
        den = self.density()
        plt.plot(self.grid, den, color="black", label="Gas density")
        ax = plt.gca()
        ax.set_xlabel("position", fontsize=16)
        ax.set_ylabel("Density", fontsize=16)
        ax.set_xlim(self.xi, self.xf)
        ax.set_ylim(0, den.max() * 1.1)
        if show_trap:
            ax_trap = ax.twinx()
            ax_trap.plot(self.grid, self.trap, color="orange")
            ax_trap.tick_params(axis="y", labelcolor="orange")
            ax_trap.set_ylim(0, self.trap.max() * 1.1)
            ax_trap.set_ylabel("Trap Potential", fontsize=16)
        if show_condensate:
            condensate_orb = abs(self.natural_orbitals()[0]) ** 2
            ax.plot(self.grid, condensate_orb, label="Condensate density")
            ax.legend()
        plt.show()

    def plot_momentum_density(self, kmin=-10, kmax=10, bound=0, gf=7):
        k, den = self.momentum_density(kmin, kmax, bound, gf)
        plt.plot(k, den, color="black")
        ax = plt.gca()
        ax.set_xlim(k.min(), k.max())
        ax.set_ylim(0, den.max() * 1.1)
        ax.set_ylabel("Momentum distribution", fontsize=16)
        ax.set_xlabel("Fourier frequency", fontsize=16)
        plt.show()

    def __data_image_view(self, data, ext, cmap):
        """General routine to plot 2d data within matplotlib imshow"""
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot()
        im = ax.imshow(
            abs(data) ** 2,
            extent=ext,
            origin="lower",
            aspect="equal",
            cmap=cmap,
        )
        fig.colorbar(im, ax=ax)
        plt.show()

    def imshow_position_abs_rdm(self):
        """Absolute square value of `self.position_rdm` mapped to colors"""
        obcorr = self.position_rdm()
        self.__data_image_view(
            abs(obcorr) ** 2,
            [self.xi, self.xf, self.xi, self.xf],
            self.colormap,
        )

    def imshow_momentum_abs_rdm(self, kmin=-10, kmax=10, bound=0, gf=7):
        """Absolute square value of `self.momentum_rdm` mapped to colors"""
        k, obcorr = self.momentum_rdm(kmin, kmax, bound, gf)
        im_min = k.min()
        im_max = k.max()
        self.__data_image_view(
            abs(obcorr) ** 2, [im_min, im_max, im_min.im_max], self.colormap
        )

    def imshow_position_onebody_correlation(self):
        """Display `self.position_onebody_correlation` mapped to colors"""
        obcorr = self.position_onebody_correlation()
        self.__data_image_view(
            obcorr,
            [self.xi, self.xf, self.xi, self.xf],
            self.colormap,
        )

    def imshow_momentum_onebody_correlation(
        self, kmin=-10, kmax=10, bound=0, gf=7
    ):
        """Display `self.momentum_onebody_correlation` mapped to colors"""
        k, obcorr = self.momentum_onebody_correlation(kmin, kmax, bound, gf)
        im_min = k.min()
        im_max = k.max()
        self.__data_image_view(
            obcorr, [im_min, im_max, im_min, im_max], self.colormap
        )

    def imshow_position_mutual_probability(self):
        """Display ``self.position_mutual_probability`` mapped to colors"""
        mutprob = self.position_mutual_probability()
        self.__data_image_view(
            mutprob,
            [self.xi, self.xf, self.xi, self.xf],
            self.colormap,
        )

    def imshow_momentum_mutual_probability(
        self, kmin=-10, kmax=10, bound=0, gf=7
    ):
        """Display ``self.momentum_mutual_probability`` mapped to colors"""
        k, mutprob = self.momentum_mutual_probability(kmin, kmax, bound, gf)
        im_min = k.min()
        im_max = k.max()
        self.__data_image_view(
            mutprob, [im_min, im_max, im_min, im_max], self.colormap
        )

    def imshow_position_twobody_correlation(self):
        """Display ``self.position_twobody_correlation`` mapped to colors"""
        tbcorr = self.position_twobody_correlation()
        self.__data_image_view(
            tbcorr,
            [self.xi, self.xf, self.xi, self.xf],
            self.colormap,
        )

    def imshow_momentum_twobody_correlation(
        self, kmin=-10, kmax=10, bound=0, gf=7
    ):
        """Display ``self.momentum_twobody_correlation`` mapped to colors"""
        k, tbcorr = self.momentum_twobody_correlation(kmin, kmax, bound, gf)
        im_min = k.min()
        im_max = k.max()
        self.__data_image_view(
            tbcorr, [im_min, im_max, im_min, im_max], self.colormap
        )
