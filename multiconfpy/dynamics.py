import os
import linecache
import numpy as np
import matplotlib.pyplot as plt

from multiconfpy import observables as obs


def gen_input_file_row(filepath, row_index=0):
    """yield linecache.getline(filepath, row_index + 1)"""
    yield linecache.getline(filepath, row_index + 1)


class DynamicsProcessing:
    def __init__(self, files_prefix, dir_path):
        """Struct for many-body state loading from main program output files

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
        eq_setup = np.loadtxt(self.path_prefix + "_mctdhb_parameters.dat")
        if eq_setup.ndim == 1:
            eq_setup = eq_setup.reshape(1, eq_setup.size)
        self.eq_setup = eq_setup
        self.njobs = eq_setup.shape[0]
        self.colormap = "gnuplot"
        self.lock_job(1)

    def lock_job(self, job_id):
        """
        Lock on specific job, reading files pattern `*_job?*` where the `?`
        mark is integer number given in argument `job_id`. The coefficients
        and orbitals read must correspond to stationary state obtained from
        line `job_id` of parameters file
        The following object attributes are set:
            `self.npar`
            `self.norb`
            `self.npts`
            `self.xi`
            `self.xf`
            `self.dx`
            `self.grid`
            `self.nframes`
            `self.orb_fname`
            `self.pot_fname`
            `self.obmat_fname`
            `self.tbmat_fname`
            `self.time_arr`
            `self.g_arr`
            `self.natocc_arr`
        """
        if job_id > self.njobs:
            print("job {} not available".format(job_id))
            print("Locked in job {} of {}".format(self.job_id, self.njobs))
            return
        self.job_id = job_id
        self.orb_fname = self.path_prefix + "_job{}_orb.dat".format(job_id)
        self.pot_fname = self.path_prefix + "_job{}_obpotential.dat".format(
            job_id
        )
        self.obmat_fname = self.path_prefix + "_job{}_obmat.dat".format(job_id)
        self.tbmat_fname = self.path_prefix + "_job{}_tbmat.dat".format(job_id)
        time_fname = self.path_prefix + "_job{}_timesteps.dat".format(job_id)
        self.time_arr = np.loadtxt(time_fname)
        g_fname = self.path_prefix + "_job{}_interaction.dat".format(job_id)
        self.g_arr = np.loadtxt(g_fname)
        self.nframes = self.time_arr.size
        row = job_id - 1
        self.npar = int(self.eq_setup[row, 0])
        self.norb = int(self.eq_setup[row, 1])
        self.npts = int(self.eq_setup[row, 2])
        self.xi = self.eq_setup[row, 3]
        self.xf = self.eq_setup[row, 4]
        self.dx = (self.xf - self.xi) / (self.npts - 1)
        self.grid = np.linspace(self.xi, self.xf, self.npts)
        self.natocc_arr = np.array(
            [
                self.natural_occupations(frame_ind)
                for frame_ind in range(self.nframes)
            ],
            dtype=np.float64,
        )

    def locked_job_info(self):
        """Print on screen current job information"""
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
        print("\t(n)particles : {}".format(self.npar))
        print("\t(n)orbitals  : {}".format(self.norb))

    def __assert_frame_index(self, frame_ind):
        if frame_ind < 0 or frame_ind >= self.nframes:
            raise IOError(
                "Invalid frame index {}. Frames from {} to {}".format(
                    frame_ind, 0, self.nframes - 1
                )
            )

    def frame_from_time(self, t):
        """Get the closest frame index of a given time instant `t`"""
        return abs(self.time_den - t).argmin()

    def ob_potential(self, frame_ind=0):
        self.__assert_frame_index(frame_ind)
        return np.loadtxt(gen_input_file_row(self.pot_fname, frame_ind))

    def raw_orbitals(self, frame_ind=0):
        self.__assert_frame_index(frame_ind)
        orb = np.loadtxt(
            gen_input_file_row(self.orb_fname, frame_ind),
            dtype=np.complex128,
        ).reshape(self.norb, self.npts)
        return orb

    def onebody_dm(self, frame_ind=0):
        self.__assert_frame_index(frame_ind)
        ob_denmat = np.loadtxt(
            gen_input_file_row(self.obmat_fname, frame_ind),
            dtype=np.complex128,
        ).reshape(self.norb, self.norb)
        return ob_denmat

    def twobody_dm(self, frame_ind=0):
        self.__assert_frame_index(frame_ind)
        tb_denmat = np.loadtxt(
            gen_input_file_row(self.tbmat_fname, frame_ind),
            dtype=np.complex128,
        )
        return tb_denmat

    def natural_occupations(self, frame_ind=0):
        return obs.natural_occupations(self.npar, self.onebody_dm(frame_ind))

    def natural_orbitals(self, frame_ind=0):
        return obs.natural_orbitals(
            self.onebody_dm(frame_ind), self.raw_orbitals(frame_ind)
        )

    def occupation_entropy(self, frame_ind=0):
        return obs.occupation_entropy(self.natocc_arr[frame_ind])

    def density(self, frame_ind=0):
        return obs.density(
            self.natocc_arr[frame_ind],
            self.natural_orbitals(frame_ind),
        )

    def condensate_density(self, frame_ind=0):
        return abs(self.natural_orbitals(frame_ind)[0]) ** 2

    def subset_density(self, subset_ind, frame_ind=0):
        return obs.density(
            self.natocc_arr[frame_ind],
            self.natural_orbitals(frame_ind),
            subset_ind,
        )

    def momentum_density(self, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7):
        return obs.momentum_density(
            self.natocc_arr[frame_ind],
            self.natural_orbitals(frame_ind),
            self.dx,
            kmin,
            kmax,
            bound,
            gf,
        )

    def subset_momentum_density(
        self, subset_ind, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7
    ):
        return obs.momentum_density(
            self.natocc_arr[frame_ind],
            self.natural_orbitals(frame_ind),
            self.dx,
            kmin,
            kmax,
            bound,
            gf,
            subset_ind,
        )

    def position_rdm(self, frame_ind=0):
        return obs.position_rdm(
            self.natocc_arr[frame_ind], self.natural_orbitals(frame_ind)
        )

    def position_onebody_correlation(self, frame_ind=0):
        return obs.position_onebody_correlation(
            self.natocc_arr[frame_ind], self.natural_orbitals(frame_ind)
        )

    def momentum_rdm(self, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7):
        return obs.momentum_rdm(
            self.natocc_arr[frame_ind],
            self.natural_orbitals(frame_ind),
            self.dx,
            kmin,
            kmax,
            bound,
            gf,
        )

    def momentum_onebody_correlation(
        self, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7
    ):
        return obs.momentum_onebody_correlation(
            self.natocc_arr[frame_ind],
            self.natural_orbitals(frame_ind),
            self.dx,
            kmin,
            kmax,
            bound,
            gf,
        )

    def position_mutual_probability(self, frame_ind=0):
        return obs.position_mutual_probability(
            self.npar, self.twobody_dm(frame_ind), self.raw_orbitals(frame_ind)
        )

    def momentum_mutual_probability(
        self, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7
    ):
        return obs.momentum_mutual_probability(
            self.npar,
            self.twobody_dm(frame_ind),
            self.raw_orbitals(frame_ind),
            self.dx,
            kmin,
            kmax,
            bound,
            gf,
        )

    def position_twobody_correlation(self, frame_ind=0):
        return obs.position_twobody_correlation(
            self.npar,
            self.twobody_dm(frame_ind),
            self.raw_orbitals(frame_ind),
            self.density(frame_ind),
        )

    def momentum_twobody_correlation(
        self, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7
    ):
        den = self.momentum_density(frame_ind, kmin, kmax, bound, gf)[1]
        extra_args = (den, self.dx, kmin, kmax, bound, gf)
        return obs.momentum_twobody_correlation(
            self.npar,
            self.twobody_dm(frame_ind),
            self.raw_orbitals(frame_ind),
            *extra_args
        )

    def average_onebody_operator(self, op_action, args=(), subset_ind=None):
        """
        Compute average of an arbritrary extensive 1-body operator

        Parameters
        ----------
        `op_action` : ``callable``
            function which apply the operator to a state
            see examples ``multiconfpy.operator_action``
        `args` : ``tuple``
            arguments needed for specific for each evaluation
        `subset_ind` : ``list[int] / numpy.array(dtype=int)``
            restrict evaluation of average to subset of natural orbitals
            by default no restriction is applied (``None``)
            `0 <= subset_ind[i] < self.norb` without repetitions if provided

        Return
        ------
        ``float``
        """
        return np.array(
            [
                obs.average_onebody_operator(
                    self.natocc_arr[frame_ind],
                    self.natural_orbitals(frame_ind),
                    op_action,
                    self.dx,
                    args,
                    subset_ind,
                )
                for frame_ind in range(self.nframes)
            ]
        )

    def covariance(self, op_left, op_right, args_left=(), args_right=()):
        """
        Compute average of an arbritrary extensive 1-body operator

        Parameters
        ----------
        `op_action` : ``callable``
            function which apply the operator to a state
            see examples ``multiconfpy.operator_action``
        `args` : ``tuple``
            arguments needed for specific for each evaluation
        `subset_ind` : ``list[int] / numpy.array(dtype=int)``
            restrict evaluation of average to subset of natural orbitals
            by default no restriction is applied (``None``)
            `0 <= subset_ind[i] < self.norb` without repetitions if provided

        Return
        ------
        ``float``
        """
        return np.array(
            [
                obs.manybody_operator_covariance(
                    self.npar,
                    self.onebody_dm(frame_ind),
                    self.twobody_dm(frame_ind),
                    self.raw_orbitals(frame_ind),
                    self.dx,
                    op_left,
                    op_right,
                    args_left,
                    args_right,
                )
                for frame_ind in range(self.nframes)
            ]
        )

    def plot_density(
        self, frame_ind=0, show_trap=False, show_condensate=False
    ):
        den = self.density(frame_ind)
        plt.plot(self.grid, den, color="black", label="Gas density")
        ax = plt.gca()
        ax.set_xlabel("position", fontsize=16)
        ax.set_ylabel("Density", fontsize=16)
        ax.set_xlim(self.xi, self.xf)
        ax.set_ylim(0, den.max() * 1.1)
        if show_trap:
            ax_trap = ax.twinx()
            trap_arr = self.ob_potential(frame_ind)
            ax_trap.plot(self.grid, trap_arr, color="orange")
            ax_trap.tick_params(axis="y", labelcolor="orange")
            ax_trap.set_ylim(0, trap_arr.max() * 1.1)
            ax_trap.set_ylabel("Trap Potential", fontsize=16)
        if show_condensate:
            condensate_orb = abs(self.natural_orbitals(frame_ind)[0]) ** 2
            ax.plot(self.grid, condensate_orb, label="Condensate density")
            ax.legend()
        plt.show()

    def plot_momentum_density(
        self, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7
    ):
        k, den = self.momentum_density(frame_ind, kmin, kmax, bound, gf)
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

    def imshow_position_abs_rdm(self, frame_ind=0):
        """Absolute square value of `self.position_rdm` mapped to colors"""
        pos_rdm = self.position_rdm(frame_ind)
        self.__data_image_view(
            abs(pos_rdm) ** 2,
            [self.xi, self.xf, self.xi, self.xf],
            self.colormap,
        )

    def imshow_momentum_abs_rdm(
        self, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7
    ):
        """Absolute square value of `self.momentum_rdm` mapped to colors"""
        k, k_rdm = self.momentum_rdm(frame_ind, kmin, kmax, bound, gf)
        im_min = k.min()
        im_max = k.max()
        self.__data_image_view(
            abs(k_rdm) ** 2, [im_min, im_max, im_min, im_max], self.colormap
        )

    def imshow_position_onebody_correlation(self, frame_ind=0):
        """Display `self.position_onebody_correlation` mapped to colors"""
        obcorr = self.position_onebody_correlation(frame_ind)
        self.__data_image_view(
            obcorr,
            [self.xi, self.xf, self.xi, self.xf],
            self.colormap,
        )

    def imshow_momentum_onebody_correlation(
        self, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7
    ):
        """Display `self.momentum_onebody_correlation` mapped to colors"""
        k, obcorr = self.momentum_onebody_correlation(
            frame_ind, kmin, kmax, bound, gf
        )
        im_min = k.min()
        im_max = k.max()
        self.__data_image_view(
            obcorr, [im_min, im_max, im_min, im_max], self.colormap
        )

    def imshow_position_mutual_probability(self, frame_ind=0):
        """Display ``self.position_mutual_probability`` mapped to colors"""
        mutprob = self.position_mutual_probability(frame_ind)
        self.__data_image_view(
            mutprob,
            [self.xi, self.xf, self.xi, self.xf],
            self.colormap,
        )

    def imshow_momentum_mutual_probability(
        self, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7
    ):
        """Display ``self.momentum_mutual_probability`` mapped to colors"""
        k, mutprob = self.momentum_mutual_probability(
            frame_ind, kmin, kmax, bound, gf
        )
        im_min = k.min()
        im_max = k.max()
        self.__data_image_view(
            mutprob, [im_min, im_max, im_min, im_max], self.colormap
        )

    def imshow_position_twobody_correlation(self, frame_ind=0):
        """Display ``self.position_twobody_correlation`` mapped to colors"""
        tbcorr = self.position_twobody_correlation(frame_ind)
        self.__data_image_view(
            tbcorr,
            [self.xi, self.xf, self.xi, self.xf],
            self.colormap,
        )

    def imshow_momentum_twobody_correlation(
        self, frame_ind=0, kmin=-10, kmax=10, bound=0, gf=7
    ):
        """Display ``self.momentum_twobody_correlation`` mapped to colors"""
        k, tbcorr = self.momentum_twobody_correlation(
            frame_ind, kmin, kmax, bound, gf
        )
        im_min = k.min()
        im_max = k.max()
        self.__data_image_view(
            tbcorr, [im_min, im_max, im_min, im_max], self.colormap
        )
