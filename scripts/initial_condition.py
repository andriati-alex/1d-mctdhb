import os
import argparse
import numpy as np
import multiconfpy.initialization as init

SCRIPT_DESC = """

SCRIPT TO GENERATE INITIAL CONDITION DATA FOR MCTDHB

The Multiconfigurational Time-Dependent Hartree method for Bosons(MCTDHB)
requires a data set corresponding to the initial condition of a many-body
state to initiate the time propagation. For a numerical approach, once a
number of particles and single particle states (a.k.a orbitals) are given
the configurational space is defined. Thereafter, a initial condition is
set by specifying all coefficients of configurational-basis expansion and
what are the orbitals that the occupation numbers corresponds. Routines
to support this initial setup are provided in `multiconfpy.initialization`
module, but they are mainly directed for imaginary time propagation which
is used to obtain the ground state within the configurational space

Create 4 files in an input folder speciefied in --input-dir option

1)  ``prefix_conf.dat``
    Provide the most basic information about the numerical setup of
    the problem basis in the following order:
    ``Npar Norb grid_steps xi xf dt ndt``

2)  ``prefix_orb.dat``
    Matrix whose each column corresponds to a orbital in the grid
    Therefore, each row is the function value in a point as below:

           Orbital #1     Orbital #2 ... Orbital #n
        .==========================================
    x1  |   f1(x1)         f2(x1)         fn(x1)
    x2  |   f1(x2)         f2(x2)         fn(x2)
    .   |     .              .              .
    .   |     .              .              .
    .   |     .              .              .
    xm  |   f1(xm)         f2(xm)         fn(xm)

3)  ``prefix_coef.dat``
    Vector of coefficients in the config. basis expansion of dimension:

     (Npar + Norb - 1)!
    --------------------
    (Npar)!  (Norb - 1)!

4)  ``fileId_eq.dat``
    File with orbital equations parameters in the following order
    ``nabla_coef gauge_coef g p1 p2 p3``
    `nabla_coef` is usually -1/2 the coefficient with second derivative
    `gauge_coef` is the imag part of coefficient with first derivative
    `g` is the contact interaction parameter in 1D
    `pj` with j = 1,2,3 are trap potential parameters

"""

if __name__ == "__main__":
    default_input_dir = os.path.join(
        os.path.expanduser("~"), "projects/1d-mctdhb/input"
    )
    p = argparse.ArgumentParser(
        usage="python %(prog)s Npar Norb nx xi xf [optional_args] ",
        description=SCRIPT_DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "Npar",
        action="store",
        type=int,
        help="Number of particles",
    )
    p.add_argument(
        "Norb",
        action="store",
        type=int,
        help="Number of orbitals",
    )
    p.add_argument(
        "nx",
        action="store",
        type=int,
        help="Number of grid points counting the boundary points",
    )
    p.add_argument(
        "xi",
        action="store",
        type=float,
        help="Domain extreme left boundary value",
    )
    p.add_argument(
        "xf",
        action="store",
        type=float,
        help="Domain extreme right boundary value",
    )
    p.add_argument(
        "--orbital-seed",
        dest="orbital_generator_name",
        action="store",
        default="trapped",
        type=str,
        help="Function name to generate orbitals from "
        "`multiconf_analysis.initialization.orbitals`",
    )
    p.add_argument(
        "--orb-extra-args",
        dest="orb_extra_args",
        nargs="+",
        type=float,
        default=(),
        help="extra arguments accepted in orbital generator "
        "besides `norb` and `x`-grid points corresponding to "
        "function given in `--orbital-seed`. See function at "
        "`multiconf_analysis.initialization.orbitals`",
    )
    p.add_argument(
        "--coefficients-seed",
        dest="coef_generator_name",
        action="store",
        default="thermal",
        type=str,
        help="Function name to generate coefficients from "
        "`multiconf_analysis.initialization.coefficients`",
    )
    p.add_argument(
        "--coef-extra-args",
        dest="coef_extra_args",
        nargs="+",
        type=float,
        default=(),
        help="Extra arguments accepted besides `Npar` and `Norb` "
        "corresponding to function given in `--coefficients-seed` "
        "See doc in `multiconf_analysis.initialization.coefficients`",
    )
    p.add_argument(
        "--input-dir",
        dest="input_dir",
        action="store",
        type=str,
        default=default_input_dir,
        help="Path to write input files generated",
    )
    p.add_argument(
        "--files-prefix",
        dest="files_prefix",
        action="store",
        type=str,
        default="initial_condition",
        help="Prefix of input files generated",
    )
    p.add_argument(
        "-preview",
        action="store_true",
        help="Show plots of orbitals before exiting script",
    )
    args = p.parse_args()
    if args.input_dir == default_input_dir:
        os.makedirs(args.input_dir, exist_ok=True)
    else:
        if not os.path.isdir(args.input_dir):
            dir_msg = "path {} does not exist, create it [y/n]? ".format(
                args.input_dir
            )
            must_create = input(dir_msg).lower() == "y"
            if must_create:
                os.makedirs(args.input_dir)
            else:
                print("process aborted")
                exit()
    x = np.linspace(args.xi, args.xf, args.nx)
    orb_generator = getattr(init.orbitals, args.orbital_generator_name)
    coef_generator = getattr(init.coefficients, args.coef_generator_name)
    orbitals = orb_generator(args.Norb, x, *tuple(args.orb_extra_args))
    C = coef_generator(args.Npar, args.Norb, *tuple(args.coef_extra_args))
    prefix_path = os.path.join(args.input_dir, args.files_prefix)
    np.savetxt(prefix_path + "_job1_orb.dat", orbitals.T, fmt="%.14E")
    np.savetxt(prefix_path + "_job1_coef.dat", C.T, fmt="%.14E")
    f = open(prefix_path + "_mctdhb_parameters.dat", "w")
    # Arbitrary value for time step and number of steps
    f.write(
        "{} {} {} {:.10f} {:.10f} 0.002 10000 -0.5 0.0 "
        "linear_ramp 1.0 1.0 0.0 0.0 0.0 harmonic 1.0 1.0 0.0 0.0 0.0".format(
            args.Npar, args.Norb, args.nx, args.xi, args.xf
        )
    )
    f.close()
    if args.preview:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = plt.gca()
        lines = [
            ax.plot(x, abs(orbitals[i]) ** 2)[0] for i in range(args.Norb)
        ]
        labels = ["orb {}".format(i) for i in range(args.Norb)]
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("position", fontsize=16)
        ax.set_ylabel("orbital density", fontsize=16)
        ax.legend(lines, labels)
        plt.show()
