# MCTDHB in 1-dimension

A package for numerical time evolution of multiconfigurational time-dependent
Hartree method for bosons(MCTDHB). This method is a generalization of the mean
field approach, where the atoms have more than one single particle state for
occupation and all the possible configurations of particles in these single
particle states (a.k.a **orbitals**) are consided. For a first contact with
the analytical derivation of the method see:

- [Phys. Rev. A, 77, 033613 (2008)](http://dx.doi.org/10.1103/PhysRevA.77.033613)
- [Phys. Reports, 324 (2000)](https://sciencedirect.com/science/article/abs/pii/S0370157399000472)
- [ultracold.org documentation](ultracold.org/menu)

## Dependencies

To properly install the package, the following dependencies are required

- `cmake --version >= 16`
- [intel oneAPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi)
  at least the `MKL` library and the classic C/C++ compiler
- [data input/output library](https://github.com/andriati-alex/cpy-dataio)
- [differential equation library](https://github.com/andriati-alex/odelib)

after the installation of these dependencies, copy the include files of the
`cpy-dataio` and `odelib` to the directory `${HOME}/.local/include` and the
libraries to `${HOME}/.local/lib` (create these dirs if needed).

## Installation

If all dependencies are satisfied, it is easy to install with `cmake`. Inside
the directory cloned just run

```
cmake -S . -B build -DCMAKE_C_COMPILER=icc && cmake --build build --target install
```

The library will be located at `lib` dir and the main executable at `bin`.

## Basic usage - main program (in development)

Some description on config files and initial data generation with python
