"""Module to generate vector of coefficients of many-body state expansion

These routines implemented here are meant to help the initialization of a
many-body state to set input for time evolution

``coefficients_thermal(Npar -> int, Norb -> int, beta -> float)``

``coefficients_condensate(Npar -> int, Norb -> int)``

"""

import numpy as np

from tqdm import trange
from math import pi, sqrt
from multiconfpy import configurational_space as cs


def thermal(Npar, Norb, beta=2.0):
    """
    Initial random coefficients with configurational-thermal decay
    A orbital is addressed with an energy equal to its number thus
    it is desirable that more energetic states have larger indexes

    Parameters
    ----------
    `Npar` : ``int``
        number of particles
    `Norb` : ``int``
        number of orbitals
    `beta` : ``float``
        analogous to beta coefficient in thermal density matrix

    Return
    ------
    ``numpy.array(dtype=numpy.complex128)``
        Coefficients of many-body state expansion
    """
    nc = cs.number_configurations(Npar, Norb)
    C = np.empty(nc, dtype=np.complex128)
    v = np.empty(Norb, dtype=np.int32)
    phase = np.exp(2 * pi * 1.0j * (np.random.random(nc) - 0.5))
    noise = np.random.random(nc) / 10
    for i in trange(nc, desc="Coefficients"):
        cs.index_to_fock(i, Npar, Norb, v)
        conf_energy = (np.arange(Norb) * v).sum() / Npar
        C[i] = (phase[i] + noise[i]) * np.exp(-beta * conf_energy)
    return C / sqrt((abs(C) ** 2).sum())


def condensate(Npar, Norb):
    """
    Return vector of coefficients with C[0] = 1. This corresponds to a
    single Fock state with all particles in the orbital of number 0
    """
    C = np.zeros(cs.number_configurations(Npar, Norb), dtype=np.complex128)
    C[0] = 1.0
    return C
