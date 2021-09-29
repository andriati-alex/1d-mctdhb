"""Module to generate suitable orbitals depending on boundaries

The orbitals must satisfy either open, boxed or periodic boundaries
Open Boundary Condition(OBC) corresponds to systems in smooth traps
Periodic Boundary Condition(PBC) to system usually trapped in rings
Boxed corresponds to infinity hard wall potential at the boundaries
There is one function to generate orbitals suitable for each case

``trapped(
    norb -> int,
    x -> numpy.array,
    omega -> float,
    nextra -> int,
    amp -> float
)``

``periodic(
    norb -> int,
    x -> numpy.array,
    nextra -> int,
    amp -> float
)``

``box(
    norb -> int,
    x-> numpy.array,
    nextra -> int,
    amp -> float
)``

"""

import numpy as np

from math import pi, sqrt
from random import shuffle
from scipy.special import eval_hermite
from multiconfpy import function_tools as ft


def __random_setup(norb, nextra):
    """
    Generated a shuffled list of indexes for excited states and random weights

    Return
    ------
    ``tuple(list[int], numpy.array(nextra, dtype=float)``
        All integers in the list are greater or equal to `norb`
    """
    norb = int(norb)
    nextra = int(nextra)
    extra_indexes = [ind for ind in range(norb, norb + nextra)]
    shuffle(extra_indexes)
    w_real = np.random.random(nextra) - 0.5
    w_imag = np.random.random(nextra) - 0.5
    extraw = w_real + 1.0j * w_imag
    return extra_indexes, extraw


def trapped(norb, x, omega=1, nextra=5, amp=None):
    """
    Generate a set of normalized functions suitable for imag time
    propagation in trapped systems. This routine uses the first
    `norb` quantum harmonic oscillator eigen functions and then
    add a mix with some grid noise and up to `nextra` excitated
    states. The orbitals are cyclic mixed with noise using:
    `[n % norb for n in range(nextra)]`

    Parameters
    ----------
    `norb` : ``int``
        number of orbitals to generate
    `x` : ``numpy(dtype=float)``
        grid points
    `omega` : ``float``
        analogous to dimensionless quantum harmonico oscillator trap parameter
    `nextra` : ``int``
        number of extra excited states (above `norb`) to mix
    `amp` : ``float``
        Control amplitude of the noise. Recommended `0 <= amp <= norb`
        In case `amp == 0`, no noise is added and `nextra` is useless

    Return
    ------
    ``numpy.ndarray([norb, x.size], dtype=complex)``
        matrix with discretized orbitals along rows
    """
    wamp = amp or norb
    extra_indexes, extraw = __random_setup(norb, nextra)
    phases = np.exp(2 * pi * 1.0j * np.random.random(norb))
    gauss = np.exp(-0.5 * omega * x ** 2) * (omega / pi) ** 0.25
    complex_hermite_list = [
        phases[n] * eval_hermite(n, sqrt(omega) * x) for n in range(norb)
    ]
    raw_orb = np.empty([norb, x.size], dtype=np.complex128)
    for n, h in enumerate(complex_hermite_list):
        base_f = gauss * h
        # dividing by sqrt( 2 ^ n * n!)
        for k in range(1, n + 1):
            base_f = base_f / sqrt(2 * k)
        raw_orb[n] = base_f
    i = 0
    while extra_indexes:
        n = extra_indexes.pop()
        weight = wamp * extraw[n - norb]
        orb_i = i % norb
        base_f = gauss * eval_hermite(n, sqrt(omega) * x) / n
        # dividing by sqrt( 2 ^ n * n!)
        for k in range(1, n + 1):
            base_f = base_f / sqrt(2 * k)
        grid_noise = np.random.random(x.size) / 10
        raw_orb[orb_i] = raw_orb[orb_i] + weight * base_f * (1 + grid_noise)
        i = i + 1
    return ft.orthonormalize(raw_orb, x[1] - x[0])


def box(norb, x, nextra=5, amp=None):
    """
    Generate a set of normalized functions suitable for imag time
    propagation in hard-wall boxed trapped systems. Set the first
    `norb` eigenstates of Schrodinger equation in a box, then add
    excited states with grid noise if `nextra > 0`

    Parameters
    ----------
    `norb` : ``int``
        number of orbitals to generate
    `x` : ``numpy(dtype=float)``
        grid points
    `nextra` : ``int``
        number of extra excited states (above `norb`) to mix
    `amp` : ``float``
        Control amplitude of the noise. Recommended `0 <= amp <= norb`
        In case `amp == 0`, no noise is added and `nextra` is useless

    Return
    ------
    ``numpy.ndarray([norb, x.size], dtype=complex)``
        matrix with discretized orbitals along rows
    """
    wamp = amp or norb
    extra_indexes, extraw = __random_setup(norb, nextra)
    phases = np.exp(2 * pi * 1.0j * np.random.random(norb))
    L = x[-1] - x[0]
    x0 = x[0]
    base_list = [
        sqrt(2 / L) * np.sin((n + 1) * pi * (x - x0) / L) for n in range(norb)
    ]
    raw_orb = np.empty([norb, x.size], dtype=np.complex128)
    for i, orb in enumerate(base_list):
        raw_orb[i] = phases[i] * orb
    i = 0
    while extra_indexes:
        n = extra_indexes.pop()
        weight = wamp * extraw[n - norb]
        orb_i = i % norb
        base = sqrt(2 / L) * np.sin((n + 1) * pi * (x - x0) / L) / n
        grid_noise = np.random.random(x.size) / 10
        raw_orb[orb_i] = raw_orb[orb_i] + weight * base * (1 + grid_noise)
        i = i + 1
    return ft.orthonormalize(raw_orb, x[1] - x[0])


def periodic(norb, x, nextra=5, amp=None):
    """
    Generate a set of normalized functions suitable for imag time
    propagation for periodic boundaries. Set the first `norb` plane
    wave states with quantized momenta then add excited states with
    grid noise if `nextra > 0`

    Parameters
    ----------
    `norb` : ``int``
        number of orbitals to generate
    `x` : ``numpy(dtype=float)``
        grid points
    `nextra` : ``int``
        number of extra excited states (above `norb`) to mix
    `amp` : ``float``
        Control amplitude of the noise. Recommended `0 <= amp <= norb`
        In case `amp == 0`, no noise is added and `nextra` is useless

    Return
    ------
    ``numpy.ndarray([norb, x.size], dtype=complex)``
        matrix with discretized orbitals along rows
    """
    wamp = amp or norb
    extra_indexes, extraw = __random_setup(norb, nextra)
    phases = np.exp(2 * pi * 1.0j * np.random.random(norb))
    dlen = x[-1] - x[0]
    x0 = 0.5 * (x[-1] + x[0])
    base_list = [
        sqrt(1 / dlen) * np.exp(2j * n * pi * (x - x0) / dlen)
        for n in range(norb)
    ]
    raw_orb = np.empty([norb, x.size], dtype=np.complex128)
    for i, orb in enumerate(base_list):
        raw_orb[i] = phases[i] * orb
    i = 0
    while extra_indexes:
        n = extra_indexes.pop()
        weight = wamp * extraw[n - norb]
        orb_i = i % norb
        base = sqrt(1 / dlen) * np.exp(2j * n * pi * (x - x0) / dlen) / n
        grid_noise = np.random.random(x.size) / 10
        grid_noise[-1] = grid_noise[0]
        raw_orb[orb_i] = raw_orb[orb_i] + weight * base * (1 + grid_noise)
        i = i + 1
    return ft.orthonormalize(raw_orb, x[1] - x[0])
