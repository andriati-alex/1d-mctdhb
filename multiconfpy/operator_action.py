"""Module with routines for operator action

This module provide some routines to act with operators over discretized
functions. These functions provide tools for ``observables`` module that
compute operators average.

All functions include a `mult`iplicative factor to provide an easy way
to suit the operator action for any system of units always as the last
parameter. The other parameter always present is the first one `state`
a ``numpy.array`` with the discretized representation

"""

from multiconfpy import function_tools as ft


def position(state, x, mult=1.0):
    """
    Apply the position operator to a function
    Use `mult` to suit to desired unit system
    `x` : ``numpy.array`` with grid points
    """
    return mult * x * state


def momentum(state, dx, bound, mult=1.0):
    """
    Apply the momentum operator to a function
    Use `mult` to suit to desired unit system
    `bound` = 0(1) is zero(periodic) boundary
    `dx` : ``float`` grid spacing
    """
    if bound == 0:
        return -mult * 1.0j * ft.dfdx_zero(state, dx)
    else:
        return -mult * 1.0j * ft.dfdx_periodic(state, dx)


def projection(state, proj_state, dx):
    """
    Return `state` projected over `proj_state`
    `proj_state` : ``numpy.array`` state to project over
    `dx` : ``float`` grid spacing
    """
    return ft.simps(proj_state.conj() * state, dx=dx) * proj_state
