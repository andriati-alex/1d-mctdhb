"""Module with routines for operator action

This module provide some routines to act with operators over discretized
functions. These functions provide tools for ``observables`` module that
compute operators average.

All functions include a `mult`iplicative factor to provide an easy way
to suit the operator action for any system of units always as the last
parameter. The other parameter always present is the first one `state`
a ``numpy.array`` with the discretized representation

Two examples included in this module

``position(state -> numpy.array, x -> numpy.array, mult ->float)``

``momentum(state -> numpy.array, dx -> float, bound -> int, mult -> float)``

"""

from multiconfpy import function_tools as ft


def position(state, x, mult=1.0):
    """
    Apply the position operator to a function

    Parameters:
    -----------
    `x` : ``numpy.array`` array with spatial grid points
    `mult` : ``float`` factor to multiply array output (change unit system)
    """
    return mult * x * state


def momentum(state, dx, bound, mult=1.0):
    """
    Apply the momentum operator to a function

    Parameters:
    -----------
    `bound` : ``int`` 0(1) define zero(periodic) boundary
    `dx` : ``float`` grid spacing
    `mult` : ``float`` factor to multiply array output (change unit system)
    """
    if bound == 0:
        return -mult * 1.0j * ft.dfdx_zero(state, dx)
    else:
        return -mult * 1.0j * ft.dfdx_periodic(state, dx)


def projection(state, proj_state, dx):
    """
    Return `state` projected over `proj_state`

    Parameters:
    -----------
    `proj_state` : ``numpy.array`` state to project over
    `dx` : ``float`` grid spacing
    """
    return ft.simps(proj_state.conj() * state, dx=dx) * proj_state


def combine(state, callable_list, args_list):
    """
    Given a list of operator functions apply them sequentially (composing)

    Parameters
    ----------
    `callable_list` : ``list[callable]``
        list of functions to call representing operators these operators
        are applied according to list index starting from 0
    `args_list` : ``list[tuple]``
        list with tuples to pass for each function in `callable_list`

    Return
    ------
    ``numpy.array(state.size, dtype=state.dtyp)``
    """
    l1 = len(callable_list)
    l2 = len(args_list)
    if l1 != l2:
        raise ValueError(
            "length of callable and args list are different: "
            "{} and {} respectively".format(l1, l2)
        )
    composed = state.copy()
    for fun, args in zip(callable_list, args_list):
        composed = fun(composed, *args)
    return composed
