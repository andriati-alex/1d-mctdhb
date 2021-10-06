/** \file builtin_potential.h
 *
 * \author Alex Andriati - andriati@if.usp.br
 * \date September/2021
 * \brief Module to implement some one-body potential functions
 *
 * All functions must follow the \c single_particle_pot signature
 *
 * \see single_particle_pot
 */

#ifndef BUILTIN_POTENTIAL_H
#define BUILTIN_POTENTIAL_H

#include "mctdhb_types.h"

single_particle_pot
get_builtin_pot(char pot_name[]);

void
potfunc_harmonic(double t, uint16_t npts, Rarray x, void* params, Rarray V);

void
potfunc_doublewell(double t, uint16_t npts, Rarray x, void* params, Rarray V);

void
potfunc_harmonicgauss(double t, uint16_t npts, Rarray x, void* params, Rarray V);

void
potfunc_barrier(double t, uint16_t M, Rarray x, void* params, Rarray V);

void
potfunc_time_trapezoid_barrier(
    double t, uint16_t npts, Rarray x, void* params, Rarray pot);

void
potfunc_opticallattice(double t, uint16_t M, Rarray x, void* params, Rarray V);

void
potfunc_time_trapezoid_opticallattice(
    double t, uint16_t npts, Rarray x, void* params, Rarray pot);

void
potfunc_step(double t, uint16_t M, Rarray x, void* params, Rarray V);

void
potfunc_square_well(double t, uint16_t M, Rarray x, void* params, Rarray V);

#endif
