/** \file synchronize.h
 *
 * \author Alex Andriati - andriati@if.usp.br
 * \date September/2021
 * \brief Module to reset secondary fields in structs reference data types
 *
 * The main datatypes have some auxiliar fields, as density matrices and
 * potential that shall be computed according to current storaged time,
 * coefficients and orbitals. These routines compute these secondaty parts
 * using the fundamental ones, which are time, coefficients and orbitals
 */

#ifndef SYNCHRONIZE_H
#define SYNCHRONIZE_H

#include "mctdhb_types.h"

/** \brief Minimum value to use in one-body density matrix regularization */
#define MIN_REGULARIZATION_FACTOR 1E-13
/** \brief Max value to use in one-body density matrix regularization */
#define MAX_REGULARIZATION_FACTOR 1E-6
/** \brief Default regularization value for one-body density matrix */
#define DEFAULT_REGULARIZATION_FACTOR 1E-9

/** \brief Overwrite default regularization factor for one-body density matrix
 *
 * The value given must lies between the ones established by the macros
 * \c MIN_REGULARIZATION_FACTOR and \c MAX_REGULARIZATION_FACTOR . The
 * default value is set by DEFAULT_REGULARIZATION_FACTOR macro. This is
 * also used as threshold ratio between min and max natural occupations
 * to indeed use the regularization process automatically in updating
 * one-body density matrix \c sync_density_matrices routine
 *
 * \param[in] reg_fac new regularization factor
 */
void
set_regulatization_factor(double reg_fac);

/** \brief Return current regularization factor set */
double
get_current_regularization_factor();

/** \brief Compute and set orbital Hamiltonian matrices in ManyBodyState */
void
sync_orbital_matrices(OrbitalEquation eq_desc, ManyBodyState psi);

/** \brief Compute and set density matrix in ManyBodyState */
void
sync_density_matrices(MultiConfiguration mc_space, ManyBodyState psi);

/** \brief Compute and set one-body potential and interaction */
void
sync_equation_params(OrbitalEquation eq_desc);

/** \brief Evaluate all sync routines given current step in time evolution */
void
sync_integration_new_step(MCTDHBDataStruct mctdhb, uint32_t current_step);

#endif
