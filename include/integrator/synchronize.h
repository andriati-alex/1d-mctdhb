#ifndef SYNCHRONIZE_H
#define SYNCHRONIZE_H

#include "mctdhb_types.h"

/** \brief Minimum value to use in one-body density matrix regularization */
#define MIN_REGULARIZATION_FACTOR 1E-14
/** \brief Max value to use in one-body density matrix regularization */
#define MAX_REGULARIZATION_FACTOR 1E-6

void
set_regulatization_factor(double reg_fac);

double
get_current_regularization_factor();

void
sync_orbital_matrices(OrbitalEquation eq_desc, ManyBodyState psi);

void
sync_density_matrices(MultiConfiguration mc_space, ManyBodyState psi);

void
sync_equation_params(OrbitalEquation eq_desc);

void
sync_integration_new_step(MCTDHBDataStruct mctdhb, uint32_t current_step);

#endif
