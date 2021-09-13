#ifndef SYNCHRONIZE_H
#define SYNCHRONIZE_H

#include "mctdhb_types.h"

void
sync_orbital_matrices(OrbitalEquation eq_desc, ManyBodyState psi);

void
sync_density_matrices(MultiConfiguration mc_space, ManyBodyState psi);

void
sync_equation_params(OrbitalEquation eq_desc);

void
sync_integration_new_step(MCTDHBDataStruct mctdhb, uint32_t current_step);

#endif
