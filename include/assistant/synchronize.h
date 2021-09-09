#ifndef SYNCHRONIZE_H
#define SYNCHRONIZE_H

#include "mctdhb_types.h"

void
sync_orbital_matrices(MCTDHBDataStruct mctdhb);

void
sync_density_matrices(MCTDHBDataStruct mctdhb);

void
sync_equation_params(double t, OrbitalEquation eq_desc);

#endif
