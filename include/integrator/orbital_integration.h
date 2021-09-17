#ifndef ORBITAL_INTEGRATION_H
#define ORBITAL_INTEGRATION_H

#include "mctdhb_types.h"
#include "odesys.h"

void
set_periodic_bounds(uint16_t norb, uint16_t grid_size, Cmatrix orb);

void
robust_multiorb_projector(
    OrbitalEquation eq_desc,
    uint16_t        norb,
    Cmatrix         orb_projector,
    Cmatrix         function_inp,
    Cmatrix         projected);

void
simple_multiorb_projector(
    OrbitalEquation eq_desc,
    uint16_t        norb,
    Cmatrix         orb_projector,
    Cmatrix         function_inp,
    Cmatrix         projected);

void
dodt_fullstep_interface(ComplexODEInputParameters odepar, Carray orb_der);

void
dodt_splitstep_nonlinear_interface(
    ComplexODEInputParameters odepar, Carray orb_der);

void
propagate_fullstep_orb_rk(MCTDHBDataStruct mctdhb, Carray orb_next);

void
propagate_splitstep_orb(MCTDHBDataStruct mctdhb, Carray orb_next);

#endif
