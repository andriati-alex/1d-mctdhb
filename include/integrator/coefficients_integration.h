#ifndef COEFFICIENTS_INTEGRATION_H
#define COEFFICIENTS_INTEGRATION_H

#include "mctdhb_types.h"
#include "odesys.h"

void
realtime_dcdt_odelib(ComplexODEInputParameters inp_params, Carray dcdt);

void
imagtime_dcdt_odelib(ComplexODEInputParameters inp_params, Carray dcdt);

void
iterative_lanczos_integrator(MCTDHBDataStruct mctdhb, Carray C);

#endif
