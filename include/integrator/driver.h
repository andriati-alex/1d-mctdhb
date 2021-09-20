#ifndef MCTDHB_INTEGRATOR_DRIVER_H
#define MCTDHB_INTEGRATOR_DRIVER_H

#include "mctdhb_types.h"

void
mctdhb_propagate_step(
    MCTDHBDataStruct mctdhb,
    Carray           orb_works,
    Carray           coef_works,
    uint32_t         curr_step);

void
integration_driver(
    MCTDHBDataStruct mctdhb,
    uint32_t         rec_nsteps,
    char             prefix[],
    double           tend,
    uint32_t         monitor_rate);

#endif
