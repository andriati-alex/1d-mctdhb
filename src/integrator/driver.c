#include "assistant/arrays_definition.h"
#include "assistant/dataio.h"
#include "cpydataio.h"
#include "integrator/coefficients_integration.h"
#include "integrator/orbital_integration.h"
#include "integrator/synchronize.h"
#include "linalg/basic_linalg.h"
#include "linalg/lapack_interface.h"
#include <stdlib.h>
#include <string.h>

void
mctdhb_propagate_step(
    MCTDHBDataStruct mctdhb,
    Carray           orb_works,
    Carray           coef_works,
    uint32_t         curr_step)
{
    switch (mctdhb->orb_integ_method)
    {
        case SPLITSTEP:
            propagate_splitstep_orb(mctdhb, orb_works);
            break;
        case FULLSTEP_RUNGEKUTTA:
            propagate_fullstep_orb_rk(mctdhb, orb_works);
            break;
    }
    switch (mctdhb->coef_integ_method)
    {
        case LANCZOS:
            propagate_coef_sil(mctdhb, coef_works);
            break;
        case RUNGEKUTTA:
            propagate_coef_rk(mctdhb, coef_works);
            break;
    }
    sync_integration_new_step(mctdhb, curr_step);
}

void
integration_driver(
    MCTDHBDataStruct mctdhb,
    uint32_t         rec_nsteps,
    char             prefix[],
    double           tend,
    uint32_t         monitor_rate,
    Verbosity        verb)
{
    uint32_t      prop_steps;
    double        curr_t;
    char          custom_prefix[STR_BUFF_SIZE], fname[STR_BUFF_SIZE];
    Carray        orb_next_work, coef_next_work;
    ManyBodyState psi;

    mctdhb->orb_eq->tend = tend; // User redefinition of final prop time

    psi = mctdhb->state;

    curr_t = 0;
    prop_steps = 0;
    orb_next_work = get_dcomplex_array(psi->norb * psi->grid_size);
    coef_next_work = get_dcomplex_array(psi->space_dim);

    screen_integration_monitor(mctdhb, verb);

    // Record initial conditions ********************************************
    switch (mctdhb->integ_type)
    {
        case REALTIME:
            record_custom_data_selection(prefix, psi);
            append_timestep_potential(prefix, mctdhb->orb_eq);
            break;
        case IMAGTIME:
            strcpy(custom_prefix, prefix);
            strcat(custom_prefix, "_init");
            record_raw_data(custom_prefix, psi);
            break;
    }
    // initial condition recorded *******************************************

    while (curr_t < tend)
    {
        mctdhb_propagate_step(
            mctdhb, orb_next_work, coef_next_work, prop_steps);

        prop_steps++;
        curr_t = prop_steps * mctdhb->orb_eq->tstep;

        if (mctdhb->integ_type == REALTIME && prop_steps % rec_nsteps == 0)
        {
            record_custom_data_selection(prefix, psi);
            append_timestep_potential(prefix, mctdhb->orb_eq);
        }

        if (prop_steps % monitor_rate == 0)
        {
            screen_integration_monitor(mctdhb, verb);
        }
    }

    printf("\n\nSteps evolved %" PRIu32 "\n\n", prop_steps);

    if (mctdhb->integ_type == IMAGTIME)
    {
        // record potential once
        strcpy(fname, out_dirname);
        strcat(fname, prefix);
        strcat(fname, "_obpotential.dat");
        rarr_column_txt(
            fname, "%.14E", psi->grid_size, mctdhb->orb_eq->pot_grid);
        // record final (hopefully) converged orbitals and coef
        record_raw_data(prefix, psi);
    }

    if (mctdhb->integ_type == REALTIME)
    {
        record_time_interaction(prefix, mctdhb->orb_eq);
        record_time_array(prefix, tend, mctdhb->orb_eq->tstep);
    }

    free(orb_next_work);
    free(coef_next_work);
}
