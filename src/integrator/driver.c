#include "integrator/driver.h"
#include "assistant/arrays_definition.h"
#include "assistant/dataio.h"
#include "assistant/integrator_monitor.h"
#include "cpydataio.h"
#include "integrator/coefficients_integration.h"
#include "integrator/orbital_integration.h"
#include "integrator/synchronize.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

static Bool    check_auto_convergence = FALSE;
static uint8_t converged_energy_digits = 8;
static double  converged_eig_residue = 1E-5;
static double  threshold_impr_ortho = 1E-6;

static void
print_time_used(double t_sec)
{
    int secs = (int) t_sec, hours = 0, mins = 0;
    if (secs / 3600 > 0)
    {
        hours = secs / 3600;
        secs = secs % 3600;
    }
    if (secs / 60 > 0)
    {
        mins = secs / 60;
        secs = secs % 60;
    }
    printf("%d hour(s) %d minute(s)", hours, mins);
}

static Bool
imagtime_check_convergence(MCTDHBDataStruct mctdhb, double* prev_e)
{
    double   total_e, curr_e, eig_res, etol;
    uint16_t npar = mctdhb->state->npar;

    total_e = creal(total_energy(mctdhb->state));
    curr_e = total_e / npar;
    etol = pow(0.1, (double) converged_energy_digits) * (*prev_e);
    eig_res = eig_residual(
        mctdhb->multiconfig_space,
        mctdhb->state->coef,
        mctdhb->state->hob,
        mctdhb->state->hint,
        total_e);
    if (fabs(curr_e - (*prev_e)) < etol && eig_res < converged_eig_residue)
    {
        return TRUE;
    }
    *prev_e = curr_e;
    return FALSE;
}

static void
realtime_check_overlap(MCTDHBDataStruct mctdhb)
{
    double o;

    o = overlap_residual(
        mctdhb->orb_eq->norb,
        mctdhb->orb_eq->grid_size,
        mctdhb->orb_eq->dx,
        mctdhb->state->orbitals);

    if (o > threshold_impr_ortho)
    {
        printf("\n\n== Improving orthogonality ==\n\n");
        mctdhb->orb_workspace->impr_ortho = TRUE;
    }

    if (o > REALTIME_OVERLAP_RESIDUE_TOL)
    {
        printf(
            "\n\nOverlap residue fell below critical value %.3lf. Aborting\n\n",
            REALTIME_OVERLAP_RESIDUE_TOL);
        exit(EXIT_FAILURE);
    }
}

void
set_autoconvergence_check(Bool must_check)
{
    check_auto_convergence = must_check;
}

void
set_energy_convergence_digits(uint8_t edig)
{
    if (edig < 4)
    {
        printf(
            "\n\nWARNING: %" PRIu8
            " digits is too small for energy convergence criteria.\n\n",
            edig);
    }
    if (edig > 13)
    {
        printf(
            "\n\nERROR: %" PRIu8
            " digits for convergence is too much due to roundoff.\n\n",
            edig);
        exit(EXIT_FAILURE);
    }
    converged_energy_digits = edig;
}

void
set_energy_convergence_eig_residual(double eig_res)
{
    if (eig_res > 0.01)
    {
        printf(
            "\n\nWARNING: Too large eigenvalue residual to define convergence "
            "%.2lf\n\n",
            eig_res);
    }
    converged_eig_residue = eig_res;
}

void
set_overlap_residue_threshold(double over_res)
{
    if (over_res > REALTIME_OVERLAP_RESIDUE_TOL)
    {
        printf(
            "\n\nERROR: Overlap residue cannot exceed %.3lf but %.3lf was "
            "given\n\n",
            REALTIME_OVERLAP_RESIDUE_TOL,
            over_res);
        exit(EXIT_FAILURE);
    }
    threshold_impr_ortho = over_res;
}

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
    if (mctdhb->orb_eq->bounds == PERIODIC_BOUNDS)
    {
        set_periodic_bounds(
            mctdhb->state->norb,
            mctdhb->state->grid_size,
            mctdhb->state->orbitals);
    }
    sync_orbital_matrices(mctdhb->orb_eq, mctdhb->state);

    switch (mctdhb->coef_integ_method)
    {
        case LANCZOS:
            propagate_coef_sil(mctdhb, coef_works);
            break;
        case RUNGEKUTTA:
            propagate_coef_rk(mctdhb, coef_works);
            break;
    }

    curr_step++; // Finish in the next step
    sync_integration_new_step(mctdhb, curr_step);
}

void
integration_driver(
    MCTDHBDataStruct mctdhb,
    uint16_t         rec_nsteps,
    char             prefix[],
    double           tend,
    uint8_t          monitor_rate)
{
    IntegratorType time_type;
    Bool           impr_ortho_active;
    uint32_t       prop_steps;
    double         curr_t, ompt_start, ompt_used, prev_e;
    char           custom_prefix[STR_BUFF_SIZE], fname[STR_BUFF_SIZE];
    Carray         orb_next_work, coef_next_work;
    ManyBodyState  psi;

    mctdhb->orb_eq->tend = tend; // User redefinition of final prop time
    psi = mctdhb->state;

    curr_t = 0;
    prop_steps = 0;
    orb_next_work = get_dcomplex_array(psi->norb * psi->grid_size);
    coef_next_work = get_dcomplex_array(psi->space_dim);
    prev_e = creal(total_energy(psi)) / psi->npar;
    time_type = mctdhb->integ_type;
    impr_ortho_active = mctdhb->orb_workspace->impr_ortho;

    screen_integration_monitor_columns();
    screen_integration_monitor(mctdhb);

    // Record initial conditions ********************************************
    switch (time_type)
    {
        case REALTIME:
            append_processed_state(prefix, psi);
            append_timestep_potential(prefix, mctdhb->orb_eq);
            break;
        case IMAGTIME:
            strcpy(custom_prefix, prefix);
            strcat(custom_prefix, "_init");
            record_raw_state(custom_prefix, psi);
            break;
    }
    // initial condition recorded *******************************************

    ompt_start = omp_get_wtime();

    while (curr_t < tend)
    {
        mctdhb_propagate_step(
            mctdhb, orb_next_work, coef_next_work, prop_steps);

        prop_steps++;
        curr_t = prop_steps * mctdhb->orb_eq->tstep;

        // Step recording for real time evolution
        if (time_type == REALTIME && prop_steps % rec_nsteps == 0)
        {
            append_processed_state(prefix, psi);
            append_timestep_potential(prefix, mctdhb->orb_eq);
        }

        // integration monitor indicators
        if (prop_steps % monitor_rate == 0)
        {
            screen_integration_monitor(mctdhb);
            if (time_type == IMAGTIME && check_auto_convergence)
            {
                if (imagtime_check_convergence(mctdhb, &prev_e)) break;
            }
            if (time_type == REALTIME && !impr_ortho_active)
            {
                realtime_check_overlap(mctdhb);
                impr_ortho_active = mctdhb->orb_workspace->impr_ortho;
            }
        }
    }

    ompt_used = (double) (omp_get_wtime() - ompt_start);

    printf(
        "\n\n%" PRIu32 " Total steps propagated in %.0lf(s): ",
        prop_steps,
        ompt_used);
    print_time_used(ompt_used);

    if (check_auto_convergence && time_type == IMAGTIME && curr_t < tend)
    {
        printf(
            "\nThe result converged before expected with "
            "t = %.1lf of %.1lf final time",
            curr_t,
            tend);
    }

    // Additional stuff to record
    switch (time_type)
    {
        case IMAGTIME:
            // record potential once (it is not updated in imagtime)
            set_output_fname(prefix, ONE_BODY_POTENTIAL_REC, fname);
            rarr_column_txt(
                fname, "%.14E", psi->grid_size, mctdhb->orb_eq->pot_grid);
            // record final (hopefully) converged orbitals and coefficients
            record_raw_state(prefix, psi);
            break;
        case REALTIME:
            record_time_interaction(prefix, mctdhb->orb_eq);
            record_time_array(prefix, tend, mctdhb->orb_eq->tstep);
            break;
    }

    free(orb_next_work);
    free(coef_next_work);
}
