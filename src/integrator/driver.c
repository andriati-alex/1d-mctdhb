#include "integrator/driver.h"
#include "assistant/arrays_definition.h"
#include "assistant/dataio.h"
#include "assistant/integrator_monitor.h"
#include "cpydataio.h"
#include "integrator/coefficients_integration.h"
#include "integrator/orbital_integration.h"
#include "integrator/synchronize.h"
#include "linalg/multiconfig_lanczos.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

static Bool    imagtime_diagonalization = TRUE;
static Bool    check_auto_convergence = FALSE;
static uint8_t converged_energy_digits = 11;
static double  converged_eig_residue = 1E-6;
static double  threshold_impr_ortho = 1E-8;

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

static uint16_t
get_appropriate_lanczos_iterations(uint32_t space_dim)
{
    if (space_dim / 2 > 50) return 50;
    if (space_dim < 4) return 0;
    return space_dim / 2;
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
    Bool impr;

    impr = mctdhb->orb_workspace->impr_ortho;
    o = overlap_residual(
        mctdhb->orb_eq->norb,
        mctdhb->orb_eq->grid_size,
        mctdhb->orb_eq->dx,
        mctdhb->state->orbitals);

    if (o > threshold_impr_ortho && !impr)
    {
        printf("\n\n== Start improving orthogonality ==\n\n");
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
set_final_diagonalization(Bool shall_diag)
{
    imagtime_diagonalization = shall_diag;
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

    // advante one time step in equation parameters
    mctdhb->orb_eq->t = (curr_step + 1) * mctdhb->orb_eq->tstep;
    if (mctdhb->integ_type == REALTIME) sync_equation_params(mctdhb->orb_eq);

    if (mctdhb->orb_eq->bounds == PERIODIC_BOUNDS)
    {
        set_periodic_bounds(
            mctdhb->state->norb,
            mctdhb->state->grid_size,
            mctdhb->state->orbitals);
    }
    sync_orbital_matrices(mctdhb->orb_eq, mctdhb->state);

    // If have only one orbital (GP-case) coefficients are trivial
    if (mctdhb->state->norb == 1) return;

    switch (mctdhb->coef_integ_method)
    {
        case LANCZOS:
            propagate_coef_sil(mctdhb, coef_works);
            break;
        case RUNGEKUTTA:
            propagate_coef_rk(mctdhb, coef_works);
            break;
    }
    sync_density_matrices(mctdhb->multiconfig_space, mctdhb->state);
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
    uint16_t       lan_it;
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
            toggle_new_append_files();
            append_processed_state(prefix, psi);
            append_timestep_potential(prefix, mctdhb->orb_eq);
            toggle_new_append_files();
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
            if (time_type == REALTIME)
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

    // Additional stuff to record
    switch (time_type)
    {
        case IMAGTIME:
            if (check_auto_convergence && curr_t < tend)
            {
                printf(
                    "\nThe result converged before expected with "
                    "t = %.1lf of %.1lf total propagation time",
                    curr_t,
                    tend);
            }
            // In case auto-convergence criteria is not fulfilled
            // finish with exact diagonalization within current orbitals
            lan_it = get_appropriate_lanczos_iterations(psi->space_dim);
            if (imagtime_diagonalization && curr_t >= tend && lan_it > 0)
            {
                lowest_state_lanczos(
                    lan_it,
                    mctdhb->multiconfig_space,
                    psi->hob,
                    psi->hint,
                    psi->coef);
                printf("\n\n== Additional coefficients diagonalization ==\n");
                screen_integration_monitor(mctdhb);
            }
            // record potential once (it is not updated in imagtime)
            set_output_fname(prefix, ONE_BODY_POTENTIAL_REC, fname);
            rarr_column_txt(
                fname, "%.14E", psi->grid_size, mctdhb->orb_eq->pot_grid);
            // record final (hopefully) converged orbitals and coefficients
            record_raw_state(prefix, psi);
            break;
        case REALTIME:
            record_time_interaction(prefix, mctdhb->orb_eq, rec_nsteps);
            record_time_array(prefix, tend, mctdhb->orb_eq->tstep, rec_nsteps);
            break;
    }

    free(orb_next_work);
    free(coef_next_work);
}
