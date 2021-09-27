#include "integrator/synchronize.h"
#include "assistant/arrays_definition.h"
#include "configurational/density_matrices.h"
#include "function_tools/orbital_matrices.h"
#include "linalg/lapack_interface.h"
#include <stdio.h>
#include <stdlib.h>

static double regular_denmat_factor = DEFAULT_REGULARIZATION_FACTOR;

void
set_regulatization_factor(double reg_fac)
{
    if (reg_fac > MAX_REGULARIZATION_FACTOR ||
        reg_fac < MIN_REGULARIZATION_FACTOR)
    {
        printf(
            "\n\nERROR: Regularization factor must lies between "
            "%.1E and %.1E but %.1E was given\n\n",
            MIN_REGULARIZATION_FACTOR,
            MAX_REGULARIZATION_FACTOR,
            reg_fac);
        exit(EXIT_FAILURE);
    }
    regular_denmat_factor = reg_fac;
}

double
get_current_regularization_factor()
{
    return regular_denmat_factor;
}

void
sync_orbital_matrices(OrbitalEquation eq_desc, ManyBodyState psi)
{
    set_orbital_hob(eq_desc, psi->norb, psi->orbitals, psi->hob);
    set_orbital_hint(eq_desc, psi->norb, psi->orbitals, psi->hint);
}

void
sync_density_matrices(MultiConfiguration mc_space, ManyBodyState psi)
{
    Rarray nat_occ;
    double scaled_reg_fac;

    nat_occ = get_double_array(psi->norb);

    set_onebody_dm(mc_space, psi->coef, psi->ob_denmat);
    set_twobody_dm(mc_space, psi->coef, psi->tb_denmat);

    cmat_hermitian_eigenvalues(psi->norb, psi->ob_denmat, nat_occ);
    if (nat_occ[0] / nat_occ[psi->norb - 1] < regular_denmat_factor)
    {
        scaled_reg_fac = nat_occ[psi->norb - 1] * regular_denmat_factor;
        cmat_regularization(psi->norb, scaled_reg_fac, psi->ob_denmat);
    }
    cmat_hermitian_inversion(psi->norb, psi->ob_denmat, psi->inv_ob_denmat);

    free(nat_occ);
}

void
sync_equation_params(OrbitalEquation eq_desc)
{
    eq_desc->g = eq_desc->inter_param(eq_desc->t, eq_desc->inter_extra_args);
    eq_desc->pot_func(
        eq_desc->t,
        eq_desc->grid_size,
        eq_desc->grid_pts,
        eq_desc->pot_extra_args,
        eq_desc->pot_grid);
}

void
sync_integration_new_step(MCTDHBDataStruct mctdhb, uint32_t current_step)
{
    double new_t = current_step * mctdhb->orb_eq->tstep;
    mctdhb->orb_eq->t = new_t;
    if (mctdhb->integ_type == REALTIME) sync_equation_params(mctdhb->orb_eq);
    sync_density_matrices(mctdhb->multiconfig_space, mctdhb->state);
    sync_orbital_matrices(mctdhb->orb_eq, mctdhb->state);
}
