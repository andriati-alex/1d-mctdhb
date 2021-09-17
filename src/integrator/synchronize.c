#include "integrator/synchronize.h"
#include "assistant/arrays_definition.h"
#include "configurational/density_matrices.h"
#include "function_tools/orbital_matrices.h"
#include "linalg/lapack_interface.h"

void
sync_orbital_matrices(OrbitalEquation eq_desc, ManyBodyState psi)
{
    set_orbital_hob(eq_desc, psi->norb, psi->orbitals, psi->hob);
    set_orbital_hint(eq_desc, psi->norb, psi->orbitals, psi->hint);
}

void
sync_density_matrices(MultiConfiguration mc_space, ManyBodyState psi)
{
    set_onebody_dm(mc_space, psi->coef, psi->ob_denmat);
    set_twobody_dm(mc_space, psi->coef, psi->tb_denmat);
    cmat_hermitian_inversion(psi->norb, psi->ob_denmat, psi->inv_ob_denmat);
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
    double new_t = (current_step + 1) * mctdhb->orb_eq->tstep;
    mctdhb->orb_eq->t = new_t;
    if (mctdhb->integ_type == REALTIME) sync_equation_params(mctdhb->orb_eq);
    sync_density_matrices(mctdhb->multiconfig_space, mctdhb->state);
    sync_orbital_matrices(mctdhb->orb_eq, mctdhb->state);
}
