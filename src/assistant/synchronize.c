#include "assistant/synchronize.h"
#include "assistant/arrays_definition.h"
#include "configurational/density_matrices.h"
#include "function_tools/orbital_matrices.h"
#include "linalg/lapack_interface.h"

char orb_read_fmt[] = " (%lf,%lf) ";
char coef_read_fmt[] = " (%lf,%lf)";

void
sync_orbital_matrices(MCTDHBDataStruct mctdhb)
{
    set_orbital_hob(
        mctdhb->orb_eq,
        mctdhb->state->norb,
        mctdhb->state->orbitals,
        mctdhb->state->hob);
    set_orbital_hint(
        mctdhb->orb_eq,
        mctdhb->state->norb,
        mctdhb->state->orbitals,
        mctdhb->state->hint);
}

void
sync_density_matrices(MCTDHBDataStruct mctdhb)
{
    set_onebody_dm(
        mctdhb->multiconfig_space,
        mctdhb->state->coef,
        mctdhb->state->ob_denmat);
    set_twobody_dm(
        mctdhb->multiconfig_space,
        mctdhb->state->coef,
        mctdhb->state->tb_denmat);
    cmat_hermitian_inversion(
        mctdhb->state->norb,
        mctdhb->state->ob_denmat,
        mctdhb->state->inv_ob_denmat);
}

void
sync_equation_params(double t, OrbitalEquation eq_desc)
{
    eq_desc->g = eq_desc->inter_param(t, eq_desc->inter_extra_args);
    eq_desc->pot_func(
        t,
        eq_desc->grid_size,
        eq_desc->grid_pts,
        eq_desc->pot_extra_args,
        eq_desc->pot_grid);
}
