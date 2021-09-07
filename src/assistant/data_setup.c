#include "configurational/density_matrices.h"
#include "function_tools/orbital_matrices.h"
#include "linalg/lapack_interface.h"
#include "mctdhb_types.h"

void
update_orbital_matrices(MCTDHBDataStruct mctdhb)
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
update_conf_matrices(MCTDHBDataStruct mctdhb)
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
