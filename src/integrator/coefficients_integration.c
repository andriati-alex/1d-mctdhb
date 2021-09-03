#include "integrator/coefficients_integration.h"
#include "configurational/density_matrices.h"
#include "configurational/hamiltonian.h"
#include "linalg/basic_linalg.h"
#include "linalg/lapack_interface.h"
#include "linalg/multiconfig_lanczos.h"
#include <mkl_lapacke.h>
#include <stdio.h>
#include <stdlib.h>

static void
dcdt_interface(ComplexODEInputParameters inp_params, Carray dcdt)
{
    MCTDHBDataStruct mctdhb = (MCTDHBDataStruct) inp_params->extra_args;
    uint32_t         dim = mctdhb->multiconfig_space->dim;
    dcomplex         type_fac = mctdhb->integ_type_num;
    apply_hamiltonian(
        mctdhb->multiconfig_space,
        inp_params->y,
        mctdhb->state->hob,
        mctdhb->state->hint,
        dcdt);
    for (uint32_t i = 0; i < dim; i++) dcdt[i] = -I * type_fac * dcdt[i];
}

static void
update_matrices(MCTDHBDataStruct mctdhb)
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
propagate_coef_rk(MCTDHBDataStruct mctdhb, Carray cnext)
{
    double             dt = mctdhb->orb_eq->tstep;
    Carray             rk_inp = mctdhb->state->coef;
    ComplexWorkspaceRK rk_work =
        (ComplexWorkspaceRK) mctdhb->coef_workspace->extern_work;
    switch (mctdhb->rk_order)
    {
        case RK2:
            cplx_rungekutta5(
                dt, 0, &dcdt_interface, mctdhb, rk_work, rk_inp, cnext);
            break;
        case RK4:
            cplx_rungekutta4(
                dt, 0, &dcdt_interface, mctdhb, rk_work, rk_inp, cnext);
            break;
        case RK5:
            cplx_rungekutta5(
                dt, 0, &dcdt_interface, mctdhb, rk_work, rk_inp, cnext);
            break;
    }
    carrCopy(mctdhb->multiconfig_space->dim, cnext, mctdhb->state->coef);
    update_matrices(mctdhb);
}

void
propagate_coef_sil(MCTDHBDataStruct mctdhb, Carray C)
{
    int     i, k, j, nc, lm, Liter;
    double  dt;
    Rarray  d, e, eigvec;
    Carray  aux, diag, offdiag, Clanczos;
    Cmatrix lvec;

    WorkspaceLanczos lanczos_work = (WorkspaceLanczos) mctdhb->coef_workspace;

    Liter = lanczos_work->iter;
    dt = mctdhb->orb_eq->tstep;
    nc = mctdhb->multiconfig_space->dim;
    lvec = lanczos_work->lanczos_vectors;
    e = lanczos_work->lapack_offd;
    d = lanczos_work->lapack_diag;
    eigvec = lanczos_work->lapack_eigvec;
    diag = lanczos_work->decomp_diag;
    offdiag = lanczos_work->decomp_offd;
    aux = lanczos_work->transform;
    Clanczos = lanczos_work->coef_lspace;

    offdiag[Liter - 1] = 0;   // Useless
    carrCopy(nc, C, lvec[0]); // Setup initial lanczos vector

    // Call Lanczos to perform tridiagonal symmetric reduction
    lm = lanczos(
        mctdhb->multiconfig_space,
        mctdhb->state->hob,
        mctdhb->state->hint,
        Liter,
        diag,
        offdiag,
        lvec);
    if (lm < Liter)
    {
        printf("\n\nWARNING : ");
        printf("lanczos iterations exit before expected - %d", lm);
        printf("\n\n");
    }

    // Transfer data to use lapack routine
    for (k = 0; k < lm; k++)
    {
        d[k] = creal(diag[k]);    // Supposed to be real
        e[k] = creal(offdiag[k]); // Supposed to be real
        for (j = 0; j < lm; j++) eigvec[k * lm + j] = 0;
    }

    k = LAPACKE_dstev(LAPACK_ROW_MAJOR, 'V', lm, d, e, eigvec, lm);
    if (k != 0)
    {
        printf("\n\nERROR IN DIAGONALIZATION\n\n");
        printf("LAPACK dstev routine returned %d\n\n", k);
        exit(EXIT_FAILURE);
    }

    // Solve exactly the equation in Lanczos vector space. The
    // transformation between the original space and the Lanczos one is
    // given by the Lanczos vectors organize in columns. When we apply such
    // a matrix to 'Clanczos' we need to get just the first Lanczos vector,
    // that is, the coefficient vector in the previous time step we load in
    // Lanczos routine.  In other words our initial condition is what we has
    // in previous time step.
    carrFill(lm, 0, Clanczos);
    Clanczos[0] = 1.0;

    for (k = 0; k < lm; k++)
    { // Solve in diagonal basis and for this apply eigvec trasformation
        aux[k] = 0;
        for (j = 0; j < lm; j++) aux[k] += eigvec[j * lm + k] * Clanczos[j];
        aux[k] = aux[k] * cexp(-I * d[k] * dt);
    }

    for (k = 0; k < lm; k++)
    { // Backward transformation from diagonal representation
        Clanczos[k] = 0;
        for (j = 0; j < lm; j++) Clanczos[k] += eigvec[k * lm + j] * aux[j];
    }

    for (i = 0; i < nc; i++)
    { // Return from Lanczos vector space to configurational
        C[i] = 0;
        for (j = 0; j < lm; j++) C[i] += lvec[j][i] * Clanczos[j];
    }
    carrCopy(nc, C, mctdhb->state->coef);
    update_matrices(mctdhb);
}
