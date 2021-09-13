#include "integrator/coefficients_integration.h"
#include "configurational/density_matrices.h"
#include "configurational/hamiltonian.h"
#include "linalg/basic_linalg.h"
#include "linalg/lapack_interface.h"
#include "linalg/multiconfig_lanczos.h"
#include "odesys.h"
#include <mkl_lapacke.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void
dcdt_interface(ComplexODEInputParameters inp_params, Carray dcdt)
{
    MCTDHBDataStruct mctdhb = (MCTDHBDataStruct) inp_params->extra_args;
    uint32_t         dim = mctdhb->multiconfig_space->dim;
    dcomplex         type_fac = mctdhb->orb_eq->time_fac;
    apply_hamiltonian(
        mctdhb->multiconfig_space,
        inp_params->y,
        mctdhb->state->hob,
        mctdhb->state->hint,
        dcdt);
    for (uint32_t i = 0; i < dim; i++) dcdt[i] = -I * type_fac * dcdt[i];
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
    if (mctdhb->integ_type == IMAGTIME)
    {
        renormalizeVector(
            mctdhb->multiconfig_space->dim, mctdhb->state->coef, 1.0);
    }
}

void
propagate_coef_sil(MCTDHBDataStruct mctdhb, Carray cnext)
{
    int      k, j, iter_done, sugg_iter;
    uint32_t i, dim;
    dcomplex prop_dt, mat_mul_sum;
    Rarray   d, e, eigvec;
    Carray   aux, diag, offdiag, Clanczos;
    Cmatrix  lvec;

    WorkspaceLanczos lanczos_work = mctdhb->coef_workspace->lan_work;

    sugg_iter = lanczos_work->iter;
    prop_dt = mctdhb->orb_eq->prop_dt;
    dim = mctdhb->multiconfig_space->dim;
    lvec = lanczos_work->lanczos_vectors;
    e = lanczos_work->lapack_offd;
    d = lanczos_work->lapack_diag;
    eigvec = lanczos_work->lapack_eigvec;
    diag = lanczos_work->decomp_diag;
    offdiag = lanczos_work->decomp_offd;
    aux = lanczos_work->transform;
    Clanczos = lanczos_work->coef_lspace;

    offdiag[sugg_iter - 1] = 0;                  // Useless
    carrCopy(dim, mctdhb->state->coef, lvec[0]); // Initial lanczos vector

    // Call Lanczos to perform tridiagonal symmetric reduction
    iter_done = lanczos(
        mctdhb->multiconfig_space,
        mctdhb->state->hob,
        mctdhb->state->hint,
        sugg_iter,
        diag,
        offdiag,
        lvec);
    if (iter_done < sugg_iter)
    {
        printf(
            "\n\nWARNING : lanczos iterations exit before "
            "expected - %d of %d iterations\n\n",
            iter_done,
            sugg_iter);
    }

    // Transfer data to use lapack routine
    for (k = 0; k < iter_done; k++)
    {
        d[k] = creal(diag[k]);    // Supposed to be real
        e[k] = creal(offdiag[k]); // Supposed to be real
        for (j = 0; j < iter_done; j++) eigvec[k * iter_done + j] = 0;
    }

    k = LAPACKE_dstev(
        LAPACK_ROW_MAJOR, 'V', iter_done, d, e, eigvec, iter_done);
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
    carrFill(iter_done, 0, Clanczos);
    Clanczos[0] = 1.0;

    // Solve in diagonal basis and for this apply eigvec trasformation
    for (k = 0; k < iter_done; k++)
    {
        aux[k] = 0;
        for (j = 0; j < iter_done; j++)
            aux[k] += eigvec[j * iter_done + k] * Clanczos[j];
        aux[k] = aux[k] * cexp(-I * d[k] * prop_dt);
    }

    // Backward transformation from diagonal representation
    for (k = 0; k < iter_done; k++)
    {
        Clanczos[k] = 0;
        for (j = 0; j < iter_done; j++)
            Clanczos[k] += eigvec[k * iter_done + j] * aux[j];
    }

    // Return from Lanczos vector space to configurational
#pragma omp parallel for private(i, mat_mul_sum) schedule(static)
    for (i = 0; i < dim; i++)
    {
        mat_mul_sum = 0;
        for (j = 0; j < iter_done; j++) mat_mul_sum += lvec[j][i] * Clanczos[j];
        cnext[i] = mat_mul_sum;
    }
    carrCopy(dim, cnext, mctdhb->state->coef);
    if (mctdhb->integ_type == IMAGTIME)
    {
        renormalizeVector(dim, mctdhb->state->coef, 1.0);
    }
}
