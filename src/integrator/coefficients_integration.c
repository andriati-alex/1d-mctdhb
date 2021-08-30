#include "integrator/coefficients_integration.h"
#include "linalg/basic_linalg.h"
#include "assistant/arrays_definition.h"
#include "configurational/hamiltonian.h"
#include "linalg/multiconfig_lanczos.h"
#include <math.h>
#include <mkl_lapacke.h>
#include <stdio.h>
#include <stdlib.h>

void
realtime_dcdt_odelib(ComplexODEInputParameters inp_params, Carray dcdt)
{
    MCTDHBDataStruct mctdhb = (MCTDHBDataStruct) inp_params->extra_args;
    uint32_t         dim = mctdhb->multiconfig_space->dim;
    apply_hamiltonian(
        mctdhb->multiconfig_space,
        inp_params->y,
        mctdhb->state->hob,
        mctdhb->state->hint,
        dcdt);
    for (uint32_t i = 0; i < dim; i++) dcdt[i] = -I * dcdt[i];
}

void
imagtime_dcdt_odelib(ComplexODEInputParameters inp_params, Carray dcdt)
{
    MCTDHBDataStruct mctdhb = (MCTDHBDataStruct) inp_params->extra_args;
    uint32_t         dim = mctdhb->multiconfig_space->dim;
    apply_hamiltonian(
        mctdhb->multiconfig_space,
        inp_params->y,
        mctdhb->state->hob,
        mctdhb->state->hint,
        dcdt);
    for (uint32_t i = 0; i < dim; i++) dcdt[i] = -dcdt[i];
}

void
iterative_lanczos_integrator(MCTDHBDataStruct mctdhb, Carray C)
{

    /** MULTICONFIGURATIONAL LINEAR SYSTEM INTEGRATION USING LANCZOS
        ============================================================
        Use lanczos to integrate the linear system of equations of the
        configurational coefficients. For more information about  this
        integrator check out:

        "Unitary quantum time evolution by iterative Lanczos recution",
        Tae Jun Park and J.C. Light, J. Chemical Physics 85, 5870, 1986
        DOI 10.1063/1.451548

        INPUT PARAMETERS
            C - initial condition
            Ho - 1-body hamiltonian matrix (coupling to orbitals)
            Hint - 2-body hamiltonian matrix (coupling to orbitals)

        OUTPUT PARAMETERS
            C - End advanced in a time step 'dt' **/

    int     i, k, j, nc, lm, Liter;
    double  dt;
    Rarray  d, e, eigvec;
    Carray  aux, diag, offdiag, Clanczos;
    Cmatrix lvec;

    Liter = mctdhb->lanczos_work->iter;
    dt = mctdhb->orb_eq->tstep;
    nc = mctdhb->multiconfig_space->dim;
    lvec = mctdhb->lanczos_work->lanczos_vectors;
    e = mctdhb->lanczos_work->lapack_offd;
    d = mctdhb->lanczos_work->lapack_diag;
    eigvec = mctdhb->lanczos_work->lapack_eigvec;
    diag = mctdhb->lanczos_work->decomp_diag;
    offdiag = mctdhb->lanczos_work->decomp_offd;
    aux = mctdhb->lanczos_work->transform;

    Clanczos = get_dcomplex_array(Liter);

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

    // Solve exactly the equation in Lanczos vector space. The transformation
    // between the original space and the Lanczos one is given by the Lanczos
    // vectors organize in columns. When we apply such a matrix to 'Clanczos'
    // we need to get just the first Lanczos vector, that is, the coefficient
    // vector in the previous time step we load in Lanczos routine.  In other
    // words our initial condition is what we has in previous time step.
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

    free(Clanczos);
}
