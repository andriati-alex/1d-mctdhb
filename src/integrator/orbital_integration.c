#include "assistant/arrays_definition.h"
#include "cpydataio.h"
#include "function_tools/calculus.h"
#include "function_tools/orbital_matrices.h"
#include "integrator/nonlinear_orbitals.h"
#include "integrator/split_linear_orbitals.h"
#include "linalg/basic_linalg.h"
#include "linalg/lapack_interface.h"
#include "mctdhb_types.h"
#include "odesys.h"
#include <stdlib.h>

static void
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
robust_multiorb_projector(
    OrbitalEquation eq_desc,
    uint16_t        M,
    Cmatrix         Orb,
    Cmatrix         Haction,
    Cmatrix         project)
{
    int      s;
    uint16_t i, k, l, Mpos;
    double   dx;
    dcomplex proj;
    Cmatrix  overlap, overlap_inv;
    dx = eq_desc->dx;
    Mpos = eq_desc->grid_size;
    overlap = get_dcomplex_matrix(M, M);
    overlap_inv = get_dcomplex_matrix(M, M);

    set_overlap_matrix(eq_desc, M, Orb, overlap);
    s = cmat_hermitian_inversion(M, overlap, overlap_inv);
    if (s != 0)
    {
        printf("\n\nFailed on Lapack inversion routine ");
        printf("for overlap matrix !\n\n");
        printf("Matrix given was :\n");
        cmat_print(M, M, overlap);
        if (s > 0)
        {
            printf("\nSingular decomposition : %d\n\n", s);
        } else
        {
            printf("\nInvalid argument given : %d\n\n", s);
        }
        exit(EXIT_FAILURE);
    }
#pragma omp parallel for private(k, i, s, l, j, proj) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            proj = 0;
            for (s = 0; s < M; s++)
            {
                for (l = 0; l < M; l++)
                {
                    proj += Orb[s][i] * overlap_inv[s][l] *
                            scalar_product(Mpos, dx, Orb[l], Haction[k]);
                }
            }
            project[k][i] = proj;
        }
    }
}

void
simple_multiorb_projector(
    OrbitalEquation eq_desc,
    uint16_t        M,
    Cmatrix         Orb,
    Cmatrix         Haction,
    Cmatrix         project)
{
    int      s;
    uint16_t i, k, Mpos;
    double   dx;
    dcomplex proj;
    dx = eq_desc->dx;
    Mpos = eq_desc->grid_size;

#pragma omp parallel for private(k, i, s, l, j, proj) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            proj = 0;
            for (s = 0; s < M; s++)
            {
                proj +=
                    Orb[s][i] * scalar_product(Mpos, dx, Orb[s], Haction[k]);
            }
            project[k][i] = proj;
        }
    }
}

void
orb_fullstep_linear_part(MCTDHBDataStruct mctdhb, Cmatrix orb, Cmatrix horb)
{
    uint16_t      n = mctdhb->orb_eq->grid_size - 1;
    uint16_t      norb = mctdhb->state->norb;
    Rarray        V = mctdhb->orb_eq->pot_grid;
    OrbDerivative der_type = mctdhb->orb_der_type;
    switch (der_type)
    {
        case FINITEDIFF:
            for (uint16_t i = 0; i < norb; i++)
            {
                linear_horb_fd(mctdhb->orb_eq, orb[i], horb[i]);
            }
            break;
        case SPECTRAL:
            for (uint16_t i = 0; i < norb; i++)
            {
                linear_horb_fft(
                    &mctdhb->orb_workspace->fft_desc,
                    mctdhb->orb_eq,
                    orb[i],
                    horb[i]);
            }
            break;
        case DVR:
            for (uint16_t i = 0; i < norb; i++)
            {
                carr_rowmajor_times_vec(
                    n, n, mctdhb->orb_workspace->dvr_mat, orb[i], horb[i]);
                for (uint16_t j = 0; j < n; j++) horb[i][j] += V[j] * orb[i][j];
                horb[i][n] = horb[i][0];
            }
            break;
    }
}

void
dodt_fullstep_interface(ComplexODEInputParameters odepar, Carray orb_der)
{
    int              k, j, M, Mpos;
    double           g, dx;
    dcomplex         interPart;
    MCTDHBDataStruct mctdhb;
    ManyBodyState    psi;
    OrbitalEquation  eq_desc;
    OrbitalWorkspace orb_work;

    mctdhb = (MCTDHBDataStruct) odepar->extra_args;
    orb_work = mctdhb->orb_workspace;
    psi = mctdhb->state;
    eq_desc = mctdhb->orb_eq;

    Cmatrix Haction, project, Orb, linhorb;

    M = psi->norb;
    Mpos = psi->grid_size;
    g = eq_desc->g;
    dx = eq_desc->dx;

    Orb = get_dcomplex_matrix(M, Mpos);
    cplx_matrix_set_from_rowmajor(M, Mpos, odepar->y, Orb);

    Haction = get_dcomplex_matrix(M, Mpos);
    project = get_dcomplex_matrix(M, Mpos);

    orb_fullstep_linear_part(mctdhb, Orb, linhorb);

#pragma omp parallel for private(k, j, i, sumMatMul, interPart) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            interPart = orb_interacting_part(k, j, g, psi);
            Haction[k][j] = interPart + linhorb[k][j];
        }
    }

    // APPLY PROJECTOR ON ORBITAL SPACE
    if (orb_work->impr_ortho)
    {
        robust_multiorb_projector(eq_desc, M, Orb, Haction, project);
    } else
    {
        simple_multiorb_projector(eq_desc, M, Orb, Haction, project);
    }

    // subtract projection on orbital space - orthogonal projection
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            orb_der[k * Mpos + j] =
                -I * mctdhb->integ_type_num * (Haction[k][j] - project[k][j]);
        }
    }

    // Release memory
    destroy_dcomplex_matrix(M, Haction);
    destroy_dcomplex_matrix(M, project);
}

void
dodt_splitstep_interface(ComplexODEInputParameters odepar, Carray orb_der)
{
    int              k, j, M, Mpos;
    double           g, dx;
    Rarray           pot;
    MCTDHBDataStruct mctdhb;
    ManyBodyState    psi;
    OrbitalEquation  eq_desc;
    OrbitalWorkspace orb_work;

    mctdhb = (MCTDHBDataStruct) odepar->extra_args;
    orb_work = mctdhb->orb_workspace;
    psi = mctdhb->state;
    eq_desc = mctdhb->orb_eq;

    Cmatrix Orb;

    M = psi->norb;
    Mpos = psi->grid_size;
    g = eq_desc->g;
    dx = eq_desc->dx;

    pot = get_double_array(Mpos);
    if (mctdhb->orb_integ_type == SPLITSTEP_FFT)
    {
        rarrCopy(Mpos, eq_desc->pot_grid, pot);
    } else
    {
        rarrFill(Mpos, 0, pot);
    }

    Orb = get_dcomplex_matrix(M, Mpos);
    cplx_matrix_set_from_rowmajor(M, Mpos, odepar->y, Orb);

#pragma omp parallel for private(k, j, i, sumMatMul, interPart) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            orb_der[k * Mpos + j] =
                -I * mctdhb->integ_type_num *
                (orb_full_nonlinear(k, j, g, psi) + pot[j] * Orb[k][j]);
        }
    }
    free(pot);
}

void
propagate_fullstep_orb_rk(MCTDHBDataStruct mctdhb, Carray orb_next)
{
    uint16_t           grid_size, norb;
    double             dt = mctdhb->orb_eq->tstep;
    ComplexWorkspaceRK rk_work =
        (ComplexWorkspaceRK) mctdhb->orb_workspace->extern_work;
    Carray rk_inp;
    norb = mctdhb->state->norb;
    grid_size = mctdhb->state->grid_size;
    rk_inp = get_dcomplex_array(norb * grid_size);
    cplx_rowmajor_set_from_matrix(
        norb, grid_size, mctdhb->state->orbitals, rk_inp);
    switch (mctdhb->rk_order)
    {
        case RK2:
            cplx_rungekutta2(
                dt,
                0,
                &dodt_fullstep_interface,
                mctdhb,
                rk_work,
                rk_inp,
                orb_next);
            break;
        case RK4:
            cplx_rungekutta4(
                dt,
                0,
                &dodt_fullstep_interface,
                mctdhb,
                rk_work,
                rk_inp,
                orb_next);
            break;
        case RK5:
            cplx_rungekutta5(
                dt,
                0,
                &dodt_fullstep_interface,
                mctdhb,
                rk_work,
                rk_inp,
                orb_next);
            break;
    }
    update_orbital_matrices(mctdhb);
    free(rk_inp);
}

void
propagate_splitstep_orb(MCTDHBDataStruct mctdhb, Carray orb_next)
{
    uint16_t           grid_size, norb;
    double             dt = mctdhb->orb_eq->tstep;
    OrbitalWorkspace   orb_work = mctdhb->orb_workspace;
    ComplexWorkspaceRK rk_work = (ComplexWorkspaceRK) orb_work->extern_work;
    Carray             rk_inp;

    norb = mctdhb->state->norb;
    grid_size = mctdhb->state->grid_size;
    rk_inp = get_dcomplex_array(norb * grid_size);

    advance_linear_crank_nicolson(
        mctdhb->orb_eq,
        norb,
        orb_work->cn_upper,
        orb_work->cn_lower,
        orb_work->cn_mid,
        mctdhb->state->orbitals);

    cplx_rowmajor_set_from_matrix(
        norb, grid_size, mctdhb->state->orbitals, rk_inp);

    switch (mctdhb->rk_order)
    {
        case RK2:
            cplx_rungekutta2(
                dt,
                0,
                &dodt_splitstep_interface,
                mctdhb,
                rk_work,
                rk_inp,
                orb_next);
            break;
        case RK4:
            cplx_rungekutta4(
                dt,
                0,
                &dodt_splitstep_interface,
                mctdhb,
                rk_work,
                rk_inp,
                orb_next);
            break;
        case RK5:
            cplx_rungekutta5(
                dt,
                0,
                &dodt_splitstep_interface,
                mctdhb,
                rk_work,
                rk_inp,
                orb_next);
            break;
    }

    cplx_matrix_set_from_rowmajor(
        norb, grid_size, orb_next, mctdhb->state->orbitals);

    advance_linear_crank_nicolson(
        mctdhb->orb_eq,
        norb,
        orb_work->cn_upper,
        orb_work->cn_lower,
        orb_work->cn_mid,
        mctdhb->state->orbitals);

    update_orbital_matrices(mctdhb);
    free(rk_inp);
}
