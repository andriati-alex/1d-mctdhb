#include "integrator/orbital_integration.h"
#include "assistant/arrays_definition.h"
#include "cpydataio.h"
#include "function_tools/calculus.h"
#include "function_tools/orbital_matrices.h"
#include "integrator/split_linear_orbitals.h"
#include "integrator/split_nonlinear_orbitals.h"
#include "integrator/synchronize.h"
#include "linalg/basic_linalg.h"
#include "linalg/lapack_interface.h"
#include <stdlib.h>

void
set_periodic_bounds(uint16_t norb, uint16_t grid_size, Cmatrix orb)
{
    for (uint16_t i = 0; i < norb; i++)
    {
        orb[i][grid_size - 1] = orb[i][0];
    }
}

void
robust_multiorb_projector(
    OrbitalEquation eq_desc,
    uint16_t        norb,
    Cmatrix         orb,
    Cmatrix         hact_orb,
    Cmatrix         project)
{
    int      s;
    uint16_t i, k, l, npts;
    double   dx;
    dcomplex proj;
    Carray   proj_overlap;
    Cmatrix  overlap, overlap_inv;

    dx = eq_desc->dx;
    npts = eq_desc->grid_size;
    proj_overlap = get_dcomplex_array(norb * norb);
    overlap = get_dcomplex_matrix(norb, norb);
    overlap_inv = get_dcomplex_matrix(norb, norb);

    set_overlap_matrix(norb, npts, dx, orb, overlap);
    s = cmat_hermitian_inversion(norb, overlap, overlap_inv);

    if (s != 0)
    {
        printf("\n\nFailed on Lapack inversion routine ");
        printf("for overlap matrix !\n\n");
        printf("Matrix given was :\n");
        cmat_print(norb, norb, overlap);
        if (s > 0)
        {
            printf("\nSingular decomposition : %d\n\n", s);
        } else
        {
            printf("\nInvalid argument given : %d\n\n", s);
        }
        exit(EXIT_FAILURE);
    }

    for (k = 0; k < norb; k++)
    {
        for (s = 0; s < norb; s++)
        {
            proj_overlap[s * norb + k] =
                scalar_product(npts, dx, orb[s], hact_orb[k]);
        }
    }

#pragma omp parallel for private(k, i, s, l, proj) schedule(static)
    for (k = 0; k < norb; k++)
    {
        for (i = 0; i < npts; i++)
        {
            proj = 0;
            for (s = 0; s < norb; s++)
            {
                for (l = 0; l < norb; l++)
                {
                    proj += orb[s][i] * overlap_inv[s][l] *
                            proj_overlap[l * norb + k];
                }
            }
            project[k][i] = proj;
        }
    }

    free(proj_overlap);
    destroy_dcomplex_matrix(norb, overlap);
    destroy_dcomplex_matrix(norb, overlap_inv);
}

void
simple_multiorb_projector(
    OrbitalEquation eq_desc,
    uint16_t        norb,
    Cmatrix         orb,
    Cmatrix         hact_orb,
    Cmatrix         project)
{
    uint16_t i, s, k, npts;
    double   dx;
    dcomplex proj;
    Carray   proj_overlap;

    dx = eq_desc->dx;
    npts = eq_desc->grid_size;
    proj_overlap = get_dcomplex_array(norb * norb);

    for (k = 0; k < norb; k++)
    {
        for (s = 0; s < norb; s++)
        {
            proj_overlap[s * norb + k] =
                scalar_product(npts, dx, orb[s], hact_orb[k]);
        }
    }

#pragma omp parallel for private(k, i, s, proj) schedule(static)
    for (k = 0; k < norb; k++)
    {
        for (i = 0; i < npts; i++)
        {
            proj = 0;
            for (s = 0; s < norb; s++)
            {
                proj += orb[s][i] * proj_overlap[s * norb + k];
            }
            project[k][i] = proj;
        }
    }
    free(proj_overlap);
}

static void
orb_fullstep_linear_part(MCTDHBDataStruct mctdhb, Cmatrix orb, Cmatrix horb)
{
    uint16_t      i, n, norb;
    Rarray        pot;
    OrbDerivative der_method;

    n = mctdhb->orb_eq->grid_size - 1;
    norb = mctdhb->state->norb;
    pot = mctdhb->orb_eq->pot_grid;
    der_method = mctdhb->orb_der_method;

    switch (der_method)
    {
        case FINITEDIFF:
            for (i = 0; i < norb; i++)
            {
                linear_horb_fd(mctdhb->orb_eq, orb[i], horb[i]);
            }
            break;
        case SPECTRAL:
            for (i = 0; i < norb; i++)
            {
                linear_horb_fft(
                    mctdhb->orb_eq, mctdhb->orb_workspace, orb[i], horb[i]);
            }
            break;
        case DVR:
            for (i = 0; i < norb; i++)
            {
                carr_rowmajor_times_vec(
                    n, n, mctdhb->orb_workspace->dvr_mat, orb[i], horb[i]);
                for (uint16_t j = 0; j < n; j++)
                {
                    horb[i][j] += pot[j] * orb[i][j];
                }
                horb[i][n] = horb[i][0];
            }
            break;
    }
}

void
dodt_fullstep_interface(ComplexODEInputParameters odepar, Carray orb_der)
{
    int              k, j, norb, npts;
    double           g, dx;
    dcomplex         time_fac;
    Cmatrix          Haction, project, linhorb;
    MCTDHBDataStruct mctdhb;
    ManyBodyState    psi;
    OrbitalEquation  eq_desc;
    OrbitalWorkspace orb_work;

    mctdhb = (MCTDHBDataStruct) odepar->extra_args;
    orb_work = mctdhb->orb_workspace;
    psi = mctdhb->state;
    eq_desc = mctdhb->orb_eq;

    time_fac = mctdhb->orb_eq->time_fac;
    norb = psi->norb;
    npts = psi->grid_size;
    g = eq_desc->g;
    dx = eq_desc->dx;

    cplx_matrix_set_from_rowmajor(norb, npts, odepar->y, psi->orbitals);

    Haction = mctdhb->orb_workspace->orb_work1;
    project = mctdhb->orb_workspace->orb_work2;
    linhorb = get_dcomplex_matrix(norb, npts);

    orb_fullstep_linear_part(mctdhb, psi->orbitals, linhorb);

#pragma omp parallel for private(k, j) schedule(static)
    for (k = 0; k < norb; k++)
    {
        for (j = 0; j < npts; j++)
        {
            Haction[k][j] = orb_interacting_part(k, j, g, psi) + linhorb[k][j];
        }
    }

    // apply projector on orbital space
    if (orb_work->impr_ortho)
    {
        robust_multiorb_projector(
            eq_desc, norb, psi->orbitals, Haction, project);
    } else
    {
        simple_multiorb_projector(
            eq_desc, norb, psi->orbitals, Haction, project);
    }

    // subtract projection on orbital space - orthogonal projection
    for (k = 0; k < norb; k++)
    {
        for (j = 0; j < npts; j++)
        {
            orb_der[k * npts + j] =
                -I * time_fac * (Haction[k][j] - project[k][j]);
        }
    }

    destroy_dcomplex_matrix(norb, linhorb);
}

void
dodt_splitstep_nonlinear_interface(
    ComplexODEInputParameters odepar, Carray orb_der)
{
    uint16_t         k, j, norb, npts;
    double           g;
    dcomplex         time_fac;
    Rarray           pot;
    Cmatrix          orb;
    MCTDHBDataStruct mctdhb;
    ManyBodyState    psi;
    OrbitalEquation  eq_desc;
    OrbitalWorkspace orb_work;

    mctdhb = (MCTDHBDataStruct) odepar->extra_args;
    orb_work = mctdhb->orb_workspace;
    psi = mctdhb->state;
    eq_desc = mctdhb->orb_eq;
    time_fac = mctdhb->orb_eq->time_fac;

    norb = psi->norb;
    npts = psi->grid_size;
    g = eq_desc->g;
    pot = eq_desc->pot_grid;

    orb = mctdhb->orb_workspace->orb_work1;
    cplx_matrix_set_from_rowmajor(norb, npts, odepar->y, orb);
    cplx_matrix_set_from_rowmajor(norb, npts, odepar->y, psi->orbitals);

#pragma omp parallel for private(k, j) schedule(static)
    for (k = 0; k < norb; k++)
    {
        for (j = 0; j < npts; j++)
        {
            orb_der[k * npts + j] =
                -I * time_fac *
                (orb_full_nonlinear(k, j, g, psi) + pot[j] * orb[k][j]);
        }
    }
}

void
propagate_fullstep_orb_rk(MCTDHBDataStruct mctdhb, Carray orb_next)
{
    uint16_t           grid_size, norb;
    double             dt;
    Carray             rk_inp;
    ComplexWorkspaceRK rk_work;

    rk_work = (ComplexWorkspaceRK) mctdhb->orb_workspace->extern_work;
    dt = mctdhb->orb_eq->tstep;
    norb = mctdhb->state->norb;
    grid_size = mctdhb->state->grid_size;
    rk_inp = get_dcomplex_array(norb * grid_size);

    if (mctdhb->orb_eq->bounds == PERIODIC_BOUNDS)
    {
        set_periodic_bounds(norb, grid_size, mctdhb->state->orbitals);
    }

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

    cplx_matrix_set_from_rowmajor(
        norb, grid_size, orb_next, mctdhb->state->orbitals);

    if (mctdhb->integ_type == IMAGTIME)
    {
        orthonormalize(
            grid_size, mctdhb->orb_eq->dx, norb, mctdhb->state->orbitals);
    }

    free(rk_inp);
}

void
propagate_splitstep_orb(MCTDHBDataStruct mctdhb, Carray orb_next)
{
    uint16_t           grid_size, norb;
    double             dt;
    OrbitalWorkspace   orb_work;
    ManyBodyState      psi;
    ComplexWorkspaceRK rk_work;
    Carray             rk_inp;

    orb_work = mctdhb->orb_workspace;
    psi = mctdhb->state;
    dt = mctdhb->orb_eq->tstep;
    norb = orb_work->norb;
    grid_size = orb_work->grid_size;
    rk_work = (ComplexWorkspaceRK) orb_work->extern_work;
    rk_inp = get_dcomplex_array(norb * grid_size);

    if (mctdhb->orb_eq->bounds == PERIODIC_BOUNDS)
    {
        set_periodic_bounds(norb, grid_size, psi->orbitals);
    }

    if (mctdhb->orb_der_method == FINITEDIFF)
    {
        advance_linear_crank_nicolson(mctdhb->orb_eq, orb_work, psi->orbitals);
    } else
    {
        advance_linear_fft(orb_work, psi->orbitals);
    }

    sync_orbital_matrices(mctdhb->orb_eq, psi);
    cplx_rowmajor_set_from_matrix(norb, grid_size, psi->orbitals, rk_inp);
    switch (mctdhb->rk_order)
    {
        case RK2:
            cplx_rungekutta2(
                dt,
                0,
                &dodt_splitstep_nonlinear_interface,
                mctdhb,
                rk_work,
                rk_inp,
                orb_next);
            break;
        case RK4:
            cplx_rungekutta4(
                dt,
                0,
                &dodt_splitstep_nonlinear_interface,
                mctdhb,
                rk_work,
                rk_inp,
                orb_next);
            break;
        case RK5:
            cplx_rungekutta5(
                dt,
                0,
                &dodt_splitstep_nonlinear_interface,
                mctdhb,
                rk_work,
                rk_inp,
                orb_next);
            break;
    }

    cplx_matrix_set_from_rowmajor(norb, grid_size, orb_next, psi->orbitals);

    if (mctdhb->orb_der_method == FINITEDIFF)
    {
        advance_linear_crank_nicolson(mctdhb->orb_eq, orb_work, psi->orbitals);
    } else
    {
        advance_linear_fft(orb_work, psi->orbitals);
    }

    if (mctdhb->integ_type == IMAGTIME)
    {
        orthonormalize(grid_size, mctdhb->orb_eq->dx, norb, psi->orbitals);
    }

    free(rk_inp);
}
