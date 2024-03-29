#include "assistant/types_definition.h"
#include "assistant/arrays_definition.h"
#include "configurational/space.h"
#include "function_tools/calculus.h"
#include "integrator/split_linear_orbitals.h"
#include "odesys.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
set_grid_points(uint16_t grid_size, double x0, double dx, Rarray pts)
{
    for (uint16_t i = 0; i < grid_size; i++) pts[i] = x0 + i * dx;
}

static void
assert_mkl_descriptor(MKL_LONG status)
{
    if (status != 0)
    {
        printf(
            "\n\nMKL_DFTI ERROR: Reported message:\n%s\n\n",
            DftiErrorMessage(status));
        exit(EXIT_FAILURE);
    }
}

OrbitalEquation
get_orbital_equation(
    char                     eq_name[],
    uint16_t                 norb,
    uint16_t                 grid_size,
    BoundaryCondition        bounds,
    IntegratorType           integ_type,
    double                   xi,
    double                   xf,
    double                   tstep,
    double                   tend,
    double                   d2coef,
    dcomplex                 d1coef,
    void*                    pot_extra_args,
    void*                    inter_extra_args,
    single_particle_pot      pot_func,
    time_dependent_parameter inter_param)
{
    if (grid_size > MAX_GRID_SIZE)
    {
        printf("\n\nERROR: Exceeded max grid size %d\n\n", MAX_GRID_SIZE);
        exit(EXIT_FAILURE);
    }
    if (grid_size < MIN_GRID_SIZE)
    {
        printf("\n\nERROR: Below min grid size %d\n\n", MIN_GRID_SIZE);
        exit(EXIT_FAILURE);
    }
    if (norb > MAX_ORBITALS)
    {
        printf(
            "\n\nERROR: Exceeded max number of orbitals %d\n\n", MAX_ORBITALS);
        exit(EXIT_FAILURE);
    }
    if (pot_func == NULL)
    {
        printf("\n\nTYPEERROR: Null pointer given for single particle "
               "(trap)potential. Require single_particle_pot type\n\n");
        exit(EXIT_FAILURE);
    }
    if (inter_param == NULL)
    {
        printf("\n\nTYPEERROR: Null pointer given for interaction parameter. "
               "Require time_dependent_parameter type\n\n");
        exit(EXIT_FAILURE);
    }
    OrbitalEquation orbeq = malloc(sizeof(_OrbitalEquation));
    strcpy(orbeq->eq_name, eq_name);
    orbeq->t = 0;
    orbeq->norb = norb;
    orbeq->grid_size = grid_size;
    orbeq->bounds = bounds;
    orbeq->xi = xi;
    orbeq->xf = xf;
    orbeq->dx = (xf - xi) / (grid_size - 1);
    orbeq->tstep = tstep;
    orbeq->tend = tend;
    if (integ_type == IMAGTIME)
    {
        orbeq->prop_dt = -I * tstep;
        orbeq->time_fac = -1.0 * I;
    } else
    {
        orbeq->prop_dt = tstep;
        orbeq->time_fac = 1.0;
    }
    orbeq->grid_pts = get_double_array(grid_size);
    set_grid_points(grid_size, xi, orbeq->dx, orbeq->grid_pts);
    orbeq->pot_grid = get_double_array(grid_size);
    pot_func(0, grid_size, orbeq->grid_pts, pot_extra_args, orbeq->pot_grid);
    orbeq->pot_extra_args = pot_extra_args;
    orbeq->inter_extra_args = inter_extra_args;
    orbeq->d2coef = d2coef;
    orbeq->d1coef = d1coef;
    orbeq->g = inter_param(0, inter_extra_args);
    orbeq->pot_func = pot_func;
    orbeq->inter_param = inter_param;
    return orbeq;
}

MultiConfiguration
get_multiconf_struct(uint16_t npar, uint16_t norb)
{
    assert_space_parameters(npar, norb);
    MultiConfiguration multiconf =
        (MultiConfiguration) malloc(sizeof(_MultiConfiguration));
    OperatorMappings op_maps =
        (OperatorMappings) malloc(sizeof(_OperatorMappings));
    multiconf->npar = npar;
    multiconf->norb = norb;
    multiconf->dim = space_dimension(npar, norb);
    multiconf->subspaces_dim = get_subspaces_dim(npar, norb);
    multiconf->hash_table = get_hash_table(npar, norb);
    op_maps->strideot = get_uint32_array(multiconf->dim);
    op_maps->stridett = get_uint32_array(multiconf->dim);
    op_maps->map = get_single_jump_map(
        npar, norb, multiconf->subspaces_dim, multiconf->hash_table);
    op_maps->maptt = get_double_diffjump_map(
        npar,
        norb,
        multiconf->subspaces_dim,
        multiconf->hash_table,
        op_maps->stridett);
    op_maps->mapot = get_double_equaljump_map(
        npar,
        norb,
        multiconf->subspaces_dim,
        multiconf->hash_table,
        op_maps->strideot);
    multiconf->op_maps = op_maps;
    return multiconf;
}

ManyBodyState
get_manybody_state(uint16_t npar, uint16_t norb, uint16_t grid_size)
{
    assert_space_parameters(npar, norb);
    ManyBodyState state = (ManyBodyState) malloc(sizeof(_ManyBodyState));
    state->npar = npar;
    state->norb = norb;
    state->grid_size = grid_size;
    state->space_dim = space_dimension(npar, norb);
    state->coef = get_dcomplex_array(state->space_dim);
    state->hint = get_dcomplex_array(norb * norb * norb * norb);
    state->tb_denmat = get_dcomplex_array(norb * norb * norb * norb);
    state->ob_denmat = get_dcomplex_matrix(norb, norb);
    state->inv_ob_denmat = get_dcomplex_matrix(norb, norb);
    state->hob = get_dcomplex_matrix(norb, norb);
    state->orbitals = get_dcomplex_matrix(norb, grid_size);
    return state;
}

WorkspaceLanczos
get_lanczos_workspace(uint16_t iter, uint32_t space_dim)
{
    WorkspaceLanczos lan_work = malloc(sizeof(_WorkspaceLanczos));
    lan_work->iter = iter;
    lan_work->space_dim = space_dim;
    lan_work->coef_lspace = get_dcomplex_array(iter);
    lan_work->decomp_diag = get_dcomplex_array(iter);
    lan_work->decomp_offd = get_dcomplex_array(iter);
    lan_work->lanczos_vectors = get_dcomplex_matrix(iter, space_dim);
    lan_work->lapack_diag = get_double_array(iter);
    lan_work->lapack_offd = get_double_array(iter);
    lan_work->lapack_eigvec = get_double_array(iter * iter);
    lan_work->transform = get_dcomplex_array(iter);
    lan_work->hc = get_dcomplex_array(space_dim);
    lan_work->reortho = get_dcomplex_array(iter);
    return lan_work;
}

CoefWorkspace
get_coef_workspace(
    CoefIntegrator coef_integ_method, uint32_t space_dim, uint16_t lan_it)
{
    CoefWorkspace cwork = (CoefWorkspace) malloc(sizeof(_CoefWorkspace));
    switch (coef_integ_method)
    {
        case LANCZOS:
            cwork->lan_work = get_lanczos_workspace(lan_it, space_dim);
            cwork->extern_work = NULL;
            break;
        case RUNGEKUTTA:
            cwork->lan_work = NULL;
            cwork->extern_work = (void*) get_cplx_rungekutta_ws(space_dim);
            break;
    }
    return cwork;
}

OrbitalWorkspace
get_orbital_workspace(OrbitalEquation eq_desc, OrbDerivative der_method)
{
    uint16_t         grid_size, norb;
    double           fft_scaling;
    MKL_LONG         desc_s;
    OrbitalWorkspace orb_work;

    grid_size = eq_desc->grid_size;
    norb = eq_desc->norb;
    fft_scaling = 1.0 / sqrt((double) grid_size - 1);

    orb_work = (OrbitalWorkspace) malloc(sizeof(_OrbitalWorkspace));
    orb_work->norb = norb;
    orb_work->grid_size = grid_size;
    orb_work->impr_ortho = FALSE;
    orb_work->orb_der_method = der_method;
    orb_work->dvr_mat = get_dcomplex_array(grid_size * grid_size);
    orb_work->cn_upper = get_dcomplex_array(grid_size);
    orb_work->cn_lower = get_dcomplex_array(grid_size);
    orb_work->cn_mid = get_dcomplex_array(grid_size);
    orb_work->orb_work1 = get_dcomplex_matrix(norb, grid_size);
    orb_work->orb_work2 = get_dcomplex_matrix(norb, grid_size);

    // MKL descriptor
    desc_s = DftiCreateDescriptor(
        &orb_work->fft_desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, grid_size - 1);
    assert_mkl_descriptor(desc_s);
    desc_s = DftiSetValue(orb_work->fft_desc, DFTI_FORWARD_SCALE, fft_scaling);
    desc_s = DftiSetValue(orb_work->fft_desc, DFTI_BACKWARD_SCALE, fft_scaling);
    desc_s = DftiCommitDescriptor(orb_work->fft_desc);

    // space required to use external odelib
    orb_work->extern_work = (void*) get_cplx_rungekutta_ws(grid_size * norb);
    orb_work->fft_freq = get_double_array(grid_size - 1);
    orb_work->fft_hder_exp = get_dcomplex_array(grid_size - 1);

    // Set constant arrays needed for linear part evaluation
    set_cn_tridiagonal(
        eq_desc, orb_work->cn_upper, orb_work->cn_lower, orb_work->cn_mid);
    set_fft_freq(grid_size - 1, eq_desc->dx, orb_work->fft_freq);
    set_hder_fftexp(eq_desc, orb_work->fft_hder_exp);
    set_expdvr_mat(eq_desc, orb_work->dvr_mat);
    return orb_work;
}

MCTDHBDataStruct
get_mctdhb_struct(
    IntegratorType           integ_type,
    CoefIntegrator           coef_integ_method,
    OrbIntegrator            orb_integ_method,
    OrbDerivative            orb_der_method,
    RungeKuttaOrder          rk_order,
    uint16_t                 lanczos_iter,
    uint16_t                 npar,
    uint16_t                 norb,
    char                     eq_name[],
    BoundaryCondition        bounds,
    double                   xi,
    double                   xf,
    uint16_t                 grid_size,
    double                   tstep,
    double                   tend,
    double                   d2coef,
    dcomplex                 d1coef,
    void*                    pot_extra_args,
    void*                    inter_extra_args,
    single_particle_pot      pot_func,
    time_dependent_parameter inter_param)
{
    uint32_t         dim = space_dimension(npar, norb);
    MCTDHBDataStruct mctdhb =
        (MCTDHBDataStruct) malloc(sizeof(_MCTDHBDataStruct));
    mctdhb->integ_type = integ_type;
    mctdhb->coef_integ_method = coef_integ_method;
    mctdhb->orb_integ_method = orb_integ_method;
    mctdhb->orb_der_method = orb_der_method;
    mctdhb->rk_order = rk_order;
    mctdhb->orb_eq = get_orbital_equation(
        eq_name,
        norb,
        grid_size,
        bounds,
        integ_type,
        xi,
        xf,
        tstep,
        tend,
        d2coef,
        d1coef,
        pot_extra_args,
        inter_extra_args,
        pot_func,
        inter_param);
    if (coef_integ_method == LANCZOS)
    {
        if (lanczos_iter > MAX_LANCZOS_ITER || lanczos_iter < 2)
        {
            printf(
                "\n\nERROR: Lanczos iterations must be between %d and %d "
                "but %u was requested\n\n",
                2,
                MAX_LANCZOS_ITER,
                lanczos_iter);
            exit(EXIT_FAILURE);
        }
    }
    mctdhb->coef_workspace =
        get_coef_workspace(coef_integ_method, dim, lanczos_iter);
    mctdhb->multiconfig_space = get_multiconf_struct(npar, norb);
    mctdhb->state = get_manybody_state(npar, norb, grid_size);
    mctdhb->orb_workspace =
        get_orbital_workspace(mctdhb->orb_eq, orb_der_method);
    return mctdhb;
}

void
set_mctdhb_integrator(
    IntegratorType    integ_type,
    CoefIntegrator    coef_integ_method,
    OrbIntegrator     orb_integ_method,
    OrbDerivative     orb_der_method,
    RungeKuttaOrder   rk_order,
    BoundaryCondition bounds,
    uint16_t          lanczos_iter,
    MCTDHBDataStruct  mctdhb)
{
    mctdhb->integ_type = integ_type;
    mctdhb->coef_integ_method = coef_integ_method;
    mctdhb->orb_integ_method = orb_integ_method;
    mctdhb->orb_der_method = orb_der_method;
    mctdhb->rk_order = rk_order;
    mctdhb->orb_eq->bounds = bounds;
    if (mctdhb->coef_integ_method == LANCZOS)
    {
        mctdhb->coef_workspace->lan_work->iter = lanczos_iter;
    }
}

void
destroy_orbital_equation(OrbitalEquation orbeq)
{
    free(orbeq->grid_pts);
    free(orbeq->pot_grid);
    free(orbeq);
}

void
destroy_multiconf_struct(MultiConfiguration multiconf)
{
    free(multiconf->subspaces_dim);
    free(multiconf->hash_table);
    free(multiconf->op_maps->strideot);
    free(multiconf->op_maps->stridett);
    free(multiconf->op_maps->map);
    free(multiconf->op_maps->mapot);
    free(multiconf->op_maps->maptt);
    free(multiconf->op_maps);
    free(multiconf);
}

void
destroy_manybody_sate(ManyBodyState state)
{
    destroy_dcomplex_matrix(state->norb, state->orbitals);
    destroy_dcomplex_matrix(state->norb, state->ob_denmat);
    destroy_dcomplex_matrix(state->norb, state->inv_ob_denmat);
    destroy_dcomplex_matrix(state->norb, state->hob);
    free(state->tb_denmat);
    free(state->hint);
    free(state->coef);
    free(state);
}

void
destroy_lanczos_workspace(WorkspaceLanczos lan_work)
{
    if (lan_work == NULL) return;
    free(lan_work->coef_lspace);
    free(lan_work->decomp_diag);
    free(lan_work->decomp_offd);
    destroy_dcomplex_matrix(lan_work->iter, lan_work->lanczos_vectors);
    free(lan_work->lapack_diag);
    free(lan_work->lapack_offd);
    free(lan_work->lapack_eigvec);
    free(lan_work->transform);
    free(lan_work->hc);
    free(lan_work->reortho);
    free(lan_work);
}

void
destroy_orbital_workspace(OrbitalWorkspace orb_work)
{
    free(orb_work->dvr_mat);
    free(orb_work->cn_upper);
    free(orb_work->cn_lower);
    free(orb_work->cn_mid);
    destroy_dcomplex_matrix(orb_work->norb, orb_work->orb_work1);
    destroy_dcomplex_matrix(orb_work->norb, orb_work->orb_work2);
    MKL_LONG free_status = DftiFreeDescriptor(&orb_work->fft_desc);
    assert_mkl_descriptor(free_status);
    destroy_cplx_rungekutta_ws((ComplexWorkspaceRK) orb_work->extern_work);
    free(orb_work->fft_freq);
    free(orb_work->fft_hder_exp);
    free(orb_work);
}

void
destroy_coef_workspace(CoefWorkspace coef_work)
{
    if (coef_work->lan_work != NULL)
    {
        destroy_lanczos_workspace(coef_work->lan_work);
    }
    if (coef_work->extern_work != NULL)
    {
        destroy_cplx_rungekutta_ws((ComplexWorkspaceRK) coef_work->extern_work);
    }
    free(coef_work);
}

void
destroy_mctdhb_struct(MCTDHBDataStruct mctdhb)
{
    destroy_orbital_equation(mctdhb->orb_eq);
    destroy_manybody_sate(mctdhb->state);
    destroy_orbital_workspace(mctdhb->orb_workspace);
    destroy_multiconf_struct(mctdhb->multiconfig_space);
    destroy_coef_workspace(mctdhb->coef_workspace);
    free(mctdhb);
}
