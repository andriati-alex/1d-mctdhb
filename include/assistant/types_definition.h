/** \file types_definition.h
 *
 * \author Alex Andriati - andriati@if.usp.br
 * \date 2021/August
 * \brief Routines for memory allocation to custom data types
 *
 * \see mctdhb_types.h
 */

#ifndef MCTDHB_TYPES_DEFINITION_H
#define MCTDHB_TYPES_DEFINITION_H

#include "mctdhb_types.h"

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
    time_dependent_parameter interaction);

MultiConfiguration
get_multiconf_struct(uint16_t npar, uint16_t norb);

ManyBodyState
get_manybody_state(uint16_t npar, uint16_t norb, uint16_t grid_size);

WorkspaceLanczos
get_lanczos_workspace(uint16_t iter, uint32_t space_dim);

OrbitalWorkspace
get_orbital_workspace(OrbitalEquation eq_desc, OrbDerivative der_method);

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
    time_dependent_parameter inter_param);

void
set_mctdhb_integrator(
    IntegratorType    integ_type,
    CoefIntegrator    coef_integ_method,
    OrbIntegrator     orb_integ_method,
    OrbDerivative     orb_der_method,
    RungeKuttaOrder   rk_order,
    BoundaryCondition bounds,
    uint16_t          lanczos_iter,
    MCTDHBDataStruct  mctdhb);

void
destroy_orbital_equation(OrbitalEquation orbeq);

void
destroy_multiconf_struct(MultiConfiguration multiconf);

void
destroy_manybody_sate(ManyBodyState state);

void
destroy_lanczos_workspace(WorkspaceLanczos lan_work);

void
destroy_orbital_workspace(OrbitalWorkspace orb_work);

void
destroy_coef_workspace(CoefWorkspace coef_work);

void
destroy_mctdhb_struct(MCTDHBDataStruct mctdhb);

#endif
