/** \file types_definition.h
 *
 * \author Alex Andriati - andriati@if.usp.br
 * \date 2021/August
 * \brief Routines for memory allocation to custom data types
 *
 * Most basic routines to create structs from fundamental parameters
 * Better description of datatypes are available in \c mctdhb_types.h
 * which are returned in most functions in this module. The routines
 * starting with \c get_ indicate that a new fresh struct is allocate
 * and \c destroy_ free the memory used in internal fiels of structs
 * and the struct itself
 *
 * \see mctdhb_types.h
 */

#ifndef MCTDHB_TYPES_DEFINITION_H
#define MCTDHB_TYPES_DEFINITION_H

#include "mctdhb_types.h"

/** \brief Allocate orbital equation descriptor struct */
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

/** \brief Allocate multiconfigurational space descriptor struct */
MultiConfiguration
get_multiconf_struct(uint16_t npar, uint16_t norb);

/** \brief Allocate many-body state descriptor struct */
ManyBodyState
get_manybody_state(uint16_t npar, uint16_t norb, uint16_t grid_size);

/** \brief Allocate workspace struct to use Lanczos routines */
WorkspaceLanczos
get_lanczos_workspace(uint16_t iter, uint32_t space_dim);

/** \brief Allocate workspace struct for orbitals propagation routines */
OrbitalWorkspace
get_orbital_workspace(OrbitalEquation eq_desc, OrbDerivative der_method);

/** \brief Allocate the main struct which reference all others */
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

/** \brief Set integrator fields in main struct \c mctdhb */
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
