#ifndef MCTDHB_TYPES_DEFINITION_H
#define MCTDHB_TYPES_DEFINITION_H

#include "mctdhb_types.h"

OrbitalEquation
get_orbital_equation(
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

MultiConfiguration
get_multiconf_struct(uint16_t npar, uint16_t norb);

ManyBodyState
get_manybody_state(uint16_t npar, uint16_t norb, uint16_t grid_size);

WorkspaceLanczos
get_lanczos_workspace(uint16_t iter, uint32_t space_dim);

MCTDHBDataStruct
get_mctdhb_struct(
    IntegratorType           integ_type,
    uint16_t                 npar,
    uint16_t                 norb,
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
    time_dependent_parameter inter_param,
    uint16_t                 lanczos_iter);

void
destroy_orbital_equation(OrbitalEquation orbeq);

void
destroy_multiconf_struct(MultiConfiguration multiconf);

void
destroy_manybody_sate(ManyBodyState state);

void
destroy_lanczos_workspace(WorkspaceLanczos lan_work);

void
destroy_mctdhb_struct(MCTDHBDataStruct mctdhb);

#endif
