#ifndef MCTDHB_DATAIO_H
#define MCTDHB_DATAIO_H

#include "mctdhb_types.h"

#define BUILTIN_TIME_PARAM_INPUT_FMT "%s %lf %lf %lf %lf %lf"
extern char orb_cplx_read_fmt[STR_BUFF_SIZE];
extern char coef_cplx_read_fmt[STR_BUFF_SIZE];

void
set_orbitals_from_file(char fname[], ManyBodyState psi);

void
set_coef_from_file(char fname[], uint32_t space_dim, ManyBodyState psi);

MCTDHBDataStruct
get_mctdhb_struct_datafile_line(
    char                     fname[],
    uint32_t                 line,
    single_particle_pot      custom_pot_fun,
    time_dependent_parameter custom_inter_fun,
    void*                    custom_pot_params,
    void*                    custom_inter_params);

#endif
