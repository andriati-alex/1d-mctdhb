#ifndef MCTDHB_DATAIO_H
#define MCTDHB_DATAIO_H

#include "mctdhb_types.h"

#define BUILTIN_TIME_PARAM_INPUT_FMT "%s %lf %lf %lf %lf %lf"
extern char orb_cplx_read_fmt[STR_BUFF_SIZE];
extern char coef_cplx_read_fmt[STR_BUFF_SIZE];
extern char inp_dirname[STR_BUFF_SIZE];
extern char out_dirname[STR_BUFF_SIZE];
extern char integrator_desc_fname[STR_BUFF_SIZE];

typedef enum
{
    COMMON_INP,
    MULTIPLE_INP,
    LAST_JOB_OUT
} JobsInputHandle;

void
set_orbitals_from_file(char fname[], ManyBodyState psi);

void
set_coef_from_file(char fname[], uint32_t space_dim, ManyBodyState psi);

MCTDHBDataStruct
get_mctdhb_datafile_line(
    char                     fname[],
    uint32_t                 line,
    single_particle_pot      custom_pot_fun,
    time_dependent_parameter custom_inter_fun,
    void*                    custom_pot_params,
    void*                    custom_inter_params);

void
set_mctdhb_integrator_from_file(char fname[], MCTDHBDataStruct mctdhb);

MCTDHBDataStruct
full_setup_mctdhb_current_dir(
    char                     fprefix[],
    uint32_t                 job_num,
    JobsInputHandle          which_inp,
    single_particle_pot      custom_pot_fun,
    time_dependent_parameter custom_inter_fun,
    void*                    custom_pot_params,
    void*                    custom_inter_params);

#endif
