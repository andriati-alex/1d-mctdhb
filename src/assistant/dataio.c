#include "assistant/dataio.h"
#include "assistant/arrays_definition.h"
#include "assistant/types_definition.h"
#include "cpydataio.h"
#include "function_tools/builtin_potential.h"
#include "function_tools/builtin_time_parameter.h"
#include <stdlib.h>

char orb_cplx_read_fmt[STR_BUFF_SIZE] = " (%lf%lfj) ";
char coef_cplx_read_fmt[STR_BUFF_SIZE] = " (%lf%lfj)";

void
set_orbitals_from_file(char fname[], ManyBodyState psi)
{
    Cmatrix orb_t = get_dcomplex_matrix(psi->grid_size, psi->norb);
    cmat_txt_read(
        fname, orb_cplx_read_fmt, 1, psi->grid_size, psi->norb, orb_t);
    for (uint16_t i = 0; i < psi->norb; i++)
    {
        for (uint16_t j = 0; j < psi->grid_size; j++)
        {
            psi->orbitals[i][j] = orb_t[j][i];
        }
    }
    destroy_dcomplex_matrix(psi->grid_size, orb_t);
}

void
set_coef_from_file(char fname[], uint32_t space_dim, ManyBodyState psi)
{
    carr_txt_read(fname, coef_cplx_read_fmt, 1, space_dim, psi->coef);
}

MCTDHBDataStruct
get_mctdhb_struct_datafile_line(
    char                     fname[],
    uint32_t                 line,
    single_particle_pot      custom_pot_fun,
    time_dependent_parameter custom_inter_fun,
    void*                    custom_pot_params,
    void*                    custom_inter_params)
{
    FILE*    f;
    int      scanf_params;
    uint32_t nlines;
    uint16_t npar, norb, grid_size;
    double   xi, xf, tstep, tend, d2coef, imagpart_d1coef;
    Rarray   gpar_read, potpar_read;
    char     g_func_name[STR_BUFF_SIZE], pot_func_name[STR_BUFF_SIZE];
    single_particle_pot      pot_func;
    time_dependent_parameter g_func;

    void *pot_params, *inter_params;

    nlines = number_of_lines(fname, 1);

    if (line == 0) line = 1;
    if (line > nlines)
    {
        printf(
            "\n\nIOERROR : Requested to read parameters from line %u "
            "but file %s has %u lines\n\n",
            line,
            fname,
            nlines);
        exit(EXIT_FAILURE);
    }

    gpar_read = get_double_array(5);
    potpar_read = get_double_array(5);

    f = open_file(fname, "r");
    jump_comment_lines(f, CURSOR_POSITION);
    while (--line > 0) jump_next_line(f);
    scanf_params = fscanf(
        f,
        "%" SCNu16 " %" SCNu16 " %" SCNu16 " %lf %lf %lf %lf"
        " %lf %lf " BUILTIN_TIME_PARAM_INPUT_FMT
        " " BUILTIN_TIME_PARAM_INPUT_FMT,
        &npar,
        &norb,
        &grid_size,
        &xi,
        &xf,
        &tstep,
        &tend,
        &d2coef,
        &imagpart_d1coef,
        g_func_name,
        &gpar_read[0],
        &gpar_read[1],
        &gpar_read[2],
        &gpar_read[3],
        &gpar_read[4],
        pot_func_name,
        &potpar_read[0],
        &potpar_read[1],
        &potpar_read[2],
        &potpar_read[3],
        &potpar_read[4]);
    pot_func = get_builtin_pot(pot_func_name);
    g_func = get_builtin_param_func(g_func_name);
    pot_params = (void*) potpar_read;
    inter_params = (void*) gpar_read;
    if (pot_func == NULL)
    {
        pot_func = custom_pot_fun;
        pot_params = custom_pot_params;
        free(potpar_read);
    }
    if (g_func == NULL)
    {
        g_func = custom_inter_fun;
        inter_params = custom_inter_params;
        free(gpar_read);
    }
    return get_mctdhb_struct(
        DEFAULT_INTEGRATION_TYPE,
        DEFAULT_COEF_INTEGRATOR,
        DEFAULT_ORB_INTEGRATOR,
        DEFAULT_ORB_DERIVATIVE,
        DEFAULT_RUNGEKUTTA_ORDER,
        npar,
        norb,
        pot_func_name,
        DEFAULT_BOUNDARY_CONDITION,
        xi,
        xf,
        grid_size,
        tstep,
        tend,
        d2coef,
        I * imagpart_d1coef,
        pot_params,
        inter_params,
        pot_func,
        g_func,
        DEFAULT_LANCZOS_ITER);
}
