#include "assistant/dataio.h"
#include "assistant/arrays_definition.h"
#include "assistant/integrator_monitor.h"
#include "assistant/synchronize.h"
#include "assistant/types_definition.h"
#include "cpydataio.h"
#include "function_tools/builtin_potential.h"
#include "function_tools/builtin_time_parameter.h"
#include "linalg/basic_linalg.h"
#include <stdlib.h>
#include <string.h>

char orb_cplx_read_fmt[STR_BUFF_SIZE] = " (%lf%lfj) ";
char coef_cplx_read_fmt[STR_BUFF_SIZE] = " (%lf%lfj)";
char inp_dirname[STR_BUFF_SIZE] = "input/";
char out_dirname[STR_BUFF_SIZE] = "output/";
char integrator_desc_fname[STR_BUFF_SIZE] = "mctdhb_integrator.conf";

static void
report_config_integ_error(char fname[], uint8_t val_read, char extra_info[])
{
    printf(
        "\n\nIOERROR: Reading %" SCNu8 " value : %s. Source file name %s\n\n",
        val_read,
        extra_info,
        fname);
    exit(EXIT_FAILURE);
}

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
get_mctdhb_datafile_line(
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
            "\n\nIOERROR : Requested to read parameters from line %" SCNu32
            " but file %s has %" SCNu32 " lines\n\n",
            line,
            fname,
            nlines);
        exit(EXIT_FAILURE);
    }

    gpar_read = get_double_array(5);
    potpar_read = get_double_array(5);

    f = open_file(fname, "r");
    jump_comment_lines(f, CURSOR_POSITION);
    for (uint32_t i = line; i > 1; i--) jump_next_line(f);
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
    if (scanf_params != 21)
    {
        printf(
            "\n\nIOERROR: Expected to read 21 in line %u of file %s. "
            "However fscanf returned %u\n\n",
            line,
            fname,
            scanf_params);
        exit(EXIT_FAILURE);
    }
    fclose(f);
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
        DEFAULT_LANCZOS_ITER,
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
        g_func);
}

void
set_mctdhb_integrator_from_file(char fname[], MCTDHBDataStruct mctdhb)
{
    uint8_t lanczos_iter;
    FILE*   f;
    if ((f = fopen(fname, "r")) == NULL)
    {
        printf(
            "\n\nWARNING: could not set integrator descriptor, "
            "file %s not found\n\n",
            fname);
        return;
    }
    jump_comment_lines(f, CURSOR_POSITION);
    if (fscanf(f, "%u", &mctdhb->integ_type) != 1)
    {
        report_config_integ_error(fname, 1, "time integration type");
    }
    jump_comment_lines(f, CURSOR_POSITION);
    if (fscanf(f, "%u", &mctdhb->coef_integ_method) != 1)
    {
        report_config_integ_error(fname, 2, "coef integration method");
    }
    jump_comment_lines(f, CURSOR_POSITION);
    if (fscanf(f, "%u", &mctdhb->orb_integ_method) != 1)
    {
        report_config_integ_error(fname, 3, "orbital integration method");
    }
    jump_comment_lines(f, CURSOR_POSITION);
    if (fscanf(f, "%u", &mctdhb->orb_der_method) != 1)
    {
        report_config_integ_error(fname, 4, "orbital integration method");
    }
    mctdhb->orb_workspace->orb_der_method = mctdhb->orb_der_method;
    jump_comment_lines(f, CURSOR_POSITION);
    if (fscanf(f, "%u", &mctdhb->rk_order) != 1)
    {
        report_config_integ_error(fname, 5, "orbital integration method");
    }
    jump_comment_lines(f, CURSOR_POSITION);
    if (fscanf(f, "%u", &mctdhb->orb_eq->bounds) != 1)
    {
        report_config_integ_error(fname, 6, "orbital integration method");
    }
    jump_comment_lines(f, CURSOR_POSITION);
    if (fscanf(f, "%" SCNu8, &lanczos_iter) != 1)
    {
        report_config_integ_error(fname, 7, "orbital integration method");
    }
    if (mctdhb->coef_integ_method == LANCZOS)
    {
        mctdhb->coef_workspace->lan_work->iter = lanczos_iter;
    }
    fclose(f);
}

MCTDHBDataStruct
full_setup_mctdhb_current_dir(
    char                     fprefix[],
    uint32_t                 job_num,
    JobsInputHandle          which_inp,
    single_particle_pot      custom_pot_fun,
    time_dependent_parameter custom_inter_fun,
    void*                    custom_pot_params,
    void*                    custom_inter_params)
{
    MCTDHBDataStruct mctdhb;
    char             fpath[STR_BUFF_SIZE], job_suffix[20];
    strcpy(fpath, inp_dirname);
    strcat(fpath, fprefix);
    strcat(fpath, "_mctdhb_parameters.dat");
    mctdhb = get_mctdhb_datafile_line(
        fpath,
        job_num,
        custom_pot_fun,
        custom_inter_fun,
        custom_pot_params,
        custom_inter_params);

    // Set orbitals
    strcpy(fpath, inp_dirname);
    strcat(fpath, fprefix);
    switch (which_inp)
    {
        case COMMON_INP:
            strcat(fpath, "_orb1.dat");
            break;
        case MULTIPLE_INP:
            sprintf(job_suffix, "_orb%" SCNu32 ".dat", job_num);
            strcat(fpath, job_suffix);
            break;
        case LAST_JOB_OUT:
            if (job_num == 1)
            {
                sprintf(job_suffix, "_orb1.dat");
            } else
            {
                strcpy(fpath, out_dirname);
                strcat(fpath, fprefix);
                sprintf(job_suffix, "_orb%" SCNu32 ".dat", job_num - 1);
            }
            strcat(fpath, job_suffix);
            break;
    }
    set_orbitals_from_file(fpath, mctdhb->state);

    // Set coefficients
    strcpy(fpath, inp_dirname);
    strcat(fpath, fprefix);
    switch (which_inp)
    {
        case COMMON_INP:
            strcat(fpath, "_coef1.dat");
            break;
        case MULTIPLE_INP:
            sprintf(job_suffix, "_coef%" SCNu32 ".dat", job_num);
            strcat(fpath, job_suffix);
            break;
        case LAST_JOB_OUT:
            if (job_num == 1)
            {
                sprintf(job_suffix, "_coef1.dat");
            } else
            {
                strcpy(fpath, out_dirname);
                strcat(fpath, fprefix);
                sprintf(job_suffix, "_coef%" SCNu32 ".dat", job_num - 1);
            }
            strcat(fpath, job_suffix);
            break;
    }
    set_coef_from_file(fpath, mctdhb->multiconfig_space->dim, mctdhb->state);

    set_mctdhb_integrator_from_file(integrator_desc_fname, mctdhb);
    sync_density_matrices(mctdhb);
    sync_orbital_matrices(mctdhb);
    return mctdhb;
}

void
screen_display_mctdhb_info(
    MCTDHBDataStruct mctdhb, Bool disp_integ, Bool disp_mem, Bool disp_monitor)
{
    OrbitalEquation orb_eq = mctdhb->orb_eq;
    sepline('*', 78, 2, 2);
    printf("*\tProblem setup parameters:");
    printf("\n*\tnpar : %" SCNu16, mctdhb->multiconfig_space->npar);
    printf("\n*\tnorb : %" SCNu16, mctdhb->multiconfig_space->norb);
    printf("\n*\tdim  : %" SCNu32, mctdhb->multiconfig_space->dim);
    printf(
        "\n*\tgrid : [%.2lf,%.2lf] with %" SCNu16 " discrete points",
        orb_eq->xi,
        orb_eq->xf,
        orb_eq->grid_size);
    printf(
        "\n*\ttime : Up to %.2lf in steps of %.6lf",
        orb_eq->tend,
        orb_eq->tstep);
    printf("\n*\tname : %s", orb_eq->eq_name);
    if (!disp_integ)
    {
        sepline('*', 78, 2, 2);
        return;
    }
    printf("\n*\n");
    printf("*\tIntegrator setup");
    switch (mctdhb->integ_type)
    {
        case IMAGTIME:
            printf("\n*\ttime : Imaginary time");
            break;
        case REALTIME:
            printf("\n*\ttime : Real time");
            break;
    }
    switch (mctdhb->coef_integ_method)
    {
        case LANCZOS:
            printf(
                "\n*\tcoef : Short Iteration Lanczos(SIL) with %" SCNu16
                " iterations per step",
                mctdhb->coef_workspace->lan_work->iter);
            break;
        case RUNGEKUTTA:
            printf("\n*\tcoef : Runge Kutta");
            break;
    }
    switch (mctdhb->orb_integ_method)
    {
        case FULLSTEP_RUNGEKUTTA:
            printf("\n*\torbs : Runge Kutta in full step");
            break;
        case SPLITSTEP:
            printf("\n*\torbs : Split-Step (Runge Kutta nonlinear)");
            break;
    }
    switch (mctdhb->orb_der_method)
    {
        case DVR:
            printf("\n*\tder  : Using Discrete Variable Representation(DVR)");
            break;
        case SPECTRAL:
            printf("\n*\tder  : Using FFT (MKL implementation)");
            break;
        case FINITEDIFF:
            printf("\n*\tder  : 2nd order finite differences scheme");
            break;
    }
    printf("\n*\tRunge-Kutta global order : %u", mctdhb->rk_order);
    if (!disp_mem)
    {
        sepline('*', 78, 2, 2);
        return;
    }
    MultiConfiguration space = mctdhb->multiconfig_space;
    uint16_t           norb = space->norb;
    uint32_t           dim = space->dim;
    uint64_t           mem_conf = 0, mem_orb = 0;
    // configurational part
    mem_conf += dim * sizeof(dcomplex);
    mem_conf += dim * norb * sizeof(uint16_t);
    mem_conf += dim * norb * norb * sizeof(uint32_t);
    mem_conf += (space->op_maps->stridett[dim - 1]) * sizeof(uint32_t);
    mem_conf +=
        (space->op_maps->strideot[dim - 1] + norb * norb) * sizeof(uint32_t);
    mem_conf += norb * norb * sizeof(dcomplex);
    mem_conf += norb * norb * norb * norb * sizeof(dcomplex);
    // orbital part
    mem_orb += 3 * norb * orb_eq->grid_size * sizeof(dcomplex);
    mem_orb += norb * norb * sizeof(dcomplex);
    mem_orb += norb * norb * norb * norb * sizeof(dcomplex);
    mem_orb += 3 * orb_eq->grid_size * sizeof(dcomplex);
    mem_orb += orb_eq->grid_size * orb_eq->grid_size * sizeof(dcomplex);
    printf("\n*\n");
    printf("*\tEstimated (minimum)memory comsumption");
    printf("\n*\tcoef : %.1lf(MB)", ((double) mem_conf) / 1E6);
    printf("\n*\torbs : %.1lf(MB)", ((double) mem_orb) / 1E6);
    if (!disp_monitor)
    {
        sepline('*', 78, 2, 2);
        return;
    }
    double ores = overlap_residual(
        norb, orb_eq->grid_size, orb_eq->dx, mctdhb->state->orbitals);
    double avg_orb_norm = avg_orbitals_norm(
        norb, orb_eq->grid_size, orb_eq->dx, mctdhb->state->orbitals);
    double cmod = carrMod(dim, mctdhb->state->coef);
    printf("\n*\n");
    printf("*\tSafety integrator indicators");
    printf("\n*\tOverlap residue : %.2E", ores);
    printf("\n*\tAverage norm    : %.10lf", avg_orb_norm);
    printf("\n*\tCoef vec norm   : %.10lf", cmod);
    sepline('*', 78, 2, 2);
}
