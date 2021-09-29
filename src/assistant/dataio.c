#include "assistant/dataio.h"
#include "assistant/arrays_definition.h"
#include "assistant/integrator_monitor.h"
#include "assistant/types_definition.h"
#include "cpydataio.h"
#include "function_tools/builtin_potential.h"
#include "function_tools/builtin_time_parameter.h"
#include "integrator/synchronize.h"
#include "linalg/basic_linalg.h"
#include "linalg/lapack_interface.h"
#include <stdlib.h>
#include <string.h>

char orb_cplx_read_fmt[STR_BUFF_SIZE] = " (%lf%lfj) ";
char coef_cplx_read_fmt[STR_BUFF_SIZE] = " (%lf%lfj)";
char inp_dirname[STR_BUFF_SIZE] = "input/";
char out_dirname[STR_BUFF_SIZE] = "output/";
char integrator_desc_fname[STR_BUFF_SIZE] = "mctdhb_integrator.conf";

uint8_t monitor_energy_digits = 10;
Bool    monitor_disp_min_occ = TRUE;
Bool    monitor_disp_momentum = TRUE;
Bool    monitor_disp_kin_energy = FALSE;
Bool    monitor_disp_int_energy = FALSE;
Bool    monitor_disp_overlap_residue = FALSE;
Bool    monitor_disp_orb_norm = FALSE;
Bool    monitor_disp_coef_norm = FALSE;
Bool    monitor_disp_eig_residue = TRUE;
Bool    new_empty_append_files = FALSE;

static void
report_integrator_warning(char fname[], uint8_t val_read, char extra_info[])
{
    printf(
        "\n\nWARNING: Reading %" SCNu8 " value : %s. Source file name %s\n\n",
        val_read,
        extra_info,
        fname);
}

void
toggle_new_append_files()
{
    if (!new_empty_append_files)
    {
        new_empty_append_files = TRUE;
        return;
    }
    new_empty_append_files = FALSE;
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
set_coef_from_file(char fname[], ManyBodyState psi)
{
    carr_txt_read(fname, coef_cplx_read_fmt, 1, psi->space_dim, psi->coef);
}

MCTDHBDataStruct
get_mctdhb_from_files(
    char                     par_fname[],
    char                     integ_fname[],
    uint32_t                 line,
    single_particle_pot      custom_pot_fun,
    time_dependent_parameter custom_inter_fun,
    void*                    custom_pot_params,
    void*                    custom_inter_params)
{
    FILE*             f;
    int               scanf_params;
    uint8_t           lanczos_iter;
    uint32_t          nlines;
    uint16_t          npar, norb, grid_size;
    double            xi, xf, tstep, tend, d2coef, imagpart_d1coef;
    Bool              had_warn;
    Rarray            gpar_read, potpar_read;
    char              g_func_name[STR_BUFF_SIZE], pot_func_name[STR_BUFF_SIZE];
    IntegratorType    integ_type;
    CoefIntegrator    coef_integ_method;
    OrbIntegrator     orb_integ_method;
    OrbDerivative     orb_der_method;
    RungeKuttaOrder   rk_order;
    BoundaryCondition bounds;

    single_particle_pot      pot_func;
    time_dependent_parameter g_func;

    void *pot_params, *inter_params;

    nlines = number_of_lines(par_fname);

    if (line > nlines)
    {
        printf(
            "\n\nIOERROR : Requested to read parameters from line %" SCNu32
            " but file %s has %" SCNu32 " lines\n\n",
            line,
            par_fname,
            nlines);
        exit(EXIT_FAILURE);
    }

    gpar_read = get_double_array(5);
    potpar_read = get_double_array(5);

    f = open_file(par_fname, "r");

    // ignore all comment lines in the beginning
    jump_comment_lines(f, CURSOR_POSITION);
    for (uint32_t i = line; i > 1; i--) jump_next_line(f);

    scanf_params = fscanf(
        f,
        "%" SCNu16 " %" SCNu16 " %" SCNu16 " %lf %lf %lf %lf"
        " %lf %lf " BUILTIN_TIME_FUNCTION_INPUT_FMT
        " " BUILTIN_TIME_FUNCTION_INPUT_FMT,
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

    fclose(f);

    if (scanf_params != MIN_PARAMS_INLINE)
    {
        printf(
            "\n\nIOERROR: Expected to read %d in line %u of file %s. "
            "However fscanf returned %u\n\n",
            MIN_PARAMS_INLINE,
            line,
            par_fname,
            scanf_params);
        exit(EXIT_FAILURE);
    }

    // choose between file or custom input function
    pot_func = get_builtin_pot(pot_func_name);
    g_func = get_builtin_param_func(g_func_name);
    pot_params = (void*) potpar_read;
    inter_params = (void*) gpar_read;
    if (pot_func == NULL)
    {
        pot_func = custom_pot_fun;
    }
    if (custom_pot_params != NULL)
    {
        pot_params = custom_pot_params;
        free(potpar_read);
    }
    if (g_func == NULL)
    {
        g_func = custom_inter_fun;
    }
    if (custom_inter_params != NULL)
    {
        inter_params = custom_inter_params;
        free(gpar_read);
    }

    had_warn = FALSE;
    if ((f = fopen(integ_fname, "r")) != NULL)
    {
        jump_comment_lines(f, CURSOR_POSITION);
        if (fscanf(f, "%u", &integ_type) != 1)
        {
            report_integrator_warning(integ_fname, 1, "time integration type");
            had_warn = TRUE;
            integ_type = DEFAULT_INTEGRATION_TYPE;
        }
        jump_comment_lines(f, CURSOR_POSITION);
        if (fscanf(f, "%u", &coef_integ_method) != 1)
        {
            report_integrator_warning(
                integ_fname, 2, "coef integration method");
            had_warn = TRUE;
            coef_integ_method = DEFAULT_COEF_INTEGRATOR;
        }
        jump_comment_lines(f, CURSOR_POSITION);
        if (fscanf(f, "%u", &orb_integ_method) != 1)
        {
            report_integrator_warning(
                integ_fname, 3, "orbital integration method");
            had_warn = TRUE;
            orb_integ_method = DEFAULT_ORB_INTEGRATOR;
        }
        jump_comment_lines(f, CURSOR_POSITION);
        if (fscanf(f, "%u", &orb_der_method) != 1)
        {
            report_integrator_warning(integ_fname, 4, "Derivatives method");
            had_warn = TRUE;
            orb_der_method = DEFAULT_ORB_DERIVATIVE;
        }
        jump_comment_lines(f, CURSOR_POSITION);
        if (fscanf(f, "%u", &rk_order) != 1)
        {
            report_integrator_warning(
                integ_fname, 5, "global runge-kutta order");
            had_warn = TRUE;
            rk_order = DEFAULT_RUNGEKUTTA_ORDER;
        }
        jump_comment_lines(f, CURSOR_POSITION);
        if (fscanf(f, "%u", &bounds) != 1)
        {
            report_integrator_warning(integ_fname, 6, "Boundary conditions");
            had_warn = TRUE;
            bounds = DEFAULT_BOUNDARY_CONDITION;
        }
        jump_comment_lines(f, CURSOR_POSITION);
        if (fscanf(f, "%" SCNu8, &lanczos_iter) != 1)
        {
            report_integrator_warning(integ_fname, 7, "Lanczos iterations");
            had_warn = TRUE;
            lanczos_iter = DEFAULT_LANCZOS_ITER;
        }
        fclose(f);
    } else
    {
        printf(
            "\n\nWARNING: could not set integrator descriptor, "
            "file %s not found. Using default values.\n\n",
            integ_fname);
        integ_type = DEFAULT_INTEGRATION_TYPE;
        coef_integ_method = DEFAULT_COEF_INTEGRATOR;
        orb_integ_method = DEFAULT_ORB_INTEGRATOR;
        orb_der_method = DEFAULT_ORB_DERIVATIVE;
        rk_order = DEFAULT_RUNGEKUTTA_ORDER;
        bounds = DEFAULT_BOUNDARY_CONDITION;
        lanczos_iter = DEFAULT_LANCZOS_ITER;
    }

    if (had_warn)
    {
        printf(
            "\n\nWARNING: some of the integration config could not be set "
            "from file %s, which were listed above\n\n",
            integ_fname);
    }

    return get_mctdhb_struct(
        integ_type,
        coef_integ_method,
        orb_integ_method,
        orb_der_method,
        rk_order,
        lanczos_iter,
        npar,
        norb,
        pot_func_name,
        bounds,
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

    char fpath[STR_BUFF_SIZE], job_suffix[20];

    strcpy(fpath, inp_dirname);
    strcat(fpath, fprefix);
    strcat(fpath, PARAMS_FNAME_SUFFIX);
    mctdhb = get_mctdhb_from_files(
        fpath,
        integrator_desc_fname,
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
            strcat(fpath, "_job1_orb.dat");
            break;
        case MULTIPLE_INP:
            sprintf(job_suffix, "_job%" SCNu32 "_orb.dat", job_num);
            strcat(fpath, job_suffix);
            break;
        case LAST_JOB_OUT:
            if (job_num == 1)
            {
                sprintf(job_suffix, "_job1_orb.dat");
            } else
            {
                strcpy(fpath, out_dirname);
                strcat(fpath, fprefix);
                sprintf(job_suffix, "_job%" SCNu32 "_orb.dat", job_num - 1);
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
            strcat(fpath, "_job1_coef.dat");
            break;
        case MULTIPLE_INP:
            sprintf(job_suffix, "_job%" SCNu32 "_coef.dat", job_num);
            strcat(fpath, job_suffix);
            break;
        case LAST_JOB_OUT:
            if (job_num == 1)
            {
                sprintf(job_suffix, "_job1_coef.dat");
            } else
            {
                strcpy(fpath, out_dirname);
                strcat(fpath, fprefix);
                sprintf(job_suffix, "_job%" SCNu32 "_coef.dat", job_num - 1);
            }
            strcat(fpath, job_suffix);
            break;
    }
    set_coef_from_file(fpath, mctdhb->state);

    sync_density_matrices(mctdhb->multiconfig_space, mctdhb->state);
    sync_orbital_matrices(mctdhb->orb_eq, mctdhb->state);
    return mctdhb;
}

void
screen_display_banner()
{
    printf("\n\n");
    printf("'||    ||'   ..|'''.| |''||''|  ||''|.   '||    ||' '||''|.\n"
           " |||  |||  .|'     ''    ||     ||   ||   ||    ||   ||   ||\n"
           " |'|.|'||  ||            ||     ||    ||  ||''''||   ||'''|.\n"
           " | '|' ||  '|.      .    ||     ||    ||  ||    ||   ||    ||\n"
           ".|     ||.  ''|....'    .||.    ||.../'  .||    ||. .||.../'\n");
    printf("\n\n");
}

void
screen_display_mctdhb_info(
    MCTDHBDataStruct mctdhb, Bool disp_integ, Bool disp_mem, Bool disp_monitor)
{
    OrbitalEquation orb_eq = mctdhb->orb_eq;

    sepline('*', 78, 2, 1);
    printf("* Problem setup parameters");
    printf("\n*\tnpar : %" PRIu16, mctdhb->multiconfig_space->npar);
    printf("\n*\tnorb : %" PRIu16, mctdhb->multiconfig_space->norb);
    printf("\n*\tdim  : %" PRIu32, mctdhb->multiconfig_space->dim);
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
        sepline('*', 78, 1, 2);
        return;
    }

    printf("\n*\n");
    printf("* Integrator setup");
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
                "\n*\tcoef : Short Iteration Lanczos(SIL) with %" PRIu16
                " iterations per step",
                mctdhb->coef_workspace->lan_work->iter);
            break;
        case RUNGEKUTTA:
            printf("\n*\tcoef : Runge Kutta of order %u", mctdhb->rk_order);
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
        sepline('*', 78, 1, 2);
        return;
    }

    MultiConfiguration space = mctdhb->multiconfig_space;
    uint16_t           norb = space->norb;
    uint32_t           dim = space->dim;
    uint64_t           mem_conf = 0, mem_orb = 0;

    // configurational part *************************************************
    mem_conf += dim * sizeof(dcomplex);
    mem_conf += dim * norb * sizeof(uint16_t);
    mem_conf += dim * norb * norb * sizeof(uint32_t);
    mem_conf += (space->op_maps->stridett[dim - 1]) * sizeof(uint32_t);
    mem_conf +=
        (space->op_maps->strideot[dim - 1] + norb * norb) * sizeof(uint32_t);
    mem_conf += norb * norb * sizeof(dcomplex);
    mem_conf += norb * norb * norb * norb * sizeof(dcomplex);

    // orbital part *********************************************************
    mem_orb += 3 * norb * orb_eq->grid_size * sizeof(dcomplex); // workspace
    mem_orb += norb * norb * sizeof(dcomplex);                  // ho matrix
    mem_orb += norb * norb * norb * norb * sizeof(dcomplex);    // hint matrix
    // crank-nicolson arrays
    mem_orb += 3 * orb_eq->grid_size * sizeof(dcomplex);
    // DVR matrix
    mem_orb += orb_eq->grid_size * orb_eq->grid_size * sizeof(dcomplex);
    printf("\n*\n");
    printf("* Estimated (minimum)memory comsumption");
    printf("\n*\tcoef : %.1lf(MB)", ((double) mem_conf) / 1E6);
    printf("\n*\torbs : %.1lf(MB)", ((double) mem_orb) / 1E6);

    if (!disp_monitor)
    {
        sepline('*', 78, 1, 2);
        return;
    }
    double ores = overlap_residual(
        norb, orb_eq->grid_size, orb_eq->dx, mctdhb->state->orbitals);
    double avg_orb_norm = avg_orbitals_norm(
        norb, orb_eq->grid_size, orb_eq->dx, mctdhb->state->orbitals);
    double cmod = carrMod(dim, mctdhb->state->coef);

    printf("\n*\n");
    printf("* Safety integrator indicators");
    printf("\n*\tOverlap residue : %.2E", ores);
    printf("\n*\tAverage norm    : %.10lf", avg_orb_norm);
    printf("\n*\tCoef vec norm   : %.10lf", cmod);
    sepline('*', 78, 1, 2);
}

uint32_t
auto_number_of_jobs(char prefix[])
{
    char params_fname[STR_BUFF_SIZE];
    strcpy(params_fname, inp_dirname);
    strcat(params_fname, prefix);
    strcat(params_fname, PARAMS_FNAME_SUFFIX);
    return number_of_lines(params_fname);
}

void
set_output_fname(char prefix[], RecordDataType id, char* fname)
{
    char specific_id[STR_BUFF_SIZE];
    switch (id)
    {
        case ORBITALS_REC:
            strcpy(specific_id, "_orb.dat");
            break;
        case COEFFICIENTS_REC:
            strcpy(specific_id, "_coef.dat");
            break;
        case PARAMETERS_REC:
            strcpy(specific_id, PARAMS_FNAME_SUFFIX);
            break;
        case ONE_BODY_MATRIX_REC:
            strcpy(specific_id, "_obmat.dat");
            break;
        case TWO_BODY_MATRIX_REC:
            strcpy(specific_id, "_tbmat.dat");
            break;
        case ONE_BODY_POTENTIAL_REC:
            strcpy(specific_id, "_obpotential.dat");
            break;
    }
    strcpy(fname, out_dirname);
    strcat(fname, prefix);
    strcat(fname, specific_id);
}

void
screen_integration_monitor_columns()
{
    uint8_t counter = 4;

    printf("\nColumns to print during time propagatio:\n");
    printf("1.[time]  2.energy  3.condensation  ");

    if (monitor_disp_min_occ)
    {
        printf("%" PRIu8 ".min occ  ", counter);
        counter++;
    }

    if (monitor_disp_momentum)
    {
        printf("%" PRIu8 ".momentum  ", counter);
        counter++;
    }

    if (monitor_disp_kin_energy)
    {
        printf("%" PRIu8 ".kinect  ", counter);
        counter++;
    }

    if (monitor_disp_int_energy)
    {
        printf("%" PRIu8 ".interacting  ", counter);
        counter++;
    }

    if (monitor_disp_overlap_residue)
    {
        printf("%" PRIu8 ".overlap residue  ", counter);
        counter++;
    }

    if (monitor_disp_orb_norm)
    {
        printf("%" PRIu8 ".orb norm  ", counter);
        counter++;
    }

    if (monitor_disp_coef_norm)
    {
        printf("%" PRIu8 ".coef norm  ", counter);
        counter++;
    }

    if (monitor_disp_eig_residue)
    {
        printf("%" PRIu8 ".eigvalue residue  ", counter);
        counter++;
    }

    printf("\n");
}

void
screen_integration_monitor(MCTDHBDataStruct mctdhb)
{
    uint16_t npar, norb, grid_size;
    uint32_t space_dim;
    dcomplex energy, kine, inte;
    double   dx, t, tend, over_res, orb_norm, coef_norm, conf_eig_residue, mom;
    char     energy_fmt[STR_BUFF_SIZE];
    Rarray   nat_occ;

    ManyBodyState psi;

    t = mctdhb->orb_eq->t;
    tend = mctdhb->orb_eq->tend;
    dx = mctdhb->orb_eq->dx;
    npar = mctdhb->state->npar;
    norb = mctdhb->state->norb;
    grid_size = mctdhb->orb_eq->grid_size;
    space_dim = mctdhb->state->space_dim;
    nat_occ = get_double_array(norb);
    psi = mctdhb->state;

    sprintf(
        energy_fmt,
        " %%%" PRIu8 ".%" PRIu8 "E",
        monitor_energy_digits + 7,
        monitor_energy_digits);

    energy = total_energy(psi);
    cmat_hermitian_eigenvalues(norb, psi->ob_denmat, nat_occ);

    printf("\n[%7.3lf / %.1lf]", t, tend);
    printf(energy_fmt, creal(energy) / npar);
    printf("%6.2lf", nat_occ[norb - 1] / npar);

    if (monitor_disp_min_occ)
    {
        printf("%10.6lf", nat_occ[0] / npar);
    }

    if (monitor_disp_momentum)
    {
        mom = creal(momentum_per_particle(mctdhb->orb_eq, psi));
        printf("%10.6lf", mom);
    }

    if (monitor_disp_kin_energy)
    {
        kine = kinect_energy(mctdhb->orb_eq, psi) / npar;
        printf(energy_fmt, creal(kine));
    }

    if (monitor_disp_int_energy)
    {
        inte = interacting_energy(psi) / npar;
        printf(energy_fmt, creal(inte));
    }

    if (monitor_disp_overlap_residue)
    {
        over_res = overlap_residual(norb, grid_size, dx, psi->orbitals);
        printf("%10.2E", over_res);
    }

    if (monitor_disp_orb_norm)
    {
        orb_norm = avg_orbitals_norm(norb, grid_size, dx, psi->orbitals);
        printf("%11.7lf", orb_norm);
    }

    if (monitor_disp_coef_norm)
    {
        coef_norm = carrMod(space_dim, psi->coef);
        printf("%11.7lf", coef_norm);
    }

    if (monitor_disp_eig_residue)
    {
        conf_eig_residue = eig_residual(
            mctdhb->multiconfig_space,
            psi->coef,
            psi->hob,
            psi->hint,
            creal(energy));
        printf("%11.7lf", conf_eig_residue);
    }
}

void
append_processed_state(char prefix[], ManyBodyState psi)
{
    uint16_t norb, grid_size;
    uint32_t norb4;
    char     fname[STR_BUFF_SIZE];

    norb = psi->norb;
    grid_size = psi->grid_size;
    norb4 = norb * norb * norb * norb;

    set_output_fname(prefix, TWO_BODY_MATRIX_REC, fname);
    if (new_empty_append_files) remove(fname);
    carr_append_stream(
        fname,
        CPLX_SCIFMT_SPACE_BEFORE,
        CURSOR_POSITION,
        LINEBREAK,
        norb4,
        psi->tb_denmat);
    set_output_fname(prefix, ONE_BODY_MATRIX_REC, fname);
    if (new_empty_append_files) remove(fname);
    cmat_rowmajor_append_stream(
        fname,
        CPLX_SCIFMT_SPACE_BEFORE,
        CURSOR_POSITION,
        LINEBREAK,
        norb,
        norb,
        psi->ob_denmat);
    set_output_fname(prefix, ORBITALS_REC, fname);
    if (new_empty_append_files) remove(fname);
    cmat_rowmajor_append_stream(
        fname,
        CPLX_SCIFMT_SPACE_BEFORE,
        CURSOR_POSITION,
        LINEBREAK,
        norb,
        grid_size,
        psi->orbitals);
}

void
record_raw_state(char prefix[], ManyBodyState psi)
{
    uint16_t norb, grid_size;
    uint32_t space_dim;
    char     fname[STR_BUFF_SIZE];

    norb = psi->norb;
    grid_size = psi->grid_size;
    space_dim = psi->space_dim;

    set_output_fname(prefix, COEFFICIENTS_REC, fname);
    carr_column_txt(fname, CPLX_SCIFMT_SPACE_BEFORE, space_dim, psi->coef);
    set_output_fname(prefix, ORBITALS_REC, fname);
    cmat_txt_transpose(
        fname, CPLX_SCIFMT_SPACE_BOTH, norb, grid_size, psi->orbitals);
}

void
append_timestep_potential(char prefix[], OrbitalEquation eq_desc)
{
    char fname[STR_BUFF_SIZE];

    set_output_fname(prefix, ONE_BODY_POTENTIAL_REC, fname);
    if (new_empty_append_files) remove(fname);

    rarr_append_stream(
        fname,
        REAL_SCIFMT_SPACE_BEFORE,
        CURSOR_POSITION,
        LINEBREAK,
        eq_desc->grid_size,
        eq_desc->pot_grid);
}

void
record_time_interaction(char prefix[], OrbitalEquation eq_desc, uint16_t recn)
{
    uint32_t prop_steps;
    double   t, g;
    char     fname[STR_BUFF_SIZE];
    FILE*    f;

    strcpy(fname, out_dirname);
    strcat(fname, prefix);
    strcat(fname, "_interaction.dat");

    f = open_file(fname, "w");

    g = eq_desc->inter_param(0, eq_desc->inter_extra_args);
    fprintf(f, "%.10E\n", g);

    prop_steps = 0;
    t = 0;
    while (t < eq_desc->tend)
    {
        prop_steps++;
        if (prop_steps % recn)
        {
            t = prop_steps * eq_desc->tstep;
            g = eq_desc->inter_param(t, eq_desc->inter_extra_args);
            fprintf(f, "%.10E\n", g);
        }
    }
    fclose(f);
}

void
record_time_array(char prefix[], double tend, double tstep, uint16_t recn)
{
    uint32_t prop_steps;
    char     fname[STR_BUFF_SIZE];
    FILE*    f;

    strcpy(fname, out_dirname);
    strcat(fname, prefix);
    strcat(fname, "_timesteps.dat");

    f = open_file(fname, "w");
    fprintf(f, "%.6lf\n", 0.0);
    prop_steps = 0;
    while (prop_steps * tstep < tend)
    {
        prop_steps++;
        if (prop_steps % recn == 0) fprintf(f, "%.6lf\n", prop_steps * tstep);
    }
    fclose(f);
}

void
append_mctdhb_parameters(char prefix[], MCTDHBDataStruct mctdhb)
{
    double energy;
    FILE*  f;
    char   fname[STR_BUFF_SIZE];

    set_output_fname(prefix, PARAMETERS_REC, fname);

    if (new_empty_append_files) remove(fname);

    energy = total_energy(mctdhb->state);

    f = open_file(fname, "a");
    fprintf(
        f,
        "%" PRIu16 " %" PRIu16 " %" PRIu16
        " %.10lf %.10lf %.10lf %.2lf %.1lf %.10lf %.10E %.10E\n",
        mctdhb->state->npar,
        mctdhb->state->norb,
        mctdhb->state->grid_size,
        mctdhb->orb_eq->xi,
        mctdhb->orb_eq->xf,
        mctdhb->orb_eq->tstep,
        mctdhb->orb_eq->tend,
        mctdhb->orb_eq->d2coef,
        cimag(mctdhb->orb_eq->d1coef),
        mctdhb->orb_eq->g,
        energy / mctdhb->state->npar);
    fclose(f);
}
