#include "assistant/dataio.h"
#include "assistant/types_definition.h"
#include "cpydataio.h"
#include "integrator/driver.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MONITOR_CONFIG_FNAME "mctdhb_monitor.conf"

static uint8_t  monitor_nsteps = 10;
static uint16_t record_nsteps = 10;

static void
report_warn_monitoring_file(uint8_t param_num)
{
    printf(
        "\nWARNING: problem reading param %" PRIu8 " from file %s, "
        "skipping the rest of file and using defaults\n\n",
        param_num,
        MONITOR_CONFIG_FNAME);
}

static void
config_monitoring()
{
    int     scan_status;
    FILE*   f;
    Bool    auto_check;
    uint8_t e_digits, params_read;
    double  eig_residue_tol, overlap_threshold;

    if ((f = fopen(MONITOR_CONFIG_FNAME, "r")) == NULL) return;

    params_read = 0;

    jump_comment_lines(f, CURSOR_POSITION);
    scan_status = fscanf(f, "%" SCNu8, &monitor_nsteps);
    params_read++;
    if (scan_status != 1)
    {
        report_warn_monitoring_file(params_read);
        return;
    }

    jump_comment_lines(f, CURSOR_POSITION);
    scan_status = fscanf(f, "%u", &auto_check);
    params_read++;
    if (scan_status != 1)
    {
        report_warn_monitoring_file(params_read);
        return;
    }
    set_autoconvergence_check(auto_check);

    jump_comment_lines(f, CURSOR_POSITION);
    scan_status = fscanf(f, "%" SCNu8, &e_digits);
    params_read++;
    if (scan_status != 1)
    {
        report_warn_monitoring_file(params_read);
        return;
    }
    set_energy_convergence_digits(e_digits);

    jump_comment_lines(f, CURSOR_POSITION);
    scan_status = fscanf(f, "%lf", &eig_residue_tol);
    params_read++;
    if (scan_status != 1)
    {
        report_warn_monitoring_file(params_read);
        return;
    }
    set_energy_convergence_eig_residual(eig_residue_tol);

    jump_comment_lines(f, CURSOR_POSITION);
    scan_status = fscanf(f, "%" SCNu16, &record_nsteps);
    params_read++;
    if (scan_status != 1)
    {
        report_warn_monitoring_file(params_read);
        return;
    }

    jump_comment_lines(f, CURSOR_POSITION);
    scan_status = fscanf(f, "%lf", &overlap_threshold);
    params_read++;
    if (scan_status != 1)
    {
        report_warn_monitoring_file(params_read);
        return;
    }
    set_overlap_residue_threshold(overlap_threshold);
}

int
main(int argc, char* argv[])
{
    uint32_t         njobs;
    FILE*            integ_desc_file;
    char             job_fmt[STR_BUFF_SIZE], out_prefix[STR_BUFF_SIZE];
    IntegratorType   time_type;
    MCTDHBDataStruct main_struct;

    omp_set_num_threads(omp_get_max_threads() / 2);

    if (argc != 2)
    {
        printf("\n\nRequire exactly 1 command line argument: "
               "the prefix of files in input directory\n\n");
        return -1;
    }

    screen_display_banner();
    config_monitoring();

    // Adjust fields to print on screen according to time type
    if ((integ_desc_file = fopen(integrator_desc_fname, "r")) != NULL)
    {
        jump_comment_lines(integ_desc_file, CURSOR_POSITION);
        fscanf(integ_desc_file, "%u", &time_type);
        if (time_type == REALTIME)
        {
            monitor_disp_coef_norm = TRUE;
            monitor_disp_orb_norm = TRUE;
            monitor_disp_overlap_residue = TRUE;
            monitor_disp_eig_residue = FALSE;
        }
        fclose(integ_desc_file);
    }

    njobs = auto_number_of_jobs(argv[1]);

    for (uint32_t job_id = 1; job_id <= njobs; job_id++)
    {
        printf("\nStarting job %" PRIu32 " ...\n", job_id);

        sprintf(job_fmt, "_job%" PRIu32, job_id);
        strcpy(out_prefix, argv[1]);
        strcat(out_prefix, job_fmt);

        main_struct = full_setup_mctdhb_current_dir(
            argv[1], job_id, COMMON_INP, NULL, NULL, NULL, NULL);

        screen_display_mctdhb_info(main_struct, TRUE, TRUE, TRUE);

        integration_driver(
            main_struct,
            record_nsteps,
            out_prefix,
            main_struct->orb_eq->tend,
            monitor_nsteps);

        free(main_struct->orb_eq->pot_extra_args);
        free(main_struct->orb_eq->inter_extra_args);
        destroy_mctdhb_struct(main_struct);

        sepline('=', 78, 1, 2);
    }

    printf("\n\nAll jobs done\n\n");
    return 0;
}
