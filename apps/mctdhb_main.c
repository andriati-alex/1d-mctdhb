#include "assistant/dataio.h"
#include "assistant/types_definition.h"
#include "cpydataio.h"
#include "integrator/driver.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
            main_struct, 10, out_prefix, main_struct->orb_eq->tend, 10);

        free(main_struct->orb_eq->pot_extra_args);
        free(main_struct->orb_eq->inter_extra_args);
        destroy_mctdhb_struct(main_struct);

        sepline('=', 78, 1, 2);
    }

    printf("\n\nAll jobs done\n\n");
    return 0;
}
