#include "assistant/dataio.h"
#include "assistant/types_definition.h"
#include "cpydataio.h"
#include "integrator/driver.h"
#include <stdio.h>
#include <string.h>

int
main(int argc, char* argv[])
{
    uint32_t         njobs;
    FILE*            finteg;
    IntegratorType   time_type;
    MCTDHBDataStruct main_struct;

    if (argc != 2)
    {
        printf("\n\nRequire exactly 1 command line argument: "
               "the prefix of files in input directory\n\n");
        return -1;
    }

    screen_display_banner();

    if ((finteg = fopen(integrator_desc_fname, "r")) != NULL)
    {
        jump_comment_lines(finteg, CURSOR_POSITION);
        fscanf(finteg, "%u", &time_type);
        if (time_type == REALTIME)
        {
            monitor_disp_coef_norm = TRUE;
            monitor_disp_orb_norm = TRUE;
            monitor_disp_overlap_residue = TRUE;
            monitor_disp_eig_residue = FALSE;
        }
        fclose(finteg);
    }

    njobs = auto_number_of_jobs(argv[1]);

    for (uint32_t job_id = 1; job_id <= njobs; job_id++)
    {
        printf("\nStarting job %" PRIu32 " ...\n", job_id);
        main_struct = full_setup_mctdhb_current_dir(
            argv[1], job_id, COMMON_INP, NULL, NULL, NULL, NULL);
        screen_display_mctdhb_info(main_struct, TRUE, TRUE, TRUE);
        integration_driver(
            main_struct, 10, argv[1], main_struct->orb_eq->tend, 10);
        destroy_mctdhb_struct(main_struct);
        sepline('=', 78, 1, 2);
    }

    return 0;
}
