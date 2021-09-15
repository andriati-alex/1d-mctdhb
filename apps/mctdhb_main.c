#include "assistant/types_definition.h"
#include "assistant/dataio.h"
#include "cpydataio.h"
#include "integrator/driver.h"
#include <string.h>

int
main(int argc, char* argv[])
{
    uint32_t         njobs;
    char             fname[STR_BUFF_SIZE];
    MCTDHBDataStruct problem_set;

    if (argc != 2)
    {
        printf("\n\nPrefix of jobs required\n\n");
        return -1;
    }

    strcpy(fname, inp_dirname);
    strcat(fname, argv[1]);
    strcat(fname, "_mctdhb_parameters");
    njobs = number_of_lines(fname);

    for (uint32_t job_id = 1; job_id <= njobs; job_id++)
    {
        problem_set = full_setup_mctdhb_current_dir(
            argv[1], job_id, COMMON_INP, NULL, NULL, NULL, NULL);
        screen_display_mctdhb_info(problem_set, TRUE, TRUE, TRUE);
        destroy_mctdhb_struct(problem_set);
    }

    return 0;
}
