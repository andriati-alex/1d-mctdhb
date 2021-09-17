#include "assistant/dataio.h"
#include "assistant/types_definition.h"
#include "cpydataio.h"
#include "integrator/driver.h"
#include <string.h>

int
main(int argc, char* argv[])
{
    uint32_t         njobs;
    char             fname[STR_BUFF_SIZE];
    MCTDHBDataStruct inp_struct;

    if (argc != 2)
    {
        printf("\n\nPrefix of jobs required\n\n");
        return -1;
    }

    strcpy(fname, inp_dirname);
    strcat(fname, argv[1]);
    strcat(fname, "_mctdhb_parameters.dat");
    njobs = number_of_lines(fname);

    for (uint32_t job_id = 1; job_id <= njobs; job_id++)
    {
        inp_struct = full_setup_mctdhb_current_dir(
            argv[1], job_id, COMMON_INP, NULL, NULL, NULL, NULL);
        screen_display_mctdhb_info(inp_struct, TRUE, TRUE, TRUE);
        integration_driver(
            inp_struct,
            10,
            argv[1],
            inp_struct->orb_eq->tend,
            10,
            MAXIMUM_VERB);
        destroy_mctdhb_struct(inp_struct);
    }

    return 0;
}
