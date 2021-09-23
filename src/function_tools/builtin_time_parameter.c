#include "function_tools/builtin_time_parameter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double
timeparam_linear_ramp(double t, void* params)
{
    double g0, gf, sweep_period;
    g0 = ((double*) params)[0];
    gf = ((double*) params)[1];
    sweep_period = ((double*) params)[2];
    if (t >= sweep_period) return gf;
    return g0 + (gf - g0) * t / sweep_period;
}

time_dependent_parameter
get_builtin_param_func(char funcname[])
{
    if (strcmp(funcname, "linear_ramp") == 0)
    {
        return &timeparam_linear_ramp;
    }
    if (strcmp(funcname, "custom") == 0)
    {
        return NULL;
    }
    printf("\n\nIOERROR : Builtin time function '%s' not found\n\n", funcname);
    exit(EXIT_FAILURE);
}
