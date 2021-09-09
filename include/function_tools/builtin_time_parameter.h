#ifndef BUILTIN_TIME_PARAMETER_H
#define BUILTIN_TIME_PARAMETER_H

#include "mctdhb_types.h"

double
timeparam_linear_ramp(double t, void* extra_params);

time_dependent_parameter
get_builtin_param_func(char function_name[]);

#endif
