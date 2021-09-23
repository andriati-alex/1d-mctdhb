/** \file builtin_time_parameter.h
 *
 * \author Alex Andriati - andriati@if.usp.br
 * \date September/2021
 * \brief Module with some example function for time dependent parameter
 *
 * The functions most follow the specific signature \c time_dependent_parameter
 *
 * \see time_dependent_parameter
 */

#ifndef BUILTIN_TIME_PARAMETER_H
#define BUILTIN_TIME_PARAMETER_H

#include "mctdhb_types.h"

double
timeparam_linear_ramp(double t, void* extra_params);

time_dependent_parameter
get_builtin_param_func(char function_name[]);

#endif
