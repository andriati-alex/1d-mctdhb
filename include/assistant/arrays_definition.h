#ifndef ARRAYS_DEFINITION_H
#define ARRAYS_DEFINITION_H

#include "mctdhb_types.h"

Iarray
get_int_array(uint32_t arr_size);

uint32_t*
get_uint_array(uint32_t arr_size);

uint16_t*
get_uint16_array(uint32_t arr_size);

uint32_t*
get_uint32_array(uint32_t arr_size);

Rarray
get_double_array(uint32_t arr_size);

Carray
get_dcomplex_array(uint32_t arr_size);

MKLCarray
get_mklcomplex16_array(uint32_t arr_size);

Rmatrix
get_double_matrix(uint32_t nrows, uint32_t ncols);

Cmatrix
get_dcomplex_matrix(uint32_t nrows, uint32_t ncols);

void
destroy_double_matrix(uint32_t nrows, Rmatrix mat);

void
destroy_dcomplex_matrix(uint32_t nrows, Cmatrix mat);

#endif
