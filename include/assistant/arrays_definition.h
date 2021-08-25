#ifndef ARRAYS_DEFINITION_H
#define ARRAYS_DEFINITION_H

#include "data_structures.h"

Iarray
get_int_array(unsigned int arr_size);

unsigned int*
get_uint_array(unsigned int arr_size);

uint16_t*
get_uint16_array(unsigned int arr_size);

uint32_t*
get_uint32_array(unsigned int arr_size);

Rarray
get_double_array(unsigned int arr_size);

Carray
get_dcomplex_array(int arr_size);

MKLCarray
get_mklcomplex16_array(int arr_size);

Rmatrix
get_double_matrix(unsigned int nrows, unsigned int ncols);

Cmatrix
get_dcomplex_matrix(unsigned int nrows, unsigned int ncols);

void
destroy_double_matrix(unsigned int nrows, Rmatrix mat);

void
destroy_dcomplex_matrix(unsigned int nrows, Cmatrix mat);

#endif
