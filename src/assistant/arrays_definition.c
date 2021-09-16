#include "assistant/arrays_definition.h"
#include <stdio.h>
#include <stdlib.h>

#define ERR_MSG_FMT "In function %s for %d elements"

static void
assert_pointer(void* ptr, char err_msg[])
{
    if (ptr == NULL)
    {
        printf("\n\nMEMORY ERROR: malloc fail. Message: %s\n\n", err_msg);
        exit(EXIT_FAILURE);
    }
}

Iarray
get_int_array(uint32_t arr_size)
{
    int* ptr;
    char err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_int_array", arr_size);
    ptr = (int*) malloc(arr_size * sizeof(int));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

uint16_t*
get_uint16_array(uint32_t arr_size)
{
    uint16_t* ptr;
    char      err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_uint16_array", arr_size);
    ptr = (uint16_t*) malloc(arr_size * sizeof(uint16_t));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

uint32_t*
get_uint32_array(uint32_t arr_size)
{
    uint32_t* ptr;
    char      err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_uint32_array", arr_size);
    ptr = (uint32_t*) malloc(arr_size * sizeof(uint32_t));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

Rarray
get_double_array(uint32_t arr_size)
{
    double* ptr;
    char    err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_double_array", arr_size);
    ptr = (double*) malloc(arr_size * sizeof(double));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

Carray
get_dcomplex_array(uint32_t arr_size)
{
    dcomplex* ptr;
    char      err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_dcomplex_array", arr_size);
    ptr = (dcomplex*) malloc(arr_size * sizeof(dcomplex));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

MKLCarray
get_mklcomplex16_array(uint32_t arr_size)
{
    MKL_Complex16* ptr;
    char           err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_mklcomplex16_array", arr_size);
    ptr = (MKL_Complex16*) malloc(arr_size * sizeof(MKL_Complex16));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

Rmatrix
get_double_matrix(uint32_t nrows, uint32_t ncols)
{
    uint32_t i;
    double** ptr;
    char     err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, "%d rows(double*) in get_double_matrix", nrows);
    ptr = (double**) malloc(nrows * sizeof(double*));
    assert_pointer((void*) ptr, err_msg);
    for (i = 0; i < nrows; i++)
    {
        ptr[i] = (double*) malloc(ncols * sizeof(double));
        sprintf(err_msg, "Row %d with %d cols in get_double_matrix", i, ncols);
        assert_pointer((void*) ptr[i], err_msg);
    }
    return ptr;
}

Cmatrix
get_dcomplex_matrix(uint32_t nrows, uint32_t ncols)
{
    uint32_t   i;
    dcomplex** ptr;
    char       err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, "%d rows(double complex*) in get_dcomplex_matrix", nrows);
    ptr = (dcomplex**) malloc(nrows * sizeof(dcomplex*));
    assert_pointer((void*) ptr, err_msg);
    for (i = 0; i < nrows; i++)
    {
        ptr[i] = (dcomplex*) malloc(ncols * sizeof(dcomplex));
        sprintf(
            err_msg, "Row %d with %d cols in get_dcomplex_matrix", i, ncols);
        assert_pointer((void*) ptr[i], err_msg);
    }
    return ptr;
}

void
destroy_double_matrix(uint32_t nrows, Rmatrix mat)
{
    for (uint32_t i = 0; i < nrows; i++) free(mat[i]);
    free(mat);
}

void
destroy_dcomplex_matrix(uint32_t nrows, Cmatrix mat)
{
    for (uint32_t i = 0; i < nrows; i++) free(mat[i]);
    free(mat);
}
