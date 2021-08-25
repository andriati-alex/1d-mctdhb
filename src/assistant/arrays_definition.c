#include "assistant/arrays_definition.h"
#include <stdio.h>
#include <stdlib.h>

#define ERR_MSG_FMT "In function %s for %d elements"

static void
assert_pointer(void* ptr, char err_msg[])
{
    if (ptr == NULL)
    {
        printf("\n\nMEMORY ERROR: malloc fail. Message: %s", err_msg);
        printf("\n\n");
        exit(EXIT_FAILURE);
    }
}

Iarray
get_int_array(unsigned int arr_size)
{
    int* ptr;
    char err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_int_array", arr_size);
    ptr = (int*) malloc(arr_size * sizeof(int));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

unsigned int*
get_uint_array(unsigned int arr_size)
{
    unsigned int* ptr;
    char err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_uint_array", arr_size);
    ptr = (unsigned int*) malloc(arr_size * sizeof(unsigned int));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

uint16_t*
get_uint16_array(unsigned int arr_size)
{
    uint16_t* ptr;
    char err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_uint16_array", arr_size);
    ptr = (uint16_t*) malloc(arr_size * sizeof(uint16_t));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

uint32_t*
get_uint32_array(unsigned int arr_size)
{
    uint32_t* ptr;
    char err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_uint32_array", arr_size);
    ptr = (uint32_t*) malloc(arr_size * sizeof(uint32_t));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

Rarray
get_double_array(unsigned int arr_size)
{
    double* ptr;
    char err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_double_array", arr_size);
    ptr = (double*) malloc(arr_size * sizeof(double));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

Carray
get_dcomplex_array(int arr_size)
{
    doublec* ptr;
    char err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_dcomplex_array", arr_size);
    ptr = (doublec*) malloc(arr_size * sizeof(doublec));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

MKLCarray
get_mklcomplex16_array(int arr_size)
{
    MKL_Complex16* ptr;
    char err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, ERR_MSG_FMT, "get_mklcomplex16_array", arr_size);
    ptr = (MKL_Complex16*) malloc(arr_size * sizeof(MKL_Complex16));
    assert_pointer((void*) ptr, err_msg);
    return ptr;
}

Rmatrix
get_double_matrix(unsigned int nrows, unsigned int ncols)
{
    unsigned int i;
    double** ptr;
    char err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, "%d rows(double*) in get_double_matrix", nrows);
    ptr = (double**) malloc(nrows * sizeof(double*));
    assert_pointer((void*) ptr, err_msg);
    for (i = 0; i < ncols; i++)
    {
        ptr[i] = (double*) malloc(ncols * sizeof(double));
        sprintf(err_msg, "Row %d of size %d in get_double_matrix", i, ncols);
        assert_pointer((void*) ptr, err_msg);
    }
    return ptr;
}

Cmatrix
get_dcomplex_matrix(unsigned int nrows, unsigned int ncols)
{
    unsigned int i;
    doublec** ptr;
    char err_msg[STR_BUFF_SIZE];
    sprintf(err_msg, "%d rows(double complex*) in get_dcomplex_matrix", nrows);
    ptr = (doublec**) malloc(nrows * sizeof(doublec*));
    assert_pointer((void*) ptr, err_msg);
    for (i = 0; i < ncols; i++)
    {
        ptr[i] = (doublec*) malloc(ncols * sizeof(doublec));
        sprintf(err_msg, "Row %d of size %d in get_dcomplex_matrix", i, ncols);
        assert_pointer((void*) ptr, err_msg);
    }
    return ptr;
}

void
destroy_double_matrix(unsigned int nrows, Rmatrix mat)
{
    for (unsigned int i = 0; i < nrows; i++) free(mat[i]);
    free(mat);
}

void
destroy_dcomplex_matrix(unsigned int nrows, Cmatrix mat)
{
    for (unsigned int i = 0; i < nrows; i++) free(mat[i]);
    free(mat);
}
