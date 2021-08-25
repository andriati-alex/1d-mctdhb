/** \file data_structure.h
 *
 * \author Alex Andriati - andriati@if.usp.br
 * \date 2021/August
 * \brief Define main data structures
 *
 * The data structures are meant to simplify the amount of arguments needed
 * in function calls. For that, zip a collection of data relevant for each
 * module in this package
 */

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <complex.h>
#include <inttypes.h>
#include <mkl_types.h>
//#include <limits.h>
//#include <math.h>
//#include <mkl_dfti.h>
//#include <omp.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>

/** \brief Double precision pi value */
#define PI            3.141592653589793
#define STR_BUFF_SIZE 128

typedef double complex doublec;
typedef int*           Iarray;
typedef int**          Imatrix;
typedef double*        Rarray;
typedef double**       Rmatrix;
typedef doublec*       Carray;
typedef doublec**      Cmatrix;
typedef MKL_Complex16* MKLCarray;

/** \brief Single particle potential function signature
 *
 * This is the general signature to evaluate one-body potential in
 * the spatial grid for a time-dependent potential at any instant.
 */
typedef void (*single_particle_pot)(
    double t, uint16_t npts, double* pts, void* params, double* grid_pot);

/** \brief Equation setup for single particle states equations */
typedef struct
{
    uint16_t            grid_size;
    double              xi, xf, dx, tstep, tend, d2coef, g;
    doublec             d1coef;
    single_particle_pot pot_func;
} _OrbitalEquation;

typedef _OrbitalEquation* OrbitalEquation;

/** \brief Mappings to track action of creation/destruction operators */
typedef struct
{
    uint32_t *map, mapot, maptt, strideot, stridett;
} _OperatorMappings;

/** \brief Multiconfigurational space setup */
typedef struct
{
    uint16_t          npar, norb;
    uint32_t          dim;
    uint16_t*         hash_table;
    uint32_t*         subspaces_dim;
    _OperatorMappings op_maps;
} _MultiConfiguration;

typedef _MultiConfiguration* MultiConfiguration;

/** \brief Many-body state in Multiconfigurational problem */
typedef struct
{
    uint16_t npar, norb, grid_size;
    uint32_t space_dim;
    Carray   coef, hint, tb_denmat;
    Cmatrix  hob, orbitals, ob_denmat, inv_ob_denmat;
} _State;

typedef _State* State;

/** \brief Extra space to avoid excessive memory allocation in runtime */
typedef struct
{
    State   inp, out;
    Carray  work_dim, work_twobody;
    Cmatrix work_npos, work_norb;
} _Workspace;

/** \brief Master struct with all information for MCTDHB numerical problem */
typedef struct
{
    MultiConfiguration multiconfig_space;
    OrbitalEquation    eq_setup;
    _Workspace         work_struct;
    State              state;
} _MCTDHBDataStruct;

typedef _MCTDHBDataStruct* MCTDHBDataStruct;

#endif
