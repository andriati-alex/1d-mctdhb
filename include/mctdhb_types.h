/** \file mctdhb_types.h
 *
 * \author Alex Andriati - andriati@if.usp.br
 * \date 2021/August
 * \brief Define main data structures
 *
 * The data structures are meant to simplify the amount of arguments needed
 * in function calls. For that, zip a collection of data relevant for each
 * module in this package
 */

#ifndef MCTDHB_TYPES_H
#define MCTDHB_TYPES_H

#include <complex.h>
#include <inttypes.h>
#include <mkl_dfti.h>
#include <mkl_types.h>

/** \brief Double precision pi value */
#define PI 3.141592653589793
/** \brief Maximum length of stack strings */
#define STR_BUFF_SIZE 128
/** \brief Maximum grid size supported */
#define MAX_GRID_SIZE 2048
/** \brief Maximum number of single particle states(orbitals) supported */
#define MAX_ORBITALS 30
/** \brief Maximum number of single particle states(orbitals) supported */
#define MAX_PARTICLES 10000
/** \brief Maximum number of iterations for Lanczos integrator **/
#define MAX_LANCZOS_ITER 10

typedef double complex dcomplex;
typedef int*           Iarray;
typedef int**          Imatrix;
typedef double*        Rarray;
typedef double**       Rmatrix;
typedef dcomplex*      Carray;
typedef dcomplex**     Cmatrix;
typedef MKL_Complex16* MKLCarray;

/** \brief Custom boolean type with enum */
typedef enum
{
    FALSE,
    TRUE
} Bool;

/** \brief Time integration type for ground state or dynamics calculation */
typedef enum
{
    IMAGTIME,
    REALTIME
} IntegratorType;

/** \brief Coefficients integration type */
typedef enum
{
    LANCZOS,
    RUNGEKUTTA
} CoefIntegrator;

/** \brief Orbital integration type */
typedef enum
{
    FULLSTEP_RUNGEKUTTA,
    SPLITSTEP,
} OrbIntegrator;

/** \brief Complementary information if OrbItegrator = FULLSTEP_RUNGEKUTTA */
typedef enum
{
    DVR,
    SPECTRAL,
    FINITEDIFF
} OrbDerivative;

/** \brief Global Runge-Kutta order for all methods using it */
typedef enum
{
    RK2 = 2,
    RK4 = 4,
    RK5 = 5
} RungeKuttaOrder;

/** \brief Specific for finite differences on how to set boundaries */
typedef enum
{
    ZERO_BOUNDS,
    PERIODIC_BOUNDS
} BoundaryCondition;

/** \brief Single particle potential function signature
 *
 * This is the general signature to evaluate one-body time-dependent
 * potential in spatial grid at any time instant
 *
 * \param[in] t time instant
 * \param[in] npts number of points in the grid
 * \param[in] pts array with grid points
 * \param[in] params extra parameters defined by the client
 * \param[out] grid_pot potential values at grid points
 */
typedef void (*single_particle_pot)(
    double t, uint16_t npts, double* pts, void* params, double* grid_pot);

/** \brief Time dependent parameter function signature
 *
 * Signature to declare time-dependent parameters. Only for interaction
 *
 * \param[in] t time instant
 * \param[in] params extra needed parameters from client
 */
typedef double (*time_dependent_parameter)(double t, void* params);

/** \brief Spatial equation setup for orbitals */
typedef struct
{
    char                     eq_name[STR_BUFF_SIZE];
    uint16_t                 norb, grid_size;
    BoundaryCondition        bounds;
    double                   t, xi, xf, dx, tstep, tend, g, d2coef;
    dcomplex                 d1coef, prop_dt, time_fac;
    Rarray                   grid_pts;
    Rarray                   pot_grid;
    void*                    pot_extra_args;
    void*                    inter_extra_args;
    single_particle_pot      pot_func;
    time_dependent_parameter inter_param;
} _OrbitalEquation;

typedef _OrbitalEquation* OrbitalEquation;

/** \brief Mappings to track action of creation/destruction operators */
typedef struct
{
    uint32_t *map, *mapot, *maptt, *strideot, *stridett;
} _OperatorMappings;

typedef _OperatorMappings* OperatorMappings;

/** \brief Multiconfigurational space setup */
typedef struct
{
    uint16_t         npar, norb;
    uint32_t         dim;
    uint16_t*        hash_table;
    uint32_t*        subspaces_dim;
    OperatorMappings op_maps;
} _MultiConfiguration;

typedef _MultiConfiguration* MultiConfiguration;

/** \brief Many-body state in Multiconfigurational problem */
typedef struct
{
    uint16_t npar, norb, grid_size;
    uint32_t space_dim;
    Carray   coef, hint, tb_denmat;
    Cmatrix  hob, orbitals, ob_denmat, inv_ob_denmat;
} _ManyBodyState;

typedef _ManyBodyState* ManyBodyState;

typedef struct
{
    uint16_t iter;
    uint32_t space_dim;
    Cmatrix  lanczos_vectors;
    Carray   decomp_diag, decomp_offd, coef_lspace, transform, hc, reortho;
    Rarray   lapack_diag, lapack_offd, lapack_eigvec;
} _WorkspaceLanczos;

typedef _WorkspaceLanczos* WorkspaceLanczos;

typedef struct
{
    uint16_t               norb, grid_size;
    OrbDerivative          orb_der_method;
    Bool                   impr_ortho;
    Carray                 dvr_mat;
    Carray                 cn_upper, cn_lower, cn_mid, cn_rhs;
    Cmatrix                orb_work1, orb_work2;
    void*                  extern_work;
    DFTI_DESCRIPTOR_HANDLE fft_desc;
    Rarray                 fft_freq;
    Carray                 fft_hder_exp;
} _OrbitalWorkspace;

typedef _OrbitalWorkspace* OrbitalWorkspace;

typedef struct
{
    WorkspaceLanczos lan_work;
    void*            extern_work;
} _CoefWorkspace;

typedef _CoefWorkspace* CoefWorkspace;

/** \brief Master struct with all information for MCTDHB numerical problem */
typedef struct
{
    IntegratorType     integ_type;
    CoefIntegrator     coef_integ_method;
    OrbIntegrator      orb_integ_method;
    OrbDerivative      orb_der_method;
    RungeKuttaOrder    rk_order;
    MultiConfiguration multiconfig_space;
    OrbitalEquation    orb_eq;
    CoefWorkspace      coef_workspace;
    OrbitalWorkspace   orb_workspace;
    ManyBodyState      state;
} _MCTDHBDataStruct;

typedef _MCTDHBDataStruct* MCTDHBDataStruct;

#endif
